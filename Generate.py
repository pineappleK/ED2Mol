import os
import copy
import random
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch as th
from tqdm import tqdm
from rdkit import Chem
from traceback import print_exc
from scipy.spatial import KDTree
from models.ED_Generator import ED_generator
from models.Mol_Generator import GrowthPointPredictionModel, TorsionAnglePredictionModel
from utils.Score import Scorer
from utils.EDExtract import FcalcPdb
from utils.CorePlacement import RetainMaxValues, GetClusterCenters, CoreId
from utils.DataProcessing import DatasetGrowthPoint, GrowthPointFilter, Attach
from utils.Utils import *


def LoadModels(configs):
    gppm_path = configs.get('model', 'GPPM')
    tapm_path = configs.get('model', 'TAPM')
    device_type = configs.get('sample', 'device_type')
    device = th.device(device_type)

    GPPM = GrowthPointPredictionModel().to(device)
    GPPM.load_state_dict(th.load(gppm_path, map_location=th.device(device_type)))
    GPPM.eval()

    TAPM = TorsionAnglePredictionModel().to(device)
    pretrained_dict = th.load(tapm_path, map_location=th.device(device_type))
    model_dict = TAPM.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    TAPM.load_state_dict(model_dict)
    TAPM.eval()

    return GPPM, TAPM


def LoadLibs(configs):
    cores = Chem.SDMolSupplier(configs.get('lib', 'Cores'))
    cores = [mol for mol in cores]
    frags = Chem.SDMolSupplier(configs.get('lib', 'Frags'))
    frags = [mol for mol in frags]
    return cores, frags


def LigEDgen(configs, output_dir, tmp_dir):
    """
    build ligand ED
    recpdb: receptor PDB file
    center_x, center_y, center_z: pocket center
    """
    pocgrid, pocED = FcalcPdb(configs.get('sample', 'receptor'), float(configs.get('sample', 'x')), 
                     float(configs.get('sample', 'y')), float(configs.get('sample', 'z')), tmp_dir)
    device = th.device("cpu")
    ED_gen = ED_generator().to(device)
    ED_gen.load_state_dict(th.load(configs.get('model', 'Pocket2ED'),map_location=th.device(device)))
    ED_gen.eval()

    pocED = th.tensor(pocED).reshape(1,1,48,48,48)
    ligED = ED_gen.forward(pocED.float())[2].detach()
    ligED[ligED<0.1]=0
    ligED = ligED.reshape(48,48,48)
    gengrid = np.hstack([pocgrid, ligED.reshape((-1,1))])
    pocED = th.where(pocED.float().reshape(48,48,48) > 0, th.tensor(0), th.tensor(1))
    ligED = ligED*pocED
    segments = Segment3DGrid(copy.deepcopy(ligED.reshape(48,48,48)))
    max_list = max(segments, key=len)
    liggridED = th.zeros_like(ligED)
    for x,y,z in max_list:
        liggridED[x,y,z] = ligED[x,y,z]
    gengrid = np.hstack([pocgrid, np.array(liggridED.reshape(-1,1))])
    np.save(os.path.join(output_dir, "ligED.npy"), gengrid)
    
    with open(os.path.join(output_dir, "ligED.pdb"), 'w') as f:
        for temp in gengrid:
            coord = temp[:3]
            intensity = temp[3]
            if intensity.item() <= 0.0:
                continue
            intensity = intensity.item()
            f.write(f'ATOM  {10000:>4}  X   MOL     1    {coord[0]:>7.3f} {coord[1]:>7.3f} {coord[2]:>7.3f} 0.00  {intensity:>6.2f}      MOL\n')
        f.write('END\n')
        
    return gengrid

    
def MolGrow(EDgrid, configs):
    '''
    Main function for generating molecules.
    '''
    initial_cores, frags = LoadLibs(configs)
    GPPM, TAPM = LoadModels(configs)
    scorer = Scorer(configs)
    ref_core = configs.get('sample', 'reference_core')
    tolerence = float(configs.get('sample','tolerence')) if configs.has_option('sample','tolerence') else 0.0

    try:
        growth_point_pos = [float(configs.get('sample', 'grow_x')), float(configs.get('sample', 'grow_y')), float(configs.get('sample', 'grow_z'))]
    except:
        growth_point_pos = None
    
    iteration = MaxIteration(EDgrid, configs.get('sample', 'iteration'))
    print("**********A {}-step Molecule Generation**********".format(str(iteration)))

    generate_molecules = []
    for step in range(iteration):
        pop_mols = []
        if not step:
            if not ref_core:
                EDvalue = th.tensor(EDgrid)[:,-1].reshape(1,48,48,48)
                max_grid = RetainMaxValues(EDvalue)
                merged_coords_max_grid = np.hstack((EDgrid[:, :-1], max_grid.numpy().reshape(-1, 1)))
                cluster_centers = GetClusterCenters(merged_coords_max_grid)
                for core in tqdm(initial_cores[:], desc="Placing Initial Fragments"):
                    core_enumeration = CoreId(core, cluster_centers)
                    core_mol_list = scorer.QScore(core_enumeration, EDgrid, tol=tolerence)
                    pop_mols.append(core_mol_list[:1])
            else:
                CheckAtomCol(Chem.SDMolSupplier(ref_core)[0])
                pop_mols = [[c] for c in tqdm(Chem.SDMolSupplier(ref_core), desc="Placing Reference Fragments") if c]
        
        else:
            for core in tqdm(cores, desc="Growing Fragments--Step {}".format(str(step+1))):
                try:
                    cleancores = Attach(TAPM, frags, core, EDgrid, configs)
                    scoredcores = scorer.QScore(cleancores, EDgrid, tol=tolerence)
                    pop_mols.append(scoredcores)
                except:
                    print_exc()
        
        grow_mols = []; candidate_mols = []; scored_mols = []
        sim_threshold = float(configs.get('sample', 'sim_threshold')) if step else 1.0
        bucket_size = float(configs.get('sample', 'bucket_size'))

        for core_mol_list in pop_mols:
            # predict growing point
            core_mol_list = [MarkAxialBonds(CleanMol(mol)) for mol in core_mol_list]
            cores_dataset = DatasetGrowthPoint(core_mol_list, EDgrid[:,-1], KDTree(EDgrid[:,:3]))
            gpcores, nogpcores = GrowthPointFilter(configs, core_mol_list, GPPM, cores_dataset, step, growth_point_pos)
            bucket = MolBucket(bucket_size=bucket_size, sim_threshold=sim_threshold)
            interact_res = configs.get('sample', 'key_res')
            if nogpcores:
                nogpcores_keep = bucket.add(nogpcores)
                if interact_res:
                    nogpcores_keep = scorer.InteractionScore(nogpcores_keep)
                    nogpcores_keep = [_c for _c,score in nogpcores_keep if score == 1 ]
                scored_mols.extend(nogpcores_keep)
            
            if gpcores:
                gpcores_keep = bucket.add(gpcores)
                if interact_res:
                    gpcores_score = scorer.InteractionScore(gpcores_keep)
                    for _c, score in gpcores_score:
                        if score:
                            grow_mols.append(_c)
                        else:
                            candidate_mols.append(_c)
                else:
                    grow_mols.extend(gpcores_keep)
                scored_mols.extend(gpcores_keep)

        if step:
            keep = int(configs.get('sample', 'retain_mols_num')) if configs.has_option('sample','retain_mols_num') else 100
        else:
            keep = int(configs.get('sample', 'retain_cores_num')) if configs.has_option('sample','retain_cores_num') else 100

        if len(grow_mols) > keep:
            cores = random.sample(grow_mols, keep)
        else:
            cores = grow_mols
            if len(candidate_mols) > keep - len(grow_mols):
                if interact_res:
                    query_coords = GetResCoords(interact_res)
                    cores.extend(SortMolByDist(candidate_mols, query_coords)[:keep-len(grow_mols)])
                else:
                    cores.extend(candidate_mols[:keep-len(grow_mols)])
            else:
                cores.extend(candidate_mols)
        
        if step:
            generate_molecules.extend(scored_mols)
    
    num_molecules = int(configs.get('sample','output_mols_num')) if configs.has_option('sample','output_mols_num') else 1000
    selected_molecules = random.sample(generate_molecules, num_molecules) if len(generate_molecules) >= num_molecules else generate_molecules

    record = Chem.SDWriter(os.path.join(configs.get('sample', 'output_dir'), f"output.sdf"))
    for idx,_c in enumerate(selected_molecules):
        _c.SetProp("generation number", str(idx))
        record.write(CleanMol(_c))
    record.close()
    print("**********Complete Molecule Generation**********")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample.yml')
    args = parser.parse_args()
    configs = GetConfigs(args.config)

    SetSeed(int(configs.get('sample', 'seed')))
    output_dir = configs.get('sample', 'output_dir')
    tmp_dir = os.path.join(output_dir, 'tmp')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    ligED = LigEDgen(configs, output_dir, tmp_dir)
    MolGrow(ligED, configs)
    shutil.rmtree(tmp_dir)
