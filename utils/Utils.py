import os
import stat
import copy
import math
import random
import subprocess
import configparser
import multiprocessing
import torch as th
import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import label
from itertools import permutations
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdMolTransforms import GetDihedralDeg, GetDihedralRad, SetBondLength, SetDihedralRad
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs import BulkTanimotoSimilarity


def SetSeed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def GetConfigs(config_path):
    configs = configparser.ConfigParser()
    configs.read(config_path)
    return configs


def MaxIteration(EDgrid, assign_iter, vol=1000):
    if assign_iter:
        iteration = int(assign_iter)
    else:
        grid_num = EDgrid[EDgrid[:,3]>0].shape[0]
        iteration = min(max(round(grid_num / vol), 2), 5)
    return iteration


def Segment3DGrid(grid):
    labeled_array, num_features = label(grid)
    regions = []
    for label_num in range(1, num_features + 1):
        region = [(x, y, z) for x, y, z in zip(*np.where(labeled_array == label_num))]
        regions.append(region)       
    return regions


def GetAtoms(pdbf):
    atomlines = []
    for line in open(pdbf).readlines():
        if line[:6] in ["ATOM  ", "HETATM"]:
            atomlines.append(line)
    return atomlines


def GetRatom(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ["R","*"]:
            break
    return atom


def BondNeighbors(mol, a1, a2):
    """
    if x-a1-a2-y, get the neighbor atoms x and y of a1 and a2
    """
    a1n = [n.GetIdx() for n in mol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2]
    a2n = [n.GetIdx() for n in mol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1]
    return a1n ,a2n


def CalcAlpha(mol, a1, a2):
    """
    x and y are neighbors of a1 and a2, alpha is the angle of anta2(s[0], s[1]), where s is sum of cosxy and sinxy
    """
    a1n, a2n = BondNeighbors(mol, a1, a2)
    s = []
    conf = mol.GetConformer()
    for x in a1n:
        for y in a2n:
            angle = GetDihedralDeg(conf, x, a1, a2, y)
            cosxy = math.cos(math.pi*angle/180)
            sinxy = math.sin(math.pi*angle/180)
            s.append([cosxy, sinxy])
    s = 1000000*np.sum(np.array(s), 0)
    alpha = -math.atan2(s[0], s[1])#顺时针度数
    return alpha


def ConnectMols(mol1, mol2, atom1, atom2):
    """
    function borrowed from https://github.com/molecularsets/moses/blob/master/moses/baselines/combinatorial.py
    """
    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()
    emol.AddBond(neighbor1_idx,
                 neighbor2_idx + mol1.GetNumAtoms(),
                 order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    if neighbor1_idx > atom1_idx:
        update_nei1_idx = neighbor1_idx - 1
    else:
        update_nei1_idx = neighbor1_idx
    if neighbor2_idx > atom2_idx:
        update_nei2_idx = mol1.GetNumAtoms() - 1 + neighbor2_idx - 1
    else:
        update_nei2_idx = mol1.GetNumAtoms() - 1 + neighbor2_idx
    return mol, update_nei1_idx, update_nei2_idx


def RotateFragment(core, frag, torsion_angle, Set_bond_length):
    atom1 = GetRatom(core)
    atom2 = GetRatom(frag)
    rdMolAlign.AlignMol(frag, core, atomMap=[(atom2.GetIdx(), atom1.GetNeighbors()[0].GetIdx()), (atom2.GetNeighbors()[0].GetIdx(), atom1.GetIdx())])
    newmol, a1, a2 = ConnectMols(core, frag, atom1, atom2)
    Chem.SanitizeMol(newmol)
    conf = newmol.GetConformer()
    x = [n.GetIdx() for n in newmol.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2][0]
    y = [n.GetIdx() for n in newmol.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1][0]
    angle = (-CalcAlpha(newmol, a1, a2)+torsion_angle)
    alpha_ori = GetDihedralRad(conf, x, a1, a2, y)
    alpha = angle + alpha_ori
    #set dihedral, change bond length
    SetDihedralRad(conf, x, a1, a2, y, alpha)
    SetBondLength(conf, a1, a2, Set_bond_length)
    return newmol


def CleanMol(mol):
    copy_mol = copy.deepcopy(mol)
    copy_mol = Chem.AddHs(copy_mol,addCoords = True)
    for a in copy_mol.GetAtoms():
        if a.GetSymbol() in ["R", "*", "H"]:
            a.SetAtomicNum(1)
    return copy_mol


def CheckAtomCol(mol):
    mol = CleanMol(mol)
    dm = Chem.Get3DDistanceMatrix(mol)

    R = []
    n = mol.GetNumAtoms()
    for a1 in mol.GetAtoms():
        rvdw1 = Chem.GetPeriodicTable().GetRvdw(a1.GetAtomicNum())
        for a2 in mol.GetAtoms():
            rvdw2 = Chem.GetPeriodicTable().GetRvdw(a2.GetAtomicNum())
            R.append((rvdw1 + rvdw2) * 0.6)
    R = np.array(R).reshape(n, n)

    rings = mol.GetRingInfo().AtomRings()
    am = Chem.GetAdjacencyMatrix(mol) + np.eye(n, dtype = int)
    for r in rings:
        row = [i[0] for i in permutations(r,2)]
        col = [i[1] for i in permutations(r,2)]
        am[row, col] = 1
    
    coll = dm[np.logical_not(am)] < R[np.logical_not(am)]
    if coll.sum() > 0:
        return False
    else:
        return True


def CalculateAngleBetweenVectors(vector1, vector2):
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle_in_radians = np.arccos(cosine_angle)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees


def Minimize(input_sdf, configs):
    smina = "./utils/smina.static"
    if not os.access(smina, os.X_OK):
        os.chmod(smina, os.stat(smina).st_mode | stat.S_IXUSR)
    pdb = configs.get('sample', 'receptor')
    tmpdir = os.path.join(configs.get('sample', 'output_dir'), 'tmp')
    output_sdf = os.path.join(tmpdir, input_sdf.split("/")[-1].split(".sdf")[0]+"_lig_score.sdf")
    command = f"{smina} -r {pdb} -l {input_sdf} -o {output_sdf} --minimize --minimize_iters 1000 --cpu 1"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return output_sdf


def MarkAxialBonds(mol):
    """
    detect axial bonds
    """
    TargetSmiles = "C1****C1"
    mol = Chem.AddHs(mol,addCoords=True)
    substructure = Chem.MolFromSmarts(TargetSmiles)
    matches = mol.GetSubstructMatches(substructure)
    Positions = np.array(mol.GetConformer().GetPositions())
    for match in matches: 
        RingPositions = Positions[np.array(match).reshape(-1)]
        x, y, z = RingPositions.T
        A = np.column_stack((x, y, np.ones_like(x)))
        coefficients, _, _, _ = lstsq(A, z)
        a, b, c = coefficients
        NormalVector = np.array([a,b,-1])
        for i in match:
            Atom = mol.GetAtoms()[i]
            AtomPosition = Positions[i]
            angles = []
            for Neighbor in Atom.GetNeighbors():
                if Neighbor.GetAtomicNum() == 1:
                    NeighborPosition = Positions[Neighbor.GetIdx()]
                    BondVector = NeighborPosition - AtomPosition
                    angle = CalculateAngleBetweenVectors(BondVector,NormalVector)
                    angle = abs(angle-90)
                    angles.append(angle)
                    if angle > 60:
                        Neighbor.SetIsotope(2)
            if len(angles) == 2:
                AxialIndex = np.argmax(angles)
                index = 0
                for Neighbor in Atom.GetNeighbors():
                    if Neighbor.GetAtomicNum() == 1:
                        if AxialIndex == index:
                            Neighbor.SetIsotope(2)
                        index+=1    
    return copy.deepcopy(mol)


def MolOpt(mols, configs):
    tmpdir = os.path.join(configs.get('sample', 'output_dir'), 'tmp')
    num_cpu = int(configs.get('sample', 'num_cpu'))
    if len(mols) > num_cpu:
        pass
    else:
        num_cpu = len(mols)
    chunksize = math.ceil(len(mols) / num_cpu)
    mol_chunks = [mols[i:i+chunksize] for i in range(0, len(mols), chunksize)]
    input_sdfs = []
    for index,mols in enumerate(mol_chunks):
        writer = Chem.SDWriter(os.path.join(tmpdir,str(index)+".sdf"))
        input_sdfs.append(os.path.join(tmpdir,str(index)+".sdf"))
        for mol in mols:
            writer.write(mol)
        writer.close()
    pool = multiprocessing.Pool()
    results = [pool.apply_async(Minimize, args=[args, configs]) for args in input_sdfs]
    pool.close()
    pool.join()
    optimized_molecules = []
    for result in results:
        output_sdf = result.get() 
        for mol in Chem.SDMolSupplier(output_sdf):
            if float(mol.GetProp("minimizedAffinity"))<0:
                optimized_molecules.append(mol)
        os.remove(output_sdf)
    for input_sdf in input_sdfs:
        os.remove(input_sdf)
    return optimized_molecules


def DistFromMol2Coords(mol, coord_array):
    mol_xyz = mol.GetConformer().GetPositions()
    distances = np.linalg.norm(coord_array - mol_xyz[:, np.newaxis], axis=2)
    min_distances = np.min(distances, axis=0)

    return np.min(min_distances)


def SortMolByDist(mols, coord_array):
    all_min_distances = [DistFromMol2Coords(mol, coord_array) for mol in mols]
    sorted_mols = [mol for _, mol in sorted(zip(all_min_distances, mols), key=lambda x: -np.max(x[0]))]
    return sorted_mols


def GetResCoords(res, recf):
    # get residue coordinates
    # e.g. res: "A809 sidechain"
    configs = GetConfigs()
    recf = configs.get('sample', 'receptor')
    chainid = res.split()[0][0]
    resnum = res.split()[0][1:]
    bbcoords = []; sccoords = []
    atmlines = GetAtoms(recf)
    for line in atmlines:
        if line[21] == chainid and line[22:26].strip() == resnum:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if line[12:16].strip() in ["N", "CA", "C", "O"]:
                bbcoords.append([x, y, z])
            else:
                sccoords.append([x, y, z])
    res_coords = np.array(bbcoords) if res.split()[1] == "backbone" else np.array(sccoords)
    return res_coords


class MolBucket:
    """
    put diverse molecules derived from the same core into the bucket with limited size
    """
    def __init__(self, bucket_size, sim_threshold):
        self.bucket_size = bucket_size
        self.sim_threshold = sim_threshold

    def update(self, idx, query_mol, bucket_fps):
        """
        Calculate ECFP similiarity and decide whether to add the query into the bucket
        """
        qfp = GetMorganFingerprint(query_mol, 2)
        # tanimoto similarity profile
        sim = BulkTanimotoSimilarity(qfp, list(bucket_fps.values()))
        if sorted(sim)[-1] <= self.sim_threshold:
            bucket_fps.update({idx: qfp})
        
        return bucket_fps

    def add(self, ranked_mols):
        """
        Add mols on the ranked list sequentially into the bucket
        """
        bucket_fps = {0: GetMorganFingerprint(Chem.RemoveHs(ranked_mols[0]), 2)}
        for idx, m in enumerate(ranked_mols):
            if m:
                m = Chem.RemoveAllHs(m)
                if len(bucket_fps) < self.bucket_size:
                    bucket_fps = self.update(idx, m, bucket_fps)
                else:
                    break
        
        return [ranked_mols[idx] for idx in bucket_fps]