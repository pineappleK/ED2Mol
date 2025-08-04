import os
import shutil
import numpy as np
from rdkit import Chem
from scipy.spatial import KDTree
from plip.structure.preparation import PDBComplex
from concurrent.futures import ProcessPoolExecutor, as_completed


class Scorer:
    def __init__(self, configs):
        self.protf = configs.get('sample', 'receptor')
        self.query_res = configs.get('sample', 'key_res')
        self.tmpdir = os.path.join(configs.get('sample', 'output_dir'), 'tmp')
        self.num_cpu = int(configs.get('sample', 'num_cpu'))
    
    def QScore(self, mols, density, tol):
        scoredmols = []

        voxpoints = density[:, :-1]
        T = KDTree(voxpoints)
        for mol in mols:
            z = []; ha_coors = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ["R", "*", "H"]:
                    z.append(atom.GetAtomicNum())
                    ha_coors.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))
            z = np.array(z)
            dis, idx = T.query(ha_coors, k = 1)
            rho = density[:, -1][idx]

            if (rho == 0).sum() / rho.size > tol:
                q_score = 0
            else:
                q_score = len(ha_coors)*np.sum(z*rho)/np.sum(z)
                mol.SetProp('qscore', str(q_score))
                scoredmols.append(mol)
        
        scoredmols = sorted(scoredmols, key=lambda x: float(x.GetProp('qscore')), reverse=True)
        return scoredmols
    
    def InteractionMatch(self, compf, query_res, lig_idx):
        """
        detect if the molecule interacts with query residue
        """
        # plip analysis
        mol = PDBComplex()
        mol.load_pdb(compf) 
        bsid = 'UNL:Z:1'
        mol.analyze()
        interaction = mol.interaction_sets[bsid]

        # list all side chains or backbones of interacting residues
        interaction_res = []
        for i in interaction.all_itypes:
            i_info = i._asdict()
            if i_info['restype'] not in ['LIG', 'HOH']:
                if i_info.get('sidechain', True):
                    portion = 'sidechain'
                else:
                    portion = 'backbone'
                res = ''.join([str(i.reschain), str(i.resnr)])
                interaction_res.append(res + " " + portion)
        interaction_res = list(set(interaction_res))

        if query_res not in interaction_res:
            return lig_idx, False
        return lig_idx, True

    def InteractionScore(self, ligs):
        """
        label each molecule in the population
        return nested array of [ligand_index, match_query]
        [[0, True], [1, False]...]
        """
        reclines = [l for l in open(self.protf).readlines() if l[:6] in ["ATOM  ", "HETATM"]]
        lig_hit = []
        plipdir = os.path.join(self.tmpdir, "plip")
        os.mkdir(plipdir)

        with ProcessPoolExecutor(max_workers=self.num_cpu) as ex:
            futures = []
            for lig_idx, l in enumerate(ligs):
                lname = str(lig_idx)
                outfname = os.path.join(plipdir, lname + ".pdb")
                l = Chem.RemoveHs(l)
                ligpdb = Chem.PDBWriter(outfname)
                ligpdb.write(l)
                ligpdb.close()
                ligatoms = [l for l in open(outfname).readlines() if l[:6] == "HETATM"]
                ligconect = [l for l in open(outfname).readlines() if l[:6] == "CONECT"]

                with open(outfname, "w") as comp:
                    for _, l in enumerate(ligatoms):
                        comp.writelines(l[:21] + "Z" + l[22:])
                    for _, l in enumerate(reclines):
                        serial = _ + len(ligatoms) + 1
                        comp.writelines(l[:6] + (5-len(str(serial)))*" " + str(serial) + l[11:])
                    comp.writelines(ligconect)
                    comp.writelines("END")
                
                futures.append(ex.submit(self.InteractionMatch, outfname, self.query_res, lig_idx))
             
            for future in as_completed(futures):
                lig_idx, hit = future.result()
                lig_hit.append([lig_idx, hit])

            mol_plipscore = []
            for idx, hit in lig_hit:
                plipscore = 1 if hit else 0
                l = ligs[idx]
                mol_plipscore.append([l, plipscore])

        shutil.rmtree(plipdir)
        return mol_plipscore