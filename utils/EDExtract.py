import os
import scipy
import numpy as np
import pandas as pd
import iotbx.pdb
import mmtbx.model
from rdkit import Chem
from utils.Utils import GetAtoms
from collections import defaultdict


def IsinGrid(x, y, z, centerx, centery, centerz, grid_size):
    """
    Check a point in the grid
    """
    if (abs(centerx -x) <= grid_size) and (abs(centery -y) <= grid_size) and (abs(centerz -z) <= grid_size):
        return 1
    else:
        return 0


def GetPocAtoms(pdbf, cx, cy, cz):
    atomlines = GetAtoms(pdbf)
    pocatoms = defaultdict(list)
    for line in atomlines:
        name = line[17: 20].strip()
        chain = line[21]
        num = int(line[22: 26].strip())
        x = float(line[30: 38].strip())
        y = float(line[38: 46].strip())
        z = float(line[46: 54].strip())
        atomtype = (line[76:78].strip())[0].upper()
        if IsinGrid(x, y, z, cx, cy, cz, 15):
            pocatoms["name"].append(name)
            pocatoms["chain"].append(chain)
            pocatoms["num"].append(num)
            pocatoms["x"].append(x)
            pocatoms["y"].append(y)
            pocatoms["z"].append(z)
            pocatoms["atomtype"].append(atomtype)
    pocatomdf = pd.DataFrame(pocatoms)
    return pocatomdf


def Voxelize(centroid, GridSize=12, SpacingCutoff=0.5):
    xv = np.arange(0, GridSize*2, SpacingCutoff)
    yv = np.arange(0, GridSize*2, SpacingCutoff)
    zv = np.arange(0, GridSize*2, SpacingCutoff)
    Grid_x, Grid_y, Grid_z = np.meshgrid(xv, yv, zv)
    Grid_x = Grid_x + centroid[0] - GridSize
    Grid_y = Grid_y + centroid[1] - GridSize
    Grid_z = Grid_z + centroid[2] - GridSize
    points = np.hstack((Grid_x.flatten().reshape((-1, 1)),
                        Grid_y.flatten().reshape((-1, 1)),
                        Grid_z.flatten().reshape((-1, 1))))
    voxpoints = points + SpacingCutoff/2
    return voxpoints


def Fcalc(pdbf,voxpoints,resolution=2.0):
    pdb_inp = iotbx.pdb.input(file_name = pdbf)
    model = mmtbx.model.manager(model_input=pdb_inp)
    xrs = model.get_xray_structure()
    fcalc = xrs.structure_factors(d_min = resolution).f_calc()
    fft_map = fcalc.fft_map(resolution_factor = 0.25)
    fft_map.apply_volume_scaling()
    fcalc_map_data = fft_map.real_map_unpadded()
    uc = fft_map.crystal_symmetry().unit_cell()
    rho_fcalc = []
    for p in voxpoints:
        frac = uc.fractionalize(p)
        density = fcalc_map_data.value_at_closest_grid_point(frac)
        rho_fcalc.append(density)
    
    rho_fcalc = np.array(rho_fcalc).reshape(-1,1)
    rho_fcalc = np.where(rho_fcalc < 0, 0, rho_fcalc)
    return rho_fcalc


def FcalcPdb(pdbf, cx, cy, cz, tmpdir):
    pdbname2 = os.path.join(tmpdir, "rec_cell.pdb")
    with open(pdbname2, "w") as op:
        op.writelines("CRYST1   90.000   90.000   90.000  90.00  90.00  90.00 P 1           3          \n")
        for lines in open(pdbf).readlines():
            if "ANISOU" not in lines:
                lines = lines[:56]+"1.00"+lines[60:61]+" 0.00"+lines[66:]
                op.writelines(lines)
        op.close()    
    
    pocatoms = GetPocAtoms(pdbf, cx, cy, cz)
    pocatomcoords = pocatoms[["x", "y", "z"]].to_numpy()
    voxpoints = Voxelize([cx, cy, cz], GridSize = 12, SpacingCutoff = 0.5)
    rho_fcalc = Fcalc(pdbname2,voxpoints)
    dist = scipy.spatial.distance.cdist(voxpoints, pocatomcoords, metric = "euclidean")
    ptable = Chem.GetPeriodicTable()
    CutOff = np.array([ptable.GetRvdw(element) for element in pocatoms["atomtype"]])
    interval = (dist-CutOff)
    poc_close_p = np.where(interval < 0)[0]
    rho_fcalc_pdb = np.zeros((48*48*48, 1))
    rho_fcalc_pdb[poc_close_p, 0] = rho_fcalc[poc_close_p, 0]
    os.remove(pdbname2)

    return voxpoints, rho_fcalc_pdb