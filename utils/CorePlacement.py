import math
import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from itertools import product
from sklearn.cluster import DBSCAN
from rdkit.Geometry import Point3D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def GetMolCenter(mol):
    center = np.mean(np.array(mol.GetConformer().GetPositions()), 0) 
    return center


def CoreTranslate(Core, TargetCenter):   
    conf = Core.GetConformer()
    positions = conf.GetPositions()
    center = GetMolCenter(Core)
    new_positions = np.array(positions)+np.array(TargetCenter)-np.array(center)
    for i in range(Core.GetNumAtoms()):
        X,Y,Z = np.array(new_positions)[i]
        Core.GetConformer().SetAtomPosition(i, Point3D(X,Y,Z)) 
    
    Core.UpdatePropertyCache()
    return Core


def CoreRotate(Core, Pitch, Yaw, Roll):
    core_copy = copy.deepcopy(Core)
    center = GetMolCenter(core_copy)
    conf = core_copy.GetConformer()
    positions = conf.GetPositions()
    positions = np.array(positions)
    
    ex_arr = np.array([[1,0,0],[0,math.cos(Pitch),-math.sin(Pitch)],[0,math.sin(Pitch),math.cos(Pitch)]])
    ey_arr = np.array([[math.cos(Yaw), 0, math.sin(Yaw)],[0, 1, 0],[-math.sin(Yaw), 0,math.cos(Yaw)]])
    ez_arr = np.array([[math.cos(Roll), -math.sin(Roll),0],[math.sin(Roll), math.cos(Roll), 0],[0, 0,1]])
    temp = np.transpose(ez_arr.dot(np.transpose(positions-center)))
    temp = np.transpose(ey_arr.dot(np.transpose(temp)))
    temp = np.transpose(ex_arr.dot(np.transpose(temp)))
    new_positions = temp+center
    
    for i in range(core_copy.GetNumAtoms()):
        X,Y,Z = np.array(new_positions)[i]
        core_copy.GetConformer().SetAtomPosition(i, Point3D(X,Y,Z)) 
    
    core_copy.UpdatePropertyCache()
    return core_copy


def RetainMaxValues(tensor, alpha=0.4):
    """
    Keep the maximum value in the tensor
    """
    stride = (1, 1, 1)
    pool_size = (4, 4, 4)

    tensor_copy = tensor.clone()
    max_value = tensor_copy.max().item()
    threshold = max_value * alpha
    tensor_copy[tensor_copy < threshold] = 0

    max_values, indices = F.max_pool3d(tensor_copy, pool_size, stride, return_indices=True)

    result = th.zeros_like(tensor_copy)
    result = result.view(-1)
    result[indices.view(-1)] = tensor_copy.view(-1)[indices.view(-1)]
    result = result.view(tensor_copy.shape)

    mask = max_values != 0
    filtered_indices = indices[mask]
    return result


def GetClusterCenters(merged_arr, eps=1, min_samples=2):
    """
    Get the centers of the clusters
    """
    center_list = []
    merged_arr = merged_arr[merged_arr[:, -1] != 0]
    merged_arr = merged_arr[:, :3]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(merged_arr)
    unique_values, counts = np.unique(labels, return_counts=True)

    sorted_indices = np.argsort(counts)[::-1]
    top_three_unique = unique_values[sorted_indices][:3]

    for label in top_three_unique:
        indices = labels == label
        cluster_points = merged_arr[indices]
        center_list.append(np.mean(cluster_points, axis=0))
    return center_list 


def CoreId(core, cluster_centers, num_divisions=9):
    new_cores = []
    for center in (cluster_centers):
        core = CoreTranslate(core, center)
        for x_a, y_a, z_a in product(range(num_divisions), range(num_divisions), range(num_divisions)): 
            core = CoreRotate(core, x_a, y_a, z_a)
            new_cores.append(core)
    return new_cores