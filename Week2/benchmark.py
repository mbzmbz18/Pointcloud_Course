import random
import math
import numpy as np
import time
import os
import struct
import open3d as o3d
from scipy import spatial
from Week2 import octree as octree
from Week2 import kdtree as kdtree
from Week2.result_set import KNNResultSet, RadiusNNResultSet


def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

########################################################################################################################
# Main Function
###############

# Path management issues
file_name = '000000.bin'
path2folder = os.path.dirname(__file__)
path2datafile = os.path.join(path2folder, file_name)

# KNN-Search Configuration
leaf_size = 32
min_extent = 0.0001
k_neighbor = 8
radius = 1

# Read the point cloud from files
#data = read_velodyne_bin(path2datafile)
#point_cloud = o3d.geometry.PointCloud()
#point_cloud.points = o3d.utility.Vector3dVector(data)
#o3d.visualization.draw_geometries([point_cloud])

################################# Octree #####################################
print("-------------- Octree --------------")
construction_time_sum = 0
knn_time_sum = 0
radius_time_sum = 0
brute_time_sum = 0
db_np = read_velodyne_bin(path2datafile)
query = db_np[0,:]
print('Query point: ')
print(query)

# Construction of Octree
print('Number of data point: '+str(len(db_np)))
begin_t = time.time()
root = octree.octree_construction(db_np, leaf_size, min_extent)
construction_time_sum += time.time() - begin_t

# K-NN Search
begin_t = time.time()
result_set = KNNResultSet(capacity=k_neighbor)
octree.octree_knn_search(root, db_np, result_set, query)
knn_time_sum += time.time() - begin_t

# Radius-NN Search
begin_t = time.time()
result_set = RadiusNNResultSet(radius=radius)
octree.octree_radius_search_fast(root, db_np, result_set, query)
radius_time_sum += time.time() - begin_t

# Brute Force search
begin_t = time.time()
diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
nn_idx = np.argsort(diff)
nn_dist = diff[nn_idx]
brute_time_sum += time.time() - begin_t

print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000,
                                                                 knn_time_sum*1000,
                                                                 radius_time_sum*1000,
                                                                 brute_time_sum*1000))

############################### KD-Tree #########################################
print("--------------- kdtree --------------")
construction_time_sum = 0
knn_time_sum = 0
radius_time_sum = 0
brute_time_sum = 0
scipy_knn_time_sum = 0
db_np = read_velodyne_bin(path2datafile)
query = db_np[0,:]
print('Query point: ')
print(query)

# Construction of KD-tree
begin_t = time.time()
root = kdtree.kdtree_construction(db_np, leaf_size)
construction_time_sum += time.time() - begin_t

# K-NN Search
begin_t = time.time()
result_set = KNNResultSet(capacity=k_neighbor)
kdtree.kdtree_knn_search(root, db_np, result_set, query)
knn_time_sum += time.time() - begin_t

# Radius-NN Search
begin_t = time.time()
result_set = RadiusNNResultSet(radius=radius)
kdtree.kdtree_radius_search(root, db_np, result_set, query)
radius_time_sum += time.time() - begin_t

# Brute Force search
begin_t = time.time()
diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
nn_idx = np.argsort(diff)
nn_dist = diff[nn_idx]
brute_time_sum += time.time() - begin_t

# Scipy.spatial KD-Tree
begin_t = time.time()
kdtree_scipy = spatial.KDTree(db_np, leafsize=k_neighbor)
result = kdtree_scipy.query(query, k=k_neighbor)
scipy_knn_time_sum += time.time() - begin_t

print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f, scipyknn %.3f" % (construction_time_sum * 1000,
                                                                 knn_time_sum * 1000,
                                                                 radius_time_sum * 1000,
                                                                 brute_time_sum * 1000,
                                                                 scipy_knn_time_sum * 1000))
                                                                 

