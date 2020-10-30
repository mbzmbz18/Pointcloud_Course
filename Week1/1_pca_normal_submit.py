import open3d as o3d
import os
import numpy as np
from pandas import DataFrame
from pyntcloud import PyntCloud

# Function to calculate PCA
def PCA(data, sort=True):
    # Calculate H
    data_mean = np.mean(data, axis=0)
    data_normalized = data - data_mean
    # data_normalized is Nx3 matrix, H is 3x3 matrix
    H = np.dot(data_normalized.T, data_normalized)
    # Call SVD for H
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)
    # Sort eigenvalues in decreasing order if necessary
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


# Path management issues
file_name = 'bathtub_0001.txt'
path2folder = os.path.dirname(__file__)
path2datafile = os.path.join(path2folder, '../data', file_name)

# Import point cloud from path and visualization
points = np.genfromtxt(path2datafile, delimiter=",")
points = DataFrame(points[:, 0:3])
points.columns = ['x', 'y', 'z']
point_cloud_pynt = PyntCloud(points)
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
#o3d.visualization.draw_geometries([point_cloud_o3d])

# Obtain point vertices as np.array
points_array = points.to_numpy()
print('Total points number is:', points_array.shape[0])

# Call PCA Function
eigenvalues, eigenvectors = PCA(points)

# Extract main direction, i.e. first column
point_cloud_vector = eigenvectors[:, 0]
print('the main orientation of this pointcloud is: ', point_cloud_vector)

# Visualization of PCA by creating Lineset
center = np.mean(points_array, axis=0)
point = [center, center+eigenvectors[:, 0], center+eigenvectors[:, 1], center+eigenvectors[:, 2]]
lines = [[0, 1], [0, 2], [0, 3]]
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point), lines=o3d.utility.Vector2iVector(lines))
line_set.colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries([point_cloud_o3d, line_set])

# Calculate surface normals
pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
normals = []
for i in range(len(points_array)):
    # Choose 10 neighboring points
    [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 10)
    k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]
    w, v = PCA(k_nearest_point)
    normals.append(v[:, 2])

normals = np.array(normals, dtype=np.float64)
point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([point_cloud_o3d])





