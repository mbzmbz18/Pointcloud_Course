import open3d as o3d
import numpy as np
import os
from pandas import DataFrame
from pyntcloud import PyntCloud

# Function for centroid computation
def compute_centroid(point_indices, point_cloud):
    x = np.mean(point_cloud[point_indices, 0])
    y = np.mean(point_cloud[point_indices, 1])
    z = np.mean(point_cloud[point_indices, 2])
    centroid = np.asarray([x, y, z])
    return centroid

# Function for random point selection
def compute_random(point_indices, point_cloud):
    n_number = len(point_indices)
    index = np.random.randint(n_number)
    random_point = point_cloud[point_indices[index]]
    return random_point

# Function for voxel grid downsampling
def voxel_filter(point_cloud, leafsize):
    # Initialize output
    filtered_cloud = []
    # Obtain the range of points
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
    # Compute the dimension of the voxel grid
    D_x = np.int(np.floor((x_max - x_min) / leafsize) + 1)
    D_y = np.int(np.floor((y_max - y_min) / leafsize) + 1)
    D_z = np.int(np.floor((z_max - z_min) / leafsize) + 1)
    print('Dimension of voxel grid: ' + str(D_x) + 'x' + str(D_y) + 'x' + str(D_z))
    # Compute voxel index for each point
    h_list = []
    for i in range(len(point_cloud)):
        x, y, z = point_cloud[i, :]
        h_x = np.floor((x - x_min) / leafsize)
        h_y = np.floor((y - y_min) / leafsize)
        h_z = np.floor((z - z_min) / leafsize)
        h = np.int(h_x + h_y * D_x + h_z * D_x * D_y)
        h_list.append(h)
    h_list = np.asarray(h_list)
    # Ordered point indices according to increasing h-value
    index_ordered = np.argsort(h_list)
    # According to ordered point indices, find corresponding h-value
    hvalue_ordered = np.full_like(index_ordered, 0)
    for i in range(len(index_ordered)):
        hvalue_ordered[i] = h_list[index_ordered[i]]
    # Selection of point in each grid
    start_index = 0
    for i in range(len(index_ordered) - 1):
        if hvalue_ordered[i] == hvalue_ordered[i + 1]:
            continue
        else:
            # Collect all indices of point which have same hvalue
            point_indices = index_ordered[start_index:i + 1]
            #sample_point = compute_centroid(point_indices, point_cloud)
            sample_point = compute_random(point_indices, point_cloud)
            filtered_cloud.append(sample_point)
            start_index = i

    filtered_points = np.asarray(filtered_cloud)
    return filtered_points


# Path management issues
file_name = 'car_0005.txt'
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

# Call Voxel Filter Function
leaf_size = 0.05
filtered_cloud = voxel_filter(points_array, leaf_size)

# Visualization filtered point cloud
point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
o3d.visualization.draw_geometries([point_cloud_o3d])









    














