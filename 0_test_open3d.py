import open3d as o3d
import numpy as np
import os
from pandas import DataFrame
from pyntcloud import PyntCloud


file_name = 'airplane_0002.txt'
path2folder = os.path.dirname(__file__)
path2datafile = os.path.join(path2folder, 'data', file_name)

"""
points = np.random.rand(10000, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])
"""

"""
pcd = o3d.io.read_point_cloud(path2datafile, format='xyzn')
print(pcd)
o3d.visualization.draw_geometries([pcd])
"""

points = np.genfromtxt(path2datafile, delimiter=",")
points = DataFrame(points[:, 0:3])
points.columns = ['x', 'y', 'z']
point_cloud_pynt = PyntCloud(points)
point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
#o3d.visualization.draw_geometries([point_cloud_o3d])

points_vertices = points.to_numpy()



