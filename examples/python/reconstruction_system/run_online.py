import cv2
import numpy as np

from os.path import join
import sys
from glob import glob
from pprint import pprint
from time import time
import open3d as o3d
sys.path.append("../utility")
from make_fragments import read_rgbd_image, pose_estimation, change_resolution
from initialize_config import initialize_config


def pairwise_registration(source, target):
    target.estimate_normals()
    # print(target.has_normals())
    # exit()
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def get_time(fun):
    s = time()
    out = fun()
    return out, time() - s


path_rgb = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "color")
path_depth = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "depth")
path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic.json")
path_dataset = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense.bag")
config = {"path_dataset": "nothing", "path_intrinsic": path_intrinsic}
initialize_config(config)

intrinsic = o3d.io.read_pinhole_camera_intrinsic(
    config["path_intrinsic"])

def get_mask_visible(rgbd, T):
    global height, width
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.transform(T)
    pcd = np.asarray(pcd.points)

    pcd2 = np.dot(intrinsic.intrinsic_matrix, pcd.T).T
    pcd2[:, 0] /= pcd2[:, 2]
    pcd2[:, 1] /= pcd2[:, 2]
    pcd2 = pcd2[:, :2]

    return np.logical_and(
        np.logical_and(0 <= pcd2[:, 0], pcd2[:, 0] <= width),
        np.logical_and(0 <= pcd2[:, 1], pcd2[:, 1] <= height))


def register_one_rgbd_pair(source_rgbd_image, target_rgbd_image, intrinsic, with_opencv, config, flag_not_close=False):

    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = config["max_depth_diff"]
    if flag_not_close:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info
                 ] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def find_local_rgbs(list_local_map, trans, pose_graph, rgbd0, target_id):
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic)

    if np.asarray(pcd0.points).shape[0] < 20:
        return
    for source_id, rgbd in list_local_map:
        mask = get_mask_visible(rgbd, trans)
        if np.average(mask) > tresh_covisibility:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            if np.asarray(pcd.points).shape[0] < 20:
                return
            transformation_icp, information_icp = pairwise_registration(pcd, pcd0)
            # print(len(pose_graph.nodes), len(pose_graph.edges), source_id, target_id)
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                         target_id,
                                                         transformation_icp,
                                                         information_icp,
                                                         uncertain=True)
            )

color_files, depth_files = sorted(glob(join(path_rgb, "*.jpg"))), sorted(glob(join(path_depth, "*.png")))

r = 0.25
voxel_size = 0.01
num_use = 20
tresh_covisibility = 0.7
n_local_map = 10
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
rgbd_prev = None

height, width = None, None

ts = [np.zeros((3,))]
rgbds = []
pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

for nr_frame, (f1, f2) in enumerate(zip(color_files[:num_use], depth_files[:num_use])):
    rgbd = read_rgbd_image(f1, f2, True, config)
    change_resolution(rgbd, r=r)
    if height is None:
        height, width = np.asarray(rgbd.color).shape[:2]
    # get_mask_visible(rgbd, np.eye(4))
    if rgbd_prev is not None:
        (success, trans, info), t = get_time(lambda: register_one_rgbd_pair(rgbd, rgbd_prev,
                                                                            intrinsic, True, config,
                                                                           flag_not_close=False))

        find_local_rgbs(rgbds[-n_local_map:], trans, pose_graph, rgbd, len(rgbds))
        ts.append(ts[-1] + trans[:3, 3])

        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans)))

    rgbds.append((len(rgbds), rgbd))
    rgbd_prev = rgbd

# print(pose_graph.edges)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
exit()
ts = np.asarray(ts)
print(ts.shape)

m, M = np.min(ts), np.max(ts)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# pprint(dir(ax))
ax.set_xlim(m, M)
ax.set_ylim(m, M)
ax.set_zlim(m, M)
# ax.scatter(mapper.point_cloud_cumulative[:, 0], mapper.point_cloud_cumulative[:, 1],
#            mapper.point_cloud_cumulative[:, 2], s=1)
ax.plot(ts[:, 0], ts[:, 1], ts[:, 2], c="red")
plt.show()