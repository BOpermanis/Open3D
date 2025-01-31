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
from scipy.spatial.transform import Rotation


def pairwise_registration(source, target):
    if isinstance(source, o3d.cpu.pybind.geometry.RGBDImage):
        source = o3d.geometry.PointCloud.create_from_rgbd_image(source, intrinsic)
        target = o3d.geometry.PointCloud.create_from_rgbd_image(target, intrinsic)

    # keypoints = o3d.geometry.keypoint.compute_iss_keypoints(source)
    if np.asarray(target.points).shape[0] < 20:
        return None, None
    target.estimate_normals()
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

# path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_100p.json"); r = 1.0
# path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_25p.json"); r = 0.25
path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_20p.json"); r = 0.2


path_dataset = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense.bag")
config = {"path_dataset": "nothing", "path_intrinsic": path_intrinsic}
initialize_config(config)

intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
# print(intrinsic)
# print(intrinsic.intrinsic_matrix)
# exit()


def get_mask_visible(rgbd, T_viewpoint, T_rgbd, flag_testing=False):
    global height, width
    if flag_testing:
        height, width = 120, 160
        pcd = rgbd
    else:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        # vispirms transformee pcd uz 0, peec tam uz viewpoint
        pcd = pcd.transform(np.matmul(T_viewpoint, np.linalg.inv(T_rgbd)))
        pcd = np.asarray(pcd.points)

    pcd2 = np.dot(intrinsic.intrinsic_matrix, pcd.T).T
    pcd2[:, 0] /= pcd2[:, 2]
    pcd2[:, 1] /= pcd2[:, 2]

    return np.logical_and(pcd2[:, 2] > 0.0, np.logical_and(
        np.logical_and(0 <= pcd2[:, 0], pcd2[:, 0] < width),
        np.logical_and(0 <= pcd2[:, 1], pcd2[:, 1] < height)))


def find_local_rgbs(list_local_map, trans, pose_graph, rgbd0, target_id):
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic)

    if np.asarray(pcd0.points).shape[0] < 20:
        return

    for source_id, rgbd in list_local_map:
        pose = np.linalg.inv(pose_graph.nodes[source_id].pose)
        mask = get_mask_visible(rgbd, trans, pose)
        # m = np.matmul(trans, np.linalg.inv(pose))[:3, :3]
        # print(2222, np.linalg.det(m), Rotation.from_matrix(m).as_rotvec(), np.average(mask))
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


nr_start = 300
nr_end = 500
voxel_size = 0.05
tresh_covisibility = 0.7
n_local_map = 10
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
rgbd_prev = None

height, width = None, None

ts = [np.zeros((3,))]
rgbds = []
cloud = None
t_start = time()
pose_graph = o3d.pipelines.registration.PoseGraph()
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

T_cumulative = np.eye(4)
rgbd = None
for nr_frame, (f1, f2) in enumerate(zip(color_files[nr_start:nr_end], depth_files[nr_start:nr_end])):

    rgbd = read_rgbd_image(f1, f2, True, config)
    change_resolution(rgbd, r=r)
    if height is None:
        height, width = np.asarray(rgbd.color).shape[:2]

    if rgbd_prev is not None:
        (trans, _), t = get_time(lambda: pairwise_registration(rgbd_prev, rgbd))
        print(nr_frame, t)
        if trans is None:
            continue
        T_cumulative = np.matmul(trans, T_cumulative)
        find_local_rgbs(rgbds[-n_local_map:], T_cumulative, pose_graph, rgbd, len(rgbds))
        ts.append(ts[-1] + trans[:3, 3])

        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(T_cumulative)))

    rgbds.append((len(rgbds), rgbd))
    rgbd_prev = rgbd

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


clouds = []
np_cloud = []
for id_rgbd, rgbd in rgbds:
    clouds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic))
    clouds[-1].transform(pose_graph.nodes[id_rgbd].pose)
    clouds[-1] = clouds[-1].voxel_down_sample(voxel_size)
    np_cloud.append(np.asarray(clouds[-1].points))
np_cloud = np.concatenate(np_cloud)

print("len(clouds)", len(clouds))
print("time: ", time() - t_start)
o3d.visualization.draw_geometries(clouds)