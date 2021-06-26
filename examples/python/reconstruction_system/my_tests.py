# from run_online import get_mask_visible
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from os.path import join
from initialize_config import initialize_config
from make_fragments import read_rgbd_image, pose_estimation, change_resolution
import cv2
from glob import glob
from scipy.spatial.transform import Rotation


path_rgb = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "color")
path_depth = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "depth")

# path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_100p.json"); r = 1.0
path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_25p.json"); r = 0.25


path_dataset = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense.bag")
config = {"path_dataset": "nothing", "path_intrinsic": path_intrinsic}
initialize_config(config)

intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])


height, width = 120, 160

color_files, depth_files = sorted(glob(join(path_rgb, "*.jpg"))), sorted(glob(join(path_depth, "*.png")))
for nr_frame, (f1, f2) in enumerate(zip(color_files, depth_files)):
    print(nr_frame)
    rgbd = read_rgbd_image(f1, f2, True, config)
    change_resolution(rgbd, r=r)

    depth = np.asarray(rgbd.depth)
    depth = np.full_like(depth, np.average(depth))
    depth = o3d.cpu.pybind.geometry.Image(depth)
    rgbd.depth = depth


    T_viewpoint = np.eye(4)
    # T_viewpoint[0, 0] = -1
    # T_viewpoint[1, 1] = -1
    # T_viewpoint[2, 2] = -1

    T_viewpoint[:3, :3] = Rotation.from_rotvec([0, np.pi / 6, 0]).as_matrix()
    # print(T_viewpoint)
    # exit()
    print(Rotation.from_matrix(T_viewpoint[:3, :3]).as_rotvec())

    T_viewpoint[:3, 3] = np.array([0, 0, 0])


    T_rgbd = np.eye(4)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # vispirms transformee pcd uz 0, peec tam uz viewpoint

    pcd1 = np.asarray(pcd.points)
    print(np.min(pcd1), np.max(pcd1), np.round(np.average(pcd1, axis=0), 3))
    pcd = pcd.transform(np.matmul(T_viewpoint, np.linalg.inv(T_rgbd)))
    pcd = np.asarray(pcd.points)
    print(np.min(pcd), np.max(pcd), np.round(np.average(pcd, axis=0), 3))

    pcd2 = np.dot(pcd, intrinsic.intrinsic_matrix.T)
    print("z_comb", np.min(pcd2[:, 2]), np.max(pcd2[:, 2]))
    pcd2[:, 0] /= pcd2[:, 2]
    pcd2[:, 1] /= pcd2[:, 2]
    # pcd2 = pcd2[:, :2]

    print(np.min(pcd2[:, 0]), np.max(pcd2[:, 0]), np.min(pcd2[:, 1]), np.max(pcd2[:, 1]))
    mask = np.logical_and(pcd2[:, 2] > 0.0, np.logical_and(
        np.logical_and(0 <= pcd2[:, 0], pcd2[:, 0] < width),
        np.logical_and(0 <= pcd2[:, 1], pcd2[:, 1] < height)))
    print(np.average(mask))

    print(intrinsic.intrinsic_matrix)

    #2*atan(height/2/fy)*180/CV_PI;
    x = np.abs(intrinsic.intrinsic_matrix[0, 2] / intrinsic.intrinsic_matrix[0, 0])
    print(x, 2 * 180 * np.arctan(x) / np.pi)

    exit()