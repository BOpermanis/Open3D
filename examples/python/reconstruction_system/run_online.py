import cv2
import numpy as np

from os.path import join
import sys
from glob import glob
from pprint import pprint
from time import time
import open3d as o3d
sys.path.append("../utility")
from make_fragments import read_rgbd_image, pose_estimation
from initialize_config import initialize_config


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


def change_resolution(rgbd, w=None, h=None, r=None):
    color = np.asarray(rgbd.color)
    depth = np.asarray(rgbd.depth)
    if w is None:
        h, w = map(lambda x: int(x * r), color.shape[:2])
    color = cv2.resize(color, (w, h))
    depth = cv2.resize(depth, (w, h))
    color = o3d.cpu.pybind.geometry.Image(color)
    depth = o3d.cpu.pybind.geometry.Image(depth)
    rgbd.color = color
    rgbd.depth = depth

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

color_files, depth_files = sorted(glob(join(path_rgb, "*.jpg"))), sorted(glob(join(path_depth, "*.png")))

r=0.25

rgbd_prev = None

for f1, f2 in zip(color_files, depth_files):
    rgbd = read_rgbd_image(f1, f2, True, config)

    change_resolution(rgbd, r=r)

    if rgbd_prev is not None:
        (success, trans, info), t = get_time(lambda: register_one_rgbd_pair(rgbd, rgbd_prev, intrinsic, True, config))
        print(success, t)
    rgbd_prev = rgbd

