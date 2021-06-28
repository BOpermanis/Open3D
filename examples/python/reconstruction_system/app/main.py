import cv2
import numpy as np

from os.path import join
import sys
from glob import glob
from pprint import pprint
from time import time
from copy import deepcopy
import open3d as o3d
# sys.path.append("../utility")
from examples.python.reconstruction_system.make_fragments import read_rgbd_image, pose_estimation, change_resolution
from examples.python.reconstruction_system.initialize_config import initialize_config
from examples.python.reconstruction_system.app.app_utils import get_time, pairwise_registration, Frame
from examples.python.reconstruction_system.app.local_mapping import Mapper



path_rgb = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "color")
path_depth = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "depth")

# path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_100p.json"); r = 1.0
# path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_25p.json"); r = 0.25
path_intrinsic = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense", "intrinsic_20p.json"); r = 0.2

path_dataset = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense.bag")
stream_config = {"path_dataset": "nothing", "path_intrinsic": path_intrinsic}
initialize_config(stream_config)

voxel_size = 0.05
stream_config.update({
    "intrinsic": o3d.io.read_pinhole_camera_intrinsic(stream_config["path_intrinsic"]),
    "voxel_size": voxel_size,
    "tresh_covisibility": 0.7,
    "tresh_too_similar": 0.9,
    "kf_angle": (0.05, 0.3),
    "kf_trans": (0.05, 0.3),
    "max_correspondence_distance_coarse": voxel_size * 15,
    "max_correspondence_distance_fine": voxel_size * 1.5
})

Frame.stream_config = stream_config
Mapper.stream_config = stream_config

nr_start = 300
nr_end = 500
color_files, depth_files = sorted(glob(join(path_rgb, "*.jpg"))), sorted(glob(join(path_depth, "*.png")))


t_start = time()
T_cumulative = np.eye(4)
mapper = Mapper(T_cumulative)

prev_frame = None
for nr_frame, (f1, f2) in enumerate(zip(color_files[nr_start:nr_end], depth_files[nr_start:nr_end])):
    rgbd = read_rgbd_image(f1, f2, True, stream_config)
    change_resolution(rgbd, r=r)
    frame = Frame(rgbd)

    if prev_frame is not None:
        trans, _ = prev_frame.get_transform_to(frame)
        if trans is None:
            continue
        T_cumulative = np.matmul(trans, T_cumulative)
        frame.T = T_cumulative
        mapper.update_map(frame)

    # if nr_frame % 50 == 0 and nr_frame > 0:
    #     mapper.optimize()
    prev_frame = frame

mapper.optimize()
print("len(clouds)", len(mapper.kfs))
print("time: ", time() - t_start)
o3d.visualization.draw_geometries([mapper.get_total_cloud()])