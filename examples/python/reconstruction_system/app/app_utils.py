import numpy as np
import open3d as o3d
from time import time


class Frame:
    stream_config = None
    cnt = 0

    def __init__(self, rgbd, **kwargs):
        self.id = Frame.cnt
        Frame.cnt += 1
        self.cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, Frame.stream_config["intrinsic"])
        self.cloud.estimate_normals()
        self.T = None
        self.__dict__.update(kwargs)

    def get_transform_to(self, other):
        return pairwise_registration(self, other, Frame.stream_config)


def get_time(fun):
    s = time()
    out = fun()
    return out, time() - s


def pairwise_registration(source, target, stream_config):
    if isinstance(source, o3d.cpu.pybind.geometry.RGBDImage):
        source = o3d.geometry.PointCloud.create_from_rgbd_image(source, stream_config["intrinsic"])
        target = o3d.geometry.PointCloud.create_from_rgbd_image(target, stream_config["intrinsic"])

    if isinstance(source, Frame):
        source = source.cloud
        target = target.cloud

    # keypoints = o3d.geometry.keypoint.compute_iss_keypoints(source)
    if np.asarray(target.points).shape[0] < 20:
        return None, None

    if not target.has_normals():
        target.estimate_normals()

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, stream_config["max_correspondence_distance_coarse"], np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, stream_config["max_correspondence_distance_fine"],
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, stream_config["max_correspondence_distance_fine"],
        icp_fine.transformation)
    return transformation_icp, information_icp

