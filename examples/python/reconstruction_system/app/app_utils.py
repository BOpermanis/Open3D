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

        self.cloud = self.cloud.voxel_down_sample(Frame.stream_config['voxel_size'])

        self.search_tree_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.stream_config["voxel_size"] * 3.0,
            max_nn=30)

        self.cloud.estimate_normals(self.search_tree_param)

        self.fpfh = o3d.pipelines.registration.compute_fpfh_feature(self.cloud, self.search_tree_param)

        self.kps = o3d.geometry.keypoint.compute_iss_keypoints(self.cloud)
        hash_cloud = np.sum(np.asarray(self.cloud.points), axis=1)
        hash_kps = np.sum(np.asarray(self.kps.points), axis=1)
        self.fpfh_at_kps = self.fpfh.data[:, hash_cloud == hash_kps]

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

