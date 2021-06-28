import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.spatial import distance_matrix
from collections import OrderedDict
from copy import deepcopy

class Mapper:
    stream_config = None

    def __init__(self, T_start):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.T_of_last_kf = None
        self.kfs = OrderedDict() #keyframes

    def update_map(self, frame):
        if len(self.kfs) == 0:
            frame.id_pose_graph = len(self.pose_graph.nodes)
            self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(frame.T)))
            self.kfs[frame.id] = frame
            self.T_of_last_kf = frame.T
            return True

        rotvec = Rotation.from_matrix(frame.T[:3, :3]).as_rotvec()
        transvec = frame.T[:3, 3]

        rotmat = np.stack([Rotation.from_matrix(frame.T[:3, :3]).as_rotvec() for frame in self.kfs.values()])
        transmat = np.stack([frame.T[:3, 3] for frame in self.kfs.values()])

        rot_dists = distance_matrix(rotmat, np.expand_dims(rotvec, 0))
        trans_dists = distance_matrix(transmat, np.expand_dims(transvec, 0))

        mr, Mr = Mapper.stream_config["kf_angle"]
        mt, Mt = Mapper.stream_config["kf_trans"]
        if mr < np.min(rot_dists) < Mr and mt < np.min(trans_dists) < Mt:
            frame.id_pose_graph = len(self.pose_graph.nodes)
            ids = list(self.kfs.keys())
            for i_kf in np.where(np.logical_and(rot_dists < Mr, trans_dists < Mt))[0]:
                trans, info = self.kfs[ids[i_kf]].get_transform_to(frame)

                self.pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(self.kfs[ids[i_kf]].id_pose_graph,
                                                             frame.id_pose_graph,
                                                             trans, info,
                                                             uncertain=True))

                # print(i_kf, np.logical_and(rot_dists < Mr, trans_dists < Mt).shape)
            self.kfs[frame.id] = frame
            self.T_of_last_kf = frame.T
            self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(frame.T)))
        return True

    def optimize(self, T_now):
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=Mapper.stream_config["max_correspondence_distance_fine"],
            edge_prune_threshold=0.25,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)

        node2frame = {frame.id_pose_graph: frame.id for frame in self.kfs.values()}
        for i_node in range(len(self.pose_graph.nodes)):
            self.kfs[node2frame[i_node]].T = np.linalg.inv(self.pose_graph.nodes[i_node].pose)
            if i_node == len(self.pose_graph.nodes) - 1:
                T_now = np.dot(self.kfs[node2frame[i_node]].T, np.dot(np.linalg.inv(self.T_of_last_kf), T_now))
        return T_now

    def get_total_cloud(self):
        total_cloud = None
        for _, frame in self.kfs.items():
            if total_cloud is None:
                total_cloud = deepcopy(frame.cloud).transform(np.linalg.inv(frame.T))
            else:
                total_cloud += deepcopy(frame.cloud).transform(np.linalg.inv(frame.T))
        total_cloud = total_cloud.voxel_down_sample(Mapper.stream_config['voxel_size'] / 10)
        return total_cloud