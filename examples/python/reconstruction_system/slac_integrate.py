# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/slac_integrate.py

import numpy as np
import open3d as o3d
import sys

sys.path.append("../utility")
from file import join, get_rgbd_file_lists

sys.path.append(".")


def run(config):
    print("slac non-rigid optimisation.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    # dataset path and slac subfolder path
    # slac default subfolder for 0.050 voxel size: `dataset/slac/0.050/`.
    path_dataset = config["path_dataset"]
    slac_folder = join(path_dataset, config["subfolder_slac"])

    # Read RGBD images.
    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    if len(color_files) != len(depth_files):
        raise ValueError(
            "The number of color images {} must equal to the number of depth images {}."
                .format(len(color_files), len(depth_files)))

    # Read optimized pose graph. [Generated by `register` stage].
    posegraph = o3d.io.read_pose_graph(
        join(slac_folder, config["template_optimized_posegraph_slac"]))

    # If camera intrinsic is not provided,
    # the default PrimeSense intrinsic is used.
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    focal_length = intrinsic.get_focal_length()
    principal_point = intrinsic.get_principal_point()

    intrinsic_t = o3d.core.Tensor([[focal_length[0], 0, principal_point[0]],
                                   [0, focal_length[1], principal_point[1]],
                                   [0, 0, 1]])

    device = o3d.core.Device(str(config["device"]))
    voxel_grid = o3d.t.geometry.TSDFVoxelGrid(
        {
            "tsdf": o3d.core.Dtype.Float32,
            "weight": o3d.core.Dtype.UInt16,
            "color": o3d.core.Dtype.UInt16
        }, config["voxel_size"], config["sdf_trunc"], 16, config["block_count"],
        device)

    # Load control grid.
    ctr_grid_keys = o3d.core.Tensor.load(slac_folder + "ctr_grid_keys.npy")
    ctr_grid_values = o3d.core.Tensor.load(slac_folder + "ctr_grid_values.npy")

    ctr_grid = o3d.t.pipelines.slac.control_grid(3.0 / 8,
                                                 ctr_grid_keys.to(device),
                                                 ctr_grid_values.to(device),
                                                 device)

    fragment_folder = join(path_dataset, config["folder_fragment"])

    k = 0
    for i in range(len(posegraph.nodes)):
        fragment_pose_graph = o3d.io.read_pose_graph(
            join(fragment_folder, "fragment_optimized_%03d.json" % i))
        for node in fragment_pose_graph.nodes:
            pose_local = node.pose
            extrinsic_local_t = o3d.core.Tensor(np.linalg.inv(pose_local))

            pose = np.dot(posegraph.nodes[i].pose, node.pose)
            extrinsic_t = o3d.core.Tensor(np.linalg.inv(pose))

            depth = o3d.t.io.read_image(depth_files[k]).to(device)
            color = o3d.t.io.read_image(color_files[k]).to(device)
            rgbd = o3d.t.geometry.RGBDImage(color, depth)

            rgbd_projected = ctr_grid.deform(rgbd, intrinsic_t,
                                             extrinsic_local_t,
                                             config["depth_scale"],
                                             config["max_depth"])
            voxel_grid.integrate(rgbd_projected.depth, rgbd_projected.color,
                                 intrinsic_t, extrinsic_t,
                                 config["depth_scale"], config["max_depth"])

            k = k + 1
            if (device.get_type() == o3d.core.Device.CUDA and k % 10 == 0):
                o3d.core.cuda.release_cache()

    if (config["save_output_as"] == "pointcloud"):
        pcd = voxel_grid.extract_surface_points().to(o3d.core.Device("CPU:0"))
        save_pcd_path = join(slac_folder, "output_slac_pointcloud.ply")
        o3d.t.io.write_point_cloud(save_pcd_path, pcd)
    else:
        mesh = voxel_grid.extract_surface_mesh().to(o3d.core.Device("CPU:0"))
        mesh_legacy = mesh.to_legacy_triangle_mesh()
        save_mesh_path = join(slac_folder, "output_slac_mesh.ply")
        o3d.io.write_triangle_mesh(save_mesh_path, mesh_legacy)
