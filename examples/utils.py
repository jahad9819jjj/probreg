import copy
import numpy as np
import open3d as o3
import transforms3d as t3d
import point_cloud_utils

def estimate_normals(pcd, params):
    pcd.estimate_normals(search_param=params)
    pcd.orient_normals_to_align_with_direction()


def prepare_source_and_target_rigid_3d(source_filename,
                                       noise_amp=0.001,
                                       n_random=500,
                                       orientation=np.deg2rad([0.0, 30.0, 60.0]),
                                       translation=np.zeros(3),
                                       voxel_size=0.005,
                                       normals=False,
                                       data_type='pcd'):
    if data_type == 'mesh':
        num_samp = 2000
        source = o3.io.read_triangle_mesh(source_filename)
        source_verts, source_tris = np.asarray(source.vertices), np.asarray(source.triangles)
        fid, bc = point_cloud_utils.sample_mesh_random(source_verts, source_tris, num_samp)
        source_ds_verts = point_cloud_utils.interpolate_barycentric_coords(source_tris, fid, bc, source_verts)
        source = o3.geometry.PointCloud()
        source.points = o3.utility.Vector3dVector(source_ds_verts)
    elif data_type == 'pcd':
        source = o3.io.read_point_cloud(source_filename)
        source = source.voxel_down_sample(voxel_size=voxel_size)
        
    print(source)
    target = copy.deepcopy(source)
    tp = np.asarray(target.points)
    np.random.shuffle(tp)
    rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))
    rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
    target.points = o3.utility.Vector3dVector(np.r_[tp + noise_amp * np.random.randn(*tp.shape), rands])
    ans = np.identity(4)
    ans[:3, :3] = t3d.euler.euler2mat(*orientation)
    ans[:3, 3] = translation
    target.transform(ans)
    if normals:
        estimate_normals(source, o3.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        estimate_normals(target, o3.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    return source, target

def prepare_source_and_target_nonrigid_2d(source_filename,
                                          target_filename):
    source = np.loadtxt(source_filename)
    target = np.loadtxt(target_filename)
    return source, target


def prepare_source_and_target_nonrigid_3d(source_filename,
                                          target_filename,
                                          ds_param:dict={'voxel_size':5.0, 'num':1000},
                                          data_type='mesh'):
    # source = o3.geometry.PointCloud()
    # target = o3.geometry.PointCloud()
    # source.points = o3.utility.Vector3dVector(np.loadtxt(source_filename))
    # target.points = o3.utility.Vector3dVector(np.loadtxt(target_filename))
    if data_type == 'pcd':
        source = o3.io.read_point_cloud(source_filename)
        target = o3.io.read_point_cloud(target_filename)
        source = source.voxel_down_sample(voxel_size=ds_param['voxel_size'])
        target = target.voxel_down_sample(voxel_size=ds_param['voxel_size'])
    elif data_type == 'txt':
        source = o3.geometry.PointCloud()
        target = o3.geometry.PointCloud()
        source.points = o3.utility.Vector3dVector(np.loadtxt(source_filename))
        target.points = o3.utility.Vector3dVector(np.loadtxt(target_filename))
        source = source.voxel_down_sample(voxel_size=ds_param['voxel_size'])
        target = target.voxel_down_sample(voxel_size=ds_param['voxel_size'])
    elif data_type == 'mesh':
        source = o3.io.read_triangle_mesh(source_filename)
        target = o3.io.read_triangle_mesh(target_filename)
        source_verts, source_tris = np.asarray(source.vertices), np.asarray(source.triangles)
        target_verts, target_tris = np.asarray(target.vertices), np.asarray(target.triangles)
        fid, bc = point_cloud_utils.sample_mesh_random(source_verts, source_tris, ds_param['num'])
        source_ds_verts = point_cloud_utils.interpolate_barycentric_coords(source_tris, fid, bc, source_verts)
        fid, bc = point_cloud_utils.sample_mesh_random(target_verts, target_tris, ds_param['num'])
        target_ds_verts = point_cloud_utils.interpolate_barycentric_coords(target_tris, fid, bc, target_verts)
        source = o3.geometry.PointCloud()
        target = o3.geometry.PointCloud()
        source.points = o3.utility.Vector3dVector(source_ds_verts)
        target.points = o3.utility.Vector3dVector(target_ds_verts)
    else:
        raise ValueError('Invalid data type')
    print(source)
    print(target)
    return source, target