import numpy as np
import open3d as o3
import transforms3d as t3d
from probreg import cpd
from probreg import callbacks
import logging
log = logging.getLogger('probreg')
log.setLevel(logging.DEBUG)

voxel_size = 0.03
source = o3.io.read_point_cloud("frag_115.ply")
source = source.voxel_down_sample(voxel_size=voxel_size)

target = o3.io.read_point_cloud("frag_116.ply")
target = target.voxel_down_sample(voxel_size=voxel_size)

cbs = [callbacks.Open3dVisualizerCallback(source, target, save=True)]
tf_param, _, _ = cpd.registration_cpd(source, target,
                                      callbacks=cbs,
                                      use_color=True,
                                      update_scale=False)

print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),
      tf_param.scale, tf_param.t)
