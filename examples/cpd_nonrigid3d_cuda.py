import numpy as np
use_cuda = False
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
import open3d as o3
from probreg import cpd
import utils
import time

source, target = utils.prepare_source_and_target_nonrigid_3d('examples/face-x.txt', 'examples/face-y.txt', 
                                                            ds_param={'voxel_size':5.0, 'num':1000},
                                                            data_type='txt')

# TODO: extract mesh vertices and triangles from previous function
# TODO: put vertices and triangles into source_pt and target_pt, source_face, target_face
# TODO: if isinstance(hoge, o3.geometry.PointCloud) or isinstance(hoge, o3.geometry.TriangleMesh)


source_pt = cp.asarray(source.points, dtype=cp.float32)
target_pt = cp.asarray(target.points, dtype=cp.float32)

acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
start = time.time()
tf_param, _, _ = acpd.registration(target_pt)
elapsed = time.time() - start
print("time: ", elapsed)

print("result: ", to_cpu(tf_param.w), to_cpu(tf_param.g))

result = tf_param.transform(source_pt)
pc = o3.geometry.PointCloud()
pc.points = o3.utility.Vector3dVector(to_cpu(result))
pc.paint_uniform_color([0, 1, 0])
target.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([pc, target])