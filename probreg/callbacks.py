import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3

try:
    import cupy as cp

    asnumpy = cp.asnumpy
except:

    def asnumpy(x):
        return x


from .transformation import Transformation


class Plot2DCallback:
    """Display the 2D registration result of each iteration.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, save: bool = False, keep_window: bool = True):
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._cnt = 0
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        plt.plot(source[:, 0], source[:, 1], "ro", label="source")
        plt.plot(target[:, 0], target[:, 1], "g^", label="target")
        plt.plot(result[:, 0], result[:, 1], "bo", label="result")
        plt.legend()
        plt.draw()

    def __call__(self, transformation: Transformation) -> None:
        self._result = transformation.transform(self._source)
        plt.cla()
        plt.axis("equal")
        source = asnumpy(self._source)
        target = asnumpy(self._target)
        result = asnumpy(self._result)
        plt.plot(source[:, 0], source[:, 1], "ro", label="source")
        plt.plot(target[:, 0], target[:, 1], "g^", label="target")
        plt.plot(result[:, 0], result[:, 1], "bo", label="result")
        plt.legend()
        if self._save:
            plt.savefig("image_%04d.png" % self._cnt)
        plt.draw()
        plt.pause(0.001)
        self._cnt += 1


class Open3dVisualizerCallback:
    """Display the 3D registration result of each iteration.

    Args:
        source (open3d.geometry.PointCloud): Source point cloud data.
        target (open3d.geometry.PointCloud): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
        keep_window (bool, optional): If this flag is True,
            the drawing window blocks after registration is finished.
        fov: Field of view (degree).
    """

    def __init__(
        self,
        source: o3.geometry.PointCloud,
        target: o3.geometry.PointCloud,
        save: bool = False,
        keep_window: bool = True,
        fov: Any = None,
    ):
        self._vis = o3.visualization.Visualizer()
        self._vis.create_window()
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._keep_window = keep_window
        if not self._source.has_colors():
            self._source.paint_uniform_color([1, 0, 0])
        if not self._target.has_colors():
            self._target.paint_uniform_color([0, 1, 0])
        if not self._result.has_colors():
            self._result.paint_uniform_color([0, 0, 1])
        self._vis.add_geometry(self._source)
        self._vis.add_geometry(self._target)
        self._vis.add_geometry(self._result)
        if not fov is None:
            ctr = self._vis.get_view_control()
            ctr.change_field_of_view(step=fov)
        self._cnt = 0

    def __del__(self):
        if self._keep_window:
            self._vis.run()
        self._vis.destroy_window()

    def __call__(self, transformation: Transformation) -> None:
        self._result.points = transformation.transform(self._source.points)
        self._vis.update_geometry(self._source)
        self._vis.update_geometry(self._target)
        self._vis.update_geometry(self._result)
        self._vis.poll_events()
        self._vis.update_renderer()
        if self._save:
            self._vis.capture_screen_image("image_%04d.jpg" % self._cnt)
        self._cnt += 1

import polyscope
class PolyscopeVisualizerCallback:
    def __init__(self,
                 source,
                 target,
                 ) -> None:
        polyscope.init()
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        if not self._source.has_colors():
            self._source.paint_uniform_color([1, 0, 0])
        if not self._target.has_colors():
            self._target.paint_uniform_color([0, 1, 0])
        if not self._result.has_colors():
            self._result.paint_uniform_color([0, 0, 1])
        self.ps_source = polyscope.register_point_cloud(name='source', points=np.asarray(self._source.points))
        self.ps_source.add_color_quantity('color', np.asarray(self._source.colors))
        self.ps_target = polyscope.register_point_cloud(name='target',points=np.asarray(self._target.points))
        self.ps_target.add_color_quantity('color', np.asarray(self._target.colors))
        self.ps_result = polyscope.register_point_cloud(name='source_dash',points=np.asarray(self._result.points))
        self.ps_result.add_color_quantity('color', np.asarray(self._result.colors))
        self._cnt = 0

        polyscope.set_user_callback(self.__call__)
        polyscope.show()
        

    def __call__(self, transformation: Transformation):
        self._result.points = transformation.transform(self._source.points)
        self.ps_result.update_point_positions(np.asarray(self._result.points))
        self._cnt += 1

    def __del__(self):
        polyscope.clear_user_callback()