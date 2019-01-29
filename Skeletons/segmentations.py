import numpy as np

from .octrees import OctreeVolume

from nodes import NodeData


class Segmentation:
    def __init__(self):
        self._contained_count = None
        self._seen_count = None
        self._distances = None

        self._sphere = None

        self._fov_shape = None
        self._res = None

    @property
    def resolution(self) -> np.ndarray:
        if self._res is None:
            raise Exception("The resolution has not been defined for this tree")
        return self._res

    @resolution.setter
    def resolution(self, res: np.ndarray):
        if self._res is None:
            self._res = res
        else:
            raise Exception("Overwriting the skeletons resolution is not supported")

    @property
    def fov_shape(self) -> np.ndarray:
        if self._fov_shape is None:
            raise Exception(
                "The field of view dimensions have not been defined for this tree"
            )
        else:
            return self._fov_shape

    @fov_shape.setter
    def fov_shape(self, shape: np.ndarray):
        if self._fov_shape is None:
            self._fov_shape = shape
        else:
            raise Exception("Overwriting the skeletons fov shape is not supported")

    @property
    def sphere(self) -> np.ndarray:
        if self._sphere is None:
            shape = self.fov_shape
            res = self.resolution
            self._sphere = self._create_sphere(shape, res)
            return self._sphere
        else:
            return self._sphere

    @property
    def contained_count(self) -> OctreeVolume:
        """
        This octree contains counts of how many times a voxel was assigned
        a value of "in" the desired volume.
        """
        if self._contained_count is None:
            self.create_octrees_from_nodes()
        return self._contained_count

    @contained_count.setter
    def contained_count(self, seg: OctreeVolume):
        if self._contained_count is None:
            self._contained_count = seg
        else:
            raise Exception("trying to overwrite contained_count octree")

    @property
    def seen_count(self) -> OctreeVolume:
        """
        This octree is simply a volume where every voxel has a value
        equal to the number of field of views containing it
        """
        if self._seen_count is None:
            self.create_octrees_from_nodes()
        return self._seen_count

    @seen_count.setter
    def seen_count(self, tree: OctreeVolume):
        if self._seen_count is None:
            self._seen_count = tree
        else:
            raise Exception("trying to overwrite seen_count octree")

    @property
    def distances(self) -> OctreeVolume:
        if self._distances is None:
            self.create_octrees_from_nodes()
        return self._distances

    @distances.setter
    def distances(self, tree: OctreeVolume):
        if self._distances is None:
            self._distances = tree
        else:
            raise Exception("trying to overwrite distances octree")

    def create_octrees_from_nodes(self, node_data: NodeData):
        def _dist_block(dimensions, resolution):
            x = (
                (
                    np.linspace(-dimensions[0] // 2, dimensions[0] // 2, dimensions[0])
                    * resolution[0]
                )
                ** 2
            ).reshape(dimensions[0], 1, 1)
            y = (
                (
                    np.linspace(-dimensions[1] // 2, dimensions[1] // 2, dimensions[1])
                    * resolution[1]
                )
                ** 2
            ).reshape(1, dimensions[1], 1)
            z = (
                (
                    np.linspace(-dimensions[2] // 2, dimensions[2] // 2, dimensions[2])
                    * resolution[2]
                )
                ** 2
            ).reshape(1, 1, dimensions[2])
            return (x + y + z) ** (0.5) / np.sum(
                (dimensions // 2 * resolution) ** 2
            ) ** (0.5)

        def _data_populator(bounds):
            return np.zeros(np.array(bounds[1]) - np.array(bounds[0]))

        def _data_populator2(bounds):
            return np.full(
                np.array(bounds[1]) - np.array(bounds[0]), fill_value=float("inf")
            )

        self.contained_count = OctreeVolume(
            self.fov_shape,
            node_data.region_bounds(self.fov_shape),
            np.uint8,
            _data_populator,
        )
        self.seen_count = OctreeVolume(
            self.fov_shape,
            node_data.region_bounds(self.fov_shape),
            np.uint8,
            _data_populator,
        )
        self.distances = OctreeVolume(
            self.fov_shape,
            node_data.region_bounds(self.fov_shape),
            float,
            _data_populator2,
        )
        dist_block = _dist_block(self.fov_shape, self.resolution)

        for nid, node in node_data.nodes.items():
            node_bounds = node.value.get_bounds(self.fov_shape)

            node_bounds = list(map(slice, node_bounds[0], node_bounds[1]))
            if node.value.mask is not None:
                self.contained_count[node_bounds] += node.value.mask
            self.seen_count[node_bounds] += 1
            self.distances[node_bounds] = np.minimum(
                self.distances[node_bounds], dist_block
            )

    def save_data_n5(self, folder_path, dataset_path):
        """
        Save the data gathered in the n5 format.

        dependent on pyn5 library which is expected to change a lot.
        """
        datasets = {"segmentation": self.contained_count, "counts": self.seen_count}
        for name, data in datasets.items():
            print("Saving {} to n5!".format(name))
            print("Num leaves = {}".format(len(list(data.iter_leaves()))))
            data.write_to_n5(folder_path, dataset_path + "/" + name)
