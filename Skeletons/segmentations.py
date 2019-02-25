import numpy as np
from typing import Dict, Any, Union, List, Tuple, Iterable
from pathlib import Path
import pickle

from .octrees import OctreeVolume

from .arbors import SpatialArbor


class SegmentationSource:
    """
    Datastructure to contain volumetric data for algorithms based on segmentations.
    Contains 3 major sources of data:
    1) segmentation_counts (OctreeVolume[uint8]):
        How many times each voxel was selected during segmentation
    2) segmentation_views (OctreeVolume[uint8]):
        How many times each voxel was contained in a nodes field of view during segmentation
    3) distances (OctreeVolume[float32]):
        Minimum distance from each voxel to a sample point during segmentation

    segmentation_counts and views are stored seperately since storing their quotient as a float
    would be a lot more memory intense and calculating values on the fly is trivial.
    For a speed up consider using extra memory to store them together in ram
    """

    def __init__(self, constants: Dict[str, Any] = {}):
        self._constants = constants
        self._sphere = None

    @property
    def start_phys(self) -> np.ndarray:
        """
        Coordinates in X,Y,Z order with units in nano-meters
        default: 403560, 121800, 158000
        """
        return self.constants.get("start_phys", np.array([403560, 121800, 158000]))

    @property
    def start_voxel(self) -> np.ndarray:
        self.start // self.voxel_resolution

    @property
    def shape_phys(self) -> np.ndarray:
        """
        Shape in X,Y,Z order with units in nano-meters
        default: 64000, 52000, 76000
        """
        return self.constants.get("shape_phys", np.array([64000, 52000, 76000]))

    @property
    def shape_voxel(self) -> np.ndarray:
        return (self.phys_shape + self.voxel_resolution - 1) // self.voxel_resolution

    @property
    def end_phys(self) -> np.ndarray:
        """
        Coordinates in X,Y,Z order with units in nano-meters
        default: start + shape
        """
        return self.start_phys + self.shape_phys

    @property
    def end_voxel(self) -> np.ndarray:
        return self.start_voxel + self.shape_voxel

    @property
    def resolution_phys(self) -> np.ndarray:
        """
        nanometers per voxel in X,Y,Z order in original image space
        default: 4, 4, 40
        """
        return self._constants.get("original_resolution", np.array([4, 4, 40]))

    @property
    def seg_phys_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.start_phys, self.end_phys)

    @property
    def seg_voxel_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.start_voxel, self.end_voxel)

    @property
    def downsample_factor(self) -> np.ndarray:
        """
        Downsample for each axis in X,Y,Z order
        """
        return self._constants.get("downsample_scale", np.array([10, 10, 1]))

    @property
    def voxel_resolution(self) -> np.ndarray:
        """
        Resolution for each voxel in the segmentation in X,Y,Z order
        """
        return self.resolution_phys * self.downsample_factor

    @property
    def fov_shape_voxels(self) -> np.ndarray:
        """
        Shape of a field of view around each node during segmentation in X,Y,Z order
        """
        shape = self._constants.get("fov_voxel_shape", np.array([31, 31, 31]))
        if any(shape % 2 == np.array([0, 0, 0])):
            raise ValueError(
                "Even fov_shapes are not supported yet since ",
                "there would be no 'middle' voxel for the sample point",
            )
        return shape

    @property
    def fov_shape_phys(self) -> np.ndarray:
        return self.fov_shape_voxels * self.voxel_resolution

    @property
    def sphere(self) -> np.ndarray:
        """
        A 3D numpy array of shape fov_shape_voxels where each index [i,j,k] contains a bool
        indicating whether it is contained in the maximum sized sphere centered at
        [i//2, j//2, k//2] that fits in fov_shape_voxels.

        This does take into account voxel resolution.
        """
        if self._sphere is None:
            self._sphere = self._create_sphere()
        return self._sphere

    @staticmethod
    def _data_populator_factory(empty_value: Any, dtype: type):
        def data_populator(bounds: Tuple[np.ndarray, np.ndarray]):
            return np.full(
                np.array(bounds[1]) - np.array(bounds[0]),
                fill_value=empty_value,
                dtype=dtype,
            )

        return data_populator

    @property
    def segmentation_views(self) -> OctreeVolume:
        """
        This octree contains counts of how many times a voxel was contained
        in a sample points field of view.
        """
        if self._segmentation_views is None:
            self._segmentation_views = OctreeVolume(
                self.fov_shape,
                self.bounds,
                np.uint8,
                self._data_populator_factory(0, np.uint8),
            )
            self.create_octrees_from_nodes()
        return self._segmentation_views

    @property
    def segmentation_counts(self) -> OctreeVolume:
        """
        This octree contains counts of how many times a voxel was assigned
        a value of "in" the desired volume.
        """
        if self._segmentation_counts is None:
            self.segmentation_counts = OctreeVolume(
                self.fov_shape,
                self.bounds,
                np.uint8,
                self._data_populator_factory(0, np.uint8),
            )
            self.create_octrees_from_nodes()
        return self._segmentation_counts

    @property
    def distances(self) -> OctreeVolume:
        """
        This octree contains the distances from each voxel to its closest
        sample point.
        """
        if self._distances is None:
            self.distances = OctreeVolume(
                self.fov_shape,
                self.bounds,
                float,
                self._data_populator_factory(float("inf"), float),
            )
            self.create_octrees_from_nodes()
        return self._distances

    @staticmethod
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
        return (x + y + z) ** (0.5) / np.sum((dimensions // 2 * resolution) ** 2) ** (
            0.5
        )

    def create_octrees_from_nodes(self, node_data: SpatialArbor, sphere: bool = True):
        dist_block = self._dist_block(self.fov_shape, self.resolution)
        if sphere:
            dist_block[not self.sphere] = float("inf")

        for nid, node in node_data.nodes.items():
            node_bounds = node.value.get_bounds(self.fov_shape, slices=True)
            if node.value.mask is not None:
                self.segmentation_views[node_bounds] += node.value.mask
            self.segmentation_counts[node_bounds] += 1
            self.distances[node_bounds] = np.minimum(
                self.distances[node_bounds], dist_block
            )

    def save_data(self, folder_path: Path):
        """
        Save all the data necessary to rebuild this class.
        OctreeVolumes are written to n5 datasets, and constants are pickled
        """
        datasets = {
            "segmentation_views": self.segmentation_views,
            "segmentation_counts": self.segmentation_counts,
            "distances": self.distances,
        }
        for name, data in datasets.items():
            print("Saving {} to n5!".format(name))
            print("Num leaves = {}".format(len(list(data.iter_leaves()))))
            data.write_to_n5(folder_path, name)
        pickle.dump(self._constants, open(Path(folder_path, "constants.obj"), "wb"))

    def load_data(self, folder_path: Path):
        self._constants = pickle.load(open(Path(folder_path, "constants.obj"), "rb"))
        self._segmentation_views = OctreeVolume.read_from_n5(
            folder_path, "segmentation_views", self.shape_voxel
        )
        self._segmentation_views = OctreeVolume.read_from_n5(
            folder_path, "segmentation_counts", self.shape_voxel
        )
        self._segmentation_views = OctreeVolume.read_from_n5(
            folder_path, "distances", self.shape_voxel
        )

    def transform_bounds(
        self, bounds: Tuple[np.ndarray, np.ndarray], slices: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], List[slice]]:
        """
        Takes bounds in tuple format ((a,b,c), (A,B,C)) and converts them into slices
        [a:A, b:B, c:C] in voxel space
        """
        assert all(
            bounds[0] < bounds[1]
        ), "Resulting shape must be positive on all axes"
        assert all(
            bounds[0] % self.voxel_resolution == 0
        ), "Lower bound does not start on a voxel"
        assert all(
            (bounds[1] - bounds[0]) % self.voxel_resolution == 0
        ), "Queried shape must be a multiple of the voxel shape"
        if slices:
            return list(
                map(
                    slice,
                    bounds[0] // self.voxel_resolution,
                    bounds[1] // self.voxel_resolution,
                )
            )
        else:
            return (
                bounds[0] // self.voxel_resolution,
                bounds[1] // self.voxel_resolution,
            )

    def boolean_mask(self, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        voxel_bounds = self.transform_bounds(bounds, slices=True)
        return self.segmentation_counts[voxel_bounds] > 0

    @staticmethod
    def _slices(bounds: Tuple(Iterable, Iterable)):
        return list(map(slice, bounds[0], bounds[1]))

    def get_roi(self, center: np.ndarray) -> Tuple:
        voxel_shape = self.voxel_resolution
        fov_shape = self.fov_shape_phys
        center_block = center - center % voxel_shape
        block_offset = (fov_shape // voxel_shape) // 2
        start = center_block - block_offset * voxel_shape
        end = center_block + (block_offset + 1) * voxel_shape
        assert (
            start - end == fov_shape
        ), "ROI does not cover the expected area: {} vs {}".format(
            start - end, fov_shape
        )
        return start, end

    def dist_weighted_boolean_mask(self, center: np.ndarray) -> np.ndarray:
        voxel_bounds = self._slices(self.get_roi(center))
        return self.boolean_mask(voxel_bounds) * self.distances[voxel_bounds]

    def view_weighted_mask(self, center: np.ndarray, incr_denom: int = 1) -> np.ndarray:
        voxel_bounds = self._slices(self.get_roi(center))
        return self.segmentation_counts[voxel_bounds] / (
            self.segmentation_views[voxel_bounds] + incr_denom
        )

    def dist_view_weighted_mask(self, center: np.ndarray) -> np.ndarray:
        voxel_bounds = self._slices(self.get_roi(center))
        return (
            self.get_view_weighted_mask(voxel_bounds, 1) * self.distances[voxel_bounds]
        )

