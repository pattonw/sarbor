import numpy as np
from typing import Dict, Any, List, Tuple, Iterable
from pathlib import Path
import pickle
import logging

from .octrees import OctreeVolume
from .config import SegmentationsConfig
from .arbors import Node

logger = logging.getLogger('sarbor')

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
    """

    def __init__(self, config: SegmentationsConfig, constants: Dict[str, Any] = {}):
        self._constants = constants
        self._sphere = None

        # Octrees
        self._segmentation_views = None
        self._segmentation_counts = None
        self._distances = None

        # Config
        self._config = config

    @property
    def resolution_phys(self) -> np.ndarray:
        return self._config.resolution_phys

    @property
    def start_phys(self) -> np.ndarray:
        return self._config.start_phys

    @property
    def shape_phys(self) -> np.ndarray:
        return self._config("shape_phys")

    @property
    def end_phys(self) -> np.ndarray:
        return self._config.end_phys

    @property
    def voxel_resolution(self) -> np.ndarray:
        return self._config.voxel_resolution

    @property
    def start_voxel(self) -> np.ndarray:
        return self._config.start_voxel

    @property
    def shape_voxel(self) -> np.ndarray:
        return self._config.shape_voxel

    @property
    def end_voxel(self) -> np.ndarray:
        return self._config.end_voxel

    @property
    def seg_phys_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._config.seg_phys_bounds

    @property
    def seg_voxel_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._config.seg_voxel_bounds

    @property
    def downsample_factor(self) -> np.ndarray:
        return self._config.downsample_factor

    @property
    def fov_shape_voxels(self) -> np.ndarray:
        return self._config.fov_shape_voxels

    @property
    def fov_shape_phys(self) -> np.ndarray:
        return self._config.fov_shape_phys

    @property
    def leaf_shape_voxels(self) -> np.ndarray:
        return self._config.leaf_shape_voxels

    @property
    def sphere(self) -> np.ndarray:
        """
        A 3D numpy array of shape fov_shape_voxels where each index [i,j,k] contains a bool
        indicating whether it is contained in the maximum sized sphere centered at
        [i//2, j//2, k//2] that fits in fov_shape_voxels.

        This does take into account voxel resolution.
        """
        if self._sphere is None:
            self._sphere = self._create_sphere(
                self.fov_shape_voxels, self.voxel_resolution
            )
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
                self.leaf_shape_voxels,
                self.seg_voxel_bounds,
                np.uint8,
                self._data_populator_factory(0, np.uint8),
            )
        return self._segmentation_views

    @property
    def segmentation_counts(self) -> OctreeVolume:
        """
        This octree contains counts of how many times a voxel was assigned
        a value of "in" the desired volume.
        """
        if self._segmentation_counts is None:
            self._segmentation_counts = OctreeVolume(
                self.leaf_shape_voxels,
                self.seg_voxel_bounds,
                np.uint8,
                self._data_populator_factory(0, np.uint8),
            )
        return self._segmentation_counts

    @property
    def distances(self) -> OctreeVolume:
        """
        This octree contains the distances from each voxel to its closest
        sample point.
        """
        if self._distances is None:
            self._distances = OctreeVolume(
                self.leaf_shape_voxels,
                self.seg_voxel_bounds,
                float,
                self._data_populator_factory(float("inf"), float),
            )
        return self._distances

    @staticmethod
    def _create_sphere(shape, resolution):
        """
        Create a roughly isotropic shpere constrained in the bounds of shape to
        avoid letting non-isotropic data bias calculations.

        Especially important when detecting missing branches since we want to be
        able to detect branches in the z direction, and not let them get over powered by
        the extended view range in the x-y directions
        """

        def dist_to_center(i, j, k, shape, resolution):
            i = (
                # scale: [0-shape-1] - [-shape-1, shape-1]
                (2 * (i - shape[0] // 2))
                # scale up by resolution to get isotropic distances
                * resolution[0]
                # scale shortest axis down to [-1,1]
                / np.min(shape * resolution)
            )
            j = (2 * (j - shape[1] // 2)) * resolution[1] / np.min(shape * resolution)
            k = (2 * (k - shape[2] // 2)) * resolution[2] / np.min(shape * resolution)
            return (i ** 2 + j ** 2 + k ** 2) ** (0.5)

        sphere = np.ones(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if dist_to_center(i, j, k, shape, resolution) > 1:
                        sphere[i, j, k] = 0
        return sphere

    @staticmethod
    def _dist_block(dimensions, resolution):
        half_dim = dimensions // 2
        x = (
            (np.linspace(-half_dim[0], half_dim[0], dimensions[0]) * resolution[0]) ** 2
        ).reshape(dimensions[0], 1, 1)
        y = (
            (np.linspace(-half_dim[1], half_dim[1], dimensions[1]) * resolution[1]) ** 2
        ).reshape(1, dimensions[1], 1)
        z = (
            (np.linspace(-half_dim[2], half_dim[2], dimensions[2]) * resolution[2]) ** 2
        ).reshape(1, 1, dimensions[2])
        return (x + y + z) ** (0.5) / np.sum((half_dim * resolution) ** 2) ** (0.5)

    def create_octrees_from_nodes(self, nodes: Iterable[Node]):
        dist_block = self._dist_block(self.fov_shape_voxels, self.voxel_resolution)
        if self._config.use_sphere:
            dist_block[np.logical_not(self.sphere)] = float("inf")

        for node in nodes:
            node_bounds = self.transform_bounds(self.get_roi(node.value.center))
            if node.value.mask is not None:
                self.segmentation_counts[node_bounds] += node.value.mask
            self.segmentation_views[node_bounds] += 1
            self.distances[node_bounds] = np.minimum(
                self.distances[node_bounds], dist_block
            )
            if (
                self._config.interpolate_distance_nodes > 0
                and node.children is not None
            ):
                for neighbor in node.children:
                    for k in range(1, self._config.interpolate_distance_nodes + 1):
                        linear_step = neighbor.value.center * k / (
                            self._config.interpolate_distance_nodes + 1
                        ) + node.value.center * (
                            self._config.interpolate_distance_nodes - k
                        ) / (
                            self._config.interpolate_distance_nodes + 1
                        )
                        mid_bounds = self.transform_bounds(self.get_roi(linear_step))
                        self.distances[mid_bounds] = np.minimum(
                            self.distances[mid_bounds], dist_block
                        )

    def save_data(self, folder_path: Path):
        """
        Save all the data necessary to rebuild this class.
        OctreeVolumes are written to n5 datasets, and constants are pickled
        
        TODO option to choose which datasets to save, and option to precompute
        and save some of the other data views (dist weighted mask etc)

        """
        datasets = {
            "segmentation_views": self.segmentation_views,
            "segmentation_counts": self.segmentation_counts,
            "distances": self.distances,
        }
        for name, data in datasets.items():
            logger.debug("Saving {} to n5!".format(name))
            logger.debug("Num leaves = {}".format(len(list(data.iter_leaves()))))
            data.write_to_n5(folder_path + "/segmentations.n5", name)
        pickle.dump(self._constants, Path(folder_path, "constants.obj").open("wb"))

    def load_data(self, folder_path: Path):
        self._constants = pickle.load(Path(folder_path, "constants.obj").open("rb"))
        self._segmentation_views = OctreeVolume.read_from_n5(
            folder_path, "segmentation_views", self.shape_voxel
        )
        self._segmentation_counts = OctreeVolume.read_from_n5(
            folder_path, "segmentation_counts", self.shape_voxel
        )
        self._distances = OctreeVolume.read_from_n5(
            folder_path, "distances", self.shape_voxel
        )

    def extract_data(self):
        return self._constants

    def transform_bounds(self, bounds: Tuple[np.ndarray, np.ndarray]) -> Tuple[slice]:
        """
        Takes bounds in tuple format ((a,b,c), (A,B,C)) and converts them into slices
        [a:A, b:B, c:C] in voxel space
        """
        # TODO: move assertions into a proper unittest
        assert all(
            bounds[0] < bounds[1]
        ), "Resulting shape must be positive on all axes"
        assert all(
            bounds[0] % self.voxel_resolution == 0
        ), "Lower bound does not start on a voxel"
        assert all(
            (bounds[1] - bounds[0]) % self.voxel_resolution == 0
        ), "Queried shape must be a multiple of the voxel shape"
        return tuple(
            map(
                slice,
                (bounds[0] // self.voxel_resolution).astype(int),
                (bounds[1] // self.voxel_resolution).astype(int),
            )
        )

    def get_roi(self, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        voxel_shape = self.voxel_resolution
        fov_shape = self.fov_shape_phys
        center_block = center - center % voxel_shape
        block_offset = (fov_shape // voxel_shape) // 2
        start = center_block - block_offset * voxel_shape
        end = center_block + (block_offset + 1) * voxel_shape
        # TODO: put this is a test file
        assert all(
            end - start == fov_shape
        ), "ROI does not cover the expected area: {} vs {}".format(
            end - start, fov_shape
        )
        return start, end

    def boolean_mask(self, center: np.ndarray) -> np.ndarray:
        bounds = self.transform_bounds(self.get_roi(center))
        mask = self.segmentation_counts[bounds] > 0
        if self._config.use_sphere:
            mask[np.logical_not(self.sphere)] = False
        return mask

    def _boolean_mask(self, bounds: List[slice]) -> np.ndarray:
        mask = self.segmentation_counts[bounds] > 0
        if np.isnan(mask).any():
            raise ValueError("boolean_mask contains NAN!")
        if np.isinf(mask).any():
            raise ValueError("boolean_mask contains INF!")
        return mask

    def dist_weighted_boolean_mask(self, center: np.ndarray) -> np.ndarray:
        bounds = self.transform_bounds(self.get_roi(center))
        mask = self._dist_weighted_boolean_mask(bounds)
        if self._config.use_sphere:
            mask[np.logical_not(self.sphere)] = 0
        return mask

    def _dist_weighted_boolean_mask(self, bounds: List[slice]):
        mask = self._boolean_mask(bounds) * self._distance_mask(bounds)
        if np.isnan(mask).any():
            raise ValueError("dist_weighted_boolean_mask contains NAN!")
        if np.isinf(mask).any():
            raise ValueError("dist_weighted_boolean_mask contains INF!")
        return mask

    def view_weighted_mask(self, center: np.ndarray) -> np.ndarray:
        bounds = self.transform_bounds(self.get_roi(center))
        mask = self._view_weighted_mask(bounds, incr_denom=self._config.incr_denom)
        if self._config.use_sphere:
            mask[np.logical_not(self.sphere)] = 0
        return mask

    def _view_weighted_mask(self, bounds: List[slice]) -> np.ndarray:
        mask = self.segmentation_counts[bounds] / (
            self.segmentation_views[bounds] + self._config.incr_denom
        )
        assert mask.max() <= 1, "Cannot have confidence above 100%"
        if np.isnan(mask).any():
            raise ValueError("view_weighted_mask contains NAN!")
        if np.isinf(mask).any():
            raise ValueError("view_weighted_mask contains INF!")
        return mask

    def dist_view_weighted_mask(self, center: np.ndarray) -> np.ndarray:
        bounds = self.transform_bounds(self.get_roi(center))
        mask = self._dist_view_weighted_mask(bounds)
        if self._config.use_sphere:
            mask[np.logical_not(self.sphere)] = 0
        return mask

    def _dist_view_weighted_mask(self, bounds: List[slice]) -> np.ndarray:
        mask = self._view_weighted_mask(bounds) * self._distance_mask(bounds)
        if np.isnan(mask).any():
            raise ValueError("dist_view_weighted_mask contains NAN!")
        if np.isinf(mask).any():
            raise ValueError("dist_view_weighted_mask contains INF!")
        return mask

    def _distance_mask(self, bounds: List[slice]) -> np.ndarray:
        distances = self.distances[bounds]
        logger.debug(
            "Percent of distances seen that are infinite is: {}".format(
                np.isinf(distances).sum() / np.prod(distances.shape)
            )
        )
        distances[np.isinf(distances)] = 0
        return distances
