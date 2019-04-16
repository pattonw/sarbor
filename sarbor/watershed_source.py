from typing import List, Dict, Tuple, Optional
import numpy as np
import queue
import logging
from pymongo.errors import OperationFailure

import daisy
import lsd
import funlib.segment

from sarbor import Skeleton

import json
import time

from multiprocessing import Process, Manager, Value


Point3 = Tuple[int, int, int]
BoundBox = List[slice]
NodeData = Dict[Point3, Tuple[np.ndarray, BoundBox]]

"""
This class contains information about the Calyx volume used by the
futusa group

TODO: Add a sensitives file that keeps track of the file location
since that is also CALYX specific
"""


class BoundingBox:
    def __init__(
        self, start: np.ndarray, end: np.ndarray = None, shape: np.ndarray = None
    ):
        if end is None and shape is None:
            raise ValueError("Shape or End must be given!")
        self.start = start
        self.end = end or start + shape

    def contains_point(self, point: np.ndarray) -> bool:
        return all(self.start <= point) and all(self.end > point)

    def contains_point_with_radius(self, point: np.ndarray, radius: np.ndarray) -> bool:
        return self.contains_point(point - radius) and self.contains_point(
            point + radius
        )

    def contains_roi(self, start: np.ndarray, shape: np.ndarray):
        return self.contains_point(start) and self.contains_point(start + shape)

    def __str__(self):
        return "{} - {}".format(self.start, self.end)


class Calyx:
    def __init__(self, constants: Dict = {}):
        self.constants = constants

    @property
    def translation_phys(self):
        """
        Data translation between input coordinates (from CATMAID) to
        segmentation coordinates

        There is a one slice difference between catmaid and the fafb_v14 n5 volume
        used by the futusa group
        """
        return self.constants.get("translation", np.array([0, 0, -40]))

    @property
    def start(self) -> np.ndarray:
        """
        Coordinates in X,Y,Z order with units in nano-meters
        default: 403560, 121800, 158000
        """
        return self.constants.get("start", np.array([403560, 121800, 158000]))

    @property
    def shape(self) -> np.ndarray:
        """
        Shape in X,Y,Z order with units in nano-meters
        default: 64000, 52000, 76000
        """
        return self.constants.get("shape", np.array([64000, 52000, 76000]))

    @property
    def end(self) -> np.ndarray:
        """
        Coordinates in X,Y,Z order with units in nano-meters
        default: start + shape
        """
        return self.start + self.shape

    @property
    def resolution(self) -> np.ndarray:
        """
        Resolution in X,Y,Z order with units in nano-meters per voxel on each axis
        default: 4x4x40nm
        """
        return self.constants.get("resolution", np.array([4, 4, 40]))

    @property
    def bounding_box(self) -> BoundingBox:
        """
        Bounding box of the segmented region
        default: BoundingBox(start, shape)
        """
        return BoundingBox(self.start, shape=self.shape)


Point3 = Tuple[int, int, int]
BoundBox = List[slice]
NodeData = Dict[Point3, Tuple[np.ndarray, BoundBox]]


class JanSegmentationSource:
    def __init__(
        self,
        constants: Dict = {},
        sensitives_file: str = "sensitives.json",
        volume=None,
    ):
        if volume is None:
            self.volume = Calyx()
        else:
            self.volume = volume

        self.constants = constants

        self.sensitives = json.load(open(sensitives_file, "r"))

        self.mongo_db = self.sensitives["mongo_db"]
        self.frag_db_host = self.sensitives["frag_db_host"]
        self.frag_db_name = self.sensitives["frag_db_name"]
        self.edges_collection = self.sensitives["edges_collection"]
        self.mount_location = self.sensitives["mount_location"]
        self.rel_fragments_file = self.sensitives["rel_fragments_file"]
        self.fragments_dataset = self.sensitives["fragments_dataset"]

        self._node_segmentations = {}

    def __getitem__(self, key: Point3) -> Optional[Tuple[np.ndarray, List[slice]]]:
        return self._node_segmentations.get(key, None)

    def __setitem__(self, key: Point3, value: Tuple[np.ndarray, List[slice]]):
        if self._node_segmentations.get(key, None) is None:
            self._node_segmentations[key] = value
        else:
            raise ValueError("Updating segmentations is not yet supported")

    @property
    def translation_phys(self):
        return self.volume.translation_phys

    @property
    def start(self) -> np.ndarray:
        return self.volume.start

    @property
    def shape(self) -> np.ndarray:
        return self.volume.shape

    @property
    def end(self) -> np.ndarray:
        return self.volume.end

    @property
    def resolution(self) -> np.ndarray:
        return self.volume.resolution

    @property
    def scale(self) -> np.ndarray:
        return self.constants.get("scale", np.array([10, 10, 1]))

    @property
    def voxel_shape(self) -> np.ndarray:
        """
        The resolution of each voxel in the segmentation space
        This is calculated by multiplying the original resolution by the scale factor
        """
        return self.scale * self.resolution

    @property
    def fov_shape_voxels(self) -> np.ndarray:
        """
        Shape of a field of view around each node during segmentation in X,Y,Z order
        """
        shape = self.constants.get("fov_shape_voxels", np.array([31, 31, 31]))
        if any(shape % 2 == np.array([0, 0, 0])):
            raise ValueError(
                "Even fov_shapes are not supported yet since ",
                "there would be no 'middle' voxel for the sample point",
            )
        return shape

    @property
    def fov_shape(self) -> np.ndarray:
        return self.fov_shape_voxels * self.voxel_shape

    def _get_roi(self, center: np.ndarray) -> Tuple:
        """
        Get a region of interest from a center coordinate. Note that coordinates
        are provided by the input data space, thus we need to take into account
        the translation.

        This is a private function since using the translation will not provide
        the expected regions when used in other contexts
        """
        voxel_shape = self.voxel_shape
        fov_shape = self.fov_shape
        center_block = center - center % voxel_shape
        block_offset = (fov_shape // voxel_shape) // 2
        start = center_block - block_offset * voxel_shape
        start += self.translation_phys
        return start, fov_shape

    @property
    def bounding_box(self) -> BoundingBox:
        return self.volume.bounding_box

    @property
    def fragments_file(self):
        return "{}/{}".format(self.mount_location, self.rel_fragments_file)

    def data_fetcher(
        self,
        worker_id: int,
        node_queue: queue.Queue,
        results_queue: queue.Queue,
        done_workers: Value,
    ):
        while True:
            try:
                node_coords = node_queue.get(False)
                logging.debug("Got node {}!".format(node_coords))
            except queue.Empty:
                logging.debug("Worker {} Done".format(worker_id))
                with done_workers.get_lock():
                    done_workers.value += 1
                break
            try:
                stored = self[(node_coords[0], node_coords[1], node_coords[2])]
                if stored is None:
                    segmentation_data, bounds = self.get_segmentation(node_coords)
                    results_queue.put(
                        (
                            (node_coords[0], node_coords[1], node_coords[2]),
                            segmentation_data,
                            bounds,
                        )
                    )
                    logging.debug("Successfully segmented node {}".format(node_coords))
                else:
                    logging.debug(
                        "Node {} has previously been segmented!".format(node_coords)
                    )
            except ValueError as e:
                # This error should only be caused by the roi being outside
                # segmented volume bounds
                logging.debug("Node failed! {}".format(e))
                pass
            except OperationFailure as e:
                logging.warn(e)
            # except Exception as e:
            #    logging.warn("Unknown Error: {}".format(e))
            #    pass

    def query_local_segmentation(self, roi, threshold):
        # open fragments
        fragments = daisy.open_ds(self.fragments_file, self.fragments_dataset)

        # open RAG DB
        rag_provider = lsd.persistence.MongoDbRagProvider(
            self.frag_db_name,
            host=self.frag_db_host,
            mode="r",
            edges_collection=self.edges_collection,
        )

        segmentation = fragments[roi]
        segmentation.materialize()
        rag = rag_provider[roi]

        if len(rag.nodes()) == 0:
            return segmentation

        components = rag.get_connected_components(threshold)

        values_map = np.array(
            [
                [fragment, i]
                for i in range(len(components))
                for fragment in components[i]
            ],
            dtype=np.uint64,
        )
        old_values = values_map[:, 0]
        new_values = values_map[:, 1]
        funlib.segment.arrays.replace_values(
            segmentation.data, old_values, new_values, inplace=True
        )

        return segmentation

    def query_center_object(self, roi, threshold):
        segmentation = self.query_local_segmentation(roi, threshold)
        center = np.array(segmentation.shape) // 2
        center_obj = segmentation.data[center[0], center[1], center[2]]
        seg_fragments = set(segmentation.data.reshape(-1))
        values_map = np.array(
            [
                [fragment, 1 if fragment == center_obj else 0]
                for fragment in seg_fragments
            ],
            dtype=np.uint64,
        )
        old_values = values_map[:, 0]
        new_values = values_map[:, 1]
        funlib.segment.arrays.replace_values(
            segmentation.data, old_values, new_values, inplace=True
        )
        return segmentation

    def get_segmentation(self, center: np.ndarray) -> Tuple[np.ndarray, List[slice]]:
        """
        Jan's segmentation and the Saalfeld N5 have a 1 slice translation
        compared to the data in CATMAID. Thus we must scale the start
        down by one slice in the z direction.
        """
        roi_start, roi_shape = self._get_roi(center)
        if self.bounding_box.contains_roi(roi_start, roi_shape):
            # ROI input in physical dimensions i.e. nano meters [Z, Y, X].
            # Output in pixel dimensions (Z/40, Y/4, X/4)
            segmentation = (
                self.query_center_object(
                    daisy.Roi(roi_start[::-1], roi_shape[::-1]), threshold=0.3
                )
                .data.transpose([2, 1, 0])
                .astype(float)
            )
            try:
                data = (
                    (255 * segmentation)
                    .reshape(
                        [
                            segmentation.shape[0] // self.scale[0],
                            self.scale[0],
                            segmentation.shape[1] // self.scale[1],
                            self.scale[1],
                            segmentation.shape[2] // self.scale[2],
                            self.scale[2],
                        ]
                    )
                    .mean(5)
                    .mean(3)
                    .mean(1)
                )
            except Exception as e:
                logging.debug("Scale: {}".format(self.scale))
                logging.debug("Shape: {}".format(segmentation.shape))
                raise e

            downsampled_bounds = list(
                map(
                    slice,
                    center // self.resolution // self.scale
                    - self.fov_shape // self.resolution // self.scale // 2,
                    center // self.resolution // self.scale
                    + self.fov_shape // self.resolution // self.scale // 2
                    + 1,
                )
            )

            return (data, downsampled_bounds)
        else:
            logging.debug(
                "Center: {}; Queried bounds: {}; volume bounds: {}".format(
                    center, (roi_start, roi_shape), self.bounding_box
                )
            )
            raise ValueError("Roi is not contained in segmented volume")

    def segment_skeleton(self, skeleton: Skeleton, num_processes: int = 8) -> int:
        manager = Manager()
        # Queue of seeds to be picked up by workers.
        node_queue = manager.Queue()
        # Queue of results from workers.
        results_queue = manager.Queue()

        done_fetchers = Value("i", 0)

        branch_nodes = list(skeleton.get_interesting_nodes(branches=True))
        all_nodes = list(skeleton.get_nodes())
        num_nodes = len(all_nodes)
        num_branches = len(branch_nodes)
        logging.info("{} nodes with {} branches".format(num_nodes, num_branches))
        for node in all_nodes[:]:
            try:
                node_queue.put(node.value.center)
            except RecursionError:
                logging.debug("Maximum recursion depth hit. Too many nodes!")
                node_queue = manager.Queue()
                continue

        fetchers = []

        logging.debug("Starting Fetchers!")
        for fetcher_id in range(num_processes):
            fetcher = Process(
                target=self.data_fetcher,
                args=(fetcher_id, node_queue, results_queue, done_fetchers),
            )
            fetcher.start()
            fetchers.append(fetcher)

        num_done = 0
        start = time.time()
        while done_fetchers.value < num_processes or not results_queue.empty():
            try:
                node, data, bounds = results_queue.get(True, 5)
                try:
                    self[node] = (data, bounds)
                except ValueError:
                    logging.warning(
                        "Trying to override segmentation data on node {}!".format(node)
                    )
                num_done += 1
                if num_done % 50 == 0:
                    logging.info(
                        "{} out of {} done! avg: {:.3f} seconds per node".format(
                            num_done, num_nodes, (time.time() - start) / num_done
                        )
                    )
            except TypeError as e:
                logging.debug("Waiting...")
                logging.debug(e)
                num_done += 1
                pass
            except queue.Empty:
                logging.debug(
                    "Empty Queue! {}/{} fetchers done".format(
                        done_fetchers.value, num_processes
                    )
                )
                pass
        logging.info(
            "{} fetchers done! {} nodes skipped!".format(
                done_fetchers.value, num_nodes - num_done
            )
        )

        for wid, fetcher in enumerate(fetchers):
            fetcher.join()
            manager.shutdown()

        return num_done
