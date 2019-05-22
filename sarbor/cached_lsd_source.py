from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import queue
import logging
from pymongo.errors import OperationFailure

import daisy
import lsd
import funlib.segment

from .sarbor import Skeleton
from .config import CachedLSDConfig

import json
import time

from multiprocessing import Process, Manager, Value

logger = logging.getLogger('sarbor')


class CachedLSDSource:
    def __init__(self, config_file, volume=None):
        self.config = CachedLSDConfig()
        self.config.from_toml(config_file)
        self.sensitives = json.load(Path(Path.home(), self.config.sensitives_file).open("r"))

        self.mongo_db = self.sensitives["mongo_db"]
        self.frag_db_host = self.sensitives["frag_db_host"]
        self.frag_db_name = self.sensitives["frag_db_name"]
        self.edges_collection = self.sensitives["edges_collection"]
        self.fragments_file = self.sensitives["fragments_file"]
        self.fragments_dataset = self.sensitives["fragments_dataset"]


    def data_fetcher(
        self,
        worker_id: int,
        node_queue: queue.Queue,
        results_queue: queue.Queue,
        done_workers: Value,
    ):
        while True:
            try:
                node_id, roi_start, roi_shape = node_queue.get(False)
                logger.debug("Got node {}!".format(node_id))
            except queue.Empty:
                logger.debug("Worker {} Done".format(worker_id))
                with done_workers.get_lock():
                    done_workers.value += 1
                break
            try:
                segmentation_data = self.get_segmentation(roi_start, roi_shape)
                results_queue.put((node_id, segmentation_data))
                logger.debug("Successfully segmented node {}".format(node_id))
            except ValueError as e:
                # This error should only be caused by the roi being outside
                # segmented volume bounds
                logger.debug("Node failed! {}".format(e))
                pass
            except OperationFailure as e:
                logger.warn(e)
            except AssertionError as e:
                results_queue.put((node_id, str(e)))
            # except Exception as e:
            #    logger.warn("Unknown Error: {}".format(e))
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

    def get_segmentation(
        self, roi_start: np.ndarray, roi_shape: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jan's segmentation and the Saalfeld N5 have a 1 slice translation
        compared to the data in CATMAID. Thus we must scale the start
        down by one slice in the z direction.

        Throws exception if ROI is out of bounds
        """
        translation = np.array([0, 0, -40])
        roi_start += translation
        return (
            self.query_center_object(
                daisy.Roi(roi_start[::-1], roi_shape[::-1]), threshold=0.3
            )
            .data.transpose([2,1,0])
            .astype(np.uint8)
        )

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
        logger.debug("{} nodes with {} branches".format(num_nodes, num_branches))
        for node in all_nodes[:]:
            try:
                roi_start, roi_stop = skeleton.seg.get_roi(node.value.center)
                roi_shape = roi_stop - roi_start
                node_queue.put((node.key, roi_start, roi_shape))
            except RecursionError:
                logger.debug("Maximum recursion depth hit. Too many nodes!")
                node_queue = manager.Queue()
                continue

        fetchers = []

        logger.debug("Starting Fetchers!")
        for fetcher_id in range(num_processes):
            fetcher = Process(
                target=self.data_fetcher,
                args=(fetcher_id, node_queue, results_queue, done_fetchers),
            )
            fetcher.start()
            fetchers.append(fetcher)

        num_done = 0
        start = time.time()
        results = {}
        while done_fetchers.value < num_processes or not results_queue.empty():
            try:
                node_id, data = results_queue.get(True, 5)
                results[node_id] = data if type(data) is not str else None
                num_done += 1
                if num_done % 50 == 0:
                    logger.debug(
                        "{} out of {} done! avg: {:.3f} seconds per node".format(
                            num_done, num_nodes, (time.time() - start) / num_done
                        )
                    )
            except TypeError as e:
                logger.debug("Waiting...")
                logger.debug(e)
                num_done += 1
                pass
            except queue.Empty:
                logger.debug(
                    "Empty Queue! {}/{} fetchers done".format(
                        done_fetchers.value, num_processes
                    )
                )
                pass
        logger.info(
            "{} fetchers done! {} nodes skipped!".format(
                done_fetchers.value, num_nodes - num_done
            )
        )

        for wid, fetcher in enumerate(fetchers):
            fetcher.join()
            manager.shutdown()

        return results
