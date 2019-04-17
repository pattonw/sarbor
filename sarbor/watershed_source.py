from typing import List, Dict, Tuple
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

"""
This class contains information about the Calyx volume used by the
futusa group

TODO: Add a sensitives file that keeps track of the file location
since that is also CALYX specific
"""


class JanSegmentationSource:
    def __init__(self, sensitives_file: str = "sensitives.json", volume=None):
        self.sensitives = json.load(open(sensitives_file, "r"))

        self.mongo_db = self.sensitives["mongo_db"]
        self.frag_db_host = self.sensitives["frag_db_host"]
        self.frag_db_name = self.sensitives["frag_db_name"]
        self.edges_collection = self.sensitives["edges_collection"]
        self.mount_location = self.sensitives["mount_location"]
        self.rel_fragments_file = self.sensitives["rel_fragments_file"]
        self.fragments_dataset = self.sensitives["fragments_dataset"]

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
                node_id, roi_start, roi_shape = node_queue.get(False)
                logging.debug("Got node {}!".format(node_id))
            except queue.Empty:
                logging.debug("Worker {} Done".format(worker_id))
                with done_workers.get_lock():
                    done_workers.value += 1
                break
            try:
                segmentation_data = self.get_segmentation(roi_start, roi_shape)
                results_queue.put((node_id, segmentation_data))
                logging.debug("Successfully segmented node {}".format(node_id))
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
            .data.transpose([2, 1, 0])
            .astype(float)
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
        logging.debug("{} nodes with {} branches".format(num_nodes, num_branches))
        for node in all_nodes[:]:
            try:
                roi_start, roi_shape = skeleton.seg.get_roi(node.value.center)
                node_queue.put((node.key, roi_start, roi_shape))
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
        results = {}
        while done_fetchers.value < num_processes or not results_queue.empty():
            try:
                node_id, data = results_queue.get(True, 5)
                results[node_id] = data
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

        return num_done, results
