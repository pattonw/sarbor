# -*- coding: utf-8 -*-
from .watershed_source import JanSegmentationSource
import numpy as np
import logging

from .config import Config
from .skeletons import Skeleton


"""Main module."""


def query_diluvian(config: Config, output_file: str):

    return None


def query_watershed(config):
    logging.basicConfig(level=logging.INFO)

    skel = Skeleton(config)

    nodes = config.skeleton.nodes
    skel.input_nid_pid_x_y_z(nodes)

    if config.skeleton.strahler_filter and False:
        skel.filter_nodes_by_strahler(
            config.skeleton.min_strahler, config.skeleton.max_strahler
        )
    if config.skeleton.resample:
        processed_skel = skel.resample_segments()
    else:
        processed_skel = skel
        
    jans_segmentations = JanSegmentationSource()

    jans_segmentations.constants["fov_shape_voxels"] = np.array([45, 45, 45])

    jans_segmentations.segment_skeleton(processed_skel, num_processes=32)
    for node in processed_skel.get_nodes():
        try:
            data, bounds = jans_segmentations[tuple(node.value.center)]
            processed_skel.fill(node.key, (data > 127).astype(np.uint8))
            logging.info(
                "Node {} had data with max value {}!".format(node.key, data.max())
            )
        except KeyError:
            logging.info("No data for node {}!".format(node.key))
        except TypeError:
            logging.info("Node {} data was None".format(node.value.center))

    processed_skel.save_data_for_CATMAID()
