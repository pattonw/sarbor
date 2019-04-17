# -*- coding: utf-8 -*-
import numpy as np
import logging

from .config import Config
from .skeletons import Skeleton


"""Main module."""


def _process_skeleton(config: Config) -> Skeleton:

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
    return processed_skel


def _process_results(skel: Skeleton, results) -> Skeleton:
    for node in skel.get_nodes():
        try:
            data = results[node.key]
            skel.fill(node.key, data.astype(np.uint8))
            logging.debug(
                "Node {} had data with max value {}!".format(node.key, data.max())
            )
        except KeyError:
            logging.debug("No data for node {}!".format(node.key))
        except TypeError:
            logging.debug("Node {} data was None".format(node.key))
    return skel


def query_diluvian(config: Config, output_file: str):
    from .diluvian_source import DiluvianSource

    processed_skel = _process_skeleton(config)

    diluvian_source = DiluvianSource()

    results = diluvian_source.segment_skeleton(processed_skel)

    processed_skel = _process_results(processed_skel, results)

    processed_skel.save_data_for_CATMAID()

    return None


def query_watershed(config):
    from .watershed_source import JanSegmentationSource

    processed_skel = _process_skeleton(config)

    jans_segmentations = JanSegmentationSource(
        sensitives_file="/groups/cardona/home/pattonw/Code/Scripts/error_detection/sensitives.json"
    )

    results = jans_segmentations.segment_skeleton(processed_skel)

    processed_skel = _process_results(processed_skel, results)

    processed_skel.save_data_for_CATMAID()
