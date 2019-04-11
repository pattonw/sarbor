#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sarbor` package."""


import unittest
import json
from pathlib import Path
import numpy as np
from sarbor import Skeleton

import sarbor

CUTOUT_OFFSET = np.array([120200, 50900, 0])
CUTOUT_DIMS = np.array([10600, 15850, 7063])
CUTOUT_RES = np.array([4, 4, 35])
VIEW_DIMS = np.array([97, 97, 25], dtype="int")
VIEW_DIMS += (VIEW_DIMS // 4) * 4
MODEL_DOWNSAMPLE = np.array([4, 4, 1])


class TestSarborToy(unittest.TestCase):
    """Tests for `sarbor` package."""

    def setUp(self):
        """
        data gathered from running diluvian on skel id 424271
        """
        self.skel = sarbor.Skeleton()
        self.skel.load("test-data/27884")

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_resample(self):
        def get_skeleton_from_json(f):
            with f.open("r") as of:
                dataset = json.load(of)
            ids, pids, users, xs, ys, zs, _, _ = zip(*dataset[0])
            nodes = zip(ids, pids, xs, ys, zs)
            lower_bound = (
                VIEW_DIMS // 2 * MODEL_DOWNSAMPLE + CUTOUT_OFFSET
            ) * CUTOUT_RES
            upper_bound = (
                CUTOUT_OFFSET + CUTOUT_DIMS - (VIEW_DIMS // 2 + 1) * MODEL_DOWNSAMPLE
            ) * CUTOUT_RES
            nodes = list(
                filter(
                    lambda x: lower_bound[0] <= x[2] < upper_bound[0]
                    and lower_bound[1] <= x[3] < upper_bound[1]
                    and lower_bound[2] <= x[4] < upper_bound[2],
                    nodes,
                )
            )
            return nodes

        og_nodes = get_skeleton_from_json(Path("test-data/27884/compact-detail.json"))
        og_skel = Skeleton(self.skel._config)
        og_skel.input_nid_pid_x_y_z(og_nodes)
        sampled_skel = og_skel.resample_segments()
        for node1, node2 in zip(self.skel.get_nodes(), sampled_skel.get_nodes()):
            assert all(
                np.isclose(node1.value.center, node2.value.center)
            ), "{} and {} are not close".format(node1, node2)

