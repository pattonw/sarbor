#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sarbor` package."""


import unittest
import json
from pathlib import Path
import numpy as np
from sarbor import Skeleton
from sarbor.config import Config


class TestSarborReal(unittest.TestCase):
    """Tests for `sarbor` package on a real skeleton."""

    def setUp(self):
        def get_skeleton_from_json(f):
            with f.open("r") as of:
                dataset = json.load(of)
            ids, pids, users, xs, ys, zs, _, _ = zip(*dataset[0])
            nodes = zip(ids, pids, xs, ys, zs)
            return nodes

        nodes = get_skeleton_from_json(Path("test-data/27884/compact-detail.json"))
        config = Config()
        config.from_toml(Path("test-data/27884/config.toml"))
        self.skel = Skeleton(config)
        self.skel.input_nid_pid_x_y_z(nodes)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_resample(self):
        sampled_skel, original_node_mapping = self.skel.resample_segments()
        # Test leaves/branches/root didn't move
        og_interesting = [
            node
            for node in self.skel.get_interesting_nodes(
                leaves=True, branches=True, root=True
            )
        ]
        sampled_interesting = [
            node
            for node in sampled_skel.get_interesting_nodes(
                leaves=True, branches=True, root=True
            )
        ]
        self.assertEqual(len(og_interesting), len(sampled_interesting))
        self.assertTrue(
            all(
                [
                    np.allclose(a.value.center, b.value.center)
                    for a, b in zip(og_interesting, sampled_interesting)
                ]
            )
        )

        # Test strahler values for leaves/branches/root are equal
        sampled_skel.calculate_strahlers()
        self.skel.calculate_strahlers()
        self.assertTrue(
            all(
                [
                    a.strahler == b.strahler
                    for a, b in zip(og_interesting, sampled_interesting)
                ]
            )
        )

        # Test distance between nodes lies between resample_delta * 0.5 and resample_delta * 1.5
        # This might not be true if both nodes are "interesting" (root, branch, leaf), thus
        # we need to check that from each pair, at least one is normal (1 child)
        distances = [
            np.linalg.norm(node.value.center - node.parent.value.center)
            for node in sampled_skel.get_nodes()
            if node.parent is not None
            and (len(node.children) == 1 or len(node.parent.children) == 1)
        ]
        assert (
            np.max(distances) < sampled_skel.config.resample_delta * 1.5
        ), "max dist is {} but should be less than {}".format(
            np.max(distances), sampled_skel.config.resample_delta * 1.5
        )
        assert (
            np.min(distances) > sampled_skel.config.resample_delta * 0.5
        ), "min dist is {} but should be greater than {}".format(
            np.min(distances), sampled_skel.config.resample_delta * 0.5
        )
