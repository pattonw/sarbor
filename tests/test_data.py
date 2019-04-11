#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sarbor` package."""


import unittest
import json
from pathlib import Path
import numpy as np
from sarbor import Skeleton
from sarbor.config import Config

import sarbor


class TestSarborToy(unittest.TestCase):
    """Tests for `sarbor` package."""

    def setUp(self):
        """
        data gathered from running diluvian on skel id 424271
        """

        def get_skeleton_from_json(f):
            with f.open("r") as of:
                dataset = json.load(of)
            ids, pids, users, xs, ys, zs, _, _ = zip(*dataset[0])
            nodes = zip(ids, pids, xs, ys, zs)
            return nodes

        nodes = get_skeleton_from_json(Path("test-data/27884/compact-detail.json"))
        self.skel = Skeleton(Config.from_toml("test-data/27884/config.toml"))
        self.skel.input_nid_pid_x_y_z(nodes)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_resample(self):
        sampled_skel = self.skel.resample_segments()
        # Test leaves/branches/root didn't move
        og_interesting = sorted(
            [
                node.value.center
                for node in self.skel.get_interesting_nodes(
                    leaves=True, branches=True, root=True
                )
            ],
            key=lambda x: tuple(x),
        )
        sampled_interesting = sorted(
            [
                node.value.center
                for node in sampled_skel.get_interesting_nodes(
                    leaves=True, branches=True, root=True
                )
            ],
            key=lambda x: tuple(x),
        )
        self.assertTrue(
            all(
                [np.allclose(a, b) for a, b in zip(og_interesting, sampled_interesting)]
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
            np.max(distances) < sampled_skel.config.resample_delta * 1.6
        ), "max dist is {} but should be less than {}".format(
            np.max(distances), sampled_skel.config.resample_delta * 1.6
        )
        assert (
            np.min(distances) > sampled_skel.config.resample_delta * 0.5
        ), "min dist is {} but should be greater than {}".format(
            np.min(distances), sampled_skel.config.resample_delta * 0.5
        )
        assert (
            np.mean(distances) > 250
        ), "mean dist is {} but should be between {} and {}".format(
            np.mean(distances),
            sampled_skel.config.resample_delta * 0.5,
            sampled_skel.config.resample_delta * 1.5,
        )

