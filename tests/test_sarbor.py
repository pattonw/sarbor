#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sarbor` package."""


import unittest

import sarbor


class TestSarborToy(unittest.TestCase):
    """Tests for `sarbor` package."""

    def setUp(self):
        self.skeleton = sarbor.Skeleton()
        self.skeleton.input_nid_pid_x_y_z(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [2, 1, 2, 0, 0],
                [3, 2, 3, 0, 0],
                [4, 3, 4, 0, 0],
                [5, 4, 5, 0, 0],
                [6, 0, 0, 1, 0],
                [7, 6, 0, 2, 0],
                [8, 7, 0, 3, 0],
                [9, 8, 0, 4, 0],
                [10, 9, 0, 5, 0],
            ]
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_get_nodes(self):
        self.assertEqual(
            set(map(lambda node: node.key, self.skeleton.get_nodes())),
            set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )
