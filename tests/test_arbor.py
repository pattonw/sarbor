#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sarbor` package."""


import unittest

import sarbor


class TestSarborToy(unittest.TestCase):
    """Tests for `sarbor` package."""

    def setUp(self):
        """
        0-1-2-3-4-5
            |   |
            6   10
            |
            7-9
            |
            8
        """
        self.skeleton = sarbor.Skeleton()
        self.skeleton.input_nid_pid_x_y_z(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [2, 1, 2, 0, 0],
                [3, 2, 3, 0, 0],
                [4, 3, 4, 0, 0],
                [5, 4, 5, 0, 0],
                [6, 2, 2, 1, 0],
                [7, 6, 2, 2, 0],
                [8, 7, 2, 3, 0],
                [9, 7, 3, 2, 0],
                [10, 4, 4, 1, 0],
            ]
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_get_nodes(self):
        self.assertEqual(
            set(map(lambda node: node.key, self.skeleton.get_nodes())),
            set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        )

    def test_get_root(self):
        self.assertEqual(self.skeleton.arbor.root.key, 0)

    def test_dfs(self):
        dfs_iter = self.skeleton.arbor.traverse()
        self.assertEqual(next(dfs_iter).key, 0)
        self.assertEqual(next(dfs_iter).key, 1)
        self.assertEqual(next(dfs_iter).key, 2)
        self.assertEqual(next(dfs_iter).key, 3)
        self.assertEqual(next(dfs_iter).key, 4)
        self.assertEqual(next(dfs_iter).key, 5)
        self.assertEqual(next(dfs_iter).key, 10)
        self.assertEqual(next(dfs_iter).key, 6)
        self.assertEqual(next(dfs_iter).key, 7)
        self.assertEqual(next(dfs_iter).key, 8)
        self.assertEqual(next(dfs_iter).key, 9)

    def test_get_strahler(self):
        self.skeleton.calculate_strahlers()
        self.assertEqual(self.skeleton.nodes[0].strahler, 3)
        self.assertEqual(self.skeleton.nodes[1].strahler, 3)
        self.assertEqual(self.skeleton.nodes[2].strahler, 3)
        self.assertEqual(self.skeleton.nodes[3].strahler, 2)
        self.assertEqual(self.skeleton.nodes[4].strahler, 2)
        self.assertEqual(self.skeleton.nodes[5].strahler, 1)
        self.assertEqual(self.skeleton.nodes[10].strahler, 1)
        self.assertEqual(self.skeleton.nodes[6].strahler, 2)
        self.assertEqual(self.skeleton.nodes[7].strahler, 2)
        self.assertEqual(self.skeleton.nodes[8].strahler, 1)
        self.assertEqual(self.skeleton.nodes[9].strahler, 1)

    @unittest.expectedFailure
    def test_new_root(self):
        raise NotImplementedError()

    def test_split_tree(self):
        chop = self.skeleton.delete_branch((2, 6))
        dfs_iter = chop.arbor.traverse()
        self.assertEqual([n.key for n in dfs_iter], [0, 1, 2, 3, 4, 5, 10])

        chop.calculate_strahlers()
        self.assertEqual(chop.nodes[0].strahler, 2)
        self.assertEqual(chop.nodes[1].strahler, 2)
        self.assertEqual(chop.nodes[2].strahler, 2)
        self.assertEqual(chop.nodes[3].strahler, 2)
        self.assertEqual(chop.nodes[4].strahler, 2)
        self.assertEqual(chop.nodes[5].strahler, 1)
        self.assertEqual(chop.nodes[10].strahler, 1)

    def test_get_segments(self):
        """
        breadth first segment iteration
        """
        segment_iter = self.skeleton.get_segments()
        self.assertEqual([node.key for node in next(segment_iter)], [0, 1, 2])
        self.assertEqual([node.key for node in next(segment_iter)], [2, 3, 4])
        self.assertEqual([node.key for node in next(segment_iter)], [2, 6, 7])
        self.assertEqual([node.key for node in next(segment_iter)], [4, 5])
        self.assertEqual([node.key for node in next(segment_iter)], [4, 10])
        self.assertEqual([node.key for node in next(segment_iter)], [7, 8])
        self.assertEqual([node.key for node in next(segment_iter)], [7, 9])

    @unittest.expectedFailure
    def test_merge_trees(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_re_index(self):
        raise NotImplementedError
