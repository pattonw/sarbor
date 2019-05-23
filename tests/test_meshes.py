#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import numpy as np

from sarbor.meshes import (
    octree_to_sparse_vtk_volume,
    contour_sparse_vtk_volume,
    write_to_stl,
    read_from_stl,
    # visualize_mesh,
)
from sarbor.octrees import OctreeVolume

from pathlib import Path


class TestMeshes(unittest.TestCase):
    """Tests for meshes module."""

    def setUp(self):
        """
        Create blockwise sparse volume. This volume consists of `k` 3x3 cubes
        that overlap on the corners.
        """

        def populator(bounds):
            return np.zeros(np.array(bounds[1]) - np.array(bounds[0]))

        k = 20
        block = OctreeVolume(
            [10, 10, 10], [[0, 0, 0], [2 * k + 3, 2 * k + 3, 2 * k + 3]], int, populator
        )
        for i in range(k):
            block[
                list(
                    map(
                        slice,
                        [2 * i + 1, 2 * i + 1, 2 * i + 1],
                        [2 * i + 4, 2 * i + 4, 2 * i + 4],
                    )
                )
            ] = 1
        self.oct_volume = block

    def tearDown(self):
        test_mesh = Path("test.stl")
        if test_mesh.exists():
            test_mesh.unlink()

    def test_countouring(self):
        vtk_volume = octree_to_sparse_vtk_volume(self.oct_volume, cutoff=0.5)
        vtk_contour = contour_sparse_vtk_volume(vtk_volume, cutoff=0.5)
        write_to_stl(vtk_contour, "test.stl")
        vtk_contour_from_file = read_from_stl("test.stl")
        # visualize_mesh(vtk_contour_from_file)
        self.assertIsNotNone(vtk_contour_from_file)
