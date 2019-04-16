# -*- coding: utf-8 -*-
"""Global configuration objects.

This module contains boilerplate configuration objects for storing and loading
configuration state.
"""

import numpy as np
import pytoml as toml

from typing import Tuple


class BaseConfig(object):
    """Base class for configuration objects.

    String representation yields TOML that should parse back to a dictionary
    that will initialize the same configuration object.
    """

    def __str__(self):
        sanitized = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                sanitized[k] = v
        return toml.dumps(sanitized)

    __repr__ = __str__


class SegmentationsConfig(BaseConfig):
    """Configuration for segmentation data. uses X,Y,Z order

    Attributes
    ----------
    resolution_phys : sequence or ndarray of int
        Nanometers per voxel in original image space.
    start_phys : sequence or ndarray of int
        Start coordinates in nm if not using whole volume
    shape_phys : sequence or ndarray of int
        Shape of volume in nm s.t. start + shape = end
    end_phys : sequence or ndarray of int
        Upper bound of volume s.t. (start:end) contains whole volume
    downsample_factor : sequence or ndarray of int
        How much to downsample each axis
    leaf_shape_voxels : sequence or ndarray of int
        Voxels per leaf block in octrees and n5 TODO: make these definable seperately
    fov_shape_phys : sequence or ndarray of int
        Shape of fov in nm
    fov_shape_voxels : sequence or ndarray of int
        Shape of fov in voxels
    incr_denom : float
        Amount to increment denominator to get confidence scores (n/m)
        where n is the number of times a voxel was selected as "contained"
        and m is the number of times a voxel could have been selected as "contained"
    use_sphere : bool
        Whether or use a spherical field of view around each sample point
    interpolate_distance_nodes : int
        How many virtual nodes to place between sample points to make the
        distance heuristic more cylindrical. More virtual points reduce
        the chance of selecting a voxel between two sample points as a potential
        missing branch
    """

    def __init__(self, settings):
        self.resolution_phys = np.array(settings.get("resolution_phys", [4, 4, 40]))
        self.start_phys = np.array(settings.get("start_phys", [403560, 121800, 158000]))
        self.shape_phys = np.array(settings.get("shape_phys", [64000, 52000, 76000]))
        self.downsample_factor = np.array(
            settings.get("downsample_factor", [10, 10, 1])
        )
        self.leaf_shape_voxels = np.array(
            settings.get("leaf_shape_voxels", [64, 64, 64])
        )
        self.fov_shape_phys = settings.get("fov_shape_phys", None)
        self.fov_shape_voxels = np.array(settings.get("fov_shape_voxels", [31, 31, 31]))
        self.validate_fov_shape()
        self.incr_denom = float(settings.get("incr_denom", 1))
        self.use_sphere = bool(settings.get("use_sphere", True))
        self.interpolate_distance_nodes = int(
            settings.get("interpolate_distance_nodes", 0)
        )

    @property
    def end_phys(self) -> np.ndarray:
        """
        end of volume in nanometers
        """
        return self.start_phys + self.shape_phys

    @property
    def voxel_resolution(self) -> np.ndarray:
        """
        nanometers per voxel
        """
        return self.resolution_phys * self.downsample_factor

    @property
    def start_voxel(self) -> np.ndarray:
        """
        start of the volume in voxel space.
        Rounded down so it can extend lower than the start in physical coordinates
        TODO: make this always align with the start coordinates in nm space
        """
        return self.start_phys // self.voxel_resolution

    @property
    def shape_voxel(self) -> np.ndarray:
        """
        Shape of the volume in voxels
        """
        return (self.shape_phys + self.voxel_resolution - 1) // self.voxel_resolution

    @property
    def end_voxel(self) -> np.ndarray:
        """
        First voxel not contained in the volume
        """
        return self.start_voxel + self.shape_voxel

    @property
    def seg_phys_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        physical bounds of the volume
        """
        return (self.start_phys, self.end_phys)

    @property
    def seg_voxel_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        voxel bounds of the volume
        """
        return (self.start_voxel, self.end_voxel)

    def validate_fov_shape(self):
        """
        make sure that the fov shape is consistent accross the physical and
        voxel shape descriptions since both are user definable.
        """
        if self.fov_shape_phys is None and self.fov_shape_voxels is None:
            raise ValueError("FOV shape must be defined in either voxels or nm")
        elif self.fov_shape_phys is None:
            self.fov_shape_phys = self.fov_shape_voxels * self.voxel_resolution
        elif self.fov_shape_voxels is None:
            assert np.equal(
                self.fov_shape_phys % self.voxel_resolution, np.zeros([3])
            ), "FOV dimensions are not a multiple of voxel dimensions: {} / {} = {}".format(
                self.fov_shape_phys,
                self.voxel_resolution,
                self.fov_shape_phys / self.voxel_resolution,
            )
            self.fov_shape_voxels = self.fov_shape_phys // self.voxel_resolution

        assert all(
            np.equal(self.fov_shape_phys, self.fov_shape_voxels * self.voxel_resolution)
        ), "Conflicting FOV shapes were defined: {} vs {}".format(
            self.fov_shape_phys, self.fov_shape_voxels * self.voxel_resolution
        )
        assert all(np.equal(self.fov_shape_voxels % 2, np.ones([3]))), (
            "FOV shape {} has to have an odd number of"
            + " voxels on each axis to have a center voxel"
        ).format(self.fov_shape_voxels)


class SkeletonConfig(BaseConfig):
    """Configuration for a Skeleton

    Attributes
    ----------
    """

    def __init__(self, settings):
        self.skeleton_csv = str(settings.get("skeleton_csv"))
        self.output_file_base = str(settings.get("output_file_base"))

        self.save_nodes = bool(settings.get("save_nodes", True))
        self.save_rankings = bool(settings.get("save_rankings", True))
        self.save_segmentations = bool(settings.get("save_segmentations", True))
        self.save_masks = bool(settings.get("save_masks", True))
        self.save_config = bool(settings.get("save_config", True))
        self.use_consensus = bool(settings.get("use_consensus", False))

        self.resample_delta = int(settings.get("resample_delta", 500))
        self.resample_steps = int(settings.get("resample_steps", 2000))
        self.resample_sigma = float(settings.get("resample_sigma", 0.05))


class Config(object):
    """A complete collection of configuration objects.

    Attributes
    ----------
    """

    def __init__(self, settings_collection=None):
        if settings_collection is not None:
            settings = settings_collection[0].copy()
            for s in settings_collection:
                for c in s:
                    if c in settings and isinstance(settings[c], dict):
                        settings[c].update(s[c])
                    else:
                        settings[c] = s[c]
        else:
            settings = {}

        self.segmentations = SegmentationsConfig(settings.get("segmentations", {}))
        self.skeleton = SkeletonConfig(settings.get("skeleton", {}))

    def __str__(self):
        sanitized = {}
        for n, c in self.__dict__.items():
            if not isinstance(c, BaseConfig):
                sanitized[n] = c
                continue
            sanitized[n] = {}
            for k, v in c.__dict__.items():
                if isinstance(v, np.ndarray):
                    sanitized[n][k] = v.tolist()
                else:
                    sanitized[n][k] = v
        return toml.dumps(sanitized)

    def from_toml(self, *filenames):
        """Reinitializes this Config from a list of TOML configuration files.

        Existing settings are discarded. When multiple files are provided,
        configuration is overridden by later files in the list.

        Parameters
        ----------
        filenames : interable of str
            Filenames of TOML configuration files to load.
        """
        settings = []
        for filename in filenames:
            with open(filename, "rb") as fin:
                settings.append(toml.load(fin))

        return self.__init__(settings)

    def to_toml(self, filename):
        with open(filename, "w") as tomlfile:
            tomlfile.write(str(self))
