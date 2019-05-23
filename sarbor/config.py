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
        Voxels per leaf block in octrees and n5 TODO: make these seperately  definable?
    fov_shape : sequence or ndarray of int
        Shape of fov in nm
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
        self.start_phys = np.array(
            settings.get("start_phys", [403560, 121800, 158000])
        )  # Should calculate this from skeleton if not provided
        self.shape_phys = np.array(
            settings.get("shape_phys", [64000, 52000, 76000])
        )  # Should calculate this from skeleton if not provided
        self.downsample_factor = np.array(
            settings.get("downsample_factor", [10, 10, 1])
        )
        self.leaf_shape_voxels = np.array(
            settings.get("leaf_shape_voxels", [64, 64, 64])
        )
        self.fov_shape_phys = np.array(settings.get("fov_shape", [1240, 1240, 1240]))
        self.validate_fov_shape()
        self.incr_denom = float(settings.get("incr_denom", 1))
        self.use_sphere = bool(settings.get("use_sphere", True))
        self.interpolate_distance_nodes = int(
            settings.get("interpolate_distance_nodes", 3)
        )

    @property
    def resolution_phys(self) -> np.ndarray:
        return self._resolution_phys.astype(float)

    @resolution_phys.setter
    def resolution_phys(self, value):
        self._resolution_phys = value

    @property
    def start_phys(self) -> np.ndarray:
        return self._start_phys.astype(int)

    @start_phys.setter
    def start_phys(self, value):
        self._start_phys = value

    @property
    def shape_phys(self) -> np.ndarray:
        return self._shape_phys.astype(int)

    @shape_phys.setter
    def shape_phys(self, value):
        self._shape_phys = value

    @property
    def downsample_factor(self) -> np.ndarray:
        return self._downsample_factor.astype(int)

    @downsample_factor.setter
    def downsample_factor(self, value):
        self._downsample_factor = value

    @property
    def leaf_shape_voxels(self) -> np.ndarray:
        return self._leaf_shape_voxels.astype(int)

    @leaf_shape_voxels.setter
    def leaf_shape_voxels(self, value):
        self._leaf_shape_voxels = value

    @property
    def fov_shape_phys(self) -> np.ndarray:
        return self._fov_shape_phys.astype(float)

    @fov_shape_phys.setter
    def fov_shape_phys(self, value):
        self._fov_shape_phys = value

    @property
    def incr_denom(self):
        return self._incr_denom

    @incr_denom.setter
    def incr_denom(self, value):
        self._incr_denom = value

    @property
    def use_sphere(self):
        return self._use_sphere

    @use_sphere.setter
    def use_sphere(self, value):
        self._use_sphere = value

    @property
    def interpolate_distance_nodes(self):
        return self._interpolate_distance_nodes

    @interpolate_distance_nodes.setter
    def interpolate_distance_nodes(self, value):
        self._interpolate_distance_nodes = value

    @property
    def fov_shape_voxels(self) -> np.ndarray:
        return (self._fov_shape_phys // self.voxel_resolution).astype(int)

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
        assert all(
            np.equal(self.fov_shape_phys % self.voxel_resolution, np.zeros([3]))
        ), "FOV dimensions are not a multiple of voxel dimensions: {} / {} = {}".format(
            self.fov_shape_phys,
            self.voxel_resolution,
            self.fov_shape_phys / self.voxel_resolution,
        )

        assert all(np.equal(self.fov_shape_voxels % 2, np.ones([3]))), (
            "FOV shape in voxels {} has to have an odd number of"
            + " voxels on each axis to have a center voxel"
        ).format(self.fov_shape_voxels)


class SkeletonConfig(BaseConfig):
    """Configuration for a Skeleton

    Attributes
    ----------
    """

    def __init__(self, settings):
        self.csv = str(settings.get("csv"))
        self.output_file_base = str(settings.get("output_file_base"))

        self.save_nodes = bool(settings.get("save_nodes", True))
        self.save_rankings = bool(settings.get("save_rankings", True))
        self.save_segmentations = bool(settings.get("save_segmentations", True))
        self.save_masks = bool(settings.get("save_masks", True))
        self.save_config = bool(settings.get("save_config", True))
        self.use_consensus = bool(settings.get("use_consensus", False))

        self.resample = bool(settings.get("resample", True))
        self.smoothing = str(settings.get("smoothing", "none"))
        self.resample_delta = int(settings.get("resample_delta", 50))
        self.resample_steps = int(settings.get("resample_steps", 2000))
        self.resample_sigma = float(settings.get("resample_sigma", 0.05))

        self.strahler_filter = bool(settings.get("strahler_filter", True))
        self.min_strahler = int(settings.get("min_strahler", 0))
        self.max_strahler = int(settings.get("max_strahler", 10000))

    @property
    def nodes(self):
        import csv

        coords = []
        ids = []
        with open(self.csv, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar="|")
            for row in reader:
                coords.append([int(float(x)) for x in row[2:]])
                if row[1].strip() == "null" or row[1].strip() == "none":
                    ids.append([int(float(row[0])), None])
                elif row[0] == row[1]:
                    ids.append([int(float(row[0])), None])
                else:
                    ids.append([int(float(x)) for x in row[:2]])
        return [ids[i] + coords[i] for i in range(len(ids))]


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
            with filename.open("rb") as fin:
                settings.append(toml.load(fin))

        self.__init__(settings)

    def to_toml(self, filename):
        with filename.open("w") as tomlfile:
            tomlfile.write(str(self))


class CachedLSDConfig(BaseConfig):
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

        self.sensitives_file = settings.get("sensitives_file", None)

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
            with filename.open("rb") as fin:
                settings.append(toml.load(fin))

        self.__init__(settings)

    def to_toml(self, filename):
        with filename.open("w") as tomlfile:
            tomlfile.write(str(self))
