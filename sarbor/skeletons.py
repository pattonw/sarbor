import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
import time
import logging
import pickle
from pathlib import Path

from .arbors import SpatialArbor, Node
from .segmentations import SegmentationSource
from .config import Config, SkeletonConfig
from .meshes import (
    octree_to_sparse_vtk_volume,
    contour_sparse_vtk_volume,
    write_to_stl,
    decimate_mesh,
)

from typing import Tuple, Dict, List, Any

logger = logging.getLogger("sarbor")

Bounds = Tuple[np.ndarray, np.ndarray]


class Skeleton:
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        # Data sources
        self._arbor = SpatialArbor()
        self._seg = SegmentationSource(config.segmentations)
        self._config = config

        # floodfilling specific properties
        self.filled = {}

    def clone(self):
        new_skeleton = Skeleton(self._config)
        return new_skeleton

    # -----PROPERTIES-----
    @property
    def config(self) -> SkeletonConfig:
        return self._config.skeleton

    @property
    def nodes(self) -> Dict[int, Node]:
        return self.arbor.nodes

    @property
    def seg(self) -> SegmentationSource:
        return self._seg

    @property
    def arbor(self) -> SpatialArbor:
        return self._arbor

    @property
    def node_bounds(self) -> Bounds:
        """
        Bounds containing all node centers
        """
        return self.arbor.node_bounds

    @property
    def region_bounds(self) -> Bounds:
        """
        Bounds containing the field of view around every node
        """
        return (
            self.node_bounds[0] - self.fov_shape // 2,
            self.node_bounds[1] + self.fov_shape // 2 + 1,
        )

    @property
    def resolution(self) -> np.ndarray:
        return self.seg.res

    @resolution.setter
    def resolution(self, res: np.ndarray):
        self.seg.res = res

    @property
    def fov_shape(self) -> np.ndarray:
        return self.seg.fov_shape

    @fov_shape.setter
    def fov_shape(self, shape: np.ndarray):
        self.seg.fov_shape = shape

    # -----Inputing Skeleton Data-----

    def input_nodes(self, nodes: List[Node]):
        """
        build the tree by providing a list of nodes taken from another source
        (Usually another tree)
        """
        id_to_data = {}
        for node in nodes:
            id_to_data[node.key] = node.data

        self.build_tree(id_to_data)

    def input_id_pairs(self, pairs: List[Tuple[int, int]]):
        """
        build the tree by providing a list of (nid, pid) pairs. This is sufficient
        to build an arbor.
        """
        id_to_data = {nid: {"pid": pid} for nid, pid in pairs}
        self.build_tree(id_to_data)

    def input_nid_pid_x_y_z(self, nodes: List[Tuple[int, int, float, float, float]]):
        """
        builds the arbor and initializes floodfilling regions with seed locations.
        """
        id_to_data = {
            nid: {"pid": pid, "center": np.array([x, y, z])}
            for nid, pid, x, y, z in nodes
        }
        self.build_tree(id_to_data)

    def input_nid_pid_x_y_z_strahler(
        self, nodes: List[Tuple[int, int, float, float, float, int]]
    ):
        """
        builds the arbor with node coordinates and strahler indicies.
        """
        id_to_data = {
            nid: {"pid": pid, "center": np.array([x, y, z]), "strahler": strahler}
            for nid, pid, x, y, z, strahler in nodes
        }
        self.build_tree(id_to_data)

    def build_tree(self, id_to_data: Dict[int, Dict[str, Any]]):
        """
        build the tree from an map of the form:
        id: node, pid, data
        Thus we can add edges to nodes to build the tree
        TODO: move node creation into this funciton so we just need id: pid, data maps
        """
        roots = []  # List[Node]
        nodes = {}  # Dict[int, Node]
        for nid, data in id_to_data.items():
            nodes[nid] = Node(
                key=nid,
                strahler=data.get("strahler", None),
                center=data.get("center", None),
                mask=data.get("mask", None),
            )
        for nid, data in id_to_data.items():
            parent = None if nid == data["pid"] else nodes.get(data["pid"], None)
            if parent is None:
                roots.append(nodes[nid])
            else:
                parent.add_child(nodes[nid])
        if len(roots) == 1:
            self.arbor.build_from_root(roots[0])
            logger.info("No nodes lost!")
        else:
            sizes = [len(list(node.traverse())) for node in roots]
            logger.warning(
                "{} nodes excluded from tree!".format(sum(sizes) - max(sizes))
            )
            self.arbor.build_from_root(roots[sizes.index(max(sizes))])

    def is_filled(self, nid: int) -> bool:
        """
        Check if node has been filled.
        """
        return self.filled.get(nid, False)

    def fill(self, nid: int, mask: np.ndarray):
        """
        fill a node/region with a mask
        """
        node = self.nodes[nid]

        if not all(self.seg.downsample_factor == np.ones([3])):
            mask = (
                (
                    (255 * mask)
                    .reshape(
                        [
                            self.seg.fov_shape_voxels[0],
                            mask.shape[0] // self.seg.fov_shape_voxels[0],
                            self.seg.fov_shape_voxels[1],
                            mask.shape[1] // self.seg.fov_shape_voxels[1],
                            self.seg.fov_shape_voxels[2],
                            mask.shape[2] // self.seg.fov_shape_voxels[2],
                        ]
                    )
                    .mean(5)
                    .mean(3)
                    .mean(1)
                )
                > 127
            ).astype(np.uint8)
        node.value.mask = mask
        self.filled[nid] = True

    # ----Extracting Skeleton Data-----

    def extract_data(self):
        nodes, masks = self.arbor.extract_data()
        segmentation_data = self.seg.extract_data()
        return nodes, masks, segmentation_data

    def save_data_for_CATMAID(self):
        Path(self.config.output_file_base).mkdir(parents=True, exist_ok=False)
        self.seg.create_octrees_from_nodes(nodes=self.get_nodes())
        if self.config.save_nodes:
            ns = [
                (
                    node.key,
                    node.parent_key,
                    node.value.center[0],
                    node.value.center[1],
                    node.value.center[2],
                )
                for node in self.get_nodes()
            ]
            pickle.dump(ns, open(self.config.output_file_base + "/nodes.obj", "wb"))

        if self.config.save_rankings:
            self.save_rankings(
                self.config.output_file_base + "/rankings",
                consensus=self.config.use_consensus,
            )
        if self.config.save_segmentations:
            self.seg.save_data_for_CATMAID(self.config.output_file_base)
        if self.config.save_meshes:
            self.save_mesh(self.config.output_file_base + "/mesh.stl")
        if self.config.save_masks:
            nid_mask_map = {node.key: node.value.mask for node in self.get_nodes()}
            pickle.dump(
                nid_mask_map, open(self.config.output_file_base + "/masks.obj", "wb")
            )
        if self.config.save_config:
            self._config.to_toml(self.config.output_file_base + "/config.toml")

    def load(self, output_file_base):
        try:
            with open(output_file_base + "/nodes.obj", "rb") as f:
                nodes = pickle.load(f)
            self.input_nid_pid_x_y_z(nodes)
        except FileNotFoundError:
            logger.warning("Node file not found")

        try:
            with open(output_file_base + "/masks.obj", "rb") as f:
                nid_mask_map = pickle.load(f)
            for nid, mask in nid_mask_map.items():
                self.nodes[nid].value.mask = mask
        except FileNotFoundError:
            logger.warning("Masks not found")

        try:
            self._config.from_toml(output_file_base + "/config.toml")
        except FileNotFoundError:
            logger.warning("Config file is necessary to get reliable results")

    def save_rankings(self, output_file="ranking_data", consensus=False):
        connectivity_rankings = self.get_node_connectivity()
        branch_rankings = self.get_nid_branch_score_map(consensus=consensus)
        ranking_data = [
            (
                "nid",
                "pid",
                "connectivity_score",
                "branch_score",
                "branch_dx",
                "branch_dy",
                "branch_dz",
            )
        ]
        for node in self.get_nodes():
            ranking_data.append(
                (
                    node.key,
                    node.parent_key,
                    connectivity_rankings.get(
                        (node.key, node.parent_key), [None, None]
                    )[1],
                    branch_rankings[node.key][1],
                    branch_rankings[node.key][0][0]
                    * self._config.segmentations.voxel_resolution[0],
                    branch_rankings[node.key][0][1]
                    * self._config.segmentations.voxel_resolution[1],
                    branch_rankings[node.key][0][2]
                    * self._config.segmentations.voxel_resolution[2],
                )
            )
        data = np.array(ranking_data)
        np.savetxt("{}.csv".format(output_file), data, delimiter=",", fmt="%s")

    def save_mesh(self, output_file):
        octree = self.seg.segmentation_views
        vtk_volume = octree_to_sparse_vtk_volume(
            octree, resolution=self.seg.voxel_resolution
        )
        vtk_contour = contour_sparse_vtk_volume(vtk_volume, 0.5)
        vtk_contour_smaller = decimate_mesh(vtk_contour)
        write_to_stl(vtk_contour_smaller, output_file)

    # ----Editing Skeleton Data-----

    def transform(self, translation, scale):
        """
        Transform all data contained in the tree by translating and then scaling.
        """
        assert (
            self._segmentation is None
            and self._segmentation_counts is None
            and self._distances is None
        ), "scaling OctreeVolumes not yet supported"
        for node in self.get_nodes():
            node.value.transform(translation, scale)
        self.transform_properties()
        return self

    def transform_properties(self):
        """
        TODO
        """
        return None

    def filter_nodes_by_bounds(self, bounds):
        """
        Filter nodes in tree by bounds.
        """
        filtered_nodes = []
        lower, upper = bounds
        num_filtered = 0
        for node in self.get_nodes():
            if (
                lower[0] <= node.value.center[0] < upper[0]
                and lower[1] <= node.value.center[1] < upper[1]
                and lower[2] <= node.value.center[2] < upper[2]
            ):
                filtered_nodes.append(
                    (
                        node.key,
                        None if node.parent is None else node.parent.key,
                        node.value.center[0],
                        node.value.center[1],
                        node.value.center[2],
                        node.strahler,
                    )
                )
            else:
                num_filtered += 1
        self.input_nid_pid_x_y_z_strahler(filtered_nodes)
        return num_filtered

    def calculate_strahlers(self):
        self.arbor.calculate_strahler_indicies()

    def split(self, segment):
        seg_root = segment[0]
        seg_tail = segment[-1]
        upper_nodes = []
        queue = [self.arbor.root]
        while len(queue) > 0:
            current = queue.pop()
            upper_nodes.append(current.key)
            if current != seg_root:
                for child in current.children:
                    queue.append(child)
            else:
                for child in current.children:
                    if child != segment[1]:
                        queue.append(child)
        lower_nodes = []
        queue = [seg_tail]
        while len(queue) > 0:
            current = queue.pop()
            lower_nodes.append(current.key)
            for child in current.children:
                queue.append(child)
        return (
            (
                self.get_minimal_subtree(upper_nodes),
                self.get_minimal_subtree(lower_nodes),
            ),
            (segment[0:2], segment[-1:-3:-1]),
        )

    # -----Retrieving Skeleton Data-----

    def get_nodes(self, fifo=False):
        return list(self.arbor.nodes.values())

    def get_interesting_nodes(
        self, root: bool = False, leaves: bool = False, branches: bool = False
    ):
        return self.arbor.get_interesting_nodes(
            root=root, leaves=leaves, branches=branches
        )

    def get_segments(self):
        """
        gets all the sections of the tree that are single chains of nodes with no branches
        """
        for segment in self.arbor.traverse_segments():
            yield segment

    def get_minimal_subtree(self, ids):
        """
        get the smallest possible subtree containing all given ids
        """
        sub_tree_nodes = self.arbor.get_minimal_subtree(ids)
        new_skeleton = self.clone()
        new_skeleton.input_nodes(sub_tree_nodes)
        return new_skeleton

    def get_c_o_m_series(self):
        segments = self.arbor.get_root_leaf_paths()
        for segment in segments:
            series = []
            for node in segment:
                series.append(
                    list(node.value.center)
                    + list(node.value.center_of_mass)
                    + [0 if len(node.children) == 1 else 1]
                )
            yield np.array(series)

    def get_topological_copy(self, keep_root=True):
        """
        removes all simple nodes with 2 adjacent edges. if keep_root is False
        the root can potentially be removed as well and will be replaced with
        the closest branch point
        """
        raise NotImplementedError

    def get_radius(self, node, radius):
        return self.nodes.get_radius(node, radius)

    def get_radius_around_group(self, keys, radius):
        return self.nodes.get_radius_around_group(keys, radius)

    def get_constrained_radius(self, node, dx, dy, dz):
        return self.nodes.get_constrained_radius(node, dx, dy, dz)

    def filter_nodes_by_strahler(self, min_strahler: int, max_strahler: int):
        self.calculate_strahlers()
        keep_nodes = filter(
            lambda node: min_strahler <= node.strahler <= max_strahler, self.get_nodes()
        )
        self.input_nodes(keep_nodes)

    def resample_segments(self):
        """
        resample tree to have evenly spaced nodes. Gaussian smooth the curve
        and then sample at regular intervals.
        """

        # store tail points so that we can use them as roots
        branch_points = {}
        # create a list of new node points of form (nid, pid, x, y, z)
        new_tree_nodes = []
        # new nodes will need new nids/pids thus we will reassign all nids starting at 0
        new_node_id = 0
        # store mapping from new node ids back to closest original node ids
        new_nid_to_orig_map = {}

        # get each straight segment
        for segment in self.get_segments():

            # handle root case. All other segment "roots" will
            # be handled as tails of previous segments
            if new_node_id == 0:
                root = (0, None, *segment[0].value.center)
                new_node_id += 1
                new_tree_nodes.append(root)
                branch_points[segment[0].key] = root
                new_nid_to_orig_map[0] = segment[0].key

            # get interpolated nodes. includes tail but not root
            root_key = branch_points[segment[0].key]
            new_interpolated_nodes, new_node_id = self.resample_segment(
                segment, new_node_id, root_key
            )

            last = 0
            for nid, pid, x, y, z in new_interpolated_nodes:
                closest = None
                for i in range(last, len(segment)):
                    node = segment[i]
                    if closest is None:
                        closest = node
                    elif np.linalg.norm(node.value.center - [x, y, z]) < np.linalg.norm(
                        closest.value.center - [x, y, z]
                    ):
                        closest = node
                new_nid_to_orig_map[nid] = closest.key

            # save tail node for future referece if it is a branch
            if len(segment[-1].children) > 1:
                branch_points[segment[-1].key] = (
                    new_interpolated_nodes[-1]
                    if len(new_interpolated_nodes) > 0
                    else root_key
                )

            new_tree_nodes = new_tree_nodes + new_interpolated_nodes

        # create a new tree with the same config and input the new nodes
        new_skeleton = self.clone()
        new_skeleton.input_nid_pid_x_y_z(new_tree_nodes)
        return new_skeleton, new_nid_to_orig_map

    def resample_segment(self, nodes, new_node_id, root):
        def get_smoothed(coords, sigma_fraction):
            max_dist = 0
            for i in range(1, len(coords)):
                max_dist = max(
                    np.linalg.norm(np.array(coords[i]) - np.array(coords[i - 1])),
                    max_dist,
                )
            steps = int(
                max(10 * len(coords) * max_dist // self.config.resample_delta, 10)
            )
            x_y_z = list(zip(*coords))
            t = np.linspace(0, 1, len(coords))
            t2 = np.linspace(0, 1, steps)

            # linearly interpolate points through each axis independently
            x_y_z_2 = list(map(lambda x: np.interp(t2, t, x), x_y_z))

            # gaussian smooth points allong each axis
            if self.config.smoothing == "gaussian":
                x_y_z_3 = list(
                    map(
                        lambda x: gaussian_filter1d(
                            x, steps * sigma_fraction, mode="nearest"
                        ),
                        x_y_z_2,
                    )
                )

                # gaussian smooth moves end points, thus we need to
                # linearly interpolate between original ends and gaussian smoothed ends
                t = np.linspace(0, 1, 2)
                t2 = np.linspace(0, 1, steps)
                start = [[x_y_z[i][0], x_y_z_3[i][0]] for i in range(3)]
                end = [[x_y_z_3[i][-1], x_y_z[i][-1]] for i in range(3)]
                start = list(map(lambda x: np.interp(t2, t, x), start))
                end = list(map(lambda x: np.interp(t2, t, x), end))
                x_y_z_3 = list(
                    zip(*(list(zip(*start)) + list(zip(*x_y_z_3)) + list(zip(*end))))
                )
            elif self.config.smoothing == "none":
                x_y_z_3 = x_y_z_2

            return zip(*x_y_z_3)

        def downsample(coords, delta, origin, end):
            previous = coords[0]
            dist_to_end = np.linalg.norm(np.array(previous) - np.array(coords[-1]))

            for coord in coords[1:]:
                if dist_to_end < delta * 1.5:
                    break

                dist = np.linalg.norm(np.array(coord) - np.array(previous))
                if dist < delta * 0.9:
                    continue
                elif dist > 1.5 * delta:
                    raise ValueError(
                        "your resampling has too few steps to support a delta this small!"
                    )
                else:
                    yield coord
                    previous = coord
                    dist_to_end = np.linalg.norm(
                        np.array(previous) - np.array(coords[-1])
                    )

            if not np.allclose(previous, coords[-1]):
                # yield end point if not already done
                yield coords[-1]

        coords = [node.value.center for node in nodes]

        # contains both end points
        smoothed_coords = list(get_smoothed(coords, self.config.resample_sigma))

        # does not contain root
        downsampled_coords = list(
            downsample(
                smoothed_coords,
                self.config.resample_delta,
                root[2:],
                nodes[-1].value.center,
            )
        )

        # return new downsampled nodes
        previous_id = root[0]
        current_id = new_node_id
        downsampled_nodes = []
        for coord in downsampled_coords:
            node = (current_id, previous_id, coord[0], coord[1], coord[2])
            previous_id = current_id
            current_id += 1
            downsampled_nodes.append(node)

        return downsampled_nodes, new_node_id + len(downsampled_nodes)

    def get_regularness(self):
        """
        Get mean, std and outliers for distance between nodes.
        Determine whether skeleton needs to be resampled or not
        """
        raise NotImplementedError("not done yet")

    def delete_branch(self, branch_chop: Tuple[int, int]):
        self.calculate_strahlers()
        keep_root = self.arbor.nodes[branch_chop[0]]
        while keep_root.parent_key not in (None, branch_chop[1]):
            keep_root = keep_root.parent

        keep_nodes = keep_root.traverse(ignore=[branch_chop[1]])
        new_skeleton = self.clone()
        new_skeleton.input_nodes(keep_nodes)
        logger.debug(
            "Original skeleton size {} vs chopped skeleton size {}".format(
                len(self.get_nodes()), len(new_skeleton.get_nodes())
            )
        )
        return new_skeleton

    def get_closest_node(self, location: np.ndarray):
        closest = None
        dist = 0
        for nid, node in self.nodes.items():
            if closest is None or np.linalg.norm(node.value.center - location) < dist:
                closest = node
                dist = np.linalg.norm(node.value.center)
        return closest

    # -----Other-----

    def get_strahlers_from(self, other):
        """
        Copy strahlers from another tree.
        Useful when you have a tree containing some subset
        of the original trees nodes, which do not allow for
        accurate recalculation of strahler indicies.

        This is done segment by segment comparing roots
        """
        other_segments = list(other.get_segments())
        for segment in self.get_segments():
            for i in range(len(other_segments)):
                other_segment = other_segments[i]
                root_dist = abs(other_segment[0].value.center - segment[0].value.center)
                tail_dist = abs(
                    other_segment[-1].value.center - segment[-1].value.center
                )
                if all(root_dist < np.array([1e-5] * 3)) or all(
                    tail_dist < np.array([1e-5] * 3)
                ):
                    strahler = other_segment[-1].strahler
                    for node in segment[1:]:
                        node.strahler = strahler
                    segment[0].strahler = other_segment[0].strahler
                    del other_segments[i]
                    break

    def input_center_of_mass(self, node, center_of_mass):
        """
        input a nodes center of mass, update the max center of mass magnitude for normalizing
        """
        node.value.center_of_mass = center_of_mass
        self.max_mass_v_mag = max(self.max_mass_v_mag, center_of_mass[1])

    # ----Rankings-----

    def _similar_vectors(self, vec_a, vec_b):
        return False and (
            self._angle_between(vec_a, vec_b) < 0.15
            or abs(np.linalg.norm(vec_a) - np.linalg.norm(vec_b)) < 0.02
        )

    @staticmethod
    def _angle_between(vec_a, vec_b):
        v1_u = vec_a / np.linalg.norm(vec_a)
        v2_u = vec_b / np.linalg.norm(vec_b)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def _distance_between(vec_a, vec_b):
        return np.linalg.norm(vec_a - vec_b)

    def _smooth_scores(self, nid_score_map):
        """
        Don't change the actual vectors, just the scores. The vectors show where
        missing branches are likely to be, making debugging easier.

        Scores are altered by how much vectors of neighboring nodes agree with this
        one. This makes debugging much harder, but may give better results.
        """
        smoothed_map = {}

        for nid in nid_score_map:
            node = self.nodes[nid]
            neighbors = node.get_neighbors()
            # average the direction vectors of this node and its neighbors
            averaged_vec = np.sum(
                [nid_score_map[n.key][0] for n in neighbors + [node]], axis=0
            ) / (len(neighbors) + 1)

            # check how much the average agrees with the original
            difference = nid_score_map[nid][0] - averaged_vec
            # 1 multiplier if difference is 0, 0 multiplier if difference is large
            multiplier = max(
                1 - np.linalg.norm(difference) / np.linalg.norm(nid_score_map[nid][0]),
                0,
            )

            smoothed_map[nid] = (
                nid_score_map[nid][0],
                nid_score_map[nid][1] * multiplier,
            )
        return smoothed_map

    def get_nid_branch_score_map(
        self, nodes=None, sphere=True, incrDenom=True, consensus=True, key=""
    ):
        """
        Create a map from node ids to branch scores. Nodes can then be sorted
        by branch_score, leaving nodes that are likely to be in the presence of
        branches to be near the top.
        """
        nid_score_map = {}

        if nodes is None:
            nodes = self.get_nodes()

        for node in nodes:
            mask = self.get_dist_weighted_mask(
                node.value.center, increment_denominator=incrDenom, sphere=sphere
            )
            assert mask.max() <= 1, (
                "max mask value is {} which is greater than 1."
                + " confidence score above 1 should not be possible"
            ).format(mask.max())
            direction, mag = self._get_max_vec(mask)

            if key == "location":
                nid_score_map[tuple(node.value.center)] = (direction, mag)
            else:
                nid_score_map[node.key] = (direction, mag)
            if any(np.isnan(x) for x in direction):
                logger.debug(mask)
                logger.debug(direction)
                logger.debug(mag)
                raise ValueError("Direction is NAN!")

        if consensus and not key == "location":
            nid_score_map = self._smooth_scores(nid_score_map)

        return nid_score_map

    def get_node_connectivity(self, ascending: bool = True):
        """
        Return a list of parent/child node pairs and their "connectivity score".
        Connectivity score is relatively arbitrary with the property
        that 1 means well connected, 0 means not connected.
        """
        logger.debug("starting node connectivity")
        nid_score_map = {}
        for node in self.get_nodes():
            if node.parent is not None:
                nid_score_map[(node.key, node.parent_key)] = (
                    (node.value.center, node.parent.value.center),
                    self.get_connection(node, node.parent),
                )
        return nid_score_map

    def get_connection(self, node_a: Node, node_b: Node):
        """
        simply look at the overlap of two neighboring masks
        """
        if node_a.value.mask is None or node_b.value.mask is None:
            return -1
        roi_a = self.seg.get_roi(node_a.value.center)
        roi_b = self.seg.get_roi(node_b.value.center)
        roi_ab = (np.maximum(roi_a[0], roi_b[0]), np.minimum(roi_a[1], roi_b[1]))
        if any(roi_ab[0] >= roi_ab[1]):
            logger.warn(
                "Nodes {} and {} do not have overlapping masks!".format(
                    node_a.key, node_b.key
                )
            )
            return -2
        try:
            a_slices = self.seg.transform_bounds(
                (roi_ab[0] - roi_a[0], roi_ab[1] - roi_a[0])
            )
            b_slices = self.seg.transform_bounds(
                (roi_ab[0] - roi_b[0], roi_ab[1] - roi_b[0])
            )
            return (
                (node_a.value.mask[a_slices] + node_b.value.mask[b_slices] == 2).sum()
            ) / (np.prod((roi_ab[1] - roi_ab[0]) / self.seg.voxel_resolution))
        except AssertionError as e:
            logger.warn(
                "Node ({}, {}) connection failed due to: ".format(
                    node_a.key, node_b.key, e
                )
            )
            return -3

    def get_connectivity_path(self, index_a: np.ndarray, index_b: np.ndarray):
        """
        Connectivity score is calculated by finding the maximum edge weight along
        the minimum cost path between two indicies through the segmentation data.
        """
        a2b_matrix = self.get_count_weighted_mask(
            list(
                map(
                    slice,
                    np.minimum(index_a, index_b),
                    np.maximum(index_a, index_b) + 1,
                )
            ),
            increment_denominator=True,
        )
        reversers = index_a < index_b
        a2b_matrix = a2b_matrix[
            slice(None, None, -1) if reversers[0] else slice(None, None),
            slice(None, None, -1) if reversers[1] else slice(None, None),
            slice(None, None, -1) if reversers[2] else slice(None, None),
        ]
        costs = np.ones(a2b_matrix.shape)
        for i in range(a2b_matrix.shape[0]):
            for j in range(a2b_matrix.shape[1]):
                for k in range(a2b_matrix.shape[2]):
                    current_score = np.min(a2b_matrix[i, j, k])
                    region = list(
                        map(
                            slice,
                            [x - 1 if x > 0 else x for x in [i, j, k]],
                            [i + 1, j + 1, k + 1],
                        )
                    )
                    if i == j == k == 0:
                        costs[i, j, k] = current_score
                    else:
                        costs[i, j, k] = min(current_score, np.max(costs[region]))
        return costs[-1, -1, -1]

    @staticmethod
    def _get_center_of_mass(data):
        x = np.linspace(-1, 1, data.shape[0])
        y = np.linspace(-1, 1, data.shape[1])
        z = np.linspace(-1, 1, data.shape[2])
        z_change = np.sum(np.dot(data, z))
        y_change = np.sum(np.dot(data.transpose(2, 0, 1), y))
        x_change = np.sum(np.dot(data.transpose(1, 2, 0), x))
        change_direction = np.array([x_change, y_change, z_change])
        change_mag = np.dot(change_direction, change_direction) ** 0.5
        if math.isclose(change_mag, 0):
            change_direction = np.array([0, 0, 0])
        else:
            change_direction = change_direction / change_mag
        change_mag = change_mag / (len(x) * len(y) * len(z) / 4)

        if any(np.isnan(x) for x in change_direction):
            logger.debug("nan")

        return (change_direction, change_mag)

    @staticmethod
    def _get_max_vec(data):
        """
        Returns the offset to get the most likely position for a missing branch
        allong with a "score" that ranks how likely this position is to
        be a missing branch.
        the "score" factors in confidence of segmentation at that point
        pluss distance to previously seen nodes
        """
        v = np.unravel_index(np.argmax(data), data.shape)
        if np.isclose(data[v], 0):
            return np.array([0, 0, 0]), 0
        else:
            v = v - np.array(data.shape) // 2
            dist_component = np.linalg.norm(v) / (
                np.linalg.norm(np.array(data.shape) // 2)
            )
            assert (
                dist_component <= 1
            ), "max distance should be 1 at the corners. It is {}".format(
                dist_component
            )
            score = np.max(data)
            return (v, score)

    def get_dist_weighted_mask(
        self,
        center: np.ndarray,
        min_overlap_count=1,
        increment_denominator=False,
        sphere=True,
    ) -> np.ndarray:
        return self.seg.dist_view_weighted_mask(center)

    def get_count_weighted_mask(
        self,
        center: np.ndarray,
        min_overlap_count=1,
        increment_denominator=False,
        sphere=True,
    ) -> np.ndarray:
        return self.seg.view_weighted_mask(
            center, incr_denom=int(increment_denominator), sphere=sphere
        )

    # ------------ TEST HELPERS -----------
    def get_sub_nid_branch_scores(
        self,
        full_mask,
        all_scores,
        smooth=True,
        sphere=True,
        incrDenom=True,
        mass=True,
        DEBUG=None,
    ):
        nodes = []
        skipped = 0
        for node in self.get_nodes():
            node_bounds = node.value.get_bounds(self.fov_shape)
            node_bounds = list(map(slice, node_bounds[0], node_bounds[1]))
            sub_mask = self.segmentation[node_bounds]
            whole_mask = full_mask[node_bounds]
            if (
                not np.array_equal(sub_mask, whole_mask)
                or tuple(node.value.center) not in all_scores
            ):
                nodes.append(node)
            else:
                skipped += 1
        logger.debug(
            "{}/{} nodes recalculated".format(len(nodes), skipped + len(nodes))
        )
        small_radius_scores = self.get_nid_branch_score_map(
            nodes=nodes, sphere=sphere, mass=mass, incrDenom=incrDenom
        )

        sub_nid_branch_score_map = {}
        for node in self.get_nodes():
            if node.key in small_radius_scores:
                sub_nid_branch_score_map[node.key] = small_radius_scores[node.key]
            else:
                sub_nid_branch_score_map[node.key] = all_scores[
                    tuple(node.value.center)
                ]

        if smooth:
            sub_nid_branch_score_map = self._smooth_scores(sub_nid_branch_score_map)

        return sub_nid_branch_score_map

    def get_sub_nid_branch_score_map(
        self,
        close_nodes,
        nid_score_map,
        smooth=True,
        sphere=True,
        incrDenom=True,
        mass=True,
        DEBUG=None,
    ):
        """
        This function is mostly used for debugging purposes. Given that you know
        a specific node is a branch point, you can delete a branch, and then find
        all nodes where the nodes field of view overlaps with any node in the now
        deleted branch. Since clearly any node whose field of view does not overlap
        will not have its branch score changed by removing distant nodes. This greatly
        reduces computation time needed to validate larger datasets.
        """
        # Assumes Two halves of the neuron do not come near each other
        large_radius = [
            self.nodes[key]
            for key in self.get_radius_around_group(close_nodes, self.fov_shape)
        ]
        t1 = time.time()
        self.seg.create_octrees_from_nodes(nodes=large_radius)
        t2 = time.time()
        if DEBUG == "TIME":
            logger.debug(
                "\tCreating Octrees took {} seconds, {} per node with {} nodes".format(
                    round(t2 - t1),
                    round((t2 - t1) / len(large_radius), 1),
                    len(large_radius),
                )
            )

        t1 = time.time()
        small_radius_scores = self.get_nid_branch_score_map(
            nodes=[self.nodes[key] for key in close_nodes],
            sphere=sphere,
            mass=mass,
            incrDenom=incrDenom,
        )
        t2 = time.time()
        if DEBUG == "TIME":
            logger.debug(
                "\tCalculating scores took {} seconds, {} per node with {} nodes".format(
                    round(t2 - t1),
                    round((t2 - t1) / len(close_nodes), 1),
                    len(close_nodes),
                )
            )

        sub_nid_branch_score_map = {}
        for node in self.get_nodes():
            if node.key in small_radius_scores:
                sub_nid_branch_score_map[node.key] = small_radius_scores[node.key]
            else:
                sub_nid_branch_score_map[node.key] = nid_score_map[node.key]

        if smooth:
            sub_nid_branch_score_map = self._smooth_scores(sub_nid_branch_score_map)

        return sub_nid_branch_score_map
