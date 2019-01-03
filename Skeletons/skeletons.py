import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
import time

from .octrees import OctreeVolume
from .arbors import Arbor
from .fovs import Region


class Skeleton:
    def __init__(self):
        # Data sources
        self.arbor = Arbor()
        self._segmentation = None
        self._segmentation_counts = None
        self._distances = None

        # arbor dependent properties
        self._bounds = None
        self._node_map = None

        # tree specific properties
        self._bounds = None
        self._fov_shape = None
        self._res = None
        self._sphere = None

        # floodfilling specific properties
        self.filled = {}
        self.max_mass_v_mag = -1

    def reset(self):
        self.reset_data_sources()
        self.reset_arbor_props()
        self.reset_tree_props()
        self.reset_floodfill_props()

    def reset_data_sources(self):
        self.arbor = Arbor()
        self._segmentation = None
        self._segmentation_counts = None
        self._distances = None

    def reset_arbor_props(self):
        self._bounds = None
        self._node_map = None

    def reset_tree_props(self):
        self._bounds = None
        self._fov_shape = None
        self._res = None
        self._sphere = None

    def reset_floodfill_props(self):
        self.filled = {}
        self.max_mass_v_mag = -1

    def clone(self):
        new_skeleton = Skeleton()
        new_skeleton._fov_shape = self._fov_shape
        new_skeleton._res = self._res
        new_skeleton._sphere = self._sphere
        return new_skeleton

    # -----PROPERTIES-----
    @property
    def node_bounds(self):
        """
        Bounds containing all node centers
        """
        if self._bounds is None:
            self.calculate_tree_bounds()
            return self._bounds
        else:
            return self._bounds

    @property
    def region_bounds(self):
        """
        Bounds containing the field of view around every node
        """
        return (
            self.node_bounds[0] - self.fov_shape // 2,
            self.node_bounds[1] + self.fov_shape // 2 + 1,
        )

    @property
    def resolution(self):
        if self._res is None:
            raise Exception("The resolution has not been defined for this tree")
        return self._res

    @resolution.setter
    def resolution(self, res):
        if self._res is None:
            self._res = res
        else:
            raise Exception("Overwriting the skeletons resolution is not supported")

    @property
    def fov_shape(self):
        if self._fov_shape is None:
            raise Exception(
                "The field of view dimensions have not been defined for this tree"
            )
        else:
            return self._fov_shape

    @fov_shape.setter
    def fov_shape(self, shape):
        if self._fov_shape is None:
            self._fov_shape = shape
        else:
            raise Exception("Overwriting the skeletons fov shape is not supported")

    @property
    def node_map(self):
        if self._node_map is None:
            self._node_map = self.arbor.get_key_map()
        return self._node_map

    @property
    def sphere(self):
        shape = self.fov_shape
        res = self.resolution
        if self._sphere is None:
            self._sphere = self._create_sphere(shape, res)
            return self._sphere
        else:
            return self._sphere

    @property
    def segmentation(self):
        return self._segmentation

    @segmentation.setter
    def segmentation(self, tree):
        if self._segmentation is None:
            self._segmentation = tree
        else:
            raise Exception("trying to overwrite segmentation octree")

    @property
    def segmentation_counts(self):
        return self._segmentation_counts

    @segmentation_counts.setter
    def segmentation_counts(self, tree):
        if self._segmentation_counts is None:
            self._segmentation_counts = tree
        else:
            raise Exception("trying to overwrite segmentation_counts octree")

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, tree):
        if self._distances is None:
            self._distances = tree
        else:
            raise Exception("trying to overwrite distances octree")

    # -----Inputing Skeleton Data-----

    def input_nodes(self, nodes):
        """
        build the tree by providing a list of nodes taken from another source
        (Usually another tree)
        """
        id_to_data = {}
        for node in nodes:
            new_node, pid, value = node.clone()
            id_to_data[new_node.key] = [new_node, pid, value]

        self.build_tree(id_to_data)

    def input_id_pairs(self, pairs):
        """
        build the tree by providing a list of (nid, pid) pairs. This is sufficient
        to build an arbor.
        """
        id_to_data = {nid: [self.arbor.Node(nid), pid, None] for nid, pid in pairs}
        self.build_tree(id_to_data)

    def input_nid_pid_x_y_z(self, nodes):
        """
        builds the arbor and initializes floodfilling regions with seed locations.
        """
        id_to_data = {
            nid: [self.arbor.Node(nid), pid, Region(center=np.array([x, y, z]))]
            for nid, pid, x, y, z in nodes
        }
        self.build_tree(id_to_data)

    def input_nid_pid_x_y_z_strahler(self, nodes):
        """
        builds the arbor with node coordinates and strahler indicies.
        """
        id_to_data = {
            nid: [
                self.arbor.Node(nid, strahler=strahler),
                pid,
                Region(center=np.array([x, y, z])),
            ]
            for nid, pid, x, y, z, strahler in nodes
        }
        self.build_tree(id_to_data)

    def build_tree(self, id_to_data):
        """
        build the tree from an map of the form:
        id: node, pid, data
        Thus we can add edges to nodes to build the tree
        TODO: move node creation into this funciton so we just need id: pid, data maps
        """
        roots = []
        for nid, data in id_to_data.items():
            node, pid, value = data
            node.value = value
            parent = None if nid == pid else id_to_data.get(pid, None)
            if parent is None:
                roots.append(node)
            else:
                parent[0].add_child(node)
        if len(roots) == 1:
            self.arbor.root = roots[0]
        else:
            sizes = [len(list(node.traverse())) for node in roots]
            self.arbor.root = roots[sizes.index(max(sizes))]
            self.reset_properties_from_arbor()
        self.reset_arbor_props()

    def is_filled(self, nid):
        """
        Check if node has been filled.
        """
        return self.filled.get(nid, False)

    def fill(self, nid, mask):
        """
        fill a node/region with a mask
        """
        node = self.node_map[nid]

        node.value.mask = mask
        self.filled[nid] = True

    def input_masks(
        self, data, axes=[2, 1, 0], c_o_m=True, tree_bounds=None, block_shape=None
    ):
        """
        c_o_m: whether or not to calculate the center of mass vectors for individual nodes
        """
        if self.segmentation is None or self.segmentation_counts is None:
            self.create_octrees(tree_bounds=tree_bounds, block_shape=block_shape)
        for mask, bounds, nid, pid in data:
            mask = mask.transpose(axes)
            bounds = [[bound[axes[i]] for i in range(3)] for bound in bounds]
            if c_o_m:
                node = self.node_map[nid]
                node.c_o_m = self._get_center_of_mass(node.mask * self.sphere)
                self.max_mass_v_mag = max(
                    self.max_mass_v_mag, node.value.center_of_mass[1]
                )
            self.segmentation[list(map(slice, bounds[0], bounds[1]))] += mask
            self.segmentation_counts[list(map(slice, bounds[0], bounds[1]))] += 1
        self.segmentation.bounds = self.segmentation.get_leaf_bounds()
        self.segmentation_counts.bounds = self.segmentation_counts.get_leaf_bounds()

    def input_masks_old(self, data):
        """
        input masks into the tree by providing a list of
        datapoints of the form (mask, bounds, nid, pid)
        TODO: This funciton should not need bounds or pids
        Should also insert mask data into the Octrees instead
        of nodes.
        NOTE: Instead of nids and masks, assuming constant shaped input,
        could aslo just take mask and bounds and insert into Octree.
        """
        id_to_data = {nid: [mask, bounds] for mask, bounds, nid, _ in data}
        for node in self.arbor.traverse():
            data = id_to_data.get(node.key, None)
            if data is not None:
                self.insert_data(node, data[1], data[0], [2, 1, 0])

    def input_n5_data(self, folder_path, skeleton_dir, datasets):
        """
        Skips directly to filling out octrees with data without storing
        any segmentation data in nodes. Useful when your analysis does
        not depend on the individual contributions of each node
        """
        do_seg = "segmentation" in datasets
        do_counts = "counts" in datasets
        if do_seg or do_counts:
            num_nodes = len(list(self.get_nodes()))
            done_nodes = 0
            section_bounds = None
            for node in self.get_nodes(fifo=True):
                node_bounds = (
                    node.value.bounds
                    if node.value.bounds is not None
                    else node.value.get_bounds(self.fov_shape)
                )
                if section_bounds is None:
                    section_bounds = node_bounds
                combined_bounds = (
                    np.minimum(section_bounds[0], node_bounds[0]),
                    np.maximum(section_bounds[1], node_bounds[1]),
                )
                if all(combined_bounds[1] - combined_bounds[0] < self.fov_shape * 2):
                    section_bounds = combined_bounds
                    done_nodes += 1
                else:
                    if do_seg:
                        self.segmentation.read_from_n5(
                            folder_path, skeleton_dir + "/segmentation", section_bounds
                        )
                    if do_counts:
                        self.segmentation_counts.read_from_n5(
                            folder_path, skeleton_dir + "/counts", section_bounds
                        )
                    section_bounds = node_bounds
                    done_nodes += 1
                    print(
                        "{}/{} ({:.2f}%) done".format(
                            done_nodes, num_nodes, done_nodes / num_nodes * 100
                        )
                    )

            if do_seg:
                self.segmentation.read_from_n5(
                    folder_path, skeleton_dir + "/segmentation", section_bounds
                )
            if do_counts:
                self.segmentation_counts.read_from_n5(
                    folder_path, skeleton_dir + "/counts", section_bounds
                )

    # ----Editing Skeleton Data-----

    def transform(self, translation, scale):
        """
        Transform all data contained in the tree by translating and then scaling.
        """
        assert (
            self.segmentation is None and self.segmentation_counts is None
        ), "scaling segmentation data is not yet supported"
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
        return self.input_nid_pid_x_y_z_strahler(filtered_nodes)

    # -----Retrieving Skeleton Data-----

    def get_nodes(self, fifo=False):
        return self.arbor.traverse(fifo=fifo)

    def get_regions(self):
        for node in self.get_nodes():
            yield node.value

    def get_interesting_nodes(self, root=False, leaves=False, branches=False):
        """
        This function extracts interesting nodes (root, leaves branches).
        TODO: move this funciton into the Arbor class since physical
        coordinates are irrelevant
        """

        if root or leaves or branches:
            for node in self.arbor.traverse():
                if root:
                    root = False
                    yield node
                elif branches and len(node.get_neighbors()) > 2:
                    yield node
                elif leaves and len(node.get_neighbors()) == 1:
                    yield node

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
        all_nodes = self.arbor.get_minimal_subtree(ids)
        new_skeleton = self.clone()
        new_skeleton.input_nodes(all_nodes)
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
        """
        get all nodes within a specific radius (physical distance) of a given node
        radius can either be a scalar or a 3d value with axis wise distances
        """
        origin = node.value.center
        all_nodes = [node]
        previous = [node]
        layer = node.get_neighbors()
        while len(layer) > 0:
            all_nodes += layer
            next_layer = []
            for node in layer:
                neighbors = node.get_neighbors()
                for neighbor in neighbors:
                    if len(radius) == 1:
                        if (
                            neighbor not in previous
                            and sum((neighbor.value.center - origin) ** 2) < radius ** 2
                        ):
                            next_layer.append(neighbor)
                    elif len(radius) == 3:
                        distance = abs(neighbor.value.center - origin)
                        if neighbor not in previous and all(distance <= radius):
                            next_layer.append(neighbor)
            previous = layer[:]
            layer = next_layer[:]
        return all_nodes

    def get_radius_around_group(self, keys, radius):
        """
        TODO: Improve this, this searches for nodes in a radius around every given node
        then combines the groups into one large group getting rid of redundencies

        Might be improved by having a single search that compares each node to see if they
        are close to a list of numbers
        """
        big_radius = set()
        centers = [self.node_map[key].value.center for key in keys]
        for node in self.get_nodes():
            for center in centers:
                if all(abs(node.value.center - center) < radius):
                    big_radius.add(node.key)
                    break
        return big_radius

    def get_constrained_radius(self, node, max_x, max_y, max_z):
        """
        get all nodes with a change in z <= max_z etc.
        """
        origin = node.value.center
        all_nodes = [node]
        previous = [node]
        layer = node.get_neighbors()
        while len(layer) > 0:
            all_nodes += layer
            next_layer = []
            for node in layer:
                neighbors = node.get_neighbors()
                for neighbor in neighbors:
                    if neighbor not in previous and all(
                        abs(np.array(neighbor.value.center) - np.array(origin)).astype(
                            "int"
                        )
                        <= np.array([max_x, max_y, max_z])
                    ):
                        next_layer.append(neighbor)
            previous = layer[:]
            layer = next_layer[:]
        return all_nodes

    # -----Analyze Skeleton Data-----

    def calculate_tree_bounds(self):
        """
        Find the minimum and maximum node center
        """
        lower = np.array([float("inf"), float("inf"), float("inf")])
        upper = -lower.copy()
        for node in self.get_nodes():
            if node.value is not None:
                upper = np.maximum(node.value.center, upper)
                lower = np.minimum(node.value.center, lower)
        self._bounds = (lower.astype(int), upper.astype(int))

    def create_octrees_from_nodes(
        self, tree_bounds=None, block_shape=None, nodes=None, dist=True
    ):
        def _dist_block(dimensions, resolution):
            x = (
                (
                    np.linspace(-dimensions[0] // 2, dimensions[0] // 2, dimensions[0])
                    * resolution[0]
                )
                ** 2
            ).reshape(dimensions[0], 1, 1)
            y = (
                (
                    np.linspace(-dimensions[1] // 2, dimensions[1] // 2, dimensions[1])
                    * resolution[1]
                )
                ** 2
            ).reshape(1, dimensions[1], 1)
            z = (
                (
                    np.linspace(-dimensions[2] // 2, dimensions[2] // 2, dimensions[2])
                    * resolution[2]
                )
                ** 2
            ).reshape(1, 1, dimensions[2])
            return (x + y + z) ** (0.5) / np.sum(
                (dimensions // 2 * resolution) ** 2
            ) ** (0.5)

        def _data_populator(bounds):
            return np.zeros(np.array(bounds[1]) - np.array(bounds[0]))

        def _data_populator2(bounds):
            return np.full(
                np.array(bounds[1]) - np.array(bounds[0]), fill_value=float("inf")
            )

        self.segmentation = OctreeVolume(
            self.fov_shape, self.region_bounds, np.uint8, _data_populator
        )
        self.segmentation_counts = OctreeVolume(
            self.fov_shape, self.region_bounds, np.uint8, _data_populator
        )
        if dist:
            self.distances = OctreeVolume(
                self.fov_shape, self.region_bounds, float, _data_populator2
            )
            dist_block = _dist_block(self.fov_shape, self.resolution)

        if nodes is None:
            nodes = self.get_nodes()
        for node in nodes:
            node_bounds = node.value.get_bounds(self.fov_shape)

            node_bounds = list(map(slice, node_bounds[0], node_bounds[1]))
            if dist:
                self.distances[node_bounds] = np.minimum(
                    self.distances[node_bounds], dist_block
                )
            if node.value.mask is not None:
                self.segmentation[node_bounds] += node.value.mask
            self.segmentation_counts[node_bounds] += 1

    # -----Other-----

    def recalculate_center_of_mass(self):
        """
        Calculate the center of mass for each node
        """
        for node in self.get_nodes():
            node.c_o_m = self._get_center_of_mass(node.mask * self.sphere)

    def get_scaled_center_of_mass(self, node):
        """
        retrieve the center of mass of a node, scaled down by the max of the tree
        """
        center_of_mass = node.value.center_of_mass
        if center_of_mass is not None:
            return center_of_mass[0], center_of_mass[1] / self.max_mass_v_mag
        else:
            return None

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
            self._angle_between(vec_a, vec_b) < .15
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
        smoothed_map = {}

        for nid in nid_score_map:
            node = self.node_map[nid]
            neighbors = node.get_neighbors()
            combined_vec = np.sum(
                [
                    nid_score_map[n.key][0] * nid_score_map[n.key][1]
                    for n in neighbors + [node]
                ]
            ) / (len(neighbors) + 1)
            smoothed_map[nid] = combined_vec, np.linalg.norm(combined_vec)
        return smoothed_map

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
        # Assumes Two halves of the neuron do not come near each other
        large_radius = [
            self.node_map[key]
            for key in self.get_radius_around_group(close_nodes, self.fov_shape)
        ]
        t1 = time.time()
        self.create_octrees_from_nodes(nodes=large_radius)
        t2 = time.time()
        if DEBUG == "TIME":
            print(
                "\tCreating Octrees took {} seconds, {} per node with {} nodes".format(
                    round(t2 - t1),
                    round((t2 - t1) / len(large_radius), 1),
                    len(large_radius),
                )
            )

        t1 = time.time()
        small_radius_scores = self.get_nid_branch_score_map(
            nodes=[self.node_map[key] for key in close_nodes],
            sphere=sphere,
            mass=mass,
            incrDenom=incrDenom,
        )
        t2 = time.time()
        if DEBUG == "TIME":
            print(
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

    def get_nid_branch_score_map(
        self, nodes=None, sphere=True, mass=True, incrDenom=True, consenus=True
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
                list(
                    map(
                        slice,
                        node.value.center - self.fov_shape // 2,
                        node.value.center + self.fov_shape // 2 + 1,
                    )
                ),
                increment_denominator=incrDenom,
            )
            if sphere:
                mask = mask * self.sphere

            if mass:
                direction, mag = self._get_center_of_mass(mask)
            else:
                direction, mag = self._get_max_vec(mask)

            """ # Add a "fullness" scaling term to lower rank of areas that have been excessively filled
            mag = mag * (
                1
                - np.sum(
                    self.get_count_weighted_mask(
                        list(
                            map(
                                slice,
                                node.value.center - self.fov_shape // 2,
                                node.value.center + self.fov_shape // 2 + 1,
                            )
                        )
                    )
                    > 0
                )
                / np.prod(self.fov_shape)
            )
            """

            nid_score_map[node.key] = (direction, mag)
            if any(np.isnan(x) for x in direction):
                raise ValueError("Direction is NAN!")

            if consenus:
                nid_score_map = self._smooth_scores(nid_score_map)

        return nid_score_map

    def get_node_connectivity(self, ascending=True):
        """
        Return a list of nodes and their "connectivity score".
        Connectivity score is relatively arbitrary with the property
        that 1 means well connected, 0 means not connected.

        Instead of returning every node and its connectivity to its neighbor,
        return nodes representing regions of low connectivity and rank them.

        TODO: make this more general
        """
        print("starting node connectivity")
        nid_score_map = {}
        for node in self.get_nodes():
            try:
                nid_score_map[node.key] = (
                    node.parent.key,
                    self.get_connection(node.value.center, node.parent.value.center),
                )
            except Exception as e:
                print(e)
        return nid_score_map

    def calculate_strahlers(self):
        queue = []
        for node in self.get_nodes():
            queue.append(node)
        while len(queue) > 0:
            current = queue.pop()
            current._calculate_strahler()

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

    def save_data_n5(self, folder_path, dataset_path):
        """
        Save the data gathered in the n5 format.

        dependent on pyn5 library which is expected to change a lot.
        """
        datasets = {
            "segmentation": self.segmentation,
            "counts": self.segmentation_counts,
        }
        for name, data in datasets.items():
            print("Saving {} to n5!".format(name))
            print("Num leaves = {}".format(len(list(data.iter_leaves()))))
            data.write_to_n5(folder_path, dataset_path + "/" + name)

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
            print("nan")

        return (change_direction, change_mag)

    @staticmethod
    def _get_max_vec(data):
        v = np.unravel_index(np.argmax(data), data.shape)
        if np.isclose(data[v], 0):
            return np.array([0, 0, 0]), 0
        else:
            v = (v - np.array(data.shape) // 2) / (np.array(data.shape) // 2)
            mag = (
                np.linalg.norm(v)
                / (np.linalg.norm(np.array(data.shape)) // 2)
                * np.max(data)
            )
            if not np.isclose(mag, 0):
                direction = v / mag
            return (direction, mag)

    @staticmethod
    def _create_sphere(shape, resolution):
        """
        Create a roughly isotropic shpere constrained in the bounds of shape to
        avoid letting non-isotropic data bias calculations.

        Especially important when detecting missing branches since we want to be
        able to detect branches in the z direction, and not let them get over powered by
        the extended view range in the x-y directions
        """

        def dist_to_center(i, j, k, shape, resolution):
            i = (
                (2 * (i - shape[0] // 2))  # scale: [0-shape-1] - [-shape-1, shape-1]
                * resolution[0]  # scale up by resolution to get isotropic distances
                / np.min(shape * resolution)  # scale shortest axis down to [-1,1]
            )
            j = (2 * (j - shape[1] // 2)) * resolution[1] / np.min(shape * resolution)
            k = (2 * (k - shape[2] // 2)) * resolution[2] / np.min(shape * resolution)
            return (i ** 2 + j ** 2 + k ** 2) ** (0.5)

        sphere = np.ones(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if dist_to_center(i, j, k, shape, resolution) > 1:
                        sphere[i, j, k] = 0
        return sphere

    def get_connection(self, index_a, index_b):
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

    def get_dist_weighted_mask(
        self, bounds, min_overlap_count=1, increment_denominator=False
    ):
        mask = self.get_count_weighted_mask(
            bounds, min_overlap_count, increment_denominator
        )

        dist = self.distances
        distances = dist[bounds]

        mask = mask * np.nan_to_num(distances)
        return mask

    def get_count_weighted_mask(
        self, bounds, min_overlap_count=1, increment_denominator=False
    ):
        segmentation = self.segmentation
        counts = self.segmentation_counts

        mask = segmentation[bounds].astype(float)
        seg_counts = counts[bounds].astype(float)
        if increment_denominator:
            mask = mask / (seg_counts + 1)

        else:
            mask[mask < min_overlap_count] = 0
            mask[mask >= min_overlap_count] = (
                mask[mask >= min_overlap_count] / seg_counts[mask >= min_overlap_count]
            )

        return mask

    def calculate_center_of_mass_vects(self):
        sphere = None
        for node in self.get_nodes():
            mask = node.value.mask
            if mask is None:
                continue
            data = np.copy(mask)
            if sphere is None:
                sphere = self._create_sphere(data.shape)
            data[sphere == 0] = 0
            direction, mag = self._get_center_of_mass(data)
            self.input_center_of_mass(node, (direction, mag))

    def resample_segments(self, delta, steps, sigma_fraction):
        """
        resample tree to have evenly spaced nodes. Gaussian smooth the curve
        and then sample at regular intervals.

        inputs: tree, nodes, regions or coords (nodes makes most sense I think)
        outputs: tree, nodes, regions or coords (coords makes most sense I think)
        io can then generate a new skeleton from coords
        """

        def handle_root(node, seen_roots, new_id, seen_tails):
            if new_id == 0:
                new_node = (
                    new_id,
                    None,
                    node.value.center[0],
                    node.value.center[1],
                    node.value.center[2],
                )
                seen_roots[node.key] = new_node
                new_id += 1
                return [new_node], seen_roots, new_id

            elif node.key in seen_roots:
                return [], seen_roots, new_id

            elif node.key in seen_tails:
                seen_roots[node.key] = seen_tails[node.key]
                del seen_tails[node.key]
                return [], seen_roots, new_id

        # store tail points so that we can use them as roots
        branch_points = {}
        # create a list of new node points of form (nid, pid, x, y, z)
        new_tree_nodes = []
        # new nodes will need new nids/pids thus we will reassign all nids starting at 0
        new_node_id = 0
        # get each straight segment
        for segment in self.get_segments():
            # make sure each point in the segment has coordinates
            assert all(
                [
                    node.value is not None and node.value.center is not None
                    for node in segment
                ]
            ), "segment contains some nodes with no center coordinates"

            if new_node_id == 0:
                root = (0, None, *segment[0].value.center)
                new_node_id += 1
                new_tree_nodes.append(root)
                branch_points[segment[0].key] = root

            # get interpolated nodes. (TAIL included but not Root)
            new_interpolated_nodes, new_node_id = self.resample_segment(
                segment,
                delta,
                steps,
                sigma_fraction,
                new_node_id,
                branch_points[segment[0].key],
            )
            # handle the tail node (new_tail will be empty list if not needed)
            branch_points[segment[-1].key] = new_interpolated_nodes[-1]

            new_tree_nodes = new_tree_nodes + new_interpolated_nodes
        new_skeleton = self.new_skeleton()
        new_skeleton.input_nid_pid_x_y_z(new_tree_nodes)
        return new_skeleton

    def resample_segment(self, nodes, delta, steps, sigma_fraction, new_node_id, root):
        def get_smoothed(coords, steps=100, sigma_fraction=.001):
            x_y_z = list(zip(*coords))
            t = np.linspace(0, 1, len(coords))
            t2 = np.linspace(0, 1, steps)

            x_y_z_2 = list(map(lambda x: np.interp(t2, t, x), x_y_z))
            x_y_z_3 = list(
                map(
                    lambda x: gaussian_filter1d(
                        x, steps * sigma_fraction, mode="nearest"
                    ),
                    x_y_z_2,
                )
            )
            return zip(*x_y_z_3)

        def downsample(coords, delta, origin, end):
            previous = origin
            for coord in coords + [end]:
                sqdist = sum((np.array(coord) - np.array(previous)) ** 2)
                if sqdist < delta ** 2:
                    continue
                elif sqdist > (2 * delta) ** 2:
                    k = (sqdist ** 0.5) // delta
                    for i in range(0, int(k)):
                        new_coords = (
                            ((k - 1) - i) * np.array(previous)
                            + (i + 1) * np.array(coord)
                        ) / k

                        if any(np.isnan(new_coords)):
                            raise Exception("NAN FOUND")
                        yield list(new_coords)
                        previous = coord
                else:
                    yield coord
                    previous = coord
            if not all(abs(np.array(previous) - np.array(end)) < np.array([1e-5] * 3)):
                yield end

        coords = [node.value.center for node in nodes]
        smoothed_coords = list(get_smoothed(coords, steps, sigma_fraction))
        downsampled_coords = list(
            downsample(smoothed_coords, delta, root[2:], nodes[-1].value.center)
        )
        if len(downsampled_coords) == 0:
            print("No downsampling?")
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
