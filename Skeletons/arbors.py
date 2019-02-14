from collections import deque
import numpy as np
from typing import Tuple, Dict

Bounds = Tuple[np.ndarray, np.ndarray]


class Node:
    """
    Basic Node datastructure, has basic getter and setter methods
    """

    def __init__(self, **kwargs):
        """
        node has only key, value, parent
        """
        self._key = kwargs.get("key", None)
        self._parent = kwargs.get("parent", None)
        self._children = kwargs.get("children", None)
        self._strahler = kwargs.get("strahler", None)

    @property
    def key(self):
        """
        A unique identifier for this Node
        """
        if self._key is None:
            raise ValueError("This node does not yet have a key")
        else:
            return self._key

    @key.setter
    def key(self, key):
        if self._key is None:
            self._key = key
        else:
            raise ValueError("Overwriting node keys is not supported")

    @property
    def parent(self):
        """
        This Nodes parent. (None if this node is the root of a tree)
        """
        return self._parent

    @property
    def parent_key(self):
        """
        Returns the parents key or None if no parent
        """
        if self.parent is None:
            return None
        else:
            return self.parent.key

    @parent.setter
    def parent(self, parent):
        if self._parent is None:
            self._parent = parent
        else:
            raise ValueError("Overwriting node parents is not supported")

    def get_children(self):
        return self.children

    def set_children(self, children):
        self.children = children

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def get_neighbors(self):
        if self.parent is not None:
            return [self.parent] + self.children
        else:
            return self.children

    @property
    def strahler(self) -> int:
        if self._strahler is not None:
            return self._strahler
        else:
            self._strahler = self._calculate_strahler()
            return self._strahler

    @strahler.setter
    def strahler(self, strahler):
        self._strahler = strahler

    def is_root(self) -> bool:
        return self.parent is None

    def is_branch(self) -> bool:
        return len(self.children) > 1

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def _calculate_strahler(self) -> int:
        if self.is_leaf():
            return 1
        child_strahlers = [child._strahler for child in self.get_children()]
        if any([strahler is None for strahler in child_strahlers]):
            raise ValueError("A child did not have a strahler index")
        max_strahler, count = 1, 0
        for strahler in child_strahlers:
            if strahler > max_strahler:
                max_strahler, count = strahler, 1
            elif strahler == max_strahler:
                count += 1
        if count > 1:
            return max_strahler + 1
        else:
            return max_strahler

    def get_data(self):
        """
        Return all the information necessary to build a clone of this node
        (key, parent_key, strahler).
        """
        return (self.key, self.parent_key, self.strahler)

    def get_following(self, previous):
        """
        get the next node from the perspective of the previous.
        i.e. given nodes:
            a--b--c
        b.get_following(a) = c
        b.get_vollowing(c) = a
        """
        neighbors = self.get_neighbors()
        if len(neighbors) > 0 and previous in neighbors:
            neighbors.remove(previous)
            return neighbors
        elif previous not in neighbors:
            return neighbors
        else:
            raise Exception("This node has {} neighbors".format(len(neighbors)))


class Arbor:
    """
    A basic arbor structure. Only has access to nodes and edges
    and provides as much functionality it can with just that.
    Any other data can be stored in the value parameter of
    individual nodes.
    """

    def __init__(self, root=None):
        """
        Initialize an empty tree
        """
        self.root = root

    def search(self, key):
        for node in self.traverse():
            if key == node.get_key():
                return node
        raise Exception("node {0} does not exist".format(key))

    def get_key_map(self):
        key_map = {}
        for node in self.traverse():
            key_map[node.key] = node
        return key_map

    def traverse(self, fifo=False):
        """
        Iterate over the elements of the tree
        traversal options:
            - first in first out (depth first)
            - first in last out (bredth first)
            """
        if self.root is None:
            raise Exception("this arbor has no root")
        else:
            if fifo:
                return self.depth_first_traversal()
            else:
                return self.breadth_first_traversal()

    def traverse_segments(self):
        queue = deque([self.root])

        while len(queue) > 0:
            root = queue.popleft()
            for child in root.get_children():
                segment = [root]
                current = child
                while True:
                    segment.append(current)
                    next_nodes = current.get_children()
                    if len(next_nodes) == 0:
                        break
                    elif len(next_nodes) > 1:
                        queue.append(current)
                        break
                    else:
                        segment.append(next_nodes[0])
                        current = next_nodes[0]

                yield segment

    def get_minimal_subtree(self, ids):
        """
        get the smallest possible subtree containing all given ids
        """
        uncovered = ids
        all_nodes = []
        potential_queue = []
        last = None
        for node in self.traverse(True):
            while node.parent != last and len(potential_queue) > 0:
                del potential_queue[0]
                if len(potential_queue) > 0:
                    last = potential_queue[0]
            potential_queue.insert(0, node)

            if node.key in uncovered:
                uncovered.remove(node.key)
                all_nodes = all_nodes + potential_queue
                potential_queue = []
                last = node

            if last is not None:
                last = node

        assert len(uncovered) == 0, "Not all nodes were found. missing: {}".format(
            uncovered
        )
        return all_nodes

    def get_root_leaf_paths(self):
        potential_queue = deque([])
        last = None
        for node in self.traverse(True):
            while node.parent != last and len(potential_queue) > 0:
                potential_queue.pop()
                if len(potential_queue) > 0:
                    last = potential_queue[-1]
            potential_queue.append(node)
            last = node
            if len(node.children) == 0:
                yield potential_queue

            if last is not None:
                last = node

    def breadth_first_traversal(self):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.popleft()
            yield current
            for child in current.get_children():
                queue.append(child)

    def depth_first_traversal(self):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.popleft()
            yield current
            for child in current.get_children():
                queue.insert(0, child)

    def get_interesting_nodes(self, root=False, leaves=False, branches=False):
        if root or leaves or branches:
            for node in self.arbor.traverse():
                if root:
                    root = False
                    yield node
                elif branches and node.is_branch():
                    yield node
                elif leaves and node.is_leaf():
                    yield node


class NodeWithData(Node):
    """
    The NodeWithData class contains a center value in nm coordinate space and a mask
    gathered via segmenting some region around that center.
    """

    def __init__(self, center=None, mask=None):
        """
        mask: np.array
        """
        super().__init__(self)
        self._center = center
        self._mask = mask

    @property
    def center(self) -> np.ndarray:
        """
        Get the center of a region.
        Throws error if center is None
        TODO: Change constructor so that center cannot be None
        """
        if self._center is None:
            raise Exception("No center available")
        else:
            return self._center.astype(int)

    @center.setter
    def center(self, center: np.ndarray):
        if self._center is None:
            self._center = center
        else:
            raise Exception("Overriding the center is not supported")

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            # mask can be None
            return None
        else:
            return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray):
        if self._mask is None:
            self._mask = mask
        else:
            raise Exception("Overriding the mask is not supported")

    def get_bounds(self, fov_shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note fov_shape should be in nm cubed
        """
        return (self.center - fov_shape // 2, self.center + fov_shape // 2 + 1)

    def clone(self):
        node_copy = type(self)(center=self._center.clone(), mask=self._mask.clone())
        return node_copy


class SpatialArbor(Arbor):
    """
    Class for storing and accessing local segmentations around each node
    """

    def __init__(self):
        Arbor.__init__(self)
        self._nodes = {}
        self._bounds = None

    @property
    def node_bounds(self) -> Bounds:
        """
        Bounds containing all node centers
        """
        if self._bounds is None:
            self._bounds = self.calculate_tree_bounds()
        return self._bounds

    @property
    def nodes(self) -> Dict[int, Node]:
        if self._nodes is None:
            raise ValueError("No node data to retrieve!")
        else:
            return self._nodes.items()

    def calculate_tree_bounds(self) -> Bounds:
        """
        Find the minimum and maximum node center
        """
        lower = np.array([float("inf"), float("inf"), float("inf")])
        upper = -lower.copy()
        for nid, node in self.nodes.items():
            if node.value is not None:
                upper = np.maximum(node.value.center, upper)
                lower = np.minimum(node.value.center, lower)
        return (lower.astype(int), upper.astype(int))

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
        Union of the nodes around each of they given nodes
        """
        big_radius = set()
        centers = [self.node_map[key].value.center for key in keys]
        for node in self.get_nodes():
            for center in centers:
                if all(abs(node.value.center - center) < radius):
                    big_radius.add(node.key)
                    break
        return big_radius

    def get_constrained_radius(self, node, dx, dy, dz):
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
                        <= np.array([dx, dy, dz])
                    ):
                        next_layer.append(neighbor)
            previous = layer[:]
            layer = next_layer[:]
        return all_nodes
