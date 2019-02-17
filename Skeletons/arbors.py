from collections import deque
import numpy as np
from typing import Tuple, Dict, Optional, Any, List

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
        self._value = kwargs.get("value", None)
        self.value.center = kwargs.get("center", None)
        self.value.mask = kwargs.get("mask", None)

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
    def value(self):
        """
        A unique identifier for this Node
        """
        if self._value is None:
            self._value = NodeData()
        return self._value

    @value.setter
    def value(self, value):
        if self._value is None:
            self._value = value
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

    @property
    def children(self):
        if self._children is None:
            self._children = []
        return self._children

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
        if self.value is not None:
            self.value.strahler = strahler

    def is_root(self) -> bool:
        return self.parent is None

    def is_branch(self) -> bool:
        return len(self.get_neighbors()) > 2

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_regular(self) -> bool:
        normal = len(self.get_neighbors()) == 2
        if normal:
            assert not (self.is_branch() or self.is_leaf())
        return normal

    def _calculate_strahler(self) -> int:
        if self.is_leaf():
            return 1
        child_strahlers = [child._strahler for child in self.children]
        if any([strahler is None for strahler in child_strahlers]):
            raise ValueError("A child did not have a strahler index")
        max_child_strahler, count = 1, 0
        for child_strahler in child_strahlers:
            if child_strahler > max_child_strahler:
                max_child_strahler, count = child_strahler, 1
            elif child_strahler == max_child_strahler:
                count += 1
        if count > 1:
            return max_child_strahler + 1
        else:
            return max_child_strahler

    @property
    def data(self):
        """
        Return all the information necessary to build a clone of this node
        (key, parent_key, strahler).
        """
        data = {"nid": self.key, "pid": self.parent_key, "strahler": self.strahler}
        if self.value is not None:
            data.update(self.value.data())
        return data

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

    def traverse(self, ignore: Optional[List[int]] = None):
        queue = deque([self])
        while len(queue) > 0:
            current = queue.pop()
            yield current
            for child in current.children:
                if ignore is None or child.key not in ignore:
                    queue.append(child)


class Arbor:
    """
    A basic arbor structure. Only has access to nodes and edges
    and provides as much functionality it can with just that.
    Any other data can be stored in the value parameter of
    individual nodes.
    """

    def __init__(self, root: Node = None):
        """
        Initialize an empty tree
        """
        self.root = root
        self._nodes = {}  # Dict[int, Node]
        if root is not None:
            self._nodes[root.key] = root

    def search(self, key: int) -> Node:
        for node in self.traverse():
            if key == node.get_key():
                return node
        raise Exception("node {0} does not exist".format(key))

    @property
    def nodes(self) -> Dict[int, Node]:
        if self._nodes is None:
            self._nodes = {}
        return self._nodes

    def build_from_root(self, root: Node):
        self.root = root
        for node in self.traverse():
            self.nodes[node.key] = node

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

    def breadth_first_traversal(self, ignore: Optional[List[int]] = None):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.pop()
            yield current
            for child in current.children:
                if ignore is not None and child.key not in ignore:
                    queue.appendleft(child)

    def depth_first_traversal(self, ignore: Optional[List[int]] = None):
        queue = deque([self.root])

        while len(queue) > 0:
            current = queue.pop()
            yield current
            for child in current.children:
                if ignore is not None and child.key not in ignore:
                    queue.append(child)

    def get_interesting_nodes(self, root=False, leaves=False, branches=False):
        if root or leaves or branches:
            for node in self.traverse():
                if root:
                    root = False
                    yield node
                elif branches and node.is_branch():
                    yield node
                elif leaves and node.is_leaf():
                    yield node


class NodeData:
    """
    Contains the data for a node
    """

    def __init__(self, data: Dict[str, Any] = {}):
        self._data = data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def center(self) -> np.ndarray:
        """
        Get the center of a region.
        """
        c = self.data.get("center", None)
        if c is None:
            return None
        else:
            return c

    @center.setter
    def center(self, center: np.ndarray):
        if self.center is None:
            self.data["center"] = center
        else:
            print("Overriding the center {} with {}".format(self.center, center))

    @property
    def mask(self) -> Optional[np.ndarray]:
        m = self.data.get("mask", None)
        if m is None:
            # mask can be None
            return None
        else:
            return m

    @mask.setter
    def mask(self, mask: np.ndarray):
        if self.data.get("mask", None) is None:
            self.data["mask"] = mask
        else:
            raise Exception("Overriding the mask is not supported")

    def get_bounds(self, fov_shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note fov_shape should be in nm cubed
        """
        return (self.center - fov_shape // 2, self.center + fov_shape // 2 + 1)

    def clone_center(self):
        if self.center is not None:
            return self.center.clone()
        else:
            return None

    def clone_mask(self):
        if self.mask is not None:
            return self.mask.clone()
        else:
            return None

    def clone(self):
        return {"mask": self.clone_mask(), "center": self.clone_center()}


class SpatialArbor(Arbor):
    """
    Class for storing and accessing local segmentations around each node
    """

    def __init__(self):
        Arbor.__init__(self)
        self._bounds = None

    @property
    def node_bounds(self) -> Bounds:
        """
        Bounds containing all node centers
        """
        if self._bounds is None:
            self._bounds = self.calculate_tree_bounds()
        return self._bounds

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

    def calculate_strahler_indicies(self):
        strahler_indicies = {}
        node_queue = deque()
        for node in self.breadth_first_traversal():
            node_queue.append(node)
        while len(node_queue) > 0:
            current = node_queue.pop()
            current.strahler = current._calculate_strahler()
            strahler_indicies[current] = current.strahler
        return strahler_indicies
