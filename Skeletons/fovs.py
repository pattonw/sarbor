from typing import Tuple, Dict

import numpy as np

Bounds = Tuple[np.ndarray, np.ndarray]


class Node:
    """
    The region class contains information and operations specific to
    the neuron skeleton nodes.
    TODO: Think of different names to make this clearer
    """

    def __init__(self, center=None, mask=None):
        """
        bounds: [np.array, np.array]
        mask: np.array
        """
        self._center = center
        self._mask = mask
        self._center_of_mass = None

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
        return (self.center - fov_shape // 2, self.center + fov_shape // 2 + 1)

    def insert_values(self, values):
        if values is not None:
            fov_shape = values.get("fov_shape", None)
            if fov_shape is not None:
                self.fov_shape = fov_shape
            center = values.get("center", None)
            if center is not None:
                self.center = center
            mask = values.get("mask", None)
            if mask is not None:
                self.mask = mask

    def transform(self, translation, scale):
        """
        The only way to change center and field of view.
        This means the coordinate system of center and
        fov_shape will always match as long as they are
        """
        self._center = (self._center + translation) * scale

    def clone(self):
        new_region = type(self)(center=self._center.clone(), mask=self._mask.clone())
        return new_region


class NodeData:
    def __init__(self):
        self._nodes = None
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
            return self._nodes

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
