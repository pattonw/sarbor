
from collections import deque


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
        potential_queue = []
        last = None
        for node in self.traverse(True):
            while node.parent != last and len(potential_queue) > 0:
                del potential_queue[0]
                if len(potential_queue) > 0:
                    last = potential_queue[0]
            potential_queue.insert(0, node)
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

    class Node:
        """
        Basic Node datastructure, has basic getter and setter methods
        """

        def __init__(self, nid, value=None, strahler=None):
            """
            node has only key, value, parent
            """
            self.key = nid
            self.value = value
            self.parent = None
            self.children = []
            self.strahler = strahler

        @property
        def key(self):
            """
            A unique identifier for this Node
            """
            return self._key

        @key.setter
        def key(self, key):
            self._key = int(key)

        @property
        def value(self):
            """
            The data stored in this Node
            """
            return self._value

        @value.setter
        def value(self, value):
            self._value = value

        @property
        def parent(self):
            """
            This Nodes parent. (None if this node is the root of a tree)
            """
            return self._parent

        @parent.setter
        def parent(self, parent):
            self._parent = parent

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
        def strahler(self):
            if self._strahler is not None:
                return self._strahler
            else:
                self._calculate_strahler()
                return self._strahler

        @strahler.setter
        def strahler(self, strahler):
            self._strahler = strahler

        @property
        def types(self):
            types = []
            if self.parent is None:
                types.append("Root")
            children = self.children
            if len(children) > 1:
                types.append("Branch")
            if len(children) == 1:
                types.append("Regular")
            if len(children) == 0:
                types.append("Leaf")
            return types

        def _calculate_strahler(self):
            queue = list(self.traverse(fifo=False))
            for node in queue[::-1]:
                children = node.children
                if len(children) == 0:
                    node.strahler = 1
                elif len(children) == 1:
                    node.strahler = children[0].strahler
                else:
                    max_strahler, count = 0, 0
                    for child in children:
                        if child.strahler > max_strahler:
                            max_strahler = child.strahler
                            count = 1
                        elif child.strahler == max_strahler:
                            count += 1
                    node.strahler = max_strahler + 1 if count > 1 else max_strahler

        def clone(self):
            """
            Return all the information necessary to build a clone of this node
            (key, strahler, parent key, and value).
            This is useful for creating sub_trees where the node ids and values should
            stay the same, but some parents or children will be removed
            """
            return (
                type(self)(self.key, strahler=self.strahler),
                self.parent.key if self.parent is not None else None,
                self.value,
            )

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

        def traverse(self, fifo=True):
            queue = deque([self])
            while len(queue) > 0:
                current = queue.popleft()
                yield current
                for child in current.get_children():
                    if fifo:
                        queue.insert(0, child)
                    else:
                        queue.append(child)

