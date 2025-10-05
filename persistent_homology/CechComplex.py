from typing import Iterable, List
from numpy.typing import NDArray
from utils import compute_proximity_graph, minimal_enclosing_ball
from SimplexTree import SimplexTree, Node
import numpy as np


class SimlexTreeForCechComplex(SimplexTree):
    def __init__(
        self, points: List[NDArray], epsilon: np.float64, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.points = points
        self.epsilon = epsilon
        self.cache = {}

    def expansion(self, dim_max: int) -> None:
        if dim_max < 2:
            return
        for node in reversed(self.root.children.values()):
            for child in node.children.values():
                self.cache[child] = minimal_enclosing_ball(
                    [self.points[node.label], self.points[child.label]]
                )
            self._siblings_expansion(list(node.children.values())[::-1], dim_max - 1)

    def missing_boundary(self, node: Node, path: List[int]) -> bool:
        for label in reversed(path):
            node = node.children.get(label)
            if node is None:
                return True
        return False

    def _check_boundaries(self, node: Node, label: int) -> bool:
        path = [label, node.label]
        node = node.parent
        while node != self.root:
            if self.missing_boundary(node.parent, path):
                return False
            path.append(node.label)
            node = node.parent
        return True

    def _minimal_enclosing_ball(self, node: Node, label: int) -> None:
        center, radius_sq = self.cache[node]
        if np.sum((center - self.points[label]) ** 2) <= radius_sq:
            return (center, radius_sq)
        path = [label]
        for i in range(node.depth):
            current = node.parent
            for label in reversed(path):
                current = current.children[label]
            center, radius_sq = self.cache[current]
            if np.sum((center - self.points[node.label]) ** 2) <= radius_sq:
                return (center, radius_sq)
            path.append(node.label)
            node = node.parent
        return minimal_enclosing_ball([self.points[label] for label in path])

    def _siblings_expansion(self, siblings: List[Node], k: int) -> None:
        """Recursively add simplices to the complex by connecting siblings. Goes k levels deeper at max. The method
        assumes that the siblings are indeed children of the same parent node and that they are passed in descending
        order (by label).

        Parameters
        ----------
        siblings : list of Node
            A list of nodes to use for expansion.
        k : int
            The maximum depth to expand relative to siblings.
        """
        if k == 0 or len(siblings) < 2:
            return
        for node in siblings:
            labels = []
            for next_node in siblings:
                if next_node == node:
                    break
                if self._check_boundaries(node, next_node.label):
                    labels.append(next_node.label)
            new_siblings = []
            for label in labels:
                center, radius_sq = self._minimal_enclosing_ball(node, label)
                new_node = self._add_child(node, label)
                self.cache[new_node] = (center, radius_sq)
                new_siblings.append(new_node)
            self._siblings_expansion(new_siblings, k - 1)


class CechComplex:
    def __init__(self, points: Iterable[NDArray], epsilon: float):
        self.points = points
        self.epsilon = epsilon
        self.proximity_graph = compute_proximity_graph(points, epsilon)

    def create_complex(self, dim_max: int | None = None) -> SimplexTree:
        if dim_max is None:
            dim_max = len(self.points) - 1
        complex = SimlexTreeForCechComplex(self.points, self.epsilon, len(self.points))
        complex.insert_simplices(self.proximity_graph)
        complex.expansion(dim_max)
        return complex
