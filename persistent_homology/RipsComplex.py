from typing import Iterable, List
from numpy.typing import NDArray
from utils import compute_proximity_graph
from SimplexTree import SimplexTree, Node
import numpy as np


class SimplexTreeForRipsComplex(SimplexTree):
    def __init__(self, points: List[NDArray], epsilon: np.float64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.points = points
        self.epsilon = epsilon

    def expansion(self, dim_max: int) -> None:
        if dim_max < 2:
            return
        
        for node in reversed(self.root.children.values()):
            self._siblings_expansion(list(node.children.values())[::-1], dim_max - 1)

    def _check_rips_condition(self, simplex_vertices: List[int]) -> bool:
        vertices = [self.points[v] for v in simplex_vertices]
        
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                distance = np.linalg.norm(vertices[i] - vertices[j])
                if distance > self.epsilon:
                    return False
        return True

    def _get_simplex_vertices(self, node: Node) -> List[int]:
        vertices = []
        current = node
        while current != self.root:
            vertices.append(current.label)
            current = current.parent
        return vertices[::-1]

    def _siblings_expansion(self, siblings: List[Node], k: int) -> None:
        if k == 0 or len(siblings) < 2:
            return
            
        for node in siblings:
            labels = []
            current_vertices = self._get_simplex_vertices(node)
            
            for next_node in siblings:
                if next_node == node:
                    break
                    
                candidate_vertices = current_vertices + [next_node.label]
                
                if self._check_rips_condition(candidate_vertices):
                    labels.append(next_node.label)
            
            new_siblings = []
            for label in labels:
                new_node = self._add_child(node, label)
                new_siblings.append(new_node)
            
            self._siblings_expansion(new_siblings, k - 1)


class RipsComplex:
    def __init__(self, points: Iterable[NDArray], epsilon: float):
        self.points = list(points)
        self.epsilon = epsilon
        self.proximity_graph = compute_proximity_graph(self.points, epsilon)

    def create_complex(self, dim_max: int | None = None) -> SimplexTree:
        if dim_max is None:
            dim_max = len(self.points) - 1
            
        complex = SimplexTreeForRipsComplex(self.points, self.epsilon, len(self.points))
        complex.insert_simplices(self.proximity_graph)
        complex.expansion(dim_max)
        return complex