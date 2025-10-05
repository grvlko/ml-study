import numpy as np
from collections import defaultdict

from typing import DefaultDict, Iterable, List, Self
from numpy.typing import NDArray


class Node:
    def __init__(self, label, depth, parent=None):
        self.label = label
        self.depth = depth
        self.parent = parent
        self.children = {}
        self.next = None


class SimplexTree:
    def __init__(self, num_vertices: int) -> None:
        """
        Initialize a SimplexTree with a given number of vertices.

        Parameters:
        num_vertices (int): The number of vertices in the simplex tree.

        Raises:
        ValueError: If num_vertices is less than 1.
        """
        if num_vertices < 1:
            raise ValueError("Vertex set must not be empty")
        self.num_vertices = num_vertices
        self.list_heads: DefaultDict[int, dict[int, Node]] = defaultdict(dict)
        self.root = Node(label=None, depth=0, parent=None)
        self.root.children = {
            i: Node(i, 1, parent=self.root) for i in range(num_vertices)
        }
        for node in self.root.children.values():
            self._add_to_circular_list(node)

    def dimension(self):
        return len(self.list_heads) - 1

    def _preprocess_simplex(self, simplex: List[int]) -> List[int]:
        if any(label < 0 or self.num_vertices < label for label in simplex):
            raise ValueError(
                f"Simplex's labels not in vertex set [0, {self.num_vertices - 1}]: {simplex}"
            )
        s = sorted(simplex)
        if any(s[i] == s[i + 1] for i in range(len(s) - 1)):
            raise ValueError(f"Duplicate labels in simplex: {simplex}")
        return s

    def insert(self, simplex: List[int]) -> None:
        """
        Insert a simplex and all its faces in the simplicial complex.

        Parameters
        ----------
        simplex : list of int
            The simplex to be inserted, represented as a list of labels.

        Raises
        ------
        ValueError
            If the passed simplex is invalid (e.g. contains duplicate labels, or a label not in the vertex set).
        """
        simplex = self._preprocess_simplex(simplex)
        for i, label in enumerate(simplex):
            self._rec_insert(simplex, self.root.children.get(label), i + 1)

    def _add_child(self, node: Node, label: int) -> Node:
        next_node = node.children.get(label)
        if next_node is None:
            next_node = Node(label, node.depth + 1, parent=node)
            node.children[label] = next_node
            self._add_to_circular_list(next_node)
        return next_node

    def insert_simplex(self, simplex: List[int]) -> None:
        """
        Insert a simplex into the simplicial complex. This method doesn't insert simplex faces, violating simplicial
        complex structure. It is used for precise simplex insertion.

        Parameters
        ----------
        simplex : list of int
            The simplex to be inserted, represented as a list of labels.

        Raises
        ------
        ValueError
            If the passed simplex is invalid (e.g. contains duplicate labels, or a label not in the vertex set).
        AttributeError
            If the complex doesn't contain all prefix simplices of sorted given simplex
        """
        if len(simplex) < 2:
            return
        simplex = self._preprocess_simplex(simplex)
        node = self.root
        for i in range(len(simplex) - 1):
            node = node.children.get(simplex[i])
        next_node = Node(simplex[-1], len(simplex), parent=node)
        node.children[simplex[-1]] = next_node
        self._add_to_circular_list(next_node)

    def insert_simplices(self, simplices: Iterable[List[int]]):
        for simplex in simplices:
            self.insert_simplex(simplex)

    def _rec_insert(self, simplex: List[int], node: Node, i: int) -> None:
        if i == len(simplex):
            return
        for j in range(i, len(simplex)):
            label = simplex[j]
            next_node = node.children.get(label)
            if next_node is None:
                next_node = Node(label, node.depth + 1, parent=node)
                node.children[label] = next_node
                self._add_to_circular_list(next_node)
            self._rec_insert(simplex, next_node, j + 1)

    def _add_to_circular_list(self, node):
        head = self.list_heads[node.depth].get(node.label)
        if head is None:
            self.list_heads[node.depth][node.label] = node
            node.next = node
        else:
            node.next = head.next
            head.next = node

    def get_simplices(self, dim):
        if dim < 0 or self.num_vertices <= dim:
            return []
        nodes = []
        for head in self.list_heads[dim + 1].values():
            nodes.append(head)
            current = head.next
            while current != head:
                nodes.append(current)
                current = current.next
        simplices = []
        for node in nodes:
            current = node
            simplex = []
            while current.parent is not None:
                simplex.append(current.label)
                current = current.parent
            simplices.append(tuple(reversed(simplex)))
        return sorted(simplices)
    
    def lower_star_filtration(self, f: List[float]) -> List[Self]:
        """
        Build lower star filtration using the simplex tree structure.
        
        Parameters
        ----------
        f : list of float
            Function values on vertices, f[i] is value for vertex i
            
        Returns
        -------
        list of tuples
            Simplices in filtration order
        """
        vertices = list(range(self.num_vertices))
        sorted_vertices = sorted(vertices, key=lambda v: f[v])
        rank = {v: i for i, v in enumerate(sorted_vertices)}
        lower_stars = {v: [] for v in vertices}
        
        self._traverse_node(self.root, [], rank, lower_stars)
        filtration = []
        for v in sorted_vertices:
            simplices = lower_stars[v]
            simplices.sort(key=len)
            filtration.extend(simplices)
            
        return filtration
    
    def _traverse_node(self, node: Node, current_simplex, rank, lower_stars):
        if node.depth > 0:
            current_simplex = current_simplex + [node.label]
            
            max_vertex = max(current_simplex, key=lambda v: rank[v])
            
            lower_stars[max_vertex].append(tuple(current_simplex))
        
        for child in node.children.values():
            self._traverse_node(child, current_simplex, rank, lower_stars)
    
    def boundary_matrix(self, n: int) -> NDArray[np.int_]:
        simplices = self.get_simplices(n)
        next_simplices = self.get_simplices(n - 1)
        matrix = np.zeros((len(next_simplices), len(simplices)), dtype=np.int_)
        if n < 1 or self.num_vertices <= n:
            return matrix
        return self._fill_boundary_matrix(matrix, simplices, next_simplices)

    def _fill_boundary_matrix(self, matrix, simplices, next_simplices):
        facet_index = {simplex: i for i, simplex in enumerate(next_simplices)}
        for j, simplex in enumerate(simplices):
            for i in range(len(simplex)):
                facet = tuple(simplex[:i] + simplex[i + 1 :])
                matrix[facet_index[facet]][j] = (-1) ** i
        return matrix

    def boundary_matrices(self) -> List[NDArray[np.int_]]:
        simplices = [self.get_simplices(n) for n in range(self.num_vertices)]
        matrices = []
        for n in range(1, self.num_vertices):
            matrix = np.zeros((len(simplices[n - 1]), len(simplices[n])), dtype=np.int_)
            matrices.append(
                self._fill_boundary_matrix(matrix, simplices[n], simplices[n - 1])
            )
        return matrices

    def chain_dim(self, n: int) -> int:
        return len(self.get_simplices(n))

    def chain_dims(self):
        return [self.chain_dim(n) for n in range(self.num_vertices)]

    def boundary_dim(self, n):
        return self._boundary_dim(self.boundary_matrix(n + 1))

    def _boundary_dim(self, n):
        try:
            return np.linalg.matrix_rank(self.boundary_matrix(n + 1))
        except ValueError:
            return 0

    def boundary_dims(self):
        dims = []
        for matrix in self.boundary_matrices():
            dims.append(self._boundary_dim(matrix))
        return dims + [0]

    def cycle_dim(self, n):
        return self.chain_dim(n) - self.boundary_dim(n - 1)

    def cycle_dims(self):
        return [self.cycle_dim(n) for n in range(self.num_vertices)]

    def betti(self, n):
        return self.cycle_dim(n) - self.boundary_dim(n)

    def bettis(self):
        return [self.betti(n) for n in range(self.num_vertices)]

    def print_tree(self, node: Node = None) -> None:
        self._print_tree(self.root if node is None else node, "")

    def _print_tree(self, node: Node, prefix: str) -> None:
        for i, child in enumerate(node.children.values(), 1):
            print(
                prefix
                + ("└── " if i == len(node.children) else "├── ")
                + str(child.label)
            )
            self._print_tree(
                child, prefix + ("    " if i == len(node.children) else "│   ")
            )
