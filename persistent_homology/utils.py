import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import cm

from typing import Callable, List, Tuple
from numpy.typing import NDArray

from SimplexTree import SimplexTree


def compute_proximity_graph(
    points: List[NDArray], threshold: float, distance: Callable = None
) -> list[list[int]]:
    if distance is None:

        def distance(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2))

    points = list(points)
    edges = []
    for i, p in enumerate(points):
        for j in range(i + 1, len(points)):
            if distance(p, points[j]) <= threshold:
                edges.append([i, j])
    return edges


def minimal_enclosing_ball(points: List[NDArray]) -> Tuple[NDArray, np.float64]:
    if len(points) == 1:
        return points[0], 0.0
    if len(points) == 2:
        return (points[0] + points[1]) / 2, np.sum((points[0] - points[1]) ** 2) / 4

    return welzl_algorithm(points, [])


def welzl_algorithm(
    points: List[NDArray], boundary: List[NDArray]
) -> Tuple[NDArray, np.float64]:
    if len(boundary) == 3 or not points:
        return miniball_from_points(boundary)
    p = points[0]
    remaining = points[1:]
    center, radius_sq = welzl_algorithm(remaining, boundary)
    if np.sum((p - center) ** 2) <= radius_sq:
        return center, radius_sq
    return welzl_algorithm(remaining, boundary + [p])


def miniball_from_points(points: List[NDArray]) -> Tuple[NDArray, np.float64]:
    if len(points) == 0:
        return 0.0, 0.0
    if len(points) == 1:
        return points[0], 0.0
    if len(points) == 2:
        return minimal_enclosing_ball(points)
    A, B, C = points
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    alpha = a**2 * (b**2 + c**2 - a**2)
    beta = b**2 * (a**2 + c**2 - b**2)
    gamma = c**2 * (a**2 + b**2 - c**2)
    total = alpha + beta + gamma
    center = (alpha * A + beta * B + gamma * C) / total
    radius_sq = np.max(np.sum((points - center) ** 2, axis=1))
    return center, radius_sq


def plot_2d(complex: SimplexTree, points: List[NDArray], alpha: float = 0.15) -> None:
    if len(points[0]) != 2:
        raise ValueError()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    colors = cm.viridis(np.linspace(0, 1, 3))
    points = np.array(points)

    ax.scatter(points[:, 0], points[:, 1], c=colors[0], s=50, zorder=3)

    segments = []
    for simplex in complex.get_simplices(1):
        segments.append([points[simplex[0]], points[simplex[1]]])
    ax.add_collection(
        LineCollection(
            segments, colors=colors[1], linewidths=1.5, alpha=alpha, zorder=2
        )
    )

    polygons = []
    for simplex in complex.get_simplices(2):
        polygons.append([points[simplex[0]], points[simplex[1]], points[simplex[2]]])
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors=colors[2],
            edgecolors="k",
            alpha=alpha,
            linewidths=0.5,
            zorder=1,
        )
    )

    min_coord = np.min(points, axis=0) * 0.99
    max_coord = np.max(points, axis=0) * 1.01
    ax.set_xlim(min_coord[0], max_coord[0])
    ax.set_ylim(min_coord[1], max_coord[1])
    plt.tight_layout()
    plt.show()
