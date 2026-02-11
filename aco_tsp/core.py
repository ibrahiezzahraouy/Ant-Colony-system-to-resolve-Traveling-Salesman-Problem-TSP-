"""
Core module for Ant Colony Optimization applied to the Traveling Salesman Problem.

This module implements the fundamental components of the ACO algorithm:
- Node: represents a city with (x, y) coordinates
- Edge: represents a weighted connection between two cities
- Graph: manages the complete graph of nodes and edges
- Ant: simulates a single ant traversing the graph
- ACO: orchestrates the colony to find the shortest tour
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Distance function
# ---------------------------------------------------------------------------

def euclidean_distance(node1: "Node", node2: "Node") -> float:
    """Return the Euclidean distance between two nodes."""
    return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """Represents a city (node) in the TSP graph.

    Each node is assigned a unique auto-incrementing index upon creation.

    Attributes:
        x: X-coordinate of the city.
        y: Y-coordinate of the city.
        name: Optional human-readable label.
        index: Unique integer identifier (auto-assigned).
    """

    _counter: int = 0  # class-level counter

    def __init__(self, x: float, y: float, name: Optional[str] = None) -> None:
        self.x = x
        self.y = y
        self.name = name or f"City-{Node._counter}"
        self.index = Node._counter
        Node._counter += 1

    def __repr__(self) -> str:
        return f"Node({self.x}, {self.y}, name='{self.name}', index={self.index})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the global node counter (useful between independent runs)."""
        cls._counter = 0


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class Edge:
    """Represents a weighted, undirected connection between two nodes.

    Attributes:
        node1: First endpoint.
        node2: Second endpoint.
        distance: Length of the edge.
        pheromone: Current pheromone level on this edge.
    """

    def __init__(
        self,
        node1: Node,
        node2: Node,
        distance_fn: Callable[[Node, Node], float] = euclidean_distance,
    ) -> None:
        self.node1 = node1
        self.node2 = node2
        self.distance: float = distance_fn(node1, node2)
        self.pheromone: float = 1.0

    def attractiveness(self, alpha: float, beta: float, d_mean: float) -> float:
        """Compute the attractiveness value used in the probability formula.

        attractiveness = pheromone^alpha * (d_mean / distance)^beta
        """
        return (self.pheromone ** alpha) * ((d_mean / self.distance) ** beta)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Graph:
    """Complete graph connecting all nodes.

    Builds all edges between every pair of nodes and provides methods for
    node selection and pheromone management.

    Attributes:
        nodes: Mapping of node index to Node objects.
        edges: Mapping of (i, j) index tuples to Edge objects (i < j).
    """

    def __init__(
        self,
        nodes: list[Node],
        distance_fn: Callable[[Node, Node], float] = euclidean_distance,
        seed: Optional[int] = None,
    ) -> None:
        self.nodes: dict[int, Node] = {node.index: node for node in nodes}
        self.edges: dict[tuple[int, int], Edge] = {}
        self.rng = np.random.default_rng(seed=seed)

        sorted_indices = sorted(self.nodes)
        for i, idx1 in enumerate(sorted_indices):
            for idx2 in sorted_indices[i + 1 :]:
                self.edges[(idx1, idx2)] = Edge(
                    self.nodes[idx1], self.nodes[idx2], distance_fn
                )

    # -- helpers -------------------------------------------------------------

    def get_edge(self, node1: Node, node2: Node) -> Edge:
        """Retrieve the edge connecting *node1* and *node2*."""
        key = (min(node1.index, node2.index), max(node1.index, node2.index))
        return self.edges[key]

    def select_next_node(
        self,
        current: Node,
        candidates: list[Node],
        alpha: float,
        beta: float,
        d_mean: float,
    ) -> Node:
        """Probabilistically select the next node to visit.

        The selection probability for each candidate is proportional to:
            pheromone^alpha * (d_mean / distance)^beta
        """
        if len(candidates) == 1:
            return candidates[0]

        weights = np.array(
            [
                self.get_edge(current, node).attractiveness(alpha, beta, d_mean)
                for node in candidates
            ]
        )
        probabilities = weights / weights.sum()
        return self.rng.choice(candidates, p=probabilities)

    def evaporate_pheromone(self, rho: float) -> None:
        """Apply global pheromone evaporation: τ ← (1 − ρ) · τ."""
        for edge in self.edges.values():
            edge.pheromone *= 1 - rho

    def get_mean_distance(self) -> float:
        """Return the mean edge distance in the graph."""
        distances = [e.distance for e in self.edges.values()]
        return float(np.mean(distances))

    def get_pheromone_matrix(self) -> dict[tuple[int, int], float]:
        """Return a copy of the current pheromone levels."""
        return {k: e.pheromone for k, e in self.edges.items()}


# ---------------------------------------------------------------------------
# Ant
# ---------------------------------------------------------------------------

class Ant:
    """Simulates a single ant traversing the graph.

    Attributes:
        graph: Reference to the underlying graph.
        position: Current node the ant is located at.
        path: Ordered list of visited nodes.
        distance: Total distance traveled so far.
    """

    def __init__(self, graph: Graph, d_mean: float = 1.0) -> None:
        self.graph = graph
        self.d_mean = d_mean
        self.position: Optional[Node] = None
        self.unvisited: list[Node] = []
        self.path: list[Node] = []
        self.edges_visited: list[Edge] = []
        self.distance: float = 0.0

    def initialize(self, start: Node) -> None:
        """Place the ant at *start* and reset its state."""
        self.position = start
        self.unvisited = [n for n in self.graph.nodes.values() if n != start]
        self.path = [start]
        self.edges_visited = []
        self.distance = 0.0

    def tour(self, alpha: float, beta: float) -> None:
        """Complete a full tour visiting every node exactly once."""
        while self.unvisited:
            next_node = self.graph.select_next_node(
                self.position, self.unvisited, alpha, beta, self.d_mean
            )
            self.unvisited.remove(next_node)
            self.path.append(next_node)
            edge = self.graph.get_edge(self.position, next_node)
            self.edges_visited.append(edge)
            self.distance += edge.distance
            self.position = next_node

        # Add return-to-start distance
        return_edge = self.graph.get_edge(self.position, self.path[0])
        self.distance += return_edge.distance

    def deposit_pheromone(self, normalization: float) -> None:
        """Deposit pheromone on visited edges proportional to tour quality."""
        for edge in self.edges_visited:
            edge.pheromone += normalization / self.distance


# ---------------------------------------------------------------------------
# ACO  –  Ant Colony Optimization solver
# ---------------------------------------------------------------------------

class ACO:
    """Ant Colony Optimization solver for the Traveling Salesman Problem.

    Usage::

        nodes = [Node(0, 0), Node(3, 4), Node(6, 1)]
        graph = Graph(nodes)
        aco = ACO(graph)
        best_path, best_distance, history = aco.solve()

    Attributes:
        graph: The TSP graph to solve.
    """

    def __init__(self, graph: Graph, seed: Optional[int] = None) -> None:
        self.graph = graph
        self.rng = np.random.default_rng(seed=seed)

    def solve(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        n_ants: int = 20,
        n_iterations: int = 100,
        verbose: bool = True,
    ) -> tuple[list[Node], float, list[float]]:
        """Run the ACO algorithm and return the best tour found.

        Args:
            alpha: Pheromone importance factor.
            beta: Heuristic (distance) importance factor.
            rho: Pheromone evaporation rate (0 < rho < 1).
            n_ants: Number of ants per iteration.
            n_iterations: Number of iterations to run.
            verbose: If True, print progress every 10 iterations.

        Returns:
            A tuple of (best_path, best_distance, convergence_history)
            where *convergence_history* stores the best distance at each iteration.
        """
        d_mean = self.graph.get_mean_distance()
        normalization = d_mean * len(self.graph.nodes)
        best_distance = float("inf")
        best_path: Optional[list[Node]] = None
        starts = list(self.graph.nodes.values())
        convergence: list[float] = []

        ants = [Ant(self.graph, d_mean) for _ in range(n_ants)]

        for iteration in range(n_iterations):
            # --- construct solutions ---
            for ant in ants:
                start = starts[self.rng.integers(len(starts))]
                ant.initialize(start)
                ant.tour(alpha, beta)

            # --- pheromone update ---
            self.graph.evaporate_pheromone(rho)

            for ant in ants:
                ant.deposit_pheromone(normalization / n_ants)
                if ant.distance < best_distance:
                    best_distance = ant.distance
                    best_path = list(ant.path)

            convergence.append(best_distance)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:>4d}/{n_iterations}  "
                      f"best distance = {best_distance:.2f}")

        if verbose:
            print(f"\n✓ Optimization complete — best distance = {best_distance:.2f}")

        return best_path, best_distance, convergence
