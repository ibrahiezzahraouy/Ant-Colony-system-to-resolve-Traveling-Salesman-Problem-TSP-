"""
Unit tests for the ACO-TSP core module.
"""

import pytest
from aco_tsp.core import Node, Edge, Graph, Ant, ACO, euclidean_distance


@pytest.fixture(autouse=True)
def reset_node_counter():
    """Reset the Node counter before each test."""
    Node.reset_counter()
    yield
    Node.reset_counter()


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestNode:
    def test_creation(self):
        n = Node(3.0, 4.0, name="A")
        assert n.x == 3.0
        assert n.y == 4.0
        assert n.name == "A"

    def test_auto_index(self):
        n1 = Node(0, 0)
        n2 = Node(1, 1)
        assert n2.index == n1.index + 1

    def test_equality(self):
        n1 = Node(0, 0)
        idx = n1.index
        n2 = Node(0, 0)
        assert n1 != n2  # different indices

    def test_reset_counter(self):
        Node(0, 0)
        Node.reset_counter()
        n = Node(5, 5)
        assert n.index == 0


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

class TestEdge:
    def test_distance(self):
        n1 = Node(0, 0)
        n2 = Node(3, 4)
        e = Edge(n1, n2)
        assert abs(e.distance - 5.0) < 1e-9

    def test_initial_pheromone(self):
        e = Edge(Node(0, 0), Node(1, 1))
        assert e.pheromone == 1.0

    def test_attractiveness(self):
        e = Edge(Node(0, 0), Node(3, 4))
        val = e.attractiveness(alpha=1, beta=1, d_mean=5.0)
        assert val > 0


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_edge_count(self):
        nodes = [Node(i, i) for i in range(5)]
        g = Graph(nodes)
        assert len(g.edges) == 10  # C(5,2) = 10

    def test_get_edge(self):
        nodes = [Node(0, 0), Node(3, 4)]
        g = Graph(nodes)
        e = g.get_edge(nodes[0], nodes[1])
        assert abs(e.distance - 5.0) < 1e-9

    def test_evaporation(self):
        nodes = [Node(0, 0), Node(1, 1)]
        g = Graph(nodes)
        initial = list(g.edges.values())[0].pheromone
        g.evaporate_pheromone(0.5)
        assert list(g.edges.values())[0].pheromone == pytest.approx(initial * 0.5)


# ---------------------------------------------------------------------------
# ACO solver tests
# ---------------------------------------------------------------------------

class TestACO:
    def test_solve_returns_valid_tour(self):
        nodes = [Node(0, 0), Node(3, 4), Node(6, 0)]
        g = Graph(nodes, seed=42)
        aco = ACO(g, seed=42)
        path, dist, conv = aco.solve(n_ants=5, n_iterations=10, verbose=False)

        assert len(path) == 3
        assert dist > 0
        assert len(conv) == 10

    def test_convergence_decreasing(self):
        """Over many iterations the best distance should not increase."""
        nodes = [Node(0, 0), Node(3, 4), Node(6, 0), Node(3, -4)]
        g = Graph(nodes, seed=0)
        aco = ACO(g, seed=0)
        _, _, conv = aco.solve(n_ants=10, n_iterations=50, verbose=False)

        # convergence should be non-increasing
        for i in range(1, len(conv)):
            assert conv[i] <= conv[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------

class TestEuclidean:
    def test_known_distance(self):
        assert euclidean_distance(Node(0, 0), Node(3, 4)) == pytest.approx(5.0)

    def test_zero_distance(self):
        n = Node(7, 7)
        # Need a second node at same coords (different index)
        m = Node(7, 7)
        assert euclidean_distance(n, m) == pytest.approx(0.0)
