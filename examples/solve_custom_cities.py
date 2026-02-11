#!/usr/bin/env python3
"""
Example: Solve a small TSP instance programmatically and display results.

This script demonstrates how to use the aco_tsp package as a library,
without the GUI.
"""

from aco_tsp.core import Node, Graph, ACO
from aco_tsp.visualization import plot_dashboard


def main() -> None:
    # Reset node counter for a clean run
    Node.reset_counter()

    # Define city coordinates (you can replace these with your own)
    city_coords = [
        (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
        (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
        (180, 100), (60, 80),  (120, 80),  (180, 60),  (100, 40),
    ]

    # Create nodes and graph
    nodes = [Node(x, y, name=f"C{i}") for i, (x, y) in enumerate(city_coords)]
    graph = Graph(nodes, seed=123)

    # Configure and run the solver
    aco = ACO(graph, seed=123)
    best_path, best_distance, convergence = aco.solve(
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        n_ants=25,
        n_iterations=200,
        verbose=True,
    )

    # Print the tour
    tour_names = [node.name for node in best_path]
    print(f"\nBest tour: {' → '.join(tour_names)} → {tour_names[0]}")
    print(f"Total distance: {best_distance:.2f}")

    # Visualize
    params = {"α": 1.0, "β": 3.0, "ρ": 0.1, "ants": 25, "iterations": 200}
    plot_dashboard(best_path, convergence, best_distance, params)


if __name__ == "__main__":
    main()
