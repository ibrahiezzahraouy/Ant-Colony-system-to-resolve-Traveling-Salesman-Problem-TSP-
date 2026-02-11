#!/usr/bin/env python3
"""
Entry point for the ACO-TSP application.

Usage:
    python main.py          # Launch the interactive GUI
    python main.py --demo   # Run a headless demo with random cities
"""

import argparse
import sys

from aco_tsp.core import Node, Graph, ACO


def run_gui() -> None:
    """Launch the interactive Tkinter GUI."""
    from aco_tsp.gui import TSPApp
    app = TSPApp()
    app.run()


def run_demo(n_cities: int = 15, seed: int = 42) -> None:
    """Run a quick headless demo and plot the results."""
    import numpy as np
    from aco_tsp.visualization import plot_dashboard

    print(f"Generating {n_cities} random cities (seed={seed}) …\n")
    rng = np.random.default_rng(seed)

    Node.reset_counter()
    nodes = [
        Node(
            x=float(rng.integers(10, 500)),
            y=float(rng.integers(10, 400)),
        )
        for _ in range(n_cities)
    ]

    graph = Graph(nodes, seed=seed)
    aco = ACO(graph, seed=seed)

    best_path, best_dist, convergence = aco.solve(
        alpha=1, beta=3, rho=0.1,
        n_ants=30, n_iterations=150,
        verbose=True,
    )

    params = {"α": 1, "β": 3, "ρ": 0.1, "ants": 30, "iterations": 150, "cities": n_cities}
    plot_dashboard(best_path, convergence, best_dist, params)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACO-TSP — Ant Colony Optimization for the Traveling Salesman Problem",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a headless demo with random cities instead of the GUI.",
    )
    parser.add_argument(
        "--cities", type=int, default=15,
        help="Number of random cities for --demo mode (default: 15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo(n_cities=args.cities, seed=args.seed)
    else:
        run_gui()


if __name__ == "__main__":
    main()
