"""
Matplotlib-based visualization utilities for ACO-TSP results.

Provides static plots for:
- City layout and best tour
- Convergence curve
- Combined dashboard
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from aco_tsp.core import Node


def plot_tour(
    path: list[Node],
    title: str = "Best Tour Found by ACO",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the TSP tour on a 2-D scatter plot.

    Args:
        path: Ordered list of nodes forming the tour.
        title: Plot title.
        ax: Matplotlib Axes to draw on (creates new figure if None).
        show: Whether to call ``plt.show()``.

    Returns:
        The Axes object used for drawing.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    xs = [n.x for n in path] + [path[0].x]
    ys = [n.y for n in path] + [path[0].y]

    ax.plot(xs, ys, "-o", color="#3498db", linewidth=1.5, markersize=7,
            markerfacecolor="#e74c3c", markeredgecolor="#2c3e50", zorder=3)

    for node in path:
        ax.annotate(
            node.name, (node.x, node.y),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color="#2c3e50",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_convergence(
    history: list[float],
    title: str = "Convergence Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the best-distance convergence over iterations.

    Args:
        history: List of best distances per iteration.
        title: Plot title.
        ax: Matplotlib Axes (creates new figure if None).
        show: Whether to call ``plt.show()``.

    Returns:
        The Axes object used for drawing.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history, color="#e74c3c", linewidth=1.5)
    ax.fill_between(range(len(history)), history, alpha=0.15, color="#e74c3c")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Distance")
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_dashboard(
    path: list[Node],
    convergence: list[float],
    best_distance: float,
    params: Optional[dict] = None,
) -> None:
    """Display a combined dashboard with tour + convergence.

    Args:
        path: Best tour as a list of Nodes.
        convergence: Convergence history.
        best_distance: Final best distance.
        params: Dict of algorithm parameters for annotation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_tour(path, title=f"Best Tour  (distance = {best_distance:.2f})", ax=ax1, show=False)
    plot_convergence(convergence, ax=ax2, show=False)

    if params:
        info = "  |  ".join(f"{k}={v}" for k, v in params.items())
        fig.suptitle(info, fontsize=9, color="gray", y=0.02)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
