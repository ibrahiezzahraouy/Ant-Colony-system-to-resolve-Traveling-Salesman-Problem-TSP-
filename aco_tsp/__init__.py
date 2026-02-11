"""
ACO-TSP: Ant Colony Optimization for the Traveling Salesman Problem.

A Python implementation of the Ant Colony System (ACS) metaheuristic
to solve the Traveling Salesman Problem, featuring an interactive
Tkinter GUI for city placement and real-time route visualization.
"""

from aco_tsp.core import Node, Edge, Graph, Ant, ACO

__version__ = "1.0.0"
__all__ = ["Node", "Edge", "Graph", "Ant", "ACO"]
