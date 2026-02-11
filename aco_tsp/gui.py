"""
Interactive Tkinter GUI for the ACO-TSP solver.

Allows users to:
- Click on a canvas to place cities (nodes).
- Configure algorithm parameters (α, β, ρ, ants, iterations).
- Run the ACO solver and visualize the best route on the canvas.
- Clear nodes/lines and undo the last placed node.
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import numpy as np

from aco_tsp.core import Node, Edge, Graph, ACO


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_COLORS = [
    "#2ecc71", "#3498db", "#e74c3c", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085",
    "#c0392b", "#2980b9",
]

_NODE_RADIUS = 6


# ---------------------------------------------------------------------------
# Canvas widget
# ---------------------------------------------------------------------------

class TSPCanvas(tk.Canvas):
    """Custom canvas for placing cities and drawing routes."""

    def __init__(self, parent: "TSPApp", width: int = 700, height: int = 500) -> None:
        super().__init__(
            parent, width=width, height=height,
            bg="#fdfdfd", relief=tk.RIDGE, bd=3,
            highlightthickness=0,
        )
        self._width = width
        self._height = height
        self._parent = parent

    # -- drawing helpers ----------------------------------------------------

    def draw_node(
        self,
        x: int,
        y: int,
        radius: int = _NODE_RADIUS,
        outline: str = "#2c3e50",
        fill: str = "#ecf0f1",
    ) -> int:
        """Draw a city dot and return its canvas item id."""
        item = self.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            outline=outline, fill=fill, width=2,
        )
        # Draw city index label
        idx = len(self._parent.node_coords)
        self.create_text(x, y - radius - 10, text=str(idx), font=("Segoe UI", 8, "bold"), fill="#2c3e50")
        return item

    def draw_line(
        self, x1: int, y1: int, x2: int, y2: int, color: str = "#3498db"
    ) -> int:
        """Draw a route segment and return its canvas item id."""
        return self.create_line(
            x1, y1, x2, y2, fill=color, width=2, smooth=True,
        )

    # -- event handler ------------------------------------------------------

    def on_click(self, event: tk.Event) -> None:
        """Handle left-click: place a new city."""
        self._parent.place_node(event.x, event.y)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class TSPApp(tk.Tk):
    """Main GUI application for ACO-TSP visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.title("ACO — Traveling Salesman Problem")
        self.resizable(False, False)
        self.configure(bg="#ecf0f1")

        # State
        self.node_coords: list[tuple[int, int]] = []
        self._canvas_items: list[int] = []
        self._route_items: list[int] = []
        self._label_items: list[int] = []
        self._color_pool = list(_COLORS)

        self._build_ui()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        # Title bar
        title = tk.Label(
            self, text="Ant Colony Optimization  –  TSP Solver",
            font=("Segoe UI", 14, "bold"), bg="#2c3e50", fg="white",
            pady=8,
        )
        title.pack(fill=tk.X)

        # Canvas
        self.canvas = TSPCanvas(self)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.canvas.on_click)

        # --- Parameters frame ------------------------------------------------
        param_frame = tk.LabelFrame(
            self, text="Parameters", font=("Segoe UI", 10, "bold"),
            bg="#ecf0f1", padx=10, pady=5,
        )
        param_frame.pack(fill=tk.X, padx=10)

        params = [
            ("α  (pheromone weight)", "1"),
            ("β  (distance weight)", "2"),
            ("ρ  (evaporation rate)", "0.1"),
            ("Ants", "20"),
            ("Iterations", "100"),
        ]
        self._entries: dict[str, tk.Entry] = {}
        for col, (label, default) in enumerate(params):
            tk.Label(param_frame, text=label, bg="#ecf0f1",
                     font=("Segoe UI", 9)).grid(row=0, column=col, padx=6)
            entry = tk.Entry(param_frame, width=7, justify=tk.CENTER,
                             font=("Segoe UI", 10))
            entry.insert(0, default)
            entry.grid(row=1, column=col, padx=6, pady=(0, 5))
            self._entries[label] = entry

        # --- Buttons frame ----------------------------------------------------
        btn_frame = tk.Frame(self, bg="#ecf0f1")
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        buttons = [
            ("▶  Solve", self._on_solve, "#27ae60", "white"),
            ("↩  Undo", self._on_undo, "#f39c12", "white"),
            ("✕  Clear Routes", self._on_clear_routes, "#e67e22", "white"),
            ("⟲  Reset All", self._on_reset, "#e74c3c", "white"),
        ]
        for text, cmd, bg, fg in buttons:
            tk.Button(
                btn_frame, text=text, command=cmd,
                bg=bg, fg=fg, activebackground=bg,
                font=("Segoe UI", 10, "bold"), padx=12, pady=4,
                relief=tk.FLAT, cursor="hand2",
            ).pack(side=tk.LEFT, padx=4)

        # --- Status bar -------------------------------------------------------
        self._status_var = tk.StringVar(value="Click on the canvas to place cities.")
        status_bar = tk.Label(
            self, textvariable=self._status_var, bg="#2c3e50", fg="#bdc3c7",
            font=("Segoe UI", 9), anchor=tk.W, padx=10, pady=4,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # -- node management ----------------------------------------------------

    def place_node(self, x: int, y: int) -> None:
        """Place a city at (x, y) on the canvas."""
        item = self.canvas.draw_node(x, y)
        self._canvas_items.append(item)
        self.node_coords.append((x, y))
        self._status_var.set(f"{len(self.node_coords)} cities placed.")

    def _on_undo(self) -> None:
        """Remove the last placed city."""
        if not self._canvas_items:
            return
        # Remove dot + label (2 canvas items per node)
        self.canvas.delete(self._canvas_items.pop())
        # Also find and delete the text label if it exists
        all_items = self.canvas.find_all()
        if all_items:
            self.canvas.delete(all_items[-1])  # the label drawn right after
        self.node_coords.pop()
        self._status_var.set(f"{len(self.node_coords)} cities placed.")

    def _on_clear_routes(self) -> None:
        """Erase all drawn route lines without removing cities."""
        for item in self._route_items:
            self.canvas.delete(item)
        self._route_items.clear()
        self._status_var.set("Routes cleared.")

    def _on_reset(self) -> None:
        """Clear everything and start fresh."""
        self.canvas.delete(tk.ALL)
        self._canvas_items.clear()
        self._route_items.clear()
        self.node_coords.clear()
        self._color_pool = list(_COLORS)
        self._status_var.set("Canvas reset — click to place cities.")

    # -- solver -------------------------------------------------------------

    def _on_solve(self) -> None:
        """Read parameters, run ACO and draw the best route."""
        if len(self.node_coords) < 3:
            messagebox.showwarning(
                "Not enough cities",
                "Please place at least 3 cities on the canvas.",
            )
            return

        # Read parameters from entries
        keys = list(self._entries.keys())
        try:
            alpha = float(self._entries[keys[0]].get())
            beta = float(self._entries[keys[1]].get())
            rho = float(self._entries[keys[2]].get())
            n_ants = int(self._entries[keys[3]].get())
            n_iters = int(self._entries[keys[4]].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric parameters.")
            return

        self._status_var.set("Running ACO solver …")
        self.update_idletasks()

        # Build graph
        Node.reset_counter()
        nodes = [Node(x, y) for x, y in self.node_coords]
        graph = Graph(nodes)
        aco = ACO(graph)

        best_path, best_dist, convergence = aco.solve(
            alpha=alpha, beta=beta, rho=rho,
            n_ants=n_ants, n_iterations=n_iters, verbose=True,
        )

        # Draw route
        self._draw_route(best_path)
        self._status_var.set(
            f"✓ Best distance = {best_dist:.2f}  |  {len(self.node_coords)} cities  |  "
            f"{n_iters} iterations, {n_ants} ants"
        )

    def _draw_route(self, path: list[Node]) -> None:
        """Draw the solution route on the canvas."""
        color = self._color_pool.pop(0) if self._color_pool else "#3498db"
        coords = [(n.x, n.y) for n in path]

        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            item = self.canvas.draw_line(x1, y1, x2, y2, color)
            self._route_items.append(item)

    # -- entry point --------------------------------------------------------

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self.mainloop()
