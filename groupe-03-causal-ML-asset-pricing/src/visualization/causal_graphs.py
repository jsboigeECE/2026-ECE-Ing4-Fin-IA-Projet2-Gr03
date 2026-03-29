"""
Causal Graph Visualization
===========================

Professional visualization of the causal DAG using networkx + matplotlib.

Node color coding:
    - Green: Treatment variable
    - Red/Coral: Outcome variable
    - Steel Blue: Confounders
    - Gold: Instrument
    - Light Gray: Effect modifiers
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def _get_node_colors(G: nx.DiGraph) -> Dict[str, str]:
    """Assign colors based on variable role."""
    colors = {}
    for node in G.nodes():
        if node == "earnings_surprise":
            colors[node] = "#2ecc71"   # green
        elif node == "stock_return":
            colors[node] = "#e74c3c"   # red
        elif node == "analyst_revision":
            colors[node] = "#f39c12"   # gold
        elif node in ["sector", "market_cap_quintile"]:
            colors[node] = "#bdc3c7"   # light gray
        else:
            colors[node] = "#3498db"   # steel blue
    return colors


def _get_node_labels(G: nx.DiGraph) -> Dict[str, str]:
    """Human-readable labels for nodes."""
    label_map = {
        "earnings_surprise": "Earnings\nSurprise (T)",
        "stock_return": "Stock\nReturn (Y)",
        "log_market_cap": "Market\nCap",
        "book_to_market": "Book-to-\nMarket",
        "momentum": "Momentum",
        "volatility": "Volatility",
        "analyst_coverage": "Analyst\nCoverage",
        "institutional_ownership": "Institutional\nOwnership",
        "analyst_revision": "Analyst\nRevision (Z)",
    }
    return {node: label_map.get(node, node) for node in G.nodes()}


def plot_causal_dag(
    G: nx.DiGraph,
    title: str = "Causal DAG: Earnings Surprise → Stock Return",
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the causal DAG with professional styling.

    Parameters
    ----------
    G : nx.DiGraph
        The causal graph.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Manual layout for aesthetics
    pos = {
        "log_market_cap":          (-2.0,  1.5),
        "book_to_market":          (-1.0,  2.5),
        "momentum":                (0.0,   2.5),
        "volatility":              (1.0,   2.5),
        "analyst_coverage":        (2.0,   1.5),
        "institutional_ownership": (3.0,   2.0),
        "analyst_revision":        (-2.5,  0.0),
        "earnings_surprise":       (0.0,   0.0),
        "stock_return":            (3.0,   0.0),
    }

    # Filter positions to only include nodes in the graph
    pos = {k: v for k, v in pos.items() if k in G.nodes()}

    colors = _get_node_colors(G)
    labels = _get_node_labels(G)
    node_colors = [colors.get(n, "#3498db") for n in G.nodes()]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#7f8c8d",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        width=2.0,
        alpha=0.7,
        min_source_margin=25,
        min_target_margin=25,
    )

    # Highlight the causal edge T→Y
    causal_edges = [("earnings_surprise", "stock_return")]
    causal_edges = [(u, v) for u, v in causal_edges if G.has_edge(u, v)]
    if causal_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=causal_edges, ax=ax,
            edge_color="#e74c3c",
            arrows=True,
            arrowsize=25,
            arrowstyle="-|>",
            width=3.5,
            alpha=0.9,
            min_source_margin=25,
            min_target_margin=25,
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2500,
        alpha=0.9,
        edgecolors="#ecf0f1",
        linewidths=2.0,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="white",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="white", label="Treatment (T)"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="white", label="Outcome (Y)"),
        mpatches.Patch(facecolor="#3498db", edgecolor="white", label="Confounders (W)"),
        mpatches.Patch(facecolor="#f39c12", edgecolor="white", label="Instrument (Z)"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="lower left",
        fontsize=11, facecolor="#16213e", edgecolor="#7f8c8d",
        labelcolor="white", framealpha=0.9,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=20)
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[VIZ] Causal DAG saved to {save_path}")

    return fig


def plot_identified_estimand(
    G: nx.DiGraph,
    adjustment_set: list[str],
    title: str = "Identified Backdoor Adjustment Set",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Highlight the adjustment set used for identification on the DAG.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    pos = {
        "log_market_cap":          (-2.0,  1.5),
        "book_to_market":          (-1.0,  2.5),
        "momentum":                (0.0,   2.5),
        "volatility":              (1.0,   2.5),
        "analyst_coverage":        (2.0,   1.5),
        "institutional_ownership": (3.0,   2.0),
        "analyst_revision":        (-2.5,  0.0),
        "earnings_surprise":       (0.0,   0.0),
        "stock_return":            (3.0,   0.0),
    }
    pos = {k: v for k, v in pos.items() if k in G.nodes()}

    # Color: adjusted nodes get highlighted
    node_colors = []
    for n in G.nodes():
        if n in adjustment_set:
            node_colors.append("#9b59b6")  # purple for adjusted
        elif n == "earnings_surprise":
            node_colors.append("#2ecc71")
        elif n == "stock_return":
            node_colors.append("#e74c3c")
        else:
            node_colors.append("#34495e")  # dim

    labels = _get_node_labels(G)

    nx.draw(
        G, pos, ax=ax,
        labels=labels,
        node_color=node_colors,
        node_size=2200,
        font_size=8,
        font_weight="bold",
        font_color="white",
        edge_color="#7f8c8d",
        arrows=True,
        arrowsize=18,
        width=1.5,
        alpha=0.8,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
