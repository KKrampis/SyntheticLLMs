"""Visualize feature hierarchy JSON files as interactive tree plots.

Usage:
    poetry run python feature_hierarchies/visualize_hierarchies.py
    poetry run python feature_hierarchies/visualize_hierarchies.py path/to/file.json

Outputs HTML files to feature_hierarchies/visualizations/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

HERE = Path(__file__).parent
OUT_DIR = HERE / "visualizations"


def _collect_nodes(
    node: dict[str, Any],
    parent_id: str,
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str]],
    counter: list[int],
    depth: int = 0,
) -> str:
    node_id = f"n{counter[0]}"
    counter[0] += 1
    label = node.get("label", node_id)
    alpha = node.get("alpha", 0.0)
    me = node.get("mutually_exclusive_children", False)
    nodes.append(
        {
            "id": node_id,
            "label": label,
            "alpha": alpha,
            "depth": depth,
            "mutually_exclusive_children": me,
        }
    )
    if parent_id:
        edges.append((parent_id, node_id))
    for child in node.get("children", []):
        _collect_nodes(child, node_id, nodes, edges, counter, depth + 1)
    return node_id


def _layout_tree(
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str]],
) -> tuple[list[float], list[float]]:
    """Assign (x, y) positions using a simple BFS level layout."""
    children_map: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for parent, child in edges:
        children_map[parent].append(child)

    id_to_node = {n["id"]: n for n in nodes}
    parent_map = {child: parent for parent, child in edges}

    roots = [n["id"] for n in nodes if n["id"] not in parent_map]

    # BFS to assign x positions (left to right among siblings) and y (depth)
    x_pos: dict[str, float] = {}
    y_pos: dict[str, float] = {}
    leaf_counter: list[float] = [0.0]

    def assign_x(node_id: str) -> float:
        kids = children_map[node_id]
        if not kids:
            pos = leaf_counter[0]
            leaf_counter[0] += 1.0
            x_pos[node_id] = pos
            return pos
        child_xs = [assign_x(k) for k in kids]
        pos = (child_xs[0] + child_xs[-1]) / 2.0
        x_pos[node_id] = pos
        return pos

    for root in roots:
        assign_x(root)
        leaf_counter[0] += 1.0  # gap between top-level trees

    for n in nodes:
        y_pos[n["id"]] = -n["depth"]

    xs = [x_pos[n["id"]] for n in nodes]
    ys = [y_pos[n["id"]] for n in nodes]
    return xs, ys


def visualize_tree(
    root: dict[str, Any],
    title: str,
) -> go.Figure:
    nodes: list[dict[str, Any]] = []
    edges: list[tuple[str, str]] = []
    _collect_nodes(root, "", nodes, edges, [0])

    xs, ys = _layout_tree(nodes, edges)
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

    # Edge traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for parent, child in edges:
        pi, ci = id_to_idx[parent], id_to_idx[child]
        edge_x += [xs[pi], xs[ci], None]
        edge_y += [ys[pi], ys[ci], None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"color": "#888", "width": 1.5},
        hoverinfo="none",
    )

    # Node colours: shade by alpha (0 = root = dark blue, 1 = very similar to parent = green)
    alphas = [n["alpha"] for n in nodes]
    node_colors = [
        f"rgb({int(30 + 180*(1-a))}, {int(100 + 120*a)}, {int(200 - 150*a)})"
        for a in alphas
    ]

    node_text = [
        (
            f"<b>{n['label']}</b><br>"
            f"α={n['alpha']:.2f}<br>"
            f"depth={n['depth']}<br>"
            + ("⊕ mutually exclusive children" if n["mutually_exclusive_children"] else "")
        )
        for n in nodes
    ]

    node_labels = [
        (n["label"] if len(n["label"]) <= 22 else n["label"][:20] + "…")
        for n in nodes
    ]

    node_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        marker={
            "size": 18,
            "color": node_colors,
            "line": {"width": 2, "color": "white"},
        },
        text=node_labels,
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={"text": title, "font": {"size": 16}},
            showlegend=False,
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            plot_bgcolor="white",
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            height=600,
        ),
    )
    return fig


def visualize_json(json_path: Path, out_dir: Path) -> None:
    data = json.loads(json_path.read_text())
    trees = data.get("trees", [])
    source = data.get("source", json_path.stem)
    description = data.get("description", "")

    figs = []
    for tree in trees:
        label = tree.get("label", "Tree")
        fig = visualize_tree(tree, f"{label} — {source}")
        figs.append((label, fig))

    if not figs:
        print(f"No trees found in {json_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # One combined HTML with all trees as tabs / stacked sections
    combined_html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>" + json_path.stem + " hierarchies</title>",
        "<style>body{font-family:sans-serif;margin:20px} h2{color:#333} .desc{color:#666;font-size:0.9em;max-width:900px}</style>",
        "</head><body>",
        f"<h1>{json_path.stem}</h1>",
        f"<p class='desc'>{description}</p>",
    ]

    for label, fig in figs:
        combined_html_parts.append(f"<h2>{label}</h2>")
        combined_html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    combined_html_parts.append("</body></html>")

    out_path = out_dir / (json_path.stem + ".html")
    out_path.write_text("\n".join(combined_html_parts))
    print(f"Wrote {out_path}  ({len(figs)} tree(s))")


def _mermaid_node_id(label: str, node_id: str) -> str:
    """Return a safe mermaid node ID."""
    safe = "".join(c if c.isalnum() else "_" for c in node_id)
    return safe


def _mermaid_label(node: dict[str, Any]) -> str:
    label = node.get("label", "?")
    alpha = node.get("alpha", 0.0)
    me = node.get("mutually_exclusive_children", False)
    me_mark = " ⊕" if me else ""
    return f"{label}\\nα={alpha:.2f}{me_mark}"


def _build_mermaid(
    node: dict[str, Any],
    parent_mid: str,
    lines: list[str],
    counter: list[int],
) -> str:
    mid = f"n{counter[0]}"
    counter[0] += 1
    label = _mermaid_label(node)
    lines.append(f'    {mid}["{label}"]')
    if parent_mid:
        lines.append(f"    {parent_mid} --> {mid}")
    for child in node.get("children", []):
        _build_mermaid(child, mid, lines, counter)
    return mid


def json_to_markdown(json_path: Path, out_dir: Path) -> None:
    data = json.loads(json_path.read_text())
    trees = data.get("trees", [])
    source = data.get("source", json_path.stem)
    description = data.get("description", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (json_path.stem + ".md")

    parts = [
        f"# {json_path.stem}",
        f"",
        f"**Source:** {source}",
        f"",
        f"{description}",
        f"",
        f"> Nodes show `α` (semantic similarity to parent). ⊕ = mutually exclusive children.",
        f"",
    ]

    for tree in trees:
        label = tree.get("label", "Tree")
        parts.append(f"## {label}")
        parts.append("")
        parts.append("```mermaid")
        parts.append("graph TD")
        lines: list[str] = []
        _build_mermaid(tree, "", lines, [0])
        parts.extend(lines)
        parts.append("```")
        parts.append("")

    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path}  ({len(trees)} tree(s))")


def main() -> None:
    if len(sys.argv) > 1:
        json_files = [Path(p) for p in sys.argv[1:]]
    else:
        json_files = sorted(HERE.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {HERE}")
        return

    for jf in json_files:
        visualize_json(jf, OUT_DIR)
        json_to_markdown(jf, OUT_DIR)


if __name__ == "__main__":
    main()
