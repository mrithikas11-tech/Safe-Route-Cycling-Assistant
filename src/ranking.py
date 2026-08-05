"""
src/ranking.py

Route Ranking, Reasoning & Map
Judging the routes, explaining the pick, and drawing the map.
"""

from typing import Dict, List, Any, Optional, Tuple
import networkx as nx


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value or list of values to float."""
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def get_edge_data(G: nx.Graph, u: Any, v: Any) -> Dict[str, Any]:
    """
    Safely extract edge attribute dictionary between nodes u and v.
    Handles MultiDiGraph (selecting the edge with lowest collision_prob)
    and standard DiGraph.
    """
    if not G.has_edge(u, v):
        return {}

    if G.is_multigraph():
        edge_dict = G[u][v]
        if not edge_dict:
            return {}
        # Select edge with lowest collision_prob (or first available)
        best_key = min(
            edge_dict.keys(),
            key=lambda k: _safe_float(edge_dict[k].get("collision_prob", 0.0)),
        )
        return dict(edge_dict[best_key])
    else:
        return dict(G[u][v])


def evaluate_route(
    G: nx.Graph,
    route_nodes: Optional[List[Any]],
    high_risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute metric summary for a single route given as a list of node IDs.

    Returns:
    {
        "total_length": float,          # meters
        "total_length_km": float,       # kilometers
        "weighted_risk_sum": float,     # length * prob sum
        "avg_risk": float,              # weighted avg collision prob per meter
        "high_risk_segments": int,      # count of high-risk edges
        "total_edges": int,             # total edge count in route
        "data_count": int,              # edges from confirmed data
        "model_count": int,             # edges predicted by model
        "data_fraction": float,         # data_count / total_edges (0.0 to 1.0)
        "model_fraction": float,        # model_count / total_edges (0.0 to 1.0)
        "edge_list": List[Dict]         # list of edge info dicts
    }
    """
    if not route_nodes or len(route_nodes) < 2:
        return {
            "total_length": 0.0,
            "total_length_km": 0.0,
            "weighted_risk_sum": 0.0,
            "avg_risk": 0.0,
            "high_risk_segments": 0,
            "total_edges": 0,
            "data_count": 0,
            "model_count": 0,
            "data_fraction": 0.0,
            "model_fraction": 0.0,
            "edge_list": [],
        }

    total_length = 0.0
    weighted_risk_sum = 0.0
    high_risk_segments = 0
    data_count = 0
    model_count = 0
    total_edges = 0
    edge_list = []

    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]

        edge_data = get_edge_data(G, u, v)

        length = _safe_float(edge_data.get("length", 1.0), default=1.0)
        collision_prob = _safe_float(
            edge_data.get("collision_prob", 0.0), default=0.0
        )
        collision_prob = max(0.0, min(1.0, collision_prob))

        label = int(_safe_float(edge_data.get("label", 0)))
        source = str(edge_data.get("source", "model")).lower()

        total_length += length
        weighted_risk_sum += length * collision_prob
        total_edges += 1

        is_high_risk = (collision_prob >= high_risk_threshold) or (label == 1)
        if is_high_risk:
            high_risk_segments += 1

        if source == "data":
            data_count += 1
        else:
            model_count += 1

        edge_list.append(
            {
                "u": u,
                "v": v,
                "length": length,
                "collision_prob": collision_prob,
                "label": label,
                "source": source,
                "is_high_risk": is_high_risk,
                "geometry": edge_data.get("geometry"),
            }
        )

    avg_risk = (
        weighted_risk_sum / total_length if total_length > 0.0 else 0.0
    )
    data_fraction = data_count / total_edges if total_edges > 0 else 0.0
    model_fraction = model_count / total_edges if total_edges > 0 else 0.0

    return {
        "total_length": round(total_length, 2),
        "total_length_km": round(total_length / 1000.0, 2),
        "weighted_risk_sum": round(weighted_risk_sum, 4),
        "avg_risk": round(avg_risk, 4),
        "high_risk_segments": high_risk_segments,
        "total_edges": total_edges,
        "data_count": data_count,
        "model_count": model_count,
        "data_fraction": round(data_fraction, 4),
        "model_fraction": round(model_fraction, 4),
        "edge_list": edge_list,
    }


if __name__ == "__main__":
    # Test metric helper functions with a synthetic NetworkX graph
    print("Testing metric functions...")

    test_G = nx.MultiDiGraph()
    test_G.add_node(1, y=37.35, x=-121.95)
    test_G.add_node(2, y=37.36, x=-121.94)
    test_G.add_node(3, y=37.37, x=-121.93)

    test_G.add_edge(1, 2, length=1000.0, collision_prob=0.1, source="data", label=0)
    test_G.add_edge(2, 3, length=2000.0, collision_prob=0.7, source="model", label=1)

    route = [1, 2, 3]
    eval_result = evaluate_route(test_G, route)

    print("Evaluation Result:")
    for k, v in eval_result.items():
        if k != "edge_list":
            print(f"  {k}: {v}")

    assert eval_result["total_length"] == 3000.0
    assert eval_result["high_risk_segments"] == 1
    assert eval_result["data_count"] == 1
    assert eval_result["model_count"] == 1
    assert round(eval_result["avg_risk"], 4) == round((1000 * 0.1 + 2000 * 0.7) / 3000, 4)

    print("Tests passed successfully!")
