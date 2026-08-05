"""
src/ranking.py

Route Ranking, Reasoning & Map
Judging the routes, explaining the pick, and drawing the map.
"""

from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import folium


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


def rank_and_explain(
    routes_dict: Dict[str, Any],
    G: nx.Graph,
    high_risk_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates candidate routes, selects the recommended route (safest),
    compares it with the shortest and alternative routes, and generates
    human-readable reasoning.

    Output format:
    {
        "recommended_route": List[node_ids] or None,
        "recommended_tag": "safest" | "shortest" | "none",
        "reasoning": str,
        "metrics": {
            "safest": eval_dict,
            "shortest": eval_dict,
            "alternatives": [eval_dict1, ...]
        },
        "ranked_routes": [
            {
                "name": str,
                "tag": str,
                "route": List[node_ids],
                "metrics": eval_dict
            },
            ...
        ],
        "error": str or None
    }
    """
    error = routes_dict.get("error") if isinstance(routes_dict, dict) else "Invalid routes object"
    safest_route = routes_dict.get("safest") if isinstance(routes_dict, dict) else None
    shortest_route = routes_dict.get("shortest") if isinstance(routes_dict, dict) else None
    alternatives = routes_dict.get("alternatives", []) if isinstance(routes_dict, dict) else []

    if error or not safest_route:
        err_msg = error or "No valid route available to rank."
        return {
            "recommended_route": None,
            "recommended_tag": "none",
            "reasoning": f"Cannot rank routes: {err_msg}",
            "metrics": {
                "safest": None,
                "shortest": None,
                "alternatives": [],
            },
            "ranked_routes": [],
            "error": err_msg,
        }

    safest_eval = evaluate_route(G, safest_route, high_risk_threshold)
    shortest_eval = (
        evaluate_route(G, shortest_route, high_risk_threshold)
        if shortest_route
        else safest_eval
    )

    alt_evals = [
        evaluate_route(G, alt_r, high_risk_threshold) for alt_r in alternatives if alt_r
    ]

    # Reasoning logic comparing safest vs shortest
    if shortest_route is None or safest_route == shortest_route:
        data_pct = int(safest_eval["data_fraction"] * 100)
        reasoning = (
            f"The recommended route ({safest_eval['total_length_km']} km) is already "
            f"the shortest path available and has a low collision risk profile. "
            f"{data_pct}% based on confirmed crash data."
        )
    else:
        len_diff = safest_eval["total_length"] - shortest_eval["total_length"]
        len_pct = (
            (len_diff / shortest_eval["total_length"]) * 100
            if shortest_eval["total_length"] > 0
            else 0.0
        )

        risk_diff = shortest_eval["avg_risk"] - safest_eval["avg_risk"]
        risk_pct = (
            (risk_diff / shortest_eval["avg_risk"]) * 100
            if shortest_eval["avg_risk"] > 0
            else 0.0
        )

        avoided_segments = (
            shortest_eval["high_risk_segments"] - safest_eval["high_risk_segments"]
        )
        data_pct = int(safest_eval["data_fraction"] * 100)

        parts = []
        if len_pct > 0:
            parts.append(f"{round(len_pct, 1)}% longer (+{int(len_diff)}m)")

        if avoided_segments > 0:
            parts.append(
                f"avoids {avoided_segments} high-risk segment{'s' if avoided_segments > 1 else ''}"
            )

        if risk_pct > 0:
            parts.append(f"reduces average collision risk by {round(risk_pct, 1)}%")

        if parts:
            reason_details = ", ".join(parts)
            reasoning = (
                f"Recommended route is {reason_details}. "
                f"{data_pct}% based on confirmed crash data."
            )
        else:
            reasoning = (
                f"Recommended route prioritizes lower collision probability segments "
                f"across {safest_eval['total_length_km']} km ({data_pct}% based on confirmed crash data)."
            )

    ranked_routes = [
        {
            "name": "Recommended (Safest Route)",
            "tag": "safest",
            "route": safest_route,
            "metrics": safest_eval,
        }
    ]

    if shortest_route and shortest_route != safest_route:
        ranked_routes.append(
            {
                "name": "Shortest Direct Route",
                "tag": "shortest",
                "route": shortest_route,
                "metrics": shortest_eval,
            }
        )

    for idx, (alt_r, alt_ev) in enumerate(zip(alternatives, alt_evals), start=1):
        ranked_routes.append(
            {
                "name": f"Alternative Option {idx}",
                "tag": f"alternative_{idx}",
                "route": alt_r,
                "metrics": alt_ev,
            }
        )

    return {
        "recommended_route": safest_route,
        "recommended_tag": "safest",
        "reasoning": reasoning,
        "metrics": {
            "safest": safest_eval,
            "shortest": shortest_eval,
            "alternatives": alt_evals,
        },
        "ranked_routes": ranked_routes,
        "error": None,
    }


def _node_latlon(G: nx.Graph, node_id: Any) -> Tuple[float, float]:
    """Return (lat, lon) for a graph node."""
    node_data = G.nodes[node_id]
    return float(node_data["y"]), float(node_data["x"])


def _extract_edge_coords(G: nx.Graph, u: Any, v: Any) -> List[Tuple[float, float]]:
    """
    Extract (lat, lon) coordinates for the edge connecting u and v.
    Flips OSMnx geometry (lon, lat) to Folium (lat, lon).
    """
    edge_data = get_edge_data(G, u, v)
    geom = edge_data.get("geometry")

    if geom is not None and hasattr(geom, "coords"):
        # Shapely geometry stores (x=lon, y=lat); flip to (lat, lon)
        return [(float(lat), float(lon)) for lon, lat in geom.coords]

    # Fallback to straight line between nodes
    return [_node_latlon(G, u), _node_latlon(G, v)]


def build_map(
    G: nx.Graph,
    routes_dict: Dict[str, Any],
    ranking_result: Optional[Dict[str, Any]] = None,
    zoom_start: int = 13,
) -> folium.Map:
    """
    Build an interactive Folium Map visualizing the routes.
    Optimized to only render edges contained in candidate routes.
    """
    if not isinstance(routes_dict, dict):
        routes_dict = {}

    safest_route = routes_dict.get("safest")
    shortest_route = routes_dict.get("shortest")
    alternatives = routes_dict.get("alternatives", []) or []

    all_routes = []
    if safest_route:
        all_routes.append(("safest", safest_route))
    if shortest_route and shortest_route != safest_route:
        all_routes.append(("shortest", shortest_route))
    for idx, alt_r in enumerate(alternatives, start=1):
        if alt_r:
            all_routes.append((f"alternative_{idx}", alt_r))

    # Determine center location
    all_lats = []
    all_lons = []
    for _, route in all_routes:
        for node in route:
            if node in G.nodes:
                lat, lon = _node_latlon(G, node)
                all_lats.append(lat)
                all_lons.append(lon)

    if all_lats and all_lons:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
    else:
        # Default fallback to Santa Clara County center
        center_lat, center_lon = 37.3541, -121.9552

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="cartodbpositron",
    )

    if not all_routes:
        return m

    # 1. Draw Alternatives first (bottom layer)
    for tag, route in all_routes:
        if tag.startswith("alternative_"):
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                coords = _extract_edge_coords(G, u, v)
                edge_info = get_edge_data(G, u, v)
                prob = _safe_float(edge_info.get("collision_prob", 0.0))
                folium.PolyLine(
                    locations=coords,
                    color="#757575",
                    weight=3,
                    opacity=0.6,
                    dash_array="4, 8",
                    tooltip=f"Alternative Route | Risk: {round(prob * 100, 1)}%",
                ).add_to(m)

    # 2. Draw Shortest Route (middle layer, if distinct)
    for tag, route in all_routes:
        if tag == "shortest":
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                coords = _extract_edge_coords(G, u, v)
                edge_info = get_edge_data(G, u, v)
                prob = _safe_float(edge_info.get("collision_prob", 0.0))
                folium.PolyLine(
                    locations=coords,
                    color="#E53935",
                    weight=4,
                    opacity=0.7,
                    dash_array="6, 6",
                    tooltip=f"Shortest Route (Direct) | Risk: {round(prob * 100, 1)}%",
                ).add_to(m)

    # 3. Draw Recommended Safest Route (top layer)
    for tag, route in all_routes:
        if tag == "safest":
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                coords = _extract_edge_coords(G, u, v)
                edge_info = get_edge_data(G, u, v)
                prob = _safe_float(edge_info.get("collision_prob", 0.0))
                src = str(edge_info.get("source", "model"))
                length = _safe_float(edge_info.get("length", 0.0))

                # Color code segment by risk: green if low risk (<0.5), red if high risk (>=0.5)
                seg_color = "#2E7D32" if prob < 0.5 else "#D32F2F"
                folium.PolyLine(
                    locations=coords,
                    color=seg_color,
                    weight=6,
                    opacity=0.9,
                    tooltip=f"Recommended Safest Route | Seg Risk: {round(prob * 100, 1)}% ({src}) | {int(length)}m",
                ).add_to(m)

    # Add Start and Destination Markers using safest_route or first available route
    primary_route = safest_route or (all_routes[0][1] if all_routes else None)
    if primary_route and len(primary_route) >= 2:
        start_latlon = _node_latlon(G, primary_route[0])
        end_latlon = _node_latlon(G, primary_route[-1])

        folium.Marker(
            location=start_latlon,
            popup="Start Location",
            tooltip="Start",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

        folium.Marker(
            location=end_latlon,
            popup="Destination",
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    return m


if __name__ == "__main__":
    # Test metric helper functions with a synthetic NetworkX graph
    print("Testing metric functions...")

    test_G = nx.MultiDiGraph()
    test_G.add_node(1, y=37.35, x=-121.95)
    test_G.add_node(2, y=37.36, x=-121.94)
    test_G.add_node(3, y=37.37, x=-121.93)
    test_G.add_node(4, y=37.365, x=-121.945)

    # Route 1 (shortest, high risk)
    test_G.add_edge(1, 2, length=1000.0, collision_prob=0.1, source="data", label=0)
    test_G.add_edge(2, 3, length=2000.0, collision_prob=0.8, source="model", label=1)

    # Route 2 (detour safest, lower risk)
    test_G.add_edge(1, 4, length=1800.0, collision_prob=0.1, source="data", label=0)
    test_G.add_edge(4, 3, length=1500.0, collision_prob=0.15, source="data", label=0)

    route_shortest = [1, 2, 3]
    route_safest = [1, 4, 3]

    eval_result = evaluate_route(test_G, route_shortest)

    print("Evaluation Result (Shortest):")
    for k, v in eval_result.items():
        if k != "edge_list":
            print(f"  {k}: {v}")

    assert eval_result["total_length"] == 3000.0
    assert eval_result["high_risk_segments"] == 1
    assert eval_result["data_count"] == 1
    assert eval_result["model_count"] == 1
    assert round(eval_result["avg_risk"], 4) == round((1000 * 0.1 + 2000 * 0.8) / 3000, 4)

    print("\nTesting rank_and_explain...")
    dummy_routes_dict = {
        "safest": route_safest,
        "shortest": route_shortest,
        "alternatives": [],
        "alpha": 5.0,
        "error": None,
    }

    ranking_output = rank_and_explain(dummy_routes_dict, test_G)
    print("Reasoning Output:")
    print(f"  {ranking_output['reasoning']}")
    print(f"  Recommended Tag: {ranking_output['recommended_tag']}")
    print(f"  Ranked Routes Count: {len(ranking_output['ranked_routes'])}")

    assert ranking_output["recommended_route"] == route_safest
    assert "longer" in ranking_output["reasoning"]
    assert "avoids 1 high-risk segment" in ranking_output["reasoning"]

    print("\nTesting build_map...")
    folium_map = build_map(test_G, dummy_routes_dict, ranking_output)
    assert isinstance(folium_map, folium.Map)
    print(f"  Map generated successfully (Type: {type(folium_map).__name__})")

    # Real graph integration test if scored_graph.graphml exists
    from pathlib import Path
    graph_file = Path("data/scored_graph.graphml")
    if graph_file.exists():
        print("\nTesting with real data/scored_graph.graphml...")
        try:
            from src.routing import get_routes, load_scored_graph
        except ModuleNotFoundError:
            from routing import get_routes, load_scored_graph
        real_G = load_scored_graph()
        nodes = list(real_G.nodes)
        if len(nodes) > 500:
            start_n, end_n = nodes[10], nodes[400]
            real_routes = get_routes(start_n, end_n, G=real_G, alpha=5.0)
            real_ranking = rank_and_explain(real_routes, real_G)
            real_map = build_map(real_G, real_routes, real_ranking)
            print("  Real Route Reasoning:", real_ranking["reasoning"])
            print("  Real Map Object Created:", isinstance(real_map, folium.Map))

    print("\nStep 3 tests passed successfully!")


