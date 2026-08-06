from pathlib import Path

import networkx as nx
import osmnx as ox


# Adjustable safety vs. distance tradeoff
ALPHA = 5.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = PROJECT_ROOT / "data" / "scored_graph.graphml"

# Maximum number of alternate routes
MAX_ALTERNATIVES = 3


def load_scored_graph():
    """
    Load the graph created by Person 2.
    """

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "Run scoring.py first. scored_graph.graphml not found."
        )

    print("Loading scored graph...")
    return ox.load_graphml(GRAPH_PATH)


def add_safe_costs(G, alpha=ALPHA):
    """
    Add a safety-adjusted cost to every road edge.

    safe_cost = length * (1 + alpha * collision_prob)
    """

    if alpha < 0:
        raise ValueError("alpha must be zero or greater.")

    missing_probabilities = 0

    for _, _, _, data in G.edges(keys=True, data=True):
        try:
            length = float(data.get("length", 1.0))
        except (TypeError, ValueError):
            length = 1.0

        try:
            collision_prob = float(data.get("collision_prob", 0.0))
        except (TypeError, ValueError):
            collision_prob = 0.0
            missing_probabilities += 1

        collision_prob = max(0.0, min(1.0, collision_prob))

        data["safe_cost"] = length * (
            1 + alpha * collision_prob
        )

    if missing_probabilities > 0:
        print(
            f"Warning: {missing_probabilities} edges had "
            "invalid collision probabilities."
        )

def get_shortest_route(G, start, end):
    """
    Return the route with the smallest total road length.
    """

    try:
        return nx.shortest_path(
            G,
            source=start,
            target=end,
            weight="length",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def get_safest_route(G, start, end):
    """
    Return the route with the smallest total safe_cost.
    """

    try:
        return nx.shortest_path(
            G,
            source=start,
            target=end,
            weight="safe_cost",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def convert_to_simple_digraph(G):
    """
    Convert the OSMnx MultiDiGraph into a DiGraph.

    When multiple edges connect the same two nodes,
    keep the edge with the smallest safe_cost.
    """

    simple_G = nx.DiGraph()
    simple_G.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        try:
            new_cost = float(data.get("safe_cost", 1.0))
        except (TypeError, ValueError):
            new_cost = 1.0

        if not simple_G.has_edge(u, v):
            simple_G.add_edge(u, v, **data)
        else:
            try:
                existing_cost = float(
                    simple_G[u][v].get(
                        "safe_cost",
                        float("inf"),
                    )
                )
            except (TypeError, ValueError):
                existing_cost = float("inf")

            if new_cost < existing_cost:
                simple_G[u][v].clear()
                simple_G[u][v].update(data)

    return simple_G


def get_alternative_routes(
    G,
    start,
    end,
    max_routes=MAX_ALTERNATIVES,
):
    """
    Return up to three routes ordered by safe_cost.
    """

    if max_routes <= 0:
        return []

    simple_G = convert_to_simple_digraph(G)

    try:
        route_generator = nx.shortest_simple_paths(
            simple_G,
            source=start,
            target=end,
            weight="safe_cost",
        )

        routes = []

        for route in route_generator:
            routes.append(route)

            if len(routes) >= max_routes:
                break

        return routes

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_routes(
    start,
    end,
    G=None,
    alpha=ALPHA,
    alternative_count=MAX_ALTERNATIVES,
):
    """
    Return the safest route, shortest route, and unique alternatives.

    Output:
    {
        "safest": [...],
        "shortest": [...],
        "alternatives": [[...], [...]],
        "alpha": 5.0,
        "error": None
    }
    """

    if G is None:
        try:
            G = load_scored_graph()
        except FileNotFoundError as error:
            return {
                "safest": None,
                "shortest": None,
                "alternatives": [],
                "alpha": alpha,
                "error": str(error),
            }

    add_safe_costs(G, alpha=alpha)

    shortest_route = get_shortest_route(
        G,
        start,
        end,
    )

    safest_route = get_safest_route(
        G,
        start,
        end,
    )

    if shortest_route is None and safest_route is None:
        return {
            "safest": None,
            "shortest": None,
            "alternatives": [],
            "alpha": alpha,
            "error": (
                "No path exists between the selected locations."
            ),
        }

    route_candidates = get_alternative_routes(
        G,
        start,
        end,
        max_routes=alternative_count + 2,
    )

    alternatives = []

    for route in route_candidates:
        if route == safest_route:
            continue

        if route == shortest_route:
            continue

        if route in alternatives:
            continue

        alternatives.append(route)

        if len(alternatives) >= alternative_count:
            break

    return {
        "safest": safest_route,
        "shortest": shortest_route,
        "alternatives": alternatives,
        "alpha": alpha,
        "error": None,
    }


if __name__ == "__main__":
    # Developer sanity tests; this block does not run when app.py imports
    # routing.py.
    G = load_scored_graph()

    node_list = list(G.nodes)
    start = node_list[0]
    end = node_list[1000]

    print("Start node:", start)
    print("End node:", end)

    print("\n--- Normal routing test ---")

    results = get_routes(
        start=start,
        end=end,
        G=G,
        alpha=5.0,
        alternative_count=3,
    )

    print("Error:", results["error"])

    shortest_route = results["shortest"]
    safest_route = results["safest"]
    alternatives = results["alternatives"]

    print(
        "Shortest route node count:",
        len(shortest_route) if shortest_route else 0,
    )

    print(
        "Safest route node count:",
        len(safest_route) if safest_route else 0,
    )

    print(
        "Safest and shortest are identical:",
        safest_route == shortest_route,
    )

    print(
        "Number of unique alternatives:",
        len(alternatives),
    )

    for index, route in enumerate(alternatives, start=1):
        print(
            f"Alternative {index} node count:",
            len(route),
        )

    print("\n--- Alpha sanity test ---")

    alpha_zero_results = get_routes(
        start=start,
        end=end,
        G=G,
        alpha=0.0,
        alternative_count=0,
    )

    high_alpha_results = get_routes(
        start=start,
        end=end,
        G=G,
        alpha=20.0,
        alternative_count=0,
    )

    print(
        "Alpha 0 safest equals shortest:",
        alpha_zero_results["safest"]
        == alpha_zero_results["shortest"],
    )

    print(
        "High alpha changes route:",
        high_alpha_results["safest"]
        != alpha_zero_results["safest"],
    )


