from pathlib import Path
import osmnx as ox

CENTER = (37.3541, -121.9552)
RADIUS = 30000
GRAPH_PATH = Path("data/base_graph.graphml")


def build():
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)

    G = ox.graph_from_point(
        CENTER,
        dist=RADIUS,
        network_type="bike"
    )

    ox.save_graphml(G, filepath=GRAPH_PATH)

    print(
        "Graph built and saved",
        "nodes:", len(G.nodes),
        "edges:", len(G.edges)
    )

    return G


def load_graph(filepath=GRAPH_PATH):
    return ox.load_graphml(filepath)


def get_graph():
    if GRAPH_PATH.exists() and GRAPH_PATH.stat().st_size > 0:
        print("Loading existing graph...")
        return load_graph()

    print("Graph not found. Building graph...")
    return build()


def geocode(address):
    try:
        lat, lon = ox.geocode(address)
        return lat, lon
    except Exception as error:
        print(f"Could not geocode {address!r}: {error}")
        return None


def snap_to_node(G, lat, lon):
    return ox.distance.nearest_nodes(
        G,
        X=lon,
        Y=lat
    )


def resolve(G, address):
    coords = geocode(address)

    if coords is None:
        return None

    lat, lon = coords

    distance = ox.distance.great_circle(
        CENTER[0],
        CENTER[1],
        lat,
        lon
    )

    if distance > RADIUS:
        print(f"{address!r} is outside the coverage area.")
        return None

    return snap_to_node(G, lat, lon)


def resolve_both(G, start_address, end_address):
    start = resolve(G, start_address)
    end = resolve(G, end_address)
    return start, end


if __name__ == "__main__":
    G = get_graph()

    start, end = resolve_both(
        G,
        "San Francisco State University, San Francisco, CA",
        "Santana Row, San Jose, CA"
    )

    print("Start node:", start)
    print("End node:", end)

    if start is not None and end is not None:
        print("Route has been mapped!")