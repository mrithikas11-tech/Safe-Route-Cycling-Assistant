import osmnx as ox
import networkx as nx

CENTER = (37.3541, -121.9552)
RADIUS = 30000

def build(): 
    center = (37.3541, -121.9552)
    G = ox.graph_from_point(center, dist=30000,network_type ="bike")
    ox.save_graphml(G, filepath="data/base_graph.graphml")
    print("graph build and saved", "nodes:", len(G.nodes) ,"edges:", len(G.edges))
    return G
    
def load_graph(filepath="data/base_graph.graphml"):
    return ox.load_graphml(filepath)

def geocode(address):
    try: 
        lat,lon = ox.geocode(address)
        return lat, lon 
    except Exception as error: 
        print(f"Could not geocode {address!r}: {error}")
        return None
    
def snap_to_node(G, lat, lon):
    return ox.nearest_nodes(G, X=lon,Y=lat)

def resolve(G, address): 
    coords = geocode(address)
    if coords is None: 
        return None
    lat, lon = coords
    distance = ox.distance.great_circle(CENTER[0], CENTER[1], lat, lon)
    if distance > RADIUS:
        print("your address was not found.") 
        return None
    return snap_to_node(G, lat, lon)

def resolve_both(G, start_address, end_address):
    start = resolve(G, start_address)
    end = resolve(G, end_address)
    return start, end

#testing
if __name__ == "__main__":
    G = load_graph()
    start, end = resolve_both(G, "San Francisco State University, San Francisco, CA", "Santana Row, San Jose, CA")
    print("Start node:", start)
    print("End node:", end)
    if start is not None and end is not None:
        print("route has been mapped!")