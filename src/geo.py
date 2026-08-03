import osmnx as ox

def build(): 
    center = (37.3541, -121.9552)
    G = ox.graph_from_point(center, dist=30000,network_type ="bike")
    ox.save_graphml(G, filepath="data/base_graph.graphml")
    print("graph build and saved", "nodes:", len(G.nodes) ,"edges:", len(G.edges))
    return G

if __name__ == "__main__":
    build()