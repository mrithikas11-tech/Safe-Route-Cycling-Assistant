import json
import joblib
import osmnx as ox
import pandas as pd

GRAPH_PATH = "data/base_graph.graphml"
OUTPUT_PATH = "data/scored_graph.graphml"

MODEL_PATH = "model/model.pkl"
SCHEMA_PATH = "model/schema.json"


##############################
# Load everything
##############################

print("Loading graph...")
G = ox.load_graphml(GRAPH_PATH)

print("Loading model...")
model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

expected_columns = schema["columns"]
threshold = schema["threshold"]


##############################
# Convert graph -> dataframe
##############################

edges = ox.graph_to_gdfs(
    G,
    nodes=False,
    fill_edge_geometry=True
)

print("Edges:", len(edges))


##############################
# Preprocessing
##############################

processed = pd.DataFrame(index=edges.index)

processed["length"] = edges["length"]

processed["lanes"] = (
    pd.to_numeric(edges["lanes"], errors="coerce")
)

processed["maxspeed"] = (
    pd.to_numeric(edges["maxspeed"], errors="coerce")
)

processed["oneway"] = (
    edges["oneway"]
    .fillna(False)
    .astype(int)
)

##############################
# Highway
##############################

highway = edges["highway"].copy()

highway = highway.apply(
    lambda x: x[0] if isinstance(x, list) else x
)

highway = pd.get_dummies(
    highway,
    prefix="highway"
)

processed = pd.concat(
    [processed, highway],
    axis=1
)

##############################
# Missing values
##############################

processed["lanes"] = processed["lanes"].fillna(2)

processed["maxspeed"] = processed["maxspeed"].fillna(35)

##############################
# Match model columns
##############################

processed = processed.reindex(
    columns=expected_columns,
    fill_value=0
)

assert list(processed.columns) == expected_columns

##############################
# Predict
##############################

print("Predicting...")

probabilities = model.predict_proba(processed)[:,1]

labels = (
    probabilities >= threshold
).astype(int)

##############################
# Write back to graph
##############################

for (u,v,key),prob,label in zip(
    edges.index,
    probabilities,
    labels
):

    G[u][v][key]["collision_prob"] = float(prob)

    G[u][v][key]["label"] = int(label)

    G[u][v][key]["source"] = "model"

##############################
# Save
##############################

ox.save_graphml(
    G,
    OUTPUT_PATH
)

print("Done!")

print(
    "Saved:",
    OUTPUT_PATH
)