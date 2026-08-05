from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import osmnx as ox
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_GRAPH_PATH = PROJECT_ROOT / "data" / "base_graph.graphml"
SCORED_GRAPH_PATH = PROJECT_ROOT / "data" / "scored_graph.graphml"

MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
SCHEMA_PATH = PROJECT_ROOT / "model" / "schema.json"


# --------------------------------------------------
# Helpers for raw OSM values
# --------------------------------------------------

def is_missing(value: Any) -> bool:
    """Safely check whether a value is missing."""
    if value is None:
        return True

    if isinstance(value, float) and np.isnan(value):
        return True

    return False


def normalize_category(value: Any) -> str:
    """
    Convert raw OSM values into the same string format used when the
    category mappings were created in Colab.

    Examples:
        None -> "unknown"
        "primary" -> "primary"
        ["residential", "secondary"]
            -> "['residential', 'secondary']"
    """
    if is_missing(value):
        return "unknown"

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return str(value)

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return "unknown"

        # GraphML may reload a list as a string representation.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple)):
                    return str(list(parsed))
            except (ValueError, SyntaxError):
                pass

        return text

    return str(value)


def encode_category(
    value: Any,
    mapping: dict[str, int],
    fallback_name: str = "unknown",
) -> int:
    """Apply one of the mappings stored in schema.json."""
    normalized = normalize_category(value)

    if normalized in mapping:
        return int(mapping[normalized])

    if fallback_name in mapping:
        return int(mapping[fallback_name])

    # The highway mapping did not contain "unknown".
    # Use unclassified for unseen road types.
    if "unclassified" in mapping:
        return int(mapping["unclassified"])

    raise ValueError(
        f"Cannot encode unseen category {normalized!r}; "
        "the schema has no valid fallback."
    )


def clean_bool(value: Any) -> bool:
    """Convert GraphML/OSM one-way values to Boolean."""
    if isinstance(value, bool):
        return value

    if is_missing(value):
        return False

    text = str(value).strip().lower()

    return text in {
        "true",
        "yes",
        "1",
        "-1",
    }


# --------------------------------------------------
# Engineered graph features
# --------------------------------------------------

def calculate_turn_angle(geometry: Any) -> float:
    """
    Reproduce the notebook's turn-angle feature:

    average absolute direction change between consecutive line segments.
    """
    if geometry is None:
        return 0.0

    try:
        coordinates = list(geometry.coords)
    except (AttributeError, TypeError):
        return 0.0

    if len(coordinates) < 3:
        return 0.0

    turn_angles: list[float] = []

    for index in range(1, len(coordinates) - 1):
        x1, y1 = coordinates[index - 1]
        x2, y2 = coordinates[index]
        x3, y3 = coordinates[index + 1]

        direction1 = math.atan2(
            y2 - y1,
            x2 - x1,
        )

        direction2 = math.atan2(
            y3 - y2,
            x3 - x2,
        )

        turn_radians = direction2 - direction1

        # Normalize to the range [-pi, pi].
        turn_radians = (
            (turn_radians + math.pi)
            % (2 * math.pi)
            - math.pi
        )

        turn_degrees = math.degrees(turn_radians)
        turn_angles.append(abs(turn_degrees))

    if not turn_angles:
        return 0.0

    return float(np.mean(turn_angles))


def calculate_average_node_degree(
    graph,
    u: int,
    v: int,
) -> float:
    """
    Reproduce the notebook's feature:

    (degree of start node + degree of end node) / 2
    """
    u_degree = graph.degree(u)
    v_degree = graph.degree(v)

    return float(
        (u_degree + v_degree) / 2
    )


# --------------------------------------------------
# Prepare model input
# --------------------------------------------------

def preprocess_edges(
    graph,
    edges: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:
    mappings = schema["category_mappings"]
    expected_columns = schema["columns"]

    processed = pd.DataFrame(index=edges.index)

    processed["highway"] = edges["highway"].apply(
        lambda value: encode_category(
            value,
            mappings["highway"],
        )
    )

    processed["length"] = pd.to_numeric(
        edges["length"],
        errors="coerce",
    ).fillna(0.0)

    processed["lanes"] = edges.get(
        "lanes",
        pd.Series(
            "unknown",
            index=edges.index,
        ),
    ).apply(
        lambda value: encode_category(
            value,
            mappings["lanes"],
        )
    )

    processed["maxspeed"] = edges.get(
        "maxspeed",
        pd.Series(
            "unknown",
            index=edges.index,
        ),
    ).apply(
        lambda value: encode_category(
            value,
            mappings["maxspeed"],
        )
    )

    processed["oneway"] = edges.get(
        "oneway",
        pd.Series(
            False,
            index=edges.index,
        ),
    ).apply(clean_bool)

    processed["bridge"] = edges.get(
        "bridge",
        pd.Series(
            "unknown",
            index=edges.index,
        ),
    ).apply(
        lambda value: encode_category(
            value,
            mappings["bridge"],
        )
    )

    processed["tunnel"] = edges.get(
        "tunnel",
        pd.Series(
            "unknown",
            index=edges.index,
        ),
    ).apply(
        lambda value: encode_category(
            value,
            mappings["tunnel"],
        )
    )

    processed["junction"] = edges.get(
        "junction",
        pd.Series(
            "unknown",
            index=edges.index,
        ),
    ).apply(
        lambda value: encode_category(
            value,
            mappings["junction"],
        )
    )

    processed["turn_angles"] = edges["geometry"].apply(
        calculate_turn_angle
    )

    processed["avg_node_degree"] = [
        calculate_average_node_degree(
            graph,
            u,
            v,
        )
        for u, v, _ in edges.index
    ]

    # Put features in exactly the same order used during training.
    processed = processed.reindex(
        columns=expected_columns
    )

    if list(processed.columns) != expected_columns:
        raise ValueError(
            "Processed columns do not match schema columns."
        )

    if processed.isna().any().any():
        missing = processed.isna().sum()
        missing = missing[missing > 0]

        raise ValueError(
            "Model input still contains missing values:\n"
            f"{missing}"
        )

    return processed


# --------------------------------------------------
# Score graph
# --------------------------------------------------

def score_graph() -> None:
    for required_path in [
        BASE_GRAPH_PATH,
        MODEL_PATH,
        SCHEMA_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

        if required_path.stat().st_size == 0:
            raise ValueError(
                f"Required file is empty: {required_path}"
            )

    print("Loading base graph...")
    graph = ox.load_graphml(BASE_GRAPH_PATH)

    print("Loading Random Forest model...")
    model = joblib.load(MODEL_PATH)

    print("Loading schema...")
    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        schema = json.load(file)

    threshold = float(schema["threshold"])

    print("Model:", schema.get("model_name"))
    print("Threshold:", threshold)
    print("Expected columns:", schema["columns"])

    print("Converting graph edges...")
    edges = ox.graph_to_gdfs(
        graph,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    )

    print("Total edges:", len(edges))

    print("Creating model features...")
    processed = preprocess_edges(
        graph,
        edges,
        schema,
    )

    print("\nProcessed feature preview:")
    print(processed.head())

    print("\nProcessed data types:")
    print(processed.dtypes)

    print("Generating collision probabilities...")
    probabilities = model.predict_proba(
        processed
    )[:, 1]

    labels = (
        probabilities >= threshold
    ).astype(int)

    print("Writing scores to graph...")

    for (
        (u, v, key),
        probability,
        label,
    ) in zip(
        edges.index,
        probabilities,
        labels,
    ):
        graph[u][v][key]["collision_prob"] = float(
            probability
        )

        graph[u][v][key]["label"] = int(label)
        graph[u][v][key]["source"] = "model"

    SCORED_GRAPH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Saving scored graph...")
    ox.save_graphml(
        graph,
        SCORED_GRAPH_PATH,
    )

    print("\nScoring complete.")
    print("Saved to:", SCORED_GRAPH_PATH)
    print(
        "Predicted collision-positive edges:",
        int(labels.sum()),
    )
    print(
        "Predicted collision-negative edges:",
        int((labels == 0).sum()),
    )

    print("\nProbability summary:")
    print(
        pd.Series(
            probabilities,
            name="collision_prob",
        ).describe()
    )


if __name__ == "__main__":
    score_graph()