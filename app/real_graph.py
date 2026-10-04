"""
Real graph analysis on the Elliptic Bitcoin dataset.
Loads elliptic_txs_features.csv, elliptic_txs_edgelist.csv, elliptic_txs_classes.csv,
runs community detection + classifier for illicit nodes with time-based split.
Writes precision/recall to metrics_real_graph.json.
Label clearly as real blockchain data, not Indian complaint data.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


# Paths to Elliptic dataset (download from Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
ELLIPTIC_DIR = Path(__file__).parent.parent / "data" / "real" / "elliptic"
FEATURES_FILE = ELLIPTIC_DIR / "elliptic_txs_features.csv"
EDGES_FILE = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE = ELLIPTIC_DIR / "elliptic_txs_classes.csv"

METRICS_FILE = Path(__file__).parent.parent / "metrics_real_graph.json"

# Labels: 1 = illicit, 2 = licit, unknown = unknown
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}
LABEL_NAMES = {1: "illicit", 0: "licit", -1: "unknown"}


def load_elliptic_data(
    features_path: Optional[Path] = None,
    edges_path: Optional[Path] = None,
    classes_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Elliptic Bitcoin dataset CSV files."""
    features_path = features_path or FEATURES_FILE
    edges_path = edges_path or EDGES_FILE
    classes_path = classes_path or CLASSES_FILE
    
    for path in [features_path, edges_path, classes_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Elliptic dataset file not found: {path}. "
                f"Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
            )
    
    print("Loading Elliptic dataset...")
    features = pd.read_csv(features_path, header=None)
    edges = pd.read_csv(edges_path)
    classes = pd.read_csv(classes_path)
    
    # Features: col 0 = txId, col 1 = time_step, cols 2-93 = local features, cols 94-165 = aggregated features
    features.columns = ["tx_id", "time_step"] + [f"feat_{i}" for i in range(2, 166)]
    
    # Edges: txId1 -> txId2 (directed)
    edges.columns = ["tx_id1", "tx_id2"]
    
    # Classes: txId, class
    classes.columns = ["tx_id", "class"]
    classes["label"] = classes["class"].map(LABEL_MAP)
    
    print(f"Loaded {len(features)} transactions, {len(edges)} edges, {len(classes)} labeled")
    return features, edges, classes


def build_transaction_graph(features: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Build directed transaction graph from features and edges."""
    G = nx.DiGraph()
    
    # Add nodes with features
    feature_cols = [c for c in features.columns if c.startswith("feat_")]
    for _, row in features.iterrows():
        G.add_node(
            row["tx_id"],
            time_step=row["time_step"],
            features=row[feature_cols].values.astype(np.float32)
        )
    
    # Add edges
    for _, row in edges.iterrows():
        if row["tx_id1"] in G and row["tx_id2"] in G:
            G.add_edge(row["tx_id1"], row["tx_id2"])
    
    print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def detect_communities(G: nx.DiGraph) -> Dict[int, int]:
    """Detect communities using Louvain on undirected version."""
    # Convert to undirected for community detection
    G_undir = G.to_undirected()
    
    # Use Louvain community detection (requires python-louvain)
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_undir)
        return partition
    except ImportError:
        # Fallback: use connected components
        warnings.warn("python-louvain not installed, using connected components")
        communities = {}
        for i, component in enumerate(nx.connected_components(G_undir)):
            for node in component:
                communities[node] = i
        return communities


def add_graph_features(G: nx.DiGraph, partition: Dict[int, int]) -> pd.DataFrame:
    """Compute graph-based features for each node."""
    print("Computing graph features...")
    
    # Basic centrality measures (sample for speed on large graphs)
    print("  Computing centrality...")
    degree_cent = nx.degree_centrality(G)
    in_degree_cent = nx.in_degree_centrality(G)
    out_degree_cent = nx.out_degree_centrality(G)
    
    # Community features
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    # Build feature DataFrame
    rows = []
    for node in G.nodes():
        comm = partition.get(node, -1)
        rows.append({
            "tx_id": node,
            "degree_centrality": degree_cent.get(node, 0),
            "in_degree_centrality": in_degree_cent.get(node, 0),
            "out_degree_centrality": out_degree_cent.get(node, 0),
            "community_id": comm,
            "community_size": community_sizes.get(comm, 0),
            "time_step": G.nodes[node].get("time_step", 0),
        })
    
    return pd.DataFrame(rows)


def prepare_ml_data(
    features: pd.DataFrame,
    graph_features: pd.DataFrame,
    classes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Merge features, graph features, and labels."""
    # Merge
    df = features.merge(graph_features, on="tx_id", how="left")
    df = df.merge(classes[["tx_id", "label"]], on="tx_id", how="left")
    
    # Split known/unknown
    known = df[df["label"] != -1].copy()
    unknown = df[df["label"] == -1].copy()
    
    # Features for training
    feature_cols = [c for c in known.columns if c not in ["tx_id", "label", "class"]]
    
    X = known[feature_cols].fillna(0)
    y = known["label"]
    
    # Time steps for temporal split
    time_steps = known["time_step"]
    
    return X, y, time_steps, unknown[feature_cols].fillna(0) if len(unknown) > 0 else None


def train_illicit_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    time_steps: pd.Series,
    test_split_step: int = 34  # Steps 1-34 train, 35-49 test (per Elliptic paper)
) -> Tuple[CalibratedClassifierCV, Dict]:
    """Train classifier with time-based split."""
    
    # Time-based split
    train_mask = time_steps <= test_split_step
    test_mask = time_steps > test_split_step
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Train: {len(X_train)} samples (steps 1-{test_split_step})")
    print(f"Test: {len(X_test)} samples (steps {test_split_step+1}-49)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train base classifier
    base_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    # Calibrate probabilities
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "illicit_in_test": int(y_test.sum()),
        "licit_in_test": int((y_test == 0).sum()),
        "train_time_steps": f"1-{test_split_step}",
        "test_time_steps": f"{test_split_step+1}-49",
        "note": "REAL BLOCKCHAIN DATA (Elliptic Bitcoin dataset) - NOT Indian complaint data"
    }
    
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    return clf, metrics


def run_real_graph_analysis(
    features_path: Optional[Path] = None,
    edges_path: Optional[Path] = None,
    classes_path: Optional[Path] = None
) -> Dict:
    """Run complete real graph analysis pipeline."""
    print("=" * 60)
    print("ELLIPTIC BITCOIN DATASET ANALYSIS")
    print("REAL BLOCKCHAIN DATA - NOT INDIAN COMPLAINT DATA")
    print("=" * 60)
    
    # Load data
    features, edges, classes = load_elliptic_data(features_path, edges_path, classes_path)
    
    # Build graph
    G = build_transaction_graph(features, edges)
    
    # Community detection
    print("Detecting communities...")
    partition = detect_communities(G)
    n_communities = len(set(partition.values()))
    print(f"Detected {n_communities} communities")
    
    # Graph features
    graph_features = add_graph_features(G, partition)
    
    # Prepare ML data
    X, y, time_steps, _ = prepare_ml_data(features, graph_features, classes)
    
    # Train classifier
    clf, metrics = train_illicit_classifier(X, y, time_steps)
    
    # Save metrics
    metrics["dataset"] = "Elliptic Bitcoin Dataset"
    metrics["data_type"] = "Real blockchain transactions"
    metrics["n_transactions"] = int(len(features))
    metrics["n_edges"] = int(len(edges))
    metrics["n_labeled"] = int(len(classes[classes["class"] != "unknown"]))
    metrics["n_communities"] = int(len(set(partition.values())))
    metrics["generated_at"] = datetime.utcnow().isoformat()
    
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {METRICS_FILE}")
    return metrics


if __name__ == "__main__":
    run_real_graph_analysis()