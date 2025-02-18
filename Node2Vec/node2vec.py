import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
from node2vec import Node2Vec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ILGR:
    def __init__(self, input_dim, regression_layers=[512, 256, 128, 64, 1]):
        self.input_dim = input_dim
        self.regression_layers = regression_layers
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        """Builds the regression module for predicting criticality scores."""
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        for units in self.regression_layers[:-1]:
            x = layers.Dense(units, activation='relu')(x)
        output = layers.Dense(self.regression_layers[-1], activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Using MSE loss for regression
            metrics=['mae']
        )
        return self.model

    def train(self, X, y, epochs=1000, batch_size=8):
        """Trains the regression model using the entire dataset."""
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """Predicts criticality scores for a given set of embeddings."""
        X = self.scaler.transform(X)
        predictions = self.model.predict(X).flatten()
        return np.clip(predictions, 0.0, 1.0)  # Ensure predictions are in [0, 1]

    def evaluate_rank_metrics(self, y_true, y_pred, top_n_percent=0.05):
        """Evaluates ranking metrics: Top-N% Accuracy."""
        print("y_true Range:", np.min(y_true), "to", np.max(y_true))
        print("y_pred Range:", np.min(y_pred), "to", np.max(y_pred))

        true_ranking = np.argsort(y_true)
        pred_ranking = np.argsort(y_pred)
        top_n = int(len(y_true) * top_n_percent)
        top_true = set(true_ranking[-top_n:])
        top_pred = set(pred_ranking[-top_n:])
        top_n_accuracy = len(top_true & top_pred) / len(top_true)

        return {
            "Top-N% Accuracy": top_n_accuracy
        }

def compute_criticality_scores(graph, metric='degree'):
    """Compute criticality scores for nodes based on a simple metric."""
    if metric == 'degree':
        degree_centrality = nx.degree_centrality(graph)
        scores = np.array([degree_centrality[node] for node in graph.nodes()])
    else:
        scores = np.zeros(len(graph.nodes()))  # Default: return zero scores if unknown metric
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))  # Normalize scores to [0, 1]

def load_graph(file_path):
    """Loads an unweighted graph from an edgelist file."""
    return nx.read_edgelist(file_path, nodetype=str)

def generate_node2vec_embeddings(graph, dimensions=128, walk_length=30, num_walks=200, window=10):
    """Generates node embeddings using Node2Vec."""
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=1)
    embeddings = {node: model.wv.get_vector(str(node)) for node in graph.nodes()}
    return np.array([embeddings[node] for node in graph.nodes()])

def main():
    file_path = "edges.txt"  # Replace with your actual graph file path
    graph = load_graph(file_path)
    print(f"Graph Loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Compute criticality scores
    criticality_scores = compute_criticality_scores(graph, metric='degree')
    print("Criticality scores computed.")

    # Generate embeddings using Node2Vec
    node_embeddings = generate_node2vec_embeddings(graph)
    print("Node embeddings generated.")

    # Combine training and testing data
    nodes = list(graph.nodes())
    indices = np.arange(len(nodes))
    np.random.shuffle(indices)

    # Use the entire dataset (train + test) for training (overfitting)
    train_embeddings = node_embeddings[indices]
    train_scores = criticality_scores[indices]

    ilgr = ILGR(input_dim=train_embeddings.shape[1])
    ilgr.build_model()
    ilgr.train(train_embeddings, train_scores, epochs=300)

    # Evaluate on the same data (overfitting)
    predicted_scores = ilgr.predict(train_embeddings)
    metrics = ilgr.evaluate_rank_metrics(train_scores, predicted_scores, top_n_percent=0.05)
    print("Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
