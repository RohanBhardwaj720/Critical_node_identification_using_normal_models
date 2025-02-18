import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # Add this import
from node2vec import Node2Vec

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

    def predict(self, X):
        """Predicts criticality scores for a given set of embeddings."""
        X = self.scaler.transform(X)
        predictions = self.model.predict(X).flatten()
        return np.clip(predictions, 0.0, 1.0)  # Ensure predictions are in [0, 1]

    def load_model(self, model_path, scaler_path):
        """Loads a trained model and scaler from files."""
        self.model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")

    def evaluate_rank_metrics(self, y_true, y_pred, top_n_percent=0.05):
        """Evaluates ranking metrics: Top-N% Accuracy, Spearman, and Kendall Tau."""
        print("y_true Range:", np.min(y_true), "to", np.max(y_true))
        print("y_pred Range:", np.min(y_pred), "to", np.max(y_pred))

        # Top-N% Accuracy
        true_ranking = np.argsort(y_true)
        pred_ranking = np.argsort(y_pred)
        top_n = int(len(y_true) * top_n_percent)
        top_true = set(true_ranking[-top_n:])
        top_pred = set(pred_ranking[-top_n:])
        top_n_accuracy = len(top_true & top_pred) / len(top_true)

        # Compute ranking correlations
        spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
        kendall_corr, kendall_p = stats.kendalltau(y_true, y_pred)

        return {
            "Top-N% Accuracy": top_n_accuracy,
            "Spearman Correlation": spearman_corr,
            "Kendall Tau Correlation": kendall_corr,
        }

def compute_criticality_scores(graph, metric='degree'):
    """Compute criticality scores for nodes based on a simple metric."""
    if metric == 'degree':
        degree_centrality = nx.degree_centrality(graph)
        scores = np.array([degree_centrality[node] for node in graph.nodes()])
    else:
        scores = np.zeros(len(graph.nodes()))  # Default: return zero scores if unknown metric
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))  # Normalize scores to [0, 1]

def generate_random_graph(num_nodes=100, probability=0.05):
    """Generates a random graph using the Erdős-Rényi model."""
    return nx.erdos_renyi_graph(num_nodes, probability)

def generate_node2vec_embeddings(graph, dimensions=128, walk_length=30, num_walks=200, window=10):
    """Generates node embeddings using Node2Vec."""
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=1)
    embeddings = {node: model.wv.get_vector(str(node)) for node in graph.nodes()}
    return np.array([embeddings[node] for node in graph.nodes()])

def main():
    # Generate a random graph
    graph = generate_random_graph()
    print(f"Random Graph Generated: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Compute ground truth criticality scores
    ground_truth_scores = compute_criticality_scores(graph, metric='degree')
    print("Ground truth criticality scores computed.")

    # Generate embeddings using Node2Vec
    node_embeddings = generate_node2vec_embeddings(graph)
    print("Node embeddings generated.")

    # Load the trained ILGR model and scaler
    ilgr = ILGR(input_dim=node_embeddings.shape[1])
    ilgr.load_model("trained_ilgr_model_node2vec.h5", "scaler_node2vec.pkl")

    # Predict criticality scores using the trained model
    predicted_scores = ilgr.predict(node_embeddings)

    # Binarize the ground truth and predicted scores
    threshold = 0.5  # You can adjust this threshold as needed
    ground_truth_binary = (ground_truth_scores >= threshold).astype(int)
    predicted_binary = (predicted_scores >= threshold).astype(int)

    # Evaluate the results
    accuracy = accuracy_score(ground_truth_binary, predicted_binary)
    precision = precision_score(ground_truth_binary, predicted_binary, average='macro')
    recall = recall_score(ground_truth_binary, predicted_binary, average='macro')
    f1 = f1_score(ground_truth_binary, predicted_binary, average='macro')

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()