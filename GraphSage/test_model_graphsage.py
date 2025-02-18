import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib  # Add this import

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))  # First layer with ReLU activation
        x = self.sage2(x, edge_index)  # Second layer (output layer)
        return x

def generate_graphsage_embeddings(graph, input_dim=16, hidden_dim=64, output_dim=128, epochs=500, lr=0.0001, target_scores=None):
    """Generates node embeddings using GraphSAGE."""
    # Convert NetworkX graph to PyTorch Geometric Data
    # Create a mapping from node labels (string) to integer indices
    node_map = {node: idx for idx, node in enumerate(graph.nodes())}
    
    # Convert edges to integer indices
    edge_index = torch.tensor(
        [[node_map[u], node_map[v]] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Initialize random features for the nodes
    x = torch.randn((len(graph.nodes()), input_dim), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    # Initialize GraphSAGE model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(input_dim, hidden_dim, output_dim).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure target_scores is a PyTorch tensor and move it to the correct device
    if target_scores is not None:
        target_scores = torch.tensor(target_scores, dtype=torch.float).to(device)

    # Train GraphSAGE
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # Forward pass
        if target_scores is not None:
            # Use target_scores (criticality scores) for training
            loss = F.mse_loss(out, target_scores.view(-1, 1))  # MSE loss with target scores
        else:
            loss = F.mse_loss(out, data.x)  # Use a dummy self-supervised loss (for illustration)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Return node embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()
    return embeddings

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

def main():
    # Generate a random graph
    graph = generate_random_graph()
    print(f"Random Graph Generated: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Compute ground truth criticality scores
    ground_truth_scores = compute_criticality_scores(graph, metric='degree')
    print("Ground truth criticality scores computed.")

    # Generate embeddings using GraphSAGE
    node_embeddings = generate_graphsage_embeddings(graph, target_scores=ground_truth_scores)
    print("Node embeddings generated.")

    # Load the trained ILGR model and scaler
    ilgr = ILGR(input_dim=node_embeddings.shape[1])
    ilgr.load_model("trained_ilgr_model_graphsage.h5", "scaler_graphsage.pkl")

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