import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # Importing GraphSAGE layer
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import joblib  # Add this import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def train(self, X, y, epochs=1000, batch_size=8):
        """Trains the regression model using the entire dataset."""
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        """Predicts criticality scores for a given set of embeddings."""
        X = self.scaler.transform(X)
        predictions = self.model.predict(X).flatten()
        return np.clip(predictions, 0.0, 1.0)  # Ensure predictions are in [0, 1]

    def save_model(self, model_path, scaler_path):
        """Saves the trained model and scaler to files."""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

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

def load_graph(file_path):
    """Loads an unweighted graph from an edgelist file."""
    return nx.read_edgelist(file_path, nodetype=str)

def main():
    file_path = "GraphSage/facebook_combined.txt"  # Update the path to the correct location
    graph = load_graph(file_path)
    print(f"Graph Loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

    # Compute criticality scores
    criticality_scores = compute_criticality_scores(graph, metric='degree')
    print("Criticality scores computed.")

    # Generate embeddings using GraphSAGE
    node_embeddings = generate_graphsage_embeddings(graph, target_scores=criticality_scores)
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

    # Save the trained model and scaler
    ilgr.save_model("trained_ilgr_model_graphsage.h5", "scaler_graphsage.pkl")

    # Evaluate on the same data (overfitting)
    predicted_scores = ilgr.predict(train_embeddings)
    metrics = ilgr.evaluate_rank_metrics(train_scores, predicted_scores, top_n_percent=0.05)
    print("Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()