import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import numpy as np
import time

import matplotlib.pyplot as plt

os.environ['PYTORCH_MPS_ENABLE_FALLBACK']='1'

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)



# Function to plot training and validation loss over epochs
def plot_loss(train_losses, val_losses, save_path='loss_plot.png'):
    """
    Plots the training and validation loss over epochs.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"Loss plot saved to {save_path}")


def preprocess_data(filepath, sample_size=10000, k=10, random_state=42):
    """
    Load the dataset, sample a subset, encode categorical features,
    normalize continuous features, and create an optimized edge list.

    Args:
        filepath (str): Path to the CSV dataset.
        sample_size (int): Number of samples to select.
        k (int): Number of neighbors to connect for each node.
        random_state (int): Seed for reproducibility.

    Returns:
        Data: PyTorch Geometric Data object with sampled data.
        dict: Encoders used for categorical features.
    """
    print("Loading dataset...")
    # Load dataset
    df = pd.read_csv(filepath)
    print(f"Original dataset size: {df.shape}")

    # Sample the dataset
    print("Sampling data...")
    if sample_size < len(df):
        df_sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        df_sampled = df.reset_index(drop=True)
    print(f"Sampled dataset size: {df_sampled.shape}")

    # Encode categorical features
    print("Encoding categorical features...")
    encoders = {}
    categorical_cols = ['disease_family', 'disease', 'smoking_status', 'cough_type']
    for col in categorical_cols:
        if col in df_sampled.columns:
            encoder = LabelEncoder()
            df_sampled[col] = encoder.fit_transform(df_sampled[col])
            encoders[col] = encoder
            print(f"Encoded column: {col}")
        else:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    # Normalize continuous features
    print("Normalizing continuous features...")
    scaler = StandardScaler()
    continuous_cols = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'BMI', 'heart_rate', 'blood_glucose']
    for col in continuous_cols:
        if col in df_sampled.columns:
            df_sampled[col] = scaler.fit_transform(df_sampled[[col]])
            print(f"Normalized column: {col}")
        else:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    # Prepare node features (X)
    print("Preparing node features...")
    if 'patient_id' in df_sampled.columns:
        feature_cols = [col for col in df_sampled.columns if col != 'patient_id']
        x = torch.tensor(df_sampled[feature_cols].values, dtype=torch.float)
    else:
        x = torch.tensor(df_sampled.values, dtype=torch.float)
    print(f"Node features shape: {x.shape}")

    # Create optimized edge list
    print("Creating edge list...")
    edge_index = create_edge_list_vectorized(df_sampled, k=k)
    print(f"Edge list created with shape: {edge_index.shape}")

    return Data(x=x, edge_index=edge_index), encoders


def create_edge_list_vectorized(df, k=10):
    """
    Optimized edge list creation using vectorized operations.
    Connect each node to k randomly selected neighbors within the same 'disease_family'.

    Args:
        df (pd.DataFrame): Sampled DataFrame.
        k (int): Number of neighbors to connect for each node.

    Returns:
        torch.Tensor: Edge index tensor of shape [2, num_edges].
    """
    print("Grouping data by 'disease_family'...")
    edges = []

    groups = df.groupby('disease_family').groups
    print(f"Number of disease families: {len(groups)}")

    for group_name, group in tqdm(groups.items(), total=len(groups), desc="Processing groups"):
        group_indices = group.values  # Assuming group is a pandas Index or similar
        num_nodes = len(group_indices)

        if num_nodes <= 1:
            print(f"Group '{group_name}' has only {num_nodes} node(s). Skipping.")
            continue  # No edges can be formed

        # Determine the actual number of neighbors
        actual_k = min(k, num_nodes - 1)

        # Shuffle the indices for randomness
        shuffled_indices = np.random.permutation(group_indices)

        # Assign the first 'actual_k' indices as neighbors
        neighbors = np.tile(shuffled_indices[:actual_k], (num_nodes, 1))

        # Create source and target edges
        source_nodes = np.repeat(group_indices, actual_k)
        target_nodes = neighbors.flatten()

        # Append to the edge list
        edges.extend(zip(source_nodes, target_nodes))

    # Convert to NumPy array for efficient processing
    edges = np.array(edges)

    print("Removing duplicate edges and self-loops...")
    # Remove self-loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]

    # Remove duplicate edges
    edges = np.unique(edges, axis=0)
    print(f"Total edges after sampling: {len(edges)}")

    if len(edges) == 0:
        raise ValueError("No edges were created. Please check the edge creation logic.")

    # Convert to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return edge_index

class PatientGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(PatientGNN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.attention = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.attention(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return x

def train(model, optimizer, criterion, data, device, batch_size=64):
    model.train()
    optimizer.zero_grad()
    
    # Create mini-batches using random sampling of nodes
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    mini_batches = indices.split(batch_size)
    
    total_loss = 0
    for batch in mini_batches:
        # Select nodes for this mini-batch
        x_batch = data.x[batch].to(device)
        
        # Create a mapping from global to local indices
        global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(batch)}
        
        # Filter positive edges for this mini-batch
        mask_pos = torch.isin(data.train_pos_edge_index[0], batch) & torch.isin(data.train_pos_edge_index[1], batch)
        pos_edge_index_batch = data.train_pos_edge_index[:, mask_pos].to(device)
        
        # Map global indices to local indices for positive edges
        pos_edge_index_batch = torch.tensor(
            [[global_to_local[node.item()] for node in pos_edge_index_batch_row]
             for pos_edge_index_batch_row in pos_edge_index_batch],
            dtype=torch.long,
            device=device,
        )
        
        # Filter negative edges for this mini-batch
        mask_neg = torch.isin(data.train_neg_edge_index[0], batch) & torch.isin(data.train_neg_edge_index[1], batch)
        neg_edge_index_batch = data.train_neg_edge_index[:, mask_neg].to(device)
        
        # Map global indices to local indices for negative edges
        neg_edge_index_batch = torch.tensor(
            [[global_to_local[node.item()] for node in neg_edge_index_batch_row]
             for neg_edge_index_batch_row in neg_edge_index_batch],
            dtype=torch.long,
            device=device,
        )
        
        # Forward pass
        out = model(x_batch, pos_edge_index_batch)
        
        # Compute loss
        pos_src, pos_dst = pos_edge_index_batch
        neg_src, neg_dst = neg_edge_index_batch
        
        pos_out_src = out[pos_src]
        pos_out_dst = out[pos_dst]
        neg_out_src = out[neg_src]
        neg_out_dst = out[neg_dst]
        
        # Compute similarity scores
        pos_scores = (pos_out_src * pos_out_dst).sum(dim=-1)
        neg_scores = (neg_out_src * neg_out_dst).sum(dim=-1)
        
        # Compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        loss = pos_loss + neg_loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(mini_batches)  # Return average loss for the mini-batch


# Evaluation Function
def evaluate(model, criterion, data, split='val'):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.train_pos_edge_index)

        if split == 'val':
            pos_edge = data.val_pos_edge_index
            neg_edge = data.val_neg_edge_index
        elif split == 'test':
            pos_edge = data.test_pos_edge_index
            neg_edge = data.test_neg_edge_index
        else:
            raise ValueError("Split must be 'val' or 'test'")

        pos_src = pos_edge[0]
        pos_dst = pos_edge[1]
        neg_src = neg_edge[0]
        neg_dst = neg_edge[1]

        pos_out_src = out[pos_src]
        pos_out_dst = out[pos_dst]
        neg_out_src = out[neg_src]
        neg_out_dst = out[neg_dst]

        # Compute similarity scores
        pos_scores = (pos_out_src * pos_out_dst).sum(dim=-1)
        neg_scores = (neg_out_src * neg_out_dst).sum(dim=-1)

        # Compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
        loss = pos_loss + neg_loss

    return loss.item()

# Recommendation Function
def recommend_similar_patients(model, data, patient_id, top_k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.train_pos_edge_index)
        embeddings = embeddings.cpu()  # Ensure embeddings are on CPU
        patient_embedding = embeddings[patient_id].unsqueeze(0)
        similarities = torch.cosine_similarity(patient_embedding, embeddings)
        recommended_patients = similarities.argsort(descending=True)[1:top_k+1]
        return recommended_patients

# Function to split edges into train, val, test
def split_edges(data, val_ratio=0.05, test_ratio=0.05, random_state=42):
    """
    Split the edges in data.edge_index into train, validation, and test sets.

    Args:
        data (Data): PyTorch Geometric Data object.
        val_ratio (float): Proportion of edges to use for validation.
        test_ratio (float): Proportion of edges to use for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        Data: Data object with train, val, and test edge indices.
    """
    print("Splitting edges into train, validation, and test sets...")
    num_edges = data.edge_index.size(1)
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test

    # Shuffle edge indices with reproducibility
    generator = torch.Generator().manual_seed(random_state)
    indices = torch.randperm(num_edges, generator=generator)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    data.train_pos_edge_index = data.edge_index[:, train_indices]
    data.val_pos_edge_index = data.edge_index[:, val_indices]
    data.test_pos_edge_index = data.edge_index[:, test_indices]

    # Generate negative samples for train, val, and test
    print("Generating negative samples for training...")
    data.train_neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        method='sparse'
    )

    print("Generating negative samples for validation...")
    data.val_neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.size(1),
        method='sparse'
    )

    print("Generating negative samples for testing...")
    data.test_neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.size(1),
        method='sparse'
    )

    return data

def main():
    
    # Filepath to dataset
    filepath = 'datasets/SynDisNet.csv'  # Update this path as needed

    # Set device to CUDA if available, else MPS or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Preprocess data and create Data object with sampling
    sample_size = 64000  # Number of samples to select
    k = 10  # Number of neighbors per node
    data, encoders = preprocess_data(filepath, sample_size=sample_size, k=k)

    # Split edges into train, val, and test
    data = split_edges(data, val_ratio=0.1, test_ratio=0.1, random_state=42)
    print("Edge splits:")
    print(f"Train edges: {data.train_pos_edge_index.size(1)}")
    print(f"Validation edges: {data.val_pos_edge_index.size(1)}")
    print(f"Test edges: {data.test_pos_edge_index.size(1)}")

    # Move data to device
    print("Moving data to device...")
    data = data.to(device)

    # Initialize model and move to device
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 32
    heads = 4
    print("Initializing model...")
    model = PatientGNN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, heads=heads)
    model = model.to(device)
    
    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Alternatively, use a different loss based on your task

    # Training loop with progress bar
    epochs = 100
    best_val_loss = float('inf')
    patience = 20000
    counter = 0
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        loss = train(model, optimizer, criterion, data, device=device, batch_size=64)

        train_losses.append(loss)

        # Evaluate on validation set
        val_loss = evaluate(model, criterion, data, split='val')
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Load the best model
    print("Loading the best model...")
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on test set
    test_loss = evaluate(model, criterion, data, split='test')
    print(f"Test Loss: {test_loss:.4f}")

    # Recommendations
    patient_id = 0  # Example patient ID (ensure it's within the sampled data range)
    if patient_id >= data.num_nodes:
        raise ValueError(f"patient_id {patient_id} is out of range for the dataset with {data.num_nodes} nodes.")

    recommended_patients = recommend_similar_patients(model, data, patient_id, top_k=5)
    print(f"Recommended patients for patient {patient_id}: {recommended_patients.tolist()}")
    # Plot training and validation loss over epochs
    plot_loss(train_losses, val_losses, save_path='loss_plot.png')

if __name__ == "__main__":
    main()
