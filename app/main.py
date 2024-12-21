import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

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

def create_edge_list_vectorized(df, k=10):
    edges = []
    groups = df.groupby('disease_family').groups

    for group_name, group in groups.items():
        group_indices = group.values
        num_nodes = len(group_indices)
        if num_nodes <= 1:
            continue

        actual_k = min(k, num_nodes - 1)
        shuffled_indices = np.random.permutation(group_indices)
        neighbors = np.tile(shuffled_indices[:actual_k], (num_nodes, 1))

        source_nodes = np.repeat(group_indices, actual_k)
        target_nodes = neighbors.flatten()
        edges.extend(zip(source_nodes, target_nodes))

    edges = np.array(edges)
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    edges = np.unique(edges, axis=0)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

def preprocess_data(filepath, sample_size=50, k=10, random_state=42):
    print("Loading dataset for small sample...")
    df = pd.read_csv(filepath)
    print(f"Original dataset size: {df.shape}")

    # Sample a small subset (50)
    if sample_size < len(df):
        df_sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        df_sampled = df.reset_index(drop=True)
    print(f"Sampled dataset size: {df_sampled.shape}")

    # We will keep track of encoders to map back from numeric codes to names
    encoders = {}

    # Encode categorical columns
    categorical_cols = ['disease_family', 'disease', 'smoking_status', 'cough_type']
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_sampled[col] = encoder.fit_transform(df_sampled[col])
        encoders[col] = encoder

    # Normalize continuous
    continuous_cols = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'BMI', 'heart_rate', 'blood_glucose']
    scaler = StandardScaler()
    for col in continuous_cols:
        df_sampled[col] = scaler.fit_transform(df_sampled[[col]])

    if 'patient_id' in df_sampled.columns:
        feature_cols = [col for col in df_sampled.columns if col != 'patient_id']
        x = torch.tensor(df_sampled[feature_cols].values, dtype=torch.float)
    else:
        x = torch.tensor(df_sampled.values, dtype=torch.float)

    edge_index = create_edge_list_vectorized(df_sampled, k=k)
    data = Data(x=x, edge_index=edge_index)
    return data, df_sampled, encoders

def recommend_similar_patients(model, data, patient_id, top_k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        patient_embedding = embeddings[patient_id].unsqueeze(0)
        similarities = torch.cosine_similarity(patient_embedding, embeddings)
        recommended_patients = similarities.argsort(descending=True)[1: top_k + 1]
    return recommended_patients

app = FastAPI()

model = None
data = None
df_small = None
encoders = {}
recommendations_log = {}
mappings = {}  # Will store string mappings for disease, disease_family, etc.

@app.on_event("startup")
async def startup_event():
    global model, data, df_small, encoders, recommendations_log, mappings
    print("=== Startup: loading data and model ===")

    dataset_path = os.path.join("datasets", "SynDisNet.csv")
    best_model_path = "best_model.pth"

    # Load small sample of data + encoders
    data, df_small, encoders = preprocess_data(dataset_path, sample_size=50, k=10, random_state=42)
    num_nodes = data.num_nodes
    print(f"Data loaded. Number of nodes: {num_nodes}")

    # Build mappings from numeric -> original label (for disease, disease_family)
    # (Only if you originally had those columns in your CSV)
    disease_encoder = encoders["disease"]   # LabelEncoder for 'disease'
    disease_mapping = { i: disease_encoder.classes_[i] for i in range(len(disease_encoder.classes_)) }

    disease_family_encoder = encoders["disease_family"]
    disease_family_mapping = { i: disease_family_encoder.classes_[i] for i in range(len(disease_family_encoder.classes_)) }

    mappings = {
        "disease": disease_mapping,
        "disease_family": disease_family_mapping
    }

    # Initialize model
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 32
    heads = 4
    model = PatientGNN(in_channels, hidden_channels, out_channels, heads)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print("Model loaded from best_model.pth.")

    # Precompute recs for first 50 nodes
    limited_nodes = min(50, num_nodes)
    for pid in range(limited_nodes):
        recs = recommend_similar_patients(model, data, pid, top_k=5)

        # Retrieve the 'patient_id, disease_family, disease' for this source node
        source_row = df_small.iloc[pid]
        source_features = {
            "patient_id": source_row["patient_id"],
            "disease_family": int(source_row["disease_family"]),
            "disease": int(source_row["disease"])
        }

        # Build recommended features
        recs_list = []
        for rec_id in recs.tolist():
            rec_row = df_small.iloc[rec_id]
            recs_list.append({
                "patient_id": rec_row["patient_id"],
                "disease_family": int(rec_row["disease_family"]),
                "disease": int(rec_row["disease"])
            })

        recommendations_log[pid] = {
            "source": source_features,
            "recommendations": recs_list
        }

    print("Precomputed recommendations, plus built string mappings.")
    print("=== Startup complete ===")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Send everything: the recs plus the mapping from numeric->string
        payload = {
            "event": "all_recommendations",
            "data": recommendations_log,
            "mappings": mappings
        }
        await websocket.send_json(payload)

        while True:
            msg = await websocket.receive_text()
            # Echo logic
            await websocket.send_text(f"Echo from server: {msg}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
