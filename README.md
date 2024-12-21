# **MetaGraphX: Scalable Meta-Adaptive Graph Neural Network Framework**

## **Overview**
MetaGraphX is a cutting-edge framework for multi-domain graph learning and knowledge representation. This project builds on advanced concepts in Graph Neural Networks (GNNs) and Graph Attention Networks (GATs), introducing a meta-learning-based architecture for scalable and domain-adaptive graph processing. With a focus on real-world applications, MetaGraphX bridges the gap between theoretical advancements and practical implementations in areas such as social network analysis, disease prediction, and semantic reasoning.

---

## **Key Features**
- **Meta-Adaptive Learning:**
  - Dynamic weighting of relationships via multi-head attention mechanisms.
  - Incorporation of domain-specific context for task adaptability.

- **Real-Time Visualization:**
  - Interactive graph rendering with real-time updates powered by D3.js.
  - Visualization of node relationships for intuitive understanding.

- **Applications:**
  - **Social Networks:** Node classification and link prediction.
  - **Healthcare:** Disease risk prediction and patient interaction modeling.
  - **Knowledge Representation:** Enhanced graph-based semantic analysis and recommendation systems.

---

## **Folder Structure**
```plaintext
MetaGraphX/
├── .venv/                       # Virtual environment
├── app/                         # Frontend application
│   ├── __init__.py              # Backend initialization
│   ├── index.html               # Real-time graph visualization (D3.js)
├── datasets/                    # Synthetic datasets
│   ├── Synthetic_Disease_Prediction_Dataset.csv
│   ├── Synthetic_Social_Media_Dataset.csv
├── .gitignore                   # Git ignore file
├── check.py                     # Model testing and debugging
├── meta_gat_model.py            # Core MetaGraphX architecture
├── model.py                     # Generic model utilities
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── training.py                  # Training script
```

---

## **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/gamedevCloudy/meta-gat
   cd meta-gat
   ```

2. **Set Up a Virtual Environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python check.py
   ```

---

## **Usage**

### **1. Train the Model**
Customize training parameters in `training.py` and execute:
```bash
python training.py
```

### **2. Run the Model**
Execute the core model logic via CLI:
```bash
python model.py
```

### **3. Visualize Graphs**
Start the visualization application:

#### Running the API
```bash
fastapi dev app
```

#### Running the Frontend
Open the `app/index.html` using Live Server, OR:
```bash
cd app
python -m http.server 1432
```
Visit [https://localhost:1432](https://localhost:1432/).

---

## **Datasets**

1. **Synthetic Disease Network Dataset (SyndisNet):**
   - Nodes represent patients; edges represent diagnoses or co-morbidities.
   - Applications: Disease prediction, link prediction.

2. **Synthetic Social Profiles Dataset:**
   - Nodes correspond to users; edges represent social interactions.
   - Applications: Node classification, recommendation systems.

---

## **Model Architecture**

### **MetaGAT**
- A meta-learning-enhanced GAT framework for dynamic, multi-domain graph processing.
- Leverages attention mechanisms to focus on task-specific node relationships.

### **Task-Specific Heads**
- **SocialMediaHead:** Multi-class classification for social network graphs.
- **DiseasePredictionHead:** Risk classification for healthcare datasets.

---

## **Research Contributions**
- **Meta-Learning Mechanisms:** Incorporates domain-specific attention for task adaptability.
- **Dynamic Graph Processing:** Handles evolving relationships and multi-relational graphs.
- **Applications:** Demonstrates utility in semantic reasoning, recommendation systems, and healthcare.

For more details, refer to the associated research paper: [Advanced Knowledge Representation using GNNs and GATs](#).

---

## **Future Work**
- Integration with real-world datasets (e.g., electronic health records, large-scale social graphs).
- Optimizations for real-time graph learning and inference.
- Enhancing explainability of attention mechanisms for better decision-making.

---

## **Contributors**
- **Om Umrania** - Lead Developer
- **Collaborators:** Aayush Chaudhary, Yashada Nalawade, Suvarna Patil, Atharva Vyas, Sneha Kanawade

---

## **License**
This project is licensed under the [MIT License](LICENSE).

