"""
Graph Transformer Pipeline Service

True Graphormer-style architecture for cow lameness detection.
Implements full self-attention with graph structural biases.

Key Features:
- Centrality encoding (node degree importance)
- Spatial encoding (shortest path distances)
- Temporal encoding (time-based positional encoding)
- Edge encoding (similarity, temporal distance)
- Virtual node for graph-level aggregation

NATS Subscriptions:
- pipeline.dinov3: Process after embeddings are computed

NATS Publications:
- pipeline.graph_transformer: Graph transformer results
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml
import torch
from torch_geometric.data import Data

from shared.utils.nats_client import NATSClient
from app.model import CowLamenessGraphormer, GraphormerGraphBuilder


class GraphTransformerPipeline:
    """Graph Transformer Pipeline Service"""

    # Feature dimensions (same as GNN pipeline)
    POSE_FEATURES = 10
    SILHOUETTE_FEATURES = 5
    EMBEDDING_DIM = 32
    META_FEATURES = 3

    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Graph builder
        self.graph_builder = GraphormerGraphBuilder(k_neighbors=5)

        # Model
        input_dim = self.POSE_FEATURES + self.SILHOUETTE_FEATURES + self.EMBEDDING_DIM + self.META_FEATURES

        self.model = CowLamenessGraphormer(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=6,
            num_heads=8,
            ffn_dim=512,
            edge_dim=3,
            dropout=0.1,
            max_degree=50,
            max_spd=10,
            use_virtual_node=True,
            use_temporal=True
        ).to(self.device)

        # Load weights
        self.model_path = Path("/app/shared/models/graph_transformer")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_model()

        # Results
        self.results_dir = Path("/app/data/results/graph_transformer")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _load_model(self):
        """Load model weights if available"""
        weights_path = self.model_path / "graphormer_lameness.pt"

        if weights_path.exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"Loaded Graphormer weights from {weights_path}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        else:
            print("No pretrained Graphormer weights. Using random initialization.")

        self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Graphormer parameters: {num_params:,}")

    def extract_node_features(self, video_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract features from pipeline results"""
        features = {}

        # T-LEAP pose features
        tleap_path = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
        if tleap_path.exists():
            with open(tleap_path) as f:
                tleap_data = json.load(f)

            loco = tleap_data.get("locomotion_features", {})
            features["pose"] = np.array([
                loco.get("back_arch_mean", 0),
                loco.get("back_arch_std", 0),
                loco.get("head_bob_magnitude", 0),
                loco.get("head_bob_frequency", 0),
                loco.get("front_leg_asymmetry", 0),
                loco.get("rear_leg_asymmetry", 0),
                loco.get("lameness_score", 0.5),
                loco.get("stride_fl_mean", 0),
                loco.get("stride_fr_mean", 0),
                loco.get("steadiness_score", 0.5)
            ], dtype=np.float32)
        else:
            features["pose"] = np.zeros(self.POSE_FEATURES, dtype=np.float32)

        # SAM3/YOLO silhouette
        sam3_path = Path(f"/app/data/results/sam3/{video_id}_sam3.json")
        yolo_path = Path(f"/app/data/results/yolo/{video_id}_yolo.json")

        silhouette = np.zeros(self.SILHOUETTE_FEATURES, dtype=np.float32)

        if sam3_path.exists():
            with open(sam3_path) as f:
                sam3_data = json.load(f)
            feats = sam3_data.get("features", {})
            silhouette[0] = feats.get("avg_area_ratio", 0)
            silhouette[1] = feats.get("avg_circularity", 0)
            silhouette[2] = feats.get("avg_aspect_ratio", 1)

        if yolo_path.exists():
            with open(yolo_path) as f:
                yolo_data = json.load(f)
            feats = yolo_data.get("features", {})
            silhouette[3] = feats.get("avg_confidence", 0.5)
            silhouette[4] = feats.get("position_stability", 0.5)

        features["silhouette"] = silhouette

        # DINOv3 embeddings
        dinov3_path = Path(f"/app/data/results/dinov3/{video_id}_dinov3.json")
        if dinov3_path.exists():
            with open(dinov3_path) as f:
                dinov3_data = json.load(f)

            embedding = dinov3_data.get("embedding", [])
            if len(embedding) > 0:
                embedding = np.array(embedding, dtype=np.float32)
                if len(embedding) > self.EMBEDDING_DIM:
                    embedding = embedding[:self.EMBEDDING_DIM]
                elif len(embedding) < self.EMBEDDING_DIM:
                    embedding = np.pad(embedding, (0, self.EMBEDDING_DIM - len(embedding)))
                features["embedding"] = embedding
            else:
                features["embedding"] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        else:
            features["embedding"] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        # Metadata
        features["meta"] = np.array([0.5, 1.0, 0.5], dtype=np.float32)

        return features

    def collect_graph_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Collect features from all videos"""
        node_features_list = []
        embeddings_list = []
        video_ids = []

        tleap_dir = Path("/app/data/results/tleap")
        if tleap_dir.exists():
            for result_file in tleap_dir.glob("*_tleap.json"):
                video_id = result_file.stem.replace("_tleap", "")

                features = self.extract_node_features(video_id)
                if features is not None:
                    node_feat = np.concatenate([
                        features["pose"],
                        features["silhouette"],
                        features["embedding"],
                        features["meta"]
                    ])

                    node_features_list.append(node_feat)
                    embeddings_list.append(features["embedding"])
                    video_ids.append(video_id)

        if not node_features_list:
            return None, None, []

        return np.stack(node_features_list), np.stack(embeddings_list), video_ids

    async def process_video(self, video_data: dict):
        """Process video through Graph Transformer"""
        video_id = video_data.get("video_id")
        if not video_id:
            return

        print(f"Graph Transformer processing video {video_id}")

        try:
            # Collect graph data
            node_features, embeddings, video_ids = self.collect_graph_data()

            if node_features is None or len(video_ids) == 0:
                print(f"  No video features available")
                return

            # Add current video if not in graph
            if video_id not in video_ids:
                features = self.extract_node_features(video_id)
                if features is None:
                    print(f"  Could not extract features for {video_id}")
                    return

                new_node = np.concatenate([
                    features["pose"],
                    features["silhouette"],
                    features["embedding"],
                    features["meta"]
                ])

                node_features = np.vstack([node_features, new_node])
                embeddings = np.vstack([embeddings, features["embedding"]])
                video_ids.append(video_id)

            target_idx = video_ids.index(video_id)

            print(f"  Graph: {len(video_ids)} nodes")

            # Build graph
            graph = self.graph_builder.build_graph(
                node_features=torch.tensor(node_features, dtype=torch.float32),
                embeddings=torch.tensor(embeddings, dtype=torch.float32)
            )
            graph = graph.to(self.device)

            # Predict with uncertainty
            mean_pred, std_pred = self.model.predict_with_uncertainty(graph, n_samples=10)

            # Get graph-level prediction
            severity_score = float(mean_pred[0, 0].cpu().numpy())
            uncertainty = float(std_pred[0, 0].cpu().numpy())

            # Get node-level predictions for target
            with torch.no_grad():
                result = self.model(graph, return_attention=True)
                node_preds = result['node_pred'].cpu().numpy()
                target_node_score = float(node_preds[target_idx, 0])

            # Get attention insights if available
            attention_info = {}
            if 'attention_weights' in result:
                # Get attention from last layer to target node
                last_attn = result['attention_weights'][-1]  # (num_heads, N, N)
                # Average attention to target node
                attn_to_target = last_attn[:, :, target_idx].mean(dim=0).cpu().numpy()
                top_attending = np.argsort(attn_to_target)[-5:][::-1]
                attention_info = {
                    "top_attending_nodes": [
                        {"video_id": video_ids[i], "attention": float(attn_to_target[i])}
                        for i in top_attending if i != target_idx
                    ]
                }

            # Save results
            results = {
                "video_id": video_id,
                "pipeline": "graph_transformer",
                "model": "CowLamenessGraphormer",
                "graph_prediction": severity_score,
                "node_prediction": target_node_score,
                "uncertainty": uncertainty,
                "prediction": int(severity_score > 0.5),
                "confidence": 1.0 - uncertainty,
                "graph_info": {
                    "num_nodes": len(video_ids),
                    "num_edges": graph.edge_index.shape[1],
                    "num_layers": self.model.num_layers,
                    "num_heads": self.model.num_heads,
                    "hidden_dim": self.model.hidden_dim
                },
                "attention_info": attention_info
            }

            results_file = self.results_dir / f"{video_id}_graph_transformer.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            # Publish results
            await self.nats_client.publish(
                "pipeline.graph_transformer",
                {
                    "video_id": video_id,
                    "pipeline": "graph_transformer",
                    "results_path": str(results_file),
                    "severity_score": severity_score,
                    "uncertainty": uncertainty
                }
            )

            print(f"  Graphormer completed: graph={severity_score:.3f}, node={target_node_score:.3f}, "
                  f"uncertainty={uncertainty:.3f}")

        except Exception as e:
            print(f"  Error in Graph Transformer: {e}")
            import traceback
            traceback.print_exc()

    async def start(self):
        """Start the service"""
        await self.nats_client.connect()

        # Subscribe to DINOv3 results
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_dinov3", "pipeline.dinov3"
        )
        print(f"Graph Transformer subscribing to: {subject}")

        await self.nats_client.subscribe(subject, self.process_video)

        print("=" * 60)
        print("Graph Transformer Pipeline Service Started")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: CowLamenessGraphormer")
        print(f"  - Layers: {self.model.num_layers}")
        print(f"  - Attention heads: {self.model.num_heads}")
        print(f"  - Hidden dim: {self.model.hidden_dim}")
        print(f"  - Encodings: Centrality, Spatial, Temporal, Edge")
        print(f"  - Virtual node: enabled")
        print(f"k-neighbors: {self.graph_builder.k_neighbors}")
        print("=" * 60)

        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = GraphTransformerPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())
