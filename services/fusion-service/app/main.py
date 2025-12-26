"""
Fusion Service
Combines predictions from all pipelines to generate final lameness score
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import yaml
from shared.utils.nats_client import NATSClient
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier


class FusionService:
    """Fusion service to combine all pipeline predictions"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Model storage
        self.models_dir = Path("/app/shared/models/fusion")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load fusion model if available
        self.fusion_model = None
        self._load_fusion_model()
        
        # Directories
        self.results_dir = Path("/app/data/results/fusion")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for pipeline results
        self.pipeline_results = {}
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_fusion_model(self):
        """Load fusion model if available"""
        fusion_file = self.models_dir / "fusion_model.pkl"
        if fusion_file.exists():
            try:
                import pickle
                with open(fusion_file, "rb") as f:
                    self.fusion_model = pickle.load(f)
                print(f"Loaded fusion model: {fusion_file}")
            except Exception as e:
                print(f"Failed to load fusion model: {e}")
    
    def collect_pipeline_predictions(self, video_id: str) -> Dict[str, Any]:
        """Collect predictions from all pipelines"""
        predictions = {}
        
        # ML pipeline predictions
        ml_file = Path(f"/app/data/results/ml/{video_id}_ml.json")
        if ml_file.exists():
            with open(ml_file) as f:
                ml_data = json.load(f)
                if "predictions" in ml_data:
                    predictions["ml"] = ml_data["predictions"]
        
        # YOLO features
        yolo_file = Path(f"/app/data/results/yolo/{video_id}_yolo.json")
        if yolo_file.exists():
            with open(yolo_file) as f:
                yolo_data = json.load(f)
                if "features" in yolo_data:
                    predictions["yolo"] = yolo_data["features"]
        
        # SAM3 features
        sam3_file = Path(f"/app/data/results/sam3/{video_id}_sam3.json")
        if sam3_file.exists():
            with open(sam3_file) as f:
                sam3_data = json.load(f)
                if "features" in sam3_data:
                    predictions["sam3"] = sam3_data["features"]
        
        # DINOv3 features
        dinov3_file = Path(f"/app/data/results/dinov3/{video_id}_dinov3.json")
        if dinov3_file.exists():
            with open(dinov3_file) as f:
                dinov3_data = json.load(f)
                predictions["dinov3"] = {
                    "neighbor_evidence": dinov3_data.get("neighbor_evidence", 0.5),
                    "similar_cases": dinov3_data.get("similar_cases", [])
                }
        
        # T-LEAP features (if available)
        tleap_file = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
        if tleap_file.exists():
            with open(tleap_file) as f:
                tleap_data = json.load(f)
                if "locomotion_traits" in tleap_data:
                    predictions["tleap"] = tleap_data["locomotion_traits"]
        
        return predictions
    
    def fuse_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse predictions from all pipelines"""
        fusion_features = []
        feature_names = []
        
        # ML pipeline ensemble prediction
        if "ml" in predictions and "ensemble" in predictions["ml"]:
            ml_prob = predictions["ml"]["ensemble"]["probability"]
            fusion_features.append(ml_prob)
            feature_names.append("ml_ensemble_prob")
            
            # Individual model predictions
            for model_name in ["catboost", "xgboost", "lightgbm"]:
                if model_name in predictions["ml"]:
                    fusion_features.append(predictions["ml"][model_name]["probability"])
                    feature_names.append(f"ml_{model_name}_prob")
        
        # YOLO confidence
        if "yolo" in predictions:
            yolo_features = predictions["yolo"]
            fusion_features.append(yolo_features.get("avg_confidence", 0.5))
            feature_names.append("yolo_confidence")
        
        # SAM3 area ratio
        if "sam3" in predictions:
            sam3_features = predictions["sam3"]
            fusion_features.append(sam3_features.get("avg_area_ratio", 0.5))
            feature_names.append("sam3_area_ratio")
        
        # DINOv3 neighbor evidence
        if "dinov3" in predictions:
            fusion_features.append(predictions["dinov3"].get("neighbor_evidence", 0.5))
            feature_names.append("dinov3_neighbor_evidence")
        
        # T-LEAP asymmetry (if available)
        if "tleap" in predictions:
            tleap = predictions["tleap"]
            fusion_features.append(tleap.get("asymmetry_score", 0.5))
            feature_names.append("tleap_asymmetry")
        
        # If no features, use default
        if not fusion_features:
            fusion_features = [0.5] * 5
            feature_names = [f"default_{i}" for i in range(5)]
        
        # Use fusion model if available, otherwise weighted average
        if self.fusion_model:
            try:
                features_array = np.array(fusion_features).reshape(1, -1)
                fusion_prob = self.fusion_model.predict_proba(features_array)[0, 1]
            except Exception as e:
                print(f"Fusion model prediction error: {e}")
                fusion_prob = np.mean(fusion_features)
        else:
            # Simple weighted average
            weights = {
                "ml_ensemble_prob": 0.5,
                "yolo_confidence": 0.15,
                "sam3_area_ratio": 0.1,
                "dinov3_neighbor_evidence": 0.15,
                "tleap_asymmetry": 0.1
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            
            for i, name in enumerate(feature_names):
                weight = weights.get(name, 0.1)
                weighted_sum += fusion_features[i] * weight
                total_weight += weight
            
            fusion_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return {
            "final_probability": float(fusion_prob),
            "final_prediction": int(fusion_prob > 0.5),
            "fusion_features": fusion_features,
            "feature_names": feature_names,
            "pipeline_contributions": {
                "ml": predictions.get("ml", {}).get("ensemble", {}).get("probability", 0.5) if "ml" in predictions else None,
                "yolo": predictions.get("yolo", {}).get("avg_confidence", 0.5) if "yolo" in predictions else None,
                "sam3": predictions.get("sam3", {}).get("avg_area_ratio", 0.5) if "sam3" in predictions else None,
                "dinov3": predictions.get("dinov3", {}).get("neighbor_evidence", 0.5) if "dinov3" in predictions else None,
                "tleap": predictions.get("tleap", {}).get("asymmetry_score", 0.5) if "tleap" in predictions else None
            }
        }
    
    async def process_video(self, video_data: dict):
        """Process video through fusion service"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"Fusion service processing video {video_id}")
        
        try:
            # Collect predictions from all pipelines
            predictions = self.collect_pipeline_predictions(video_id)
            
            if not predictions:
                print(f"No pipeline predictions found for {video_id}")
                return
            
            # Fuse predictions
            fusion_result = self.fuse_predictions(predictions)
            
            # Save results
            results = {
                "video_id": video_id,
                "fusion_result": fusion_result,
                "pipeline_predictions": predictions,
                "timestamp": video_data.get("timestamp", "")
            }
            
            results_file = self.results_dir / f"{video_id}_fusion.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish analysis complete event
            analysis_result = {
                "video_id": video_id,
                "final_probability": fusion_result["final_probability"],
                "final_prediction": fusion_result["final_prediction"],
                "results_path": str(results_file),
                "pipeline_contributions": fusion_result["pipeline_contributions"]
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["analysis_complete"],
                analysis_result
            )
            
            print(f"Fusion service completed for {video_id}")
            
        except Exception as e:
            print(f"Error in fusion service for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the fusion service"""
        await self.nats_client.connect()
        
        # Subscribe to ML pipeline results (last in sequence)
        subject = self.config["nats"]["subjects"]["pipeline_ml"]
        print(f"Fusion service subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("Fusion service started. Waiting for pipeline results...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    service = FusionService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())

