"""
Training endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime
import json

router = APIRouter()

TRAINING_DIR = Path("/app/data/training")
RESULTS_DIR = Path("/app/data/results")


class LabelRequest(BaseModel):
    label: int  # 0 = sound, 1 = lame
    confidence: Optional[str] = "certain"  # certain, uncertain


@router.post("/videos/{video_id}/label")
async def label_video(video_id: str, label_request: LabelRequest):
    """Submit label for a video"""
    # Store label
    labels_dir = TRAINING_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_file = labels_dir / f"{video_id}_label.json"
    
    label_data = {
        "video_id": video_id,
        "label": label_request.label,
        "confidence": label_request.confidence,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(label_file, "w") as f:
        json.dump(label_data, f)
    
    return {
        "video_id": video_id,
        "label": label_request.label,
        "status": "saved"
    }


@router.get("/queue")
async def get_training_queue():
    """Get videos that need labeling (active learning queue)"""
    # For now, return videos with analysis but no label
    videos = []
    
    fusion_dir = RESULTS_DIR / "fusion"
    labels_dir = TRAINING_DIR / "labels"
    
    for fusion_file in fusion_dir.glob("*_fusion.json"):
        video_id = fusion_file.stem.replace("_fusion", "")
        label_file = labels_dir / f"{video_id}_label.json"
        
        if not label_file.exists():
            with open(fusion_file) as f:
                fusion_data = json.load(f)
                fusion_result = fusion_data.get("fusion_result", {})
                
                # Prioritize uncertain predictions
                prob = fusion_result.get("final_probability", 0.5)
                uncertainty = abs(0.5 - prob)  # Lower uncertainty = more uncertain
                
                videos.append({
                    "video_id": video_id,
                    "predicted_probability": prob,
                    "uncertainty": uncertainty
                })
    
    # Sort by uncertainty (most uncertain first)
    videos.sort(key=lambda x: x["uncertainty"])
    
    return {
        "videos": videos[:50],  # Top 50 most uncertain
        "total": len(videos)
    }


@router.get("/stats")
async def get_training_stats():
    """Get training dataset statistics"""
    labels_dir = TRAINING_DIR / "labels"
    
    total_labels = 0
    sound_count = 0
    lame_count = 0
    
    for label_file in labels_dir.glob("*_label.json"):
        with open(label_file) as f:
            label_data = json.load(f)
            total_labels += 1
            if label_data.get("label") == 0:
                sound_count += 1
            elif label_data.get("label") == 1:
                lame_count += 1
    
    return {
        "total_labels": total_labels,
        "sound_count": sound_count,
        "lame_count": lame_count,
        "balance_ratio": sound_count / lame_count if lame_count > 0 else 0
    }


@router.post("/yolo/start")
async def start_yolo_training():
    """Trigger YOLO training"""
    # TODO: Implement training trigger via NATS
    return {
        "status": "training_requested",
        "message": "YOLO training will start when sufficient data is available"
    }


@router.post("/ml/start")
async def start_ml_training():
    """Trigger ML training"""
    # TODO: Implement training trigger via NATS
    return {
        "status": "training_requested",
        "message": "ML training will start when sufficient data is available"
    }


@router.get("/status")
async def get_training_status():
    """Get training job status"""
    # TODO: Implement training status tracking
    return {
        "active_jobs": [],
        "recent_completions": []
    }

