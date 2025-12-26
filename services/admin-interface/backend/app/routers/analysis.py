"""
Analysis endpoints
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter()

RESULTS_DIR = Path("/app/data/results")


@router.get("/{video_id}")
async def get_analysis(video_id: str):
    """Get complete analysis results for a video"""
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    
    if not fusion_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    with open(fusion_file) as f:
        fusion_data = json.load(f)
    
    # Load individual pipeline results
    pipeline_results = {}
    
    for pipeline in ["yolo", "sam3", "dinov3", "ml", "tleap"]:
        result_file = RESULTS_DIR / pipeline / f"{video_id}_{pipeline}.json"
        if result_file.exists():
            with open(result_file) as f:
                pipeline_results[pipeline] = json.load(f)
    
    return {
        "video_id": video_id,
        "fusion": fusion_data.get("fusion_result", {}),
        "pipelines": pipeline_results
    }


@router.get("/{video_id}/summary")
async def get_analysis_summary(video_id: str):
    """Get analysis summary"""
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    
    if not fusion_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    with open(fusion_file) as f:
        fusion_data = json.load(f)
    
    fusion_result = fusion_data.get("fusion_result", {})
    
    return {
        "video_id": video_id,
        "final_probability": fusion_result.get("final_probability", 0.5),
        "final_prediction": fusion_result.get("final_prediction", 0),
        "prediction_label": "lame" if fusion_result.get("final_prediction", 0) == 1 else "sound",
        "pipeline_contributions": fusion_result.get("pipeline_contributions", {})
    }

