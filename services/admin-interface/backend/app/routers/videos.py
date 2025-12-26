"""
Video management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import json
import uuid
from datetime import datetime

router = APIRouter()

VIDEOS_DIR = Path("/app/data/videos")
RESULTS_DIR = Path("/app/data/results")


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    file_path: str
    file_size: int
    uploaded_at: str
    status: str


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file"""
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate video ID
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{file_ext}"
    file_path = VIDEOS_DIR / filename
    
    # Save file
    file_size = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                file_size += len(chunk)
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{video_id}")
async def get_video(video_id: str):
    """Get video information"""
    # Find video file
    video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_path = video_files[0]
    file_size = file_path.stat().st_size
    
    # Check for analysis results
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    has_analysis = fusion_file.exists()
    
    return {
        "video_id": video_id,
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_size": file_size,
        "has_analysis": has_analysis,
        "status": "analyzed" if has_analysis else "uploaded"
    }


@router.get("")
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all videos"""
    videos = []
    
    for video_file in list(VIDEOS_DIR.glob("*.*"))[:skip+limit]:
        if video_file.is_file():
            video_id = video_file.stem.split("_")[0]  # Extract ID from filename
            fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
            
            videos.append({
                "video_id": video_id,
                "filename": video_file.name,
                "file_size": video_file.stat().st_size,
                "has_analysis": fusion_file.exists()
            })
    
    return {
        "videos": videos[skip:skip+limit],
        "total": len(videos),
        "skip": skip,
        "limit": limit
    }

