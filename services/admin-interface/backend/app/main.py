"""
Admin Interface Backend
FastAPI backend for admin interface
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from app.routers import videos, analysis, training, models, shap

app = FastAPI(
    title="Lameness Detection Admin API",
    description="Admin interface API for cow lameness detection system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(shap.router, prefix="/api/shap", tags=["shap"])

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "admin-backend"}

# Root
@app.get("/")
async def root():
    return {
        "message": "Lameness Detection Admin API",
        "docs": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

