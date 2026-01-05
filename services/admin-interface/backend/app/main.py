"""
Admin Interface Backend
FastAPI backend for admin interface with authentication and real-time updates
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import os
import traceback
import logging

from app.routers import videos, analysis, training, models, shap, cows
from app.routers import auth, pipeline, health, ml_config, elo_ranking, tutorial
from app.database import init_db, close_db
from app.websocket.handler import ws_manager, websocket_endpoint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events"""
    # Startup: Initialize database tables
    await init_db()
    print("Database initialized")
    yield
    # Shutdown: Close database connections
    await close_db()
    print("Database connections closed")


app = FastAPI(
    title="Lameness Detection Admin API",
    description="Admin interface API for cow lameness detection system with authentication and real-time updates",
    version="2.0.0",
    lifespan=lifespan
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions"""
    error_detail = traceback.format_exc()
    logger.error(f"Unhandled exception: {exc}\n{error_detail}")
    print(f"ERROR: {exc}\n{error_detail}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_detail}
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
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(shap.router, prefix="/api/shap", tags=["shap"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(ml_config.router, prefix="/api/ml-config", tags=["ml-config"])
app.include_router(elo_ranking.router, prefix="/api/elo", tags=["elo-ranking"])
app.include_router(tutorial.router, prefix="/api/tutorial", tags=["tutorial"])
app.include_router(cows.router, prefix="/api/cows", tags=["cows"])


# ============== WEBSOCKET ENDPOINTS ==============

@app.websocket("/api/ws/pipeline")
async def ws_pipeline(websocket: WebSocket):
    """WebSocket endpoint for pipeline status updates"""
    await websocket_endpoint(websocket, "pipeline")


@app.websocket("/api/ws/health")
async def ws_health(websocket: WebSocket):
    """WebSocket endpoint for system health updates"""
    await websocket_endpoint(websocket, "health")


@app.websocket("/api/ws/queue")
async def ws_queue(websocket: WebSocket):
    """WebSocket endpoint for processing queue updates"""
    await websocket_endpoint(websocket, "queue")


@app.websocket("/api/ws/rater")
async def ws_rater(websocket: WebSocket):
    """WebSocket endpoint for rater activity updates"""
    await websocket_endpoint(websocket, "rater")


# Health check
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "admin-backend",
        "websocket_connections": ws_manager.get_connection_count()
    }


# Database seed endpoint (one-time use for initial setup)
@app.post("/api/seed-db")
async def seed_database():
    """
    Seed the database with initial admin user.
    This endpoint should only be called once during initial deployment.
    """
    from app.database import get_db, User
    from app.middleware.auth import get_password_hash
    from sqlalchemy import select
    import uuid
    from datetime import datetime

    async for db in get_db():
        try:
            # Check if admin already exists
            result = await db.execute(
                select(User).where(User.email == "admin@example.com")
            )
            if result.scalar_one_or_none():
                return {"message": "Database already seeded", "status": "skipped"}

            # Create admin user
            admin = User(
                id=uuid.UUID("a0000000-0000-0000-0000-000000000001"),
                email="admin@example.com",
                username="admin",
                password_hash=get_password_hash("adminpass123"),
                role="admin",
                is_active=True,
                rater_tier="gold",
                created_at=datetime.utcnow()
            )
            db.add(admin)

            # Create researcher user
            researcher = User(
                id=uuid.UUID("a0000000-0000-0000-0000-000000000002"),
                email="researcher@example.com",
                username="researcher",
                password_hash=get_password_hash("researcher123"),
                role="researcher",
                is_active=True,
                rater_tier="gold",
                created_at=datetime.utcnow()
            )
            db.add(researcher)

            # Create rater user
            rater = User(
                id=uuid.UUID("a0000000-0000-0000-0000-000000000003"),
                email="rater@example.com",
                username="rater",
                password_hash=get_password_hash("rater123"),
                role="rater",
                is_active=True,
                rater_tier="bronze",
                created_at=datetime.utcnow()
            )
            db.add(rater)

            await db.commit()

            return {
                "message": "Database seeded successfully",
                "status": "success",
                "users_created": [
                    {"email": "admin@example.com", "role": "admin"},
                    {"email": "researcher@example.com", "role": "researcher"},
                    {"email": "rater@example.com", "role": "rater"}
                ]
            }
        except Exception as e:
            await db.rollback()
            return {"message": f"Error seeding database: {str(e)}", "status": "error"}


# Root
@app.get("/")
async def root():
    return {
        "message": "Lameness Detection Admin API",
        "docs": "/docs",
        "version": "2.0.0",
        "features": [
            "Authentication with JWT",
            "Role-based access control (RBAC)",
            "WebSocket real-time updates",
            "Pipeline monitoring",
            "Processing queue management"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
