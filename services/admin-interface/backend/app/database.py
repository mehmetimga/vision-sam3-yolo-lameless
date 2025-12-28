"""
Database connection and session management
Async SQLAlchemy setup for PostgreSQL
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Boolean, DateTime, Float, Integer, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

# Database URL from environment
_raw_url = os.getenv(
    "POSTGRES_URL",
    os.getenv("DATABASE_URL", "postgresql://lameness_user:lameness_pass@postgres:5432/lameness_db")
)
# Convert to asyncpg URL
DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


# ============== MODELS ==============

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="rater")
    is_active = Column(Boolean, default=True)
    rater_tier = Column(String(10), nullable=True)  # gold, silver, bronze
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("role IN ('admin', 'researcher', 'rater')", name="valid_role"),
    )


class Session(Base):
    """Session model for token management"""
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessingJob(Base):
    """Processing job model for queue management"""
    __tablename__ = "processing_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String(100), nullable=False, index=True)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed, cancelled
    priority = Column(Integer, default=0)
    progress = Column(Float, default=0.0)
    current_pipeline = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')",
            name="valid_job_status"
        ),
    )


class GoldTask(Base):
    """Gold task model for rater validation"""
    __tablename__ = "gold_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id_1 = Column(String(100), nullable=False)
    video_id_2 = Column(String(100), nullable=False)
    correct_winner = Column(Integer, nullable=False)  # 1 or 2
    difficulty = Column(String(10), default="medium")  # easy, medium, hard
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("correct_winner IN (1, 2)", name="valid_winner"),
        CheckConstraint("difficulty IN ('easy', 'medium', 'hard')", name="valid_difficulty"),
    )


class RaterStats(Base):
    """Rater statistics model"""
    __tablename__ = "rater_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    total_comparisons = Column(Integer, default=0)
    gold_task_accuracy = Column(Float, default=0.0)
    agreement_rate = Column(Float, default=0.0)
    weight = Column(Float, default=1.0)
    tier = Column(String(10), default="bronze")  # gold, silver, bronze
    last_activity = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ServiceHeartbeat(Base):
    """Service heartbeat for monitoring"""
    __tablename__ = "service_heartbeats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(50), unique=True, nullable=False, index=True)
    status = Column(String(20), default="unknown")  # healthy, degraded, down, unknown
    last_heartbeat = Column(DateTime, nullable=True)
    active_jobs = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    extra_data = Column(Text, nullable=True)  # JSON string for extra info


# ============== DATABASE FUNCTIONS ==============

async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()
