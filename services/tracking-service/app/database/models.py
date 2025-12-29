"""
Database models for cow tracking and Re-ID.

Provides SQLAlchemy models for:
- CowIdentity: Persistent cow identities
- TrackHistory: Track records per video
- LamenessRecord: Lameness observations per cow over time
"""
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class CowIdentityDB(Base):
    """
    Persistent cow identity record.

    Stores metadata about known cows. The actual embedding is stored
    in Qdrant for efficient similarity search.
    """
    __tablename__ = "cow_identities"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    cow_id = Column(String(100), unique=True, nullable=False, index=True)
    tag_number = Column(String(50), nullable=True)
    total_sightings = Column(Integer, default=0)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    embedding_version = Column(String(20), default="dinov3-base")
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    track_history = relationship("TrackHistoryDB", back_populates="cow_identity")
    lameness_records = relationship("LamenessRecordDB", back_populates="cow_identity")

    def __repr__(self):
        return f"<CowIdentity {self.cow_id} (sightings={self.total_sightings})>"


class TrackHistoryDB(Base):
    """
    Track record for a cow in a specific video.

    Links tracks detected in videos to their corresponding cow identities.
    """
    __tablename__ = "track_history"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id = Column(String(100), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    cow_id = Column(PGUUID(as_uuid=True), ForeignKey("cow_identities.id"), nullable=True, index=True)
    reid_confidence = Column(Float, nullable=True)
    start_frame = Column(Integer, nullable=True)
    end_frame = Column(Integer, nullable=True)
    total_frames = Column(Integer, nullable=True)
    avg_confidence = Column(Float, nullable=True)
    track_embedding = Column(Text, nullable=True)  # JSON serialized
    created_at = Column(DateTime, default=datetime.utcnow)

    # Unique constraint on video_id + track_id
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )

    # Relationships
    cow_identity = relationship("CowIdentityDB", back_populates="track_history")

    def __repr__(self):
        return f"<TrackHistory video={self.video_id} track={self.track_id}>"


class LamenessRecordDB(Base):
    """
    Lameness observation record for a cow.

    Stores lameness predictions and scores over time, enabling
    longitudinal analysis of cow health.
    """
    __tablename__ = "lameness_records"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    cow_id = Column(PGUUID(as_uuid=True), ForeignKey("cow_identities.id"), nullable=False, index=True)
    video_id = Column(String(100), nullable=False, index=True)
    observation_date = Column(DateTime, default=datetime.utcnow)

    # Prediction scores from different pipelines
    fusion_score = Column(Float, nullable=True)
    tleap_score = Column(Float, nullable=True)
    tcn_score = Column(Float, nullable=True)
    transformer_score = Column(Float, nullable=True)
    gnn_score = Column(Float, nullable=True)
    ml_ensemble_score = Column(Float, nullable=True)

    # Final prediction
    is_lame = Column(Boolean, nullable=True)
    confidence = Column(Float, nullable=True)
    severity_level = Column(String(20), nullable=True)  # "healthy", "mild", "moderate", "severe"

    # Human validation
    human_validated = Column(Boolean, default=False)
    human_label = Column(Boolean, nullable=True)
    validator_id = Column(PGUUID(as_uuid=True), nullable=True)
    validation_date = Column(DateTime, nullable=True)

    # Relationships
    cow_identity = relationship("CowIdentityDB", back_populates="lameness_records")

    def __repr__(self):
        return f"<LamenessRecord cow={self.cow_id} score={self.fusion_score}>"


# SQL for creating tables (for reference/migrations)
CREATE_TABLES_SQL = """
-- Cow Identities Table
CREATE TABLE IF NOT EXISTS cow_identities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cow_id VARCHAR(100) UNIQUE NOT NULL,
    tag_number VARCHAR(50),
    total_sightings INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding_version VARCHAR(20) DEFAULT 'dinov3-base',
    notes TEXT,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX IF NOT EXISTS idx_cow_identities_cow_id ON cow_identities(cow_id);

-- Track History Table
CREATE TABLE IF NOT EXISTS track_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(100) NOT NULL,
    track_id INTEGER NOT NULL,
    cow_id UUID REFERENCES cow_identities(id),
    reid_confidence FLOAT,
    start_frame INTEGER,
    end_frame INTEGER,
    total_frames INTEGER,
    avg_confidence FLOAT,
    track_embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(video_id, track_id)
);

CREATE INDEX IF NOT EXISTS idx_track_history_video_id ON track_history(video_id);
CREATE INDEX IF NOT EXISTS idx_track_history_cow_id ON track_history(cow_id);

-- Lameness Records Table
CREATE TABLE IF NOT EXISTS lameness_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cow_id UUID NOT NULL REFERENCES cow_identities(id),
    video_id VARCHAR(100) NOT NULL,
    observation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Pipeline scores
    fusion_score FLOAT,
    tleap_score FLOAT,
    tcn_score FLOAT,
    transformer_score FLOAT,
    gnn_score FLOAT,
    ml_ensemble_score FLOAT,

    -- Prediction
    is_lame BOOLEAN,
    confidence FLOAT,
    severity_level VARCHAR(20),

    -- Human validation
    human_validated BOOLEAN DEFAULT false,
    human_label BOOLEAN,
    validator_id UUID,
    validation_date TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lameness_records_cow_id ON lameness_records(cow_id);
CREATE INDEX IF NOT EXISTS idx_lameness_records_video_id ON lameness_records(video_id);
CREATE INDEX IF NOT EXISTS idx_lameness_records_date ON lameness_records(observation_date);
"""
