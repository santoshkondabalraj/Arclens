from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, HttpUrl


class IngestRequest(BaseModel):
    playlist_url: str
    user_id: str


class ChatRequest(BaseModel):
    question: str
    playlist_id: str
    user_id: str
    filter: Optional[dict] = None


class Citation(BaseModel):
    title: str
    channel: str
    timestamp_seconds: int
    video_id: str = ""


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    citations: list[Citation] = []
    refusal: bool = False
    reason: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    last_ingested: Optional[str] = None
    stale: bool = False
    warning: Optional[str] = None
