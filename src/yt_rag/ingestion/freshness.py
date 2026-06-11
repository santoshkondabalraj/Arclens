from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

from yt_rag.config import settings


def _meta_path(playlist_id: str) -> pathlib.Path:
    return pathlib.Path(settings.ingestion_meta_dir) / f"{playlist_id}.json"


def record_ingestion(playlist_id: str, user_id: str, playlist_title: str = "") -> None:
    """Write ingestion timestamp to disk."""
    path = _meta_path(playlist_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "playlist_id": playlist_id,
        "playlist_title": playlist_title or playlist_id,
        "user_id": user_id,
        "last_ingested": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_ingestion_meta(playlist_id: str) -> dict:
    """Load stored ingestion metadata; returns empty dict if not found."""
    path = _meta_path(playlist_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_stale(playlist_id: str) -> bool:
    """Return True if the playlist was last ingested more than staleness_days ago."""
    meta = load_ingestion_meta(playlist_id)
    if not meta.get("last_ingested"):
        return True
    last = datetime.fromisoformat(meta["last_ingested"])
    age_days = (datetime.now(timezone.utc) - last).days
    return age_days > settings.staleness_days


def list_all_ingested() -> list[dict]:
    """Return all ingestion records sorted by most recently ingested."""
    directory = pathlib.Path(settings.ingestion_meta_dir)
    if not directory.exists():
        return []
    records = []
    for path in directory.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    records.sort(key=lambda r: r.get("last_ingested", ""), reverse=True)
    return records


def staleness_warning(playlist_id: str) -> str | None:
    """Return a warning string if stale, else None."""
    meta = load_ingestion_meta(playlist_id)
    if not meta.get("last_ingested"):
        return f"Playlist '{playlist_id}' has never been ingested."
    last = datetime.fromisoformat(meta["last_ingested"])
    age_days = (datetime.now(timezone.utc) - last).days
    if age_days > settings.staleness_days:
        return (
            f"Warning: playlist '{playlist_id}' was last ingested {age_days} days ago "
            f"(threshold: {settings.staleness_days} days). Consider refreshing."
        )
    return None
