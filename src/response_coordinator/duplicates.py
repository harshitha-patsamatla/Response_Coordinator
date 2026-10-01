"""Time-, distance-, and semantic-similarity duplicate matching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from math import asin, cos, radians, sin, sqrt

from .config import DEDUP_RADIUS_KM, DEDUP_SIMILARITY, DEDUP_WINDOW_MINUTES


class SemanticSimilarity:
    def __init__(self) -> None:
        self._model = None
        self.backend = "sequence-ratio-fallback"

    def score(self, left: str, right: str) -> float:
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                self.backend = "multilingual-sentence-embeddings"
            vectors = self._model.encode([left, right], normalize_embeddings=True)
            return float(vectors[0] @ vectors[1])
        except (ImportError, OSError, RuntimeError, ValueError):
            return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    value = sin(delta_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lon / 2) ** 2
    return 2 * radius * asin(sqrt(value))


def find_duplicate(
    report_text: str,
    lat: float | None,
    lon: float | None,
    existing: list[dict],
    similarity: SemanticSimilarity,
    now: datetime | None = None,
) -> dict | None:
    if lat is None or lon is None:
        return None
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(minutes=DEDUP_WINDOW_MINUTES)
    matches = []
    for record in existing:
        incident = record["incident"]
        created_at = incident.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at < cutoff or incident.location.lat is None or incident.location.lon is None:
            continue
        distance = haversine_km(lat, lon, incident.location.lat, incident.location.lon)
        if distance > DEDUP_RADIUS_KM:
            continue
        semantic_score = similarity.score(report_text, record["report_text"])
        if semantic_score >= DEDUP_SIMILARITY:
            matches.append((semantic_score, record))
    return max(matches, key=lambda item: item[0])[1] if matches else None