"""Resource recommendations and non-automatic dispatch decisions."""

from __future__ import annotations

from math import ceil

from .schemas import ResourceRecommendation


def recommend_resources(
    incident_types: list[str],
    severity_score: int,
    victim_count: int | None,
    hazards: list[str],
) -> list[ResourceRecommendation]:
    resources: list[ResourceRecommendation] = []
    is_medical = bool({"Road Accident", "Medical"}.intersection(incident_types))
    if is_medical or (victim_count is not None and victim_count > 0):
        ambulance_type = "Ambulance (ALS)" if severity_score >= 60 or any(
            item in hazards for item in ("unconscious", "severe_bleeding")
        ) else "Ambulance (BLS)"
        quantity = max(1, ceil((victim_count or 1) / 3))
        resources.append(ResourceRecommendation(
            type=ambulance_type,
            qty=quantity,
            reason="Medical/transport support based on incident type and reported count.",
        ))
    if "Fire" in incident_types or "fire_or_smoke" in hazards:
        resources.append(ResourceRecommendation(type="Fire engine", qty=1, reason="Fire or smoke reported."))
    if "Crime" in incident_types or "Road Accident" in incident_types:
        resources.append(ResourceRecommendation(type="Police unit", qty=1, reason="Scene safety and access control."))
    if any(kind in incident_types for kind in ("Building Collapse", "Flood")):
        resources.append(ResourceRecommendation(
            type="Search and rescue team", qty=1, reason="Potential entrapment or water rescue risk."
        ))
    if not resources:
        resources.append(ResourceRecommendation(
            type="Dispatcher assessment", qty=1, reason="Incident details require human triage."
        ))
    return resources


def credibility_score(text: str, has_location: bool, victim_count: int | None) -> tuple[float, list[str]]:
    score = 0.55
    signals = []
    if has_location:
        score += 0.15
        signals.append("Location detail provided")
    if victim_count is not None:
        score += 0.15
        signals.append("Victim count provided")
    if len(text.split()) >= 6:
        score += 0.08
        signals.append("Report contains contextual detail")
    if text.count("!") >= 4 or text.isupper() and len(text) > 20:
        score -= 0.2
        signals.append("Highly emphatic formatting; verify with caller")
    return round(max(0.0, min(1.0, score)), 2), signals


def needs_verification(credibility: float) -> bool:
    return credibility < 0.55