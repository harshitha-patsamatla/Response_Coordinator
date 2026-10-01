"""Transparent weighted severity scoring and SLA priority mapping."""

from datetime import datetime


TYPE_RISK = {
    "Road Accident": 8,
    "Fire": 15,
    "Medical": 12,
    "Crime": 8,
    "Flood": 10,
    "Building Collapse": 18,
    "Other": 0,
}


def score_severity(
    victim_count: int | None,
    vulnerable: list[str],
    hazards: list[str],
    injury_mentions: int,
    incident_types: list[str],
    type_confidence: float,
    now: datetime | None = None,
) -> dict:
    score = 8
    reasons = ["Baseline uncertainty allowance: +8"]
    if victim_count is not None:
        points = min(victim_count * 8, 32)
        score += points
        reasons.append(f"{victim_count} reported/detected victim(s): +{points}")
    if injury_mentions:
        score += 10
        reasons.append("Injury mentioned: +10")
    hazard_weights = {
        "unconscious": 30,
        "severe_bleeding": 24,
        "fire_or_smoke": 18,
        "fuel_hazard": 16,
    }
    for hazard in hazards:
        points = hazard_weights.get(hazard, 0)
        score += points
        reasons.append(f"Hazard {hazard.replace('_', ' ')}: +{points}")
    if vulnerable:
        score += 10
        reasons.append(f"Vulnerable person(s) ({', '.join(vulnerable)}): +10")
    if incident_types:
        risk = max(TYPE_RISK.get(item, 0) for item in incident_types)
        points = round(risk * max(0.35, min(type_confidence, 1.0)) * 0.35)
        score += points
        reasons.append(f"Classified incident risk contribution: +{points}")
    current = now or datetime.now().astimezone()
    if current.hour >= 22 or current.hour < 6:
        score += 3
        reasons.append("Reduced night-time response availability: +3")
    score = min(score, 100)
    level = "Critical" if score >= 80 else "High" if score >= 60 else "Medium" if score >= 35 else "Low"
    evidence = sum((victim_count is not None, bool(hazards), bool(incident_types), bool(injury_mentions)))
    confidence = min(0.92, 0.35 + evidence * 0.12 + min(type_confidence, 1.0) * 0.1)
    return {"score": score, "level": level, "confidence": round(confidence, 2), "reasons": reasons}


def priority_for(score: int, independent_reports: int = 1) -> tuple[str, int]:
    boosted = min(100, score + max(0, independent_reports - 1) * 5)
    base_priority = "Immediate" if boosted >= 80 else "High" if boosted >= 60 else "Elevated" if boosted >= 35 else "Routine"
    ordered = ["Immediate", "High", "Elevated", "Routine"]
    index = ordered.index(base_priority)
    index = max(0, index - max(0, independent_reports - 1))
    priority = ordered[index]
    sla = {"Immediate": 5, "High": 15, "Elevated": 30, "Routine": 60}[priority]
    return priority, sla