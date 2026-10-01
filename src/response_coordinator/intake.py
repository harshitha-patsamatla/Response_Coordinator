"""Orchestrate one report through classification, extraction, and triage."""

from __future__ import annotations

from uuid import uuid4

from .classifier import EmergencyClassifier
from .dispatch import credibility_score, recommend_resources
from .first_aid import tips_for
from .geo import OpenStreetMapProvider
from .schemas import AssignedTeam, Incident, Location, Severity, Victims
from .severity import priority_for, score_severity


class ResponseCoordinator:
    def __init__(
        self,
        classifier: EmergencyClassifier,
        geo_provider: OpenStreetMapProvider,
        landmarks_path,
    ) -> None:
        self.classifier = classifier
        self.geo_provider = geo_provider
        self.landmarks_path = landmarks_path

    def analyze(
        self,
        text: str,
        source: str = "text",
        image_result: dict | None = None,
        voice_result: dict | None = None,
    ) -> Incident:
        from .extraction import extract_report

        classification = self.classifier.predict(text)
        extracted = extract_report(text, self.landmarks_path)
        image_result = image_result or {}
        voice_result = voice_result or {}
        image_hazards = []
        if image_result.get("fire_smoke_count", 0) > 0:
            image_hazards.append("fire_or_smoke")
        if image_result.get("crash_damage_count", 0) > 0:
            classification.labels.append(("Road Accident", 0.65))
        extracted["hazards"] = list(dict.fromkeys(extracted["hazards"] + image_hazards))
        victim_count = extracted["victim_count"]
        count_source = extracted["victim_count_source"]
        if victim_count is None and image_result.get("people_count", 0) > 0:
            victim_count = int(image_result["people_count"])
            count_source = "image_estimate"
        labels = _merge_labels(classification.labels)
        type_confidence = max((score for _, score in labels), default=0.25)
        location = self.geo_provider.resolve_location(
            extracted["location_text"],
            extracted["location_source"],
            extracted["landmark_coordinates"],
            image_result.get("gps"),
            extracted["reported_gps"],
        )
        severity_data = score_severity(
            victim_count=victim_count,
            vulnerable=extracted["vulnerable"],
            hazards=extracted["hazards"],
            injury_mentions=extracted["injury_mentions"],
            incident_types=[name for name, _ in labels],
            type_confidence=type_confidence,
        )
        if voice_result.get("urgency_indicator") == "High":
            severity_data["score"] = min(100, severity_data["score"] + 5)
            severity_data["reasons"].append("Elevated voice urgency proxy: +5 (unvalidated)")
            severity_data["level"] = _severity_level(severity_data["score"])
        credibility, credibility_reasons = credibility_score(
            text, location.lat is not None and location.lon is not None, victim_count
        )
        priority, sla = priority_for(severity_data["score"])
        missing_location = location.lat is None or location.lon is None
        missing_victim_count = victim_count is None
        follow_up = _follow_up_question(missing_location, missing_victim_count)
        team = AssignedTeam()
        nearby_facilities = []
        if follow_up is None:
            team = self.geo_provider.nearest_team(location.lat, location.lon, [name for name, _ in labels])
            if hasattr(self.geo_provider, "find_nearby_facilities"):
                nearby_facilities = self.geo_provider.find_nearby_facilities(location.lat, location.lon)
        tips, aid_sources = tips_for(extracted["hazards"])
        resources = recommend_resources(
            [name for name, _ in labels],
            severity_data["score"],
            victim_count,
            extracted["hazards"],
        )
        return Incident(
            incident_id=f"INC-{uuid4().hex[:8].upper()}",
            incident_types=[name for name, _ in labels],
            type_confidence=type_confidence,
            location=location,
            severity=Severity(**severity_data),
            victims=Victims(count=victim_count, vulnerable=extracted["vulnerable"], count_source=count_source),
            resources=resources,
            priority=priority,
            sla_minutes=sla,
            assigned_team=team,
            first_aid_tips=tips,
            follow_up_question=follow_up,
            credibility=credibility,
            source=source,
            evidence={
                "classifier_backend": classification.backend,
                "type_scores": {name: score for name, score in labels},
                "hazards": extracted["hazards"],
                "injury_mentions": extracted["injury_mentions"],
                "credibility_reasons": credibility_reasons,
                "first_aid_sources": aid_sources,
                "nearby_facilities": nearby_facilities,
                "image": image_result,
                "voice": voice_result,
            },
        )


def _merge_labels(labels: list[tuple[str, float]]) -> list[tuple[str, float]]:
    merged: dict[str, float] = {}
    for name, score in labels:
        merged[name] = max(merged.get(name, 0.0), float(score))
    if len(merged) > 1:
        merged.pop("Other", None)
    return sorted(merged.items(), key=lambda item: item[1], reverse=True) or [("Other", 0.25)]


def _follow_up_question(missing_location: bool, missing_victim_count: bool) -> str | None:
    if missing_location and missing_victim_count:
        return "Where exactly is the incident, and about how many people are affected?"
    if missing_location:
        return "Could you share the exact address, a nearby landmark, or a GPS pin?"
    if missing_victim_count:
        return "About how many people are affected?"
    return None


def _severity_level(score: int) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"