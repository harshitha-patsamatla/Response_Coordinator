"""Shared report models used by the analysis and UI layers."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Location(BaseModel):
    text: str | None = None
    lat: float | None = None
    lon: float | None = None
    confidence: float = 0.0
    source: str | None = None


class Severity(BaseModel):
    score: int = Field(ge=0, le=100)
    level: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class Victims(BaseModel):
    count: int | None = None
    vulnerable: list[str] = Field(default_factory=list)
    count_source: str | None = None


class ResourceRecommendation(BaseModel):
    type: str
    qty: int = Field(ge=0)
    reason: str


class AssignedTeam(BaseModel):
    name: str | None = None
    eta_min: int | None = None
    source: str | None = None


class Incident(BaseModel):
    incident_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    incident_types: list[str] = Field(default_factory=list)
    type_confidence: float = 0.0
    location: Location = Field(default_factory=Location)
    severity: Severity
    victims: Victims = Field(default_factory=Victims)
    resources: list[ResourceRecommendation] = Field(default_factory=list)
    priority: str = "Routine"
    sla_minutes: int = 60
    assigned_team: AssignedTeam = Field(default_factory=AssignedTeam)
    first_aid_tips: list[str] = Field(default_factory=list)
    status: str = "Reported"
    follow_up_question: str | None = None
    credibility: float = 0.7
    independent_reports: int = 1
    duplicate_of: str | None = None
    source: str = "text"
    evidence: dict[str, Any] = Field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        """Serialize the user-facing output without internal evidence details."""
        return {
            "incident_id": self.incident_id,
            "incident_types": self.incident_types,
            "location": self.location.model_dump(),
            "severity": self.severity.model_dump(),
            "victims": {
                "count": self.victims.count,
                "vulnerable": self.victims.vulnerable,
            },
            "resources": [item.model_dump() for item in self.resources],
            "priority": self.priority,
            "assigned_team": self.assigned_team.model_dump(),
            "first_aid_tips": self.first_aid_tips,
            "status": self.status,
        }