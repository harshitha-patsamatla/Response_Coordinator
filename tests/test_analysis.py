"""Focused tests for multilingual parsing and the report-completion gate."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from response_coordinator.classifier import EmergencyClassifier
from response_coordinator.intake import ResponseCoordinator
from response_coordinator.schemas import AssignedTeam, Location
from response_coordinator.severity import priority_for, score_severity


class FakeClassifier:
    def predict(self, text):
        return type("Result", (), {"labels": [("Road Accident", 0.85)], "backend": "test"})()


class FakeGeo:
    def resolve_location(self, text, source, landmark_coordinates=None, image_gps=None, reported_gps=None):
        if landmark_coordinates:
            return Location(text=text, lat=landmark_coordinates["lat"], lon=landmark_coordinates["lon"], confidence=0.9)
        if image_gps:
            return Location(text=text, lat=image_gps[0], lon=image_gps[1], confidence=0.6)
        if reported_gps:
            return Location(text=text, lat=reported_gps[0], lon=reported_gps[1], confidence=0.9)
        return Location(text=text)

    def nearest_team(self, lat, lon, incident_types=None):
        return AssignedTeam(name="Test ambulance", eta_min=7, source="test")


class AnalysisTests(unittest.TestCase):
    def test_sample_report_extracts_location_and_victims(self):
        coordinator = ResponseCoordinator(FakeClassifier(), FakeGeo(), ROOT / "data" / "landmarks.json")
        incident = coordinator.analyze("Bike accident near college gate. Two people injured.")
        self.assertEqual(incident.incident_types, ["Road Accident"])
        self.assertEqual(incident.victims.count, 2)
        self.assertEqual(incident.location.text.casefold(), "college gate")
        self.assertIsNone(incident.follow_up_question)
        self.assertEqual(incident.assigned_team.name, "Test ambulance")

    def test_missing_location_and_count_produces_one_question_and_no_team(self):
        coordinator = ResponseCoordinator(FakeClassifier(), FakeGeo(), ROOT / "data" / "landmarks.json")
        incident = coordinator.analyze("Something happened.")
        self.assertEqual(incident.follow_up_question.count("?"), 1)
        self.assertIsNone(incident.victims.count)
        self.assertIsNone(incident.assigned_team.name)

    def test_rules_fallback_classifies_sample_without_model_download(self):
        classifier = EmergencyClassifier(ROOT / "missing-checkpoint", allow_download=False)
        result = classifier.predict("Bike accident near college gate. Two people injured.")
        self.assertIn("Road Accident", [label for label, _ in result.labels])

    def test_telugu_victim_count_is_extracted(self):
        coordinator = ResponseCoordinator(FakeClassifier(), FakeGeo(), ROOT / "data" / "landmarks.json")
        incident = coordinator.analyze("కాలేజ్ గేట్ దగ్గర బైక్ ప్రమాదం. ఇద్దరు గాయపడ్డారు.")
        self.assertEqual(incident.victims.count, 2)
        self.assertIsNone(incident.follow_up_question)

    def test_severity_explains_factors_and_duplicate_priority_rises(self):
        low = score_severity(1, [], [], 0, ["Road Accident"], 0.8)
        high = score_severity(2, ["child"], ["unconscious", "severe_bleeding"], 1, ["Medical"], 0.9)
        self.assertGreater(high["score"], low["score"])
        self.assertTrue(any("unconscious" in reason for reason in high["reasons"]))
        self.assertLess(priority_for(60, 3)[1], priority_for(60, 1)[1])


if __name__ == "__main__":
    unittest.main()