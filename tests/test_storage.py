"""Storage tests for masking, dispatcher updates, and audit events."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from response_coordinator.schemas import Incident, Severity, Victims
from response_coordinator.storage import IncidentStore, mask_phone_numbers


class StorageTests(unittest.TestCase):
    def test_phone_numbers_are_masked(self):
        self.assertIn("[PHONE REDACTED]", mask_phone_numbers("Call me at +1 415 555 0134"))

    def test_priority_override_and_status_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IncidentStore(Path(directory) / "test.sqlite3")
            incident = Incident(
                incident_id="INC-TEST1234",
                severity=Severity(score=40, level="Medium", confidence=0.7),
                victims=Victims(count=1),
                evidence={"voice": {"text": "Call back at 415 555 0134"}},
            )
            store.save_incident(incident, "One injured near station")
            store.set_priority(incident.incident_id, "High")
            store.update_status(incident.incident_id, "En route")
            loaded = store.list_incidents()[0]["incident"]
            self.assertEqual(loaded.priority, "High")
            self.assertEqual(loaded.status, "En route")
            self.assertIn("[PHONE REDACTED]", loaded.evidence["voice"]["text"])
            actions = {event["action"] for event in store.audit_events()}
            self.assertIn("priority_override", actions)
            self.assertIn("status_updated", actions)


if __name__ == "__main__":
    unittest.main()