"""Mock-first notification adapters; real delivery requires explicit secrets."""

from __future__ import annotations

import os

import requests

from .schemas import Incident


def send_notification(incident: Incident, destination: str | None = None) -> dict:
    message = (
        f"{incident.priority} incident {incident.incident_id}: "
        f"{', '.join(incident.incident_types) or 'Unclassified'}, "
        f"severity {incident.severity.score}/100. Dispatcher approval required."
    )
    mode = os.getenv("NOTIFICATION_MODE", "mock").lower()
    if mode == "telegram":
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = destination or os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return {"mode": mode, "sent": False, "error": "Telegram credentials are not configured."}
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message},
                timeout=5,
            )
            response.raise_for_status()
            return {"mode": mode, "sent": True, "message": "Telegram notification sent."}
        except requests.RequestException as error:
            return {"mode": mode, "sent": False, "error": str(error)}
    return {"mode": "mock", "sent": False, "message": message}