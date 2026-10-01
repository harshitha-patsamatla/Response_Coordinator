"""Rule-based extraction for locations, people, and hazard evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path


NUMBER_WORDS = {
    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "ఒకరు": 1, "ఇద్దరు": 2, "ముగ్గురు": 3, "एक": 1, "दो": 2, "तीन": 3,
    "चार": 4, "पांच": 5,
}
PERSON_PATTERN = r"(?:people|persons?|victims?|injured|wounded|hurt|పేరు|వ్యక్తులు|మంది|గాయపడ్డారు|గాయపడిన|చిక్కుకున్నారు|लोग|व्यक्ति|घायल)"
VULNERABLE_TERMS = {
    "child": ["child", "children", "kid", "పిల్ల", "बच्चा", "बच्चे"],
    "elderly": ["elderly", "old person", "వృద్ధ", "बुजుర్గ"],
    "pregnant": ["pregnant", "ప్రసవ", "गर्भवती"],
}
HAZARD_TERMS = {
    "unconscious": ["unconscious", "not waking", "స్పృహ లేకుండా", "बेहोश"],
    "severe_bleeding": ["bleeding heavily", "severe bleeding", "blood everywhere", "ఎక్కువ రక్తం", "खून बह"],
    "fire_or_smoke": ["fire", "smoke", "burning", "మంట", "आग", "धुआँ"],
    "fuel_hazard": ["fuel", "petrol", "gas leak", "chemical", "ఇంధనం", "पेट्रोल", "गैस लीक"],
}


def _load_landmarks(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def extract_report(text: str, landmarks_path: Path | None = None) -> dict:
    normalized = text.casefold()
    landmarks = _load_landmarks(landmarks_path)
    landmark_match = next(
        ((name, coords) for name, coords in landmarks.items() if name.casefold() in normalized),
        None,
    )
    explicit_location = _extract_address(text)
    exact_landmark = bool(
        landmark_match and explicit_location and explicit_location.casefold() == landmark_match[0].casefold()
    )
    location_text = landmark_match[0] if exact_landmark else explicit_location or (landmark_match[0] if landmark_match else None)
    reported_gps = _extract_gps(text)

    victim_count, count_source = _extract_victim_count(text)
    vulnerable = [
        group
        for group, phrases in VULNERABLE_TERMS.items()
        if any(phrase.casefold() in normalized for phrase in phrases)
    ]
    hazards = [
        hazard
        for hazard, phrases in HAZARD_TERMS.items()
        if any(phrase.casefold() in normalized for phrase in phrases)
    ]
    injury_count = len(re.findall(r"\b(?:injured|wounded|hurt)\b|గాయపడిన|घायल", normalized))
    return {
        "location_text": location_text,
        "location_source": "landmark" if exact_landmark or not explicit_location and landmark_match else "explicit" if explicit_location else "reported_gps" if reported_gps else None,
        "landmark_coordinates": landmark_match[1] if landmark_match else None,
        "reported_gps": reported_gps,
        "victim_count": victim_count,
        "victim_count_source": count_source,
        "vulnerable": vulnerable,
        "hazards": hazards,
        "injury_mentions": injury_count,
        "has_address_candidate": bool(explicit_location),
    }


def _extract_address(text: str) -> str | None:
    match = re.search(
        r"\b(?:near|at|in|opposite|behind|beside|వద్ద|దగ్గర|के पास|में)\s+([^,.!?;]{2,70})",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" .") if match else None


def _extract_victim_count(text: str) -> tuple[int | None, str | None]:
    normalized = text.casefold()
    pattern = rf"\b(\d{{1,2}}|{'|'.join(map(re.escape, NUMBER_WORDS))})\s+{PERSON_PATTERN}"
    match = re.search(pattern, normalized, flags=re.IGNORECASE)
    if match:
        token = match.group(1).casefold()
        return int(token) if token.isdigit() else NUMBER_WORDS[token], "reported"
    match = re.search(r"\b(\d{1,2})\s+(?:injured|hurt|wounded)\b", normalized)
    if match:
        return int(match.group(1)), "reported"
    return None, None


def _extract_gps(text: str) -> tuple[float, float] | None:
    match = re.search(r"(?<!\d)(-?\d{1,2}\.\d+)\s*[,;/ ]\s*(-?\d{1,3}\.\d+)(?!\d)", text)
    if not match:
        return None
    latitude, longitude = float(match.group(1)), float(match.group(2))
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return latitude, longitude
    return None