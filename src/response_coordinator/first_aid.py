"""Vetted, static first-aid prompts; no generative medical instructions."""

GUIDANCE = {
    "cpr": {
        "triggers": ["unconscious"],
        "tips": [
            "Call local emergency services and ask someone to bring an AED.",
            "If the person is unresponsive and not breathing normally, begin chest compressions if you can do so safely.",
            "Push hard and fast in the center of the chest, about 100-120 compressions per minute; follow AED prompts if available.",
        ],
        "source": "American Heart Association, Hands-Only CPR guidance (static summary)",
    },
    "bleeding": {
        "triggers": ["severe_bleeding"],
        "tips": [
            "Call local emergency services for severe bleeding.",
            "Apply firm, steady direct pressure with clean cloth or gauze; do not remove an embedded object.",
            "Keep the person still and warm while waiting for trained responders.",
        ],
        "source": "American Red Cross, First Aid for Severe Bleeding (static summary)",
    },
    "burns": {
        "triggers": ["fire_or_smoke"],
        "tips": [
            "Move away from the source only if it is safe and call emergency services for serious burns or breathing difficulty.",
            "Cool a thermal burn under cool clean running water for 20 minutes when available.",
            "Do not use ice, butter, or creams; cover loosely with a clean non-stick dressing.",
        ],
        "source": "British Red Cross, burns and scalds first aid (static summary)",
    },
}


def tips_for(hazards: list[str]) -> tuple[list[str], list[str]]:
    selected = [
        entry for entry in GUIDANCE.values()
        if any(trigger in hazards for trigger in entry["triggers"])
    ]
    tips = [tip for entry in selected for tip in entry["tips"]]
    sources = list(dict.fromkeys(entry["source"] for entry in selected))
    return tips, sources