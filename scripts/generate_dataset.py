"""Generate a deterministic multilingual synthetic training set (700 rows)."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

LABEL_TEMPLATES = {
    "Road Accident": {
        "en": ["Bike accident near College Gate. Two people injured.", "A car crash at the market road; one person is bleeding.", "Two vehicles collided near the bus stop and people need help."],
        "te": ["కాలేజ్ గేట్ దగ్గర బైక్ ప్రమాదం. ఇద్దరు గాయపడ్డారు.", "బస్ స్టాప్ వద్ద వాహనాలు ఢీకొన్నాయి, గాయపడిన వారికి సహాయం కావాలి."],
        "hi": ["कॉलेज गेट के पास बाइक दुर्घटना हुई, दो लोग घायल हैं।", "बस स्टॉप के पास गाड़ी टकराई है, घायलों को मदद चाहिए।"],
    },
    "Fire": {
        "en": ["Fire and smoke near College Gate; two people are trapped.", "A shop is burning and smoke is spreading near the bus stop."],
        "te": ["కాలేజ్ గేట్ దగ్గర మంటలు, పొగ వస్తోంది; ఇద్దరు లోపల ఉన్నారు.", "దుకాణంలో అగ్ని ప్రమాదం జరిగింది, సహాయం పంపండి."],
        "hi": ["कॉलेज गेट के पास आग और धुआँ है, दो लोग अंदर फँसे हैं।", "दुकान में आग लगी है और धुआँ फैल रहा है।"],
    },
    "Medical": {
        "en": ["A person is unconscious near Gandhi Hospital and needs an ambulance.", "Someone has severe bleeding near the bus stop."],
        "te": ["గాంధీ ఆసుపత్రి దగ్గర ఒక వ్యక్తి స్పృహలో లేరు, అంబులెన్స్ కావాలి.", "బస్ స్టాప్ దగ్గర ఒకరికి తీవ్ర రక్తస్రావం అవుతోంది."],
        "hi": ["गांधी अस्पताल के पास एक व्यक्ति बेहोश है, एम्बुलेंस चाहिए।", "बस स्टॉप के पास एक व्यक्ति का बहुत खून बह रहा है।"],
    },
    "Crime": {
        "en": ["An assault was reported near College Gate; police assistance is needed.", "A robbery is happening by the market road. Please send help."],
        "te": ["కాలేజ్ గేట్ దగ్గర దాడి జరిగింది, పోలీసు సహాయం కావాలి.", "మార్కెట్ రోడ్డుపై దోపిడీ జరుగుతోంది."],
        "hi": ["कॉलेज गेट के पास हमला हुआ है, पुलिस सहायता चाहिए।", "बाजार रोड पर लूट हो रही है, मदद भेजें।"],
    },
    "Flood": {
        "en": ["Flood water is rising near the bus stop and two people are stranded.", "The road is submerged near College Gate; residents need rescue."],
        "te": ["బస్ స్టాప్ దగ్గర వరద నీరు పెరుగుతోంది, ఇద్దరు చిక్కుకున్నారు.", "కాలేజ్ గేట్ రోడ్డుపై నీరు చేరింది, సహాయం కావాలి."],
        "hi": ["बस स्टॉप के पास बाढ़ का पानी बढ़ रहा है, दो लोग फँसे हैं।", "कॉलेज गेट की सड़क पानी में डूबी है, बचाव दल चाहिए।"],
    },
    "Building Collapse": {
        "en": ["A building collapsed near College Gate; people may be trapped inside.", "Part of a house fell near the market, and two people are missing."],
        "te": ["కాలేజ్ గేట్ దగ్గర భవనం కూలింది, లోపల వ్యక్తులు చిక్కుకున్నారు.", "ఇంటి భాగం కూలింది, ఇద్దరు కనిపించడం లేదు."],
        "hi": ["कॉलेज गेट के पास इमारत गिर गई है, लोग अंदर फँसे हो सकते हैं।", "मकान का हिस्सा गिरा है और दो लोग लापता हैं।"],
    },
    "Other": {
        "en": ["A damaged streetlight is sparking near the market; please check it.", "There is an unusual hazard near the bus stop; responders should assess."],
        "te": ["మార్కెట్ దగ్గర వీధి దీపం నుండి నిప్పురవ్వలు వస్తున్నాయి.", "బస్ స్టాప్ దగ్గర ప్రమాదకర పరిస్థితి ఉంది, పరిశీలించండి."],
        "hi": ["बाजार के पास स्ट्रीट लाइट से चिंगारी निकल रही है।", "बस स्टॉप के पास खतरा है, जाँच के लिए मदद भेजें।"],
    },
}
LANGUAGES = ("en", "te", "hi")


def generate_rows(rows_per_label: int = 100, seed: int = 41) -> list[dict[str, str]]:
    randomizer = random.Random(seed)
    rows = []
    for label, translations in LABEL_TEMPLATES.items():
        for index in range(rows_per_label):
            language = LANGUAGES[index % len(LANGUAGES)]
            text = randomizer.choice(translations[language])
            labels = [label]
            if index % 10 == 0:
                labels.append(randomizer.choice([name for name in LABEL_TEMPLATES if name != label]))
            rows.append({"text": text, "labels": "|".join(labels), "language": language})
    randomizer.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_reports.csv"))
    parser.add_argument("--rows-per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    arguments = parser.parse_args()
    rows = generate_rows(arguments.rows_per_label, arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["text", "labels", "language"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic reports to {arguments.output}")


if __name__ == "__main__":
    main()