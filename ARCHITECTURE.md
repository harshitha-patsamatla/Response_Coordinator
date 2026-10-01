# AI Emergency Response Coordinator Architecture

## Project Layout

```text
response-coordinator/
|-- app.py
|-- ARCHITECTURE.md
|-- README.md
|-- requirements.txt
|-- data/
|   |-- landmarks.json
|   `-- mock_facilities.json
|-- models/                         # Optional locally fine-tuned checkpoint
|-- scripts/
|   |-- generate_dataset.py
|   `-- train_classifier.py
|-- src/response_coordinator/
|   |-- __init__.py
|   |-- config.py
|   |-- schemas.py
|   |-- intake.py
|   |-- classifier.py
|   |-- extraction.py
|   |-- severity.py
|   |-- duplicates.py
|   |-- geo.py
|   |-- vision.py
|   |-- voice.py
|   |-- dispatch.py
|   |-- first_aid.py
|   |-- notifications.py
|   `-- storage.py
`-- tests/
    |-- test_analysis.py
    `-- test_storage.py
```

## Data Flow

1. Streamlit accepts text, audio, or image. Voice is transcribed by Whisper; images are inspected by optional YOLO weights and EXIF GPS is read when available.
2. A multilingual transformer classifier returns zero or more emergency labels. A transparent rules layer extracts people/vulnerable-person mentions, location candidates, hazards, and severity factors.
3. Location resolution follows explicit address, configured landmark, then image GPS. If location or victim count is absent, the report is held as a draft and exactly one follow-up question is shown; no missing value is fabricated.
4. Resolved reports are checked for nearby/time-bounded semantic duplicates, assigned a priority and SLA, and matched to suitable facilities/response teams. OSM/Nominatim, Overpass, and OSRM are best-effort adapters; deterministic mock data keeps the demo usable offline.
5. SQLite stores minimal incident fields and a masked audit trail. The UI may recommend dispatch, but only a human approval action records a dispatch decision. External notification adapters are mock-only unless explicitly configured.

## Safety and Operating Boundaries

- This is decision support, not an emergency service. It does not contact real responders or replace a caller's contact with local emergency services.
- Missing critical facts block dispatch recommendations. Low model confidence and unsupported image classes are surfaced rather than silently treated as facts.
- Generic COCO YOLO weights identify people and common vehicles only. Fire, smoke, and crash-damage detection require a compatible custom checkpoint; the UI reports these capabilities as unavailable without one.
- Geospatial mock coordinates and facility availability are illustrative. Public map services can be unavailable, rate-limited, or inaccurate.
- Synthetic reports are for exercising the pipeline and training experiments only; they do not establish real-world model performance.