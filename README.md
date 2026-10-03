# AI Emergency Response Coordinator

A local Streamlit decision-support prototype for turning text, voice, or image reports into structured triage cards. It is not an emergency service and does not automatically contact responders.

## Project Structure

```text
app.py                         Streamlit intake, queue, map, analytics, audit UI
ARCHITECTURE.md                Components, data flow, and safety boundaries
requirements.txt               Runtime and optional model dependencies
data/landmarks.json            Editable local landmark-to-coordinate lookup
data/mock_facilities.json      Illustrative offline facilities and response teams
scripts/generate_dataset.py    700-row synthetic English/Telugu/Hindi generator
scripts/train_classifier.py    XLM-R multi-label training and evaluation
src/response_coordinator/      Classifier, extraction, scoring, maps, storage, adapters
tests/                         Pipeline and SQLite tests
```

## Setup (Windows PowerShell)

Python 3.11 or 3.12 is recommended for broad compatibility with PyTorch, Whisper, and Ultralytics. Install FFmpeg for Whisper audio decoding.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` from the included template and replace the placeholder with your own restricted Google Maps API key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

The `.env` file is ignored by Git. Never commit or paste a real API key into source code or chat. In Google Cloud, restrict the key to the Geocoding API, Places API, and Directions API (and restrict it to your local/deployment environment as appropriate). `MAP_PROVIDER=google` selects this adapter; without a valid key, the app falls back to local mock data.

Install FFmpeg separately (for example, with `winget install Gyan.FFmpeg.Shared`) and reopen the terminal so `ffmpeg` is on `PATH`.

Start the dashboard from the project directory:

```powershell
python -m streamlit run app.py
```

Do not launch the UI with `python app.py`. That executes the file outside Streamlit's runtime, causing the `missing ScriptRunContext` and session-state warnings shown in your terminal. Keep that PowerShell process open while using the printed local URL; stop it with `Ctrl+C`.

The first report may download the multilingual zero-shot classifier and YOLO COCO weights. Sentence embeddings download only when duplicate matching first needs them. These models can consume substantial disk, memory, and CPU; a GPU is optional. Set `ALLOW_MODEL_DOWNLOAD=false` to use the built-in rule classifier unless a trained checkpoint is present.

The initial landmark and mock-response data is illustrative and centered around Hyderabad. Replace `data/landmarks.json` and `data/mock_facilities.json` before using another demo location. Local landmarks work offline. Public Nominatim, Overpass, and OSRM lookups require internet access, can be rate-limited, and are best-effort. `NOMINATIM_USER_AGENT`, `ENABLE_OSM_LOOKUPS`, and `COORDINATOR_DB_PATH` are configurable environment variables. To use the alternative Google Maps geocoding/Places/Directions adapter, set `MAP_PROVIDER=google` and `GOOGLE_MAPS_API_KEY`; enable the corresponding Google APIs and billing in your Google Cloud project. The default provider remains OSM.

## Try It

The text tab starts with this sample:

```text
Bike accident near college gate. Two people injured.
```

Expected behavior: Road Accident classification, the configured College Gate coordinates, two reported victims, severity factors, resource suggestions, and a nearest-team ETA using a live route when available or an explicitly labelled mock route. Reports without a geocoded location or victim count receive one follow-up question and no team assignment. Dispatcher approval only records a demo decision.

Voice input supports microphone recording or WAV/MP3/M4A/OGG upload. Whisper detects language automatically. Install FFmpeg before using audio. The displayed urgency indicator is a weak loudness/speaking-rate proxy, not a validated speech-stress detector.

Image input reads EXIF GPS and detects people/common vehicles using generic COCO YOLO weights. Fire, smoke, and damaged/crashed-vehicle detection are unavailable unless you provide compatible custom weights through `COORDINATOR_VISION_WEIGHTS`. Detection counts are not reliable victim counts and must be verified.

## Generate and Train

Generate 700 reproducible synthetic reports (100 for each of seven labels, in English, Telugu, Hindi, including some multi-label examples):

```powershell
python scripts/generate_dataset.py
```

Fine-tune the multilingual XLM-R base model and save it where the app will find it:

```powershell
python scripts/train_classifier.py --epochs 2 --batch-size 8
```

Training downloads model weights and is resource-intensive. Evaluation writes macro precision, recall, F1, micro F1, and one confusion matrix per label to `models/emergency-xlm-roberta/evaluation.json`. The app prefers this checkpoint when present; otherwise it uses multilingual zero-shot classification. Synthetic-only scores are not a measure of real-world performance.

Run the focused tests:

```powershell
python -m unittest discover -s tests -v
```

## Safety, Privacy, and Limitations

- This is a decision-support prototype, not a replacement for official emergency services, a dispatcher, or clinical judgment. In a real emergency, contact local emergency services directly.
- No automatic dispatch is implemented. A human must approve a recommendation; the default notification mode is mock-only. Telegram delivery is opt-in and requires configured credentials.
- Classification/extraction quality is unvalidated for real emergencies, Telugu/Hindi code-mixing, regional expressions, and adversarial inputs. The 700 generated examples are intentionally small and synthetic.
- Location, ETA, facility specialty, bed availability, and responder positions may be missing, stale, simulated, or inaccurate. The mock Hyderabad coordinates and capacity values are examples, not verified services.
- Generic YOLO does not identify fire, smoke, or collision damage. Image-person counts do not prove victim counts. Voice urgency is experimental and can be biased by microphones, accents, disability, and environment.
- Credibility is a weak completeness heuristic, not a fraud detector. It must not be used to deny care or lower priority without human review.
- SQLite stores structured reports and phone-masked report text locally. Data retention defaults to 30 days and cleanup runs at app startup or by dispatcher action. Secure the machine and database file; this prototype has no authentication or role-based access control.
- First-aid instructions are short static summaries attributed in the UI to American Heart Association, American Red Cross, and British Red Cross materials. They are not diagnosis or a substitute for dispatcher guidance.

Future work: evaluate with consented and independently labelled local-language data; add calibrated confidence and fairness testing; verify facility and ETA feeds; add authentication, encryption, access controls, and retention policy review; integrate only with an authorized dispatch system after safety validation; and have local medical/emergency experts review translations and first-aid content.