"""Streamlit dispatcher dashboard for the decision-support prototype."""

from __future__ import annotations

import sys
import tempfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

if __name__ == "__main__" and not st.runtime.exists():
    raise SystemExit(
        "This is a Streamlit app. Start it with: "
        "python -m streamlit run app.py"
    )

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from response_coordinator.classifier import EmergencyClassifier
from response_coordinator.config import ALLOW_MODEL_DOWNLOAD, DATA_DIR, DB_PATH, MODEL_DIR, RETENTION_DAYS
from response_coordinator.dispatch import needs_verification
from response_coordinator.duplicates import SemanticSimilarity, find_duplicate
from response_coordinator.geo import create_geo_provider
from response_coordinator.intake import ResponseCoordinator
from response_coordinator.notifications import send_notification
from response_coordinator.severity import priority_for
from response_coordinator.storage import IncidentStore

st.set_page_config(page_title="Response Coordinator", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#16292c; --muted:#607276; --green:#12665c; --red:#ba3c32; --amber:#d98625; }
    html,body,[class*="css"] { font-family:'Manrope',sans-serif; color:var(--ink); }
    .stApp { background:radial-gradient(circle at 92% 2%,#dcebe1 0,transparent 29%),linear-gradient(135deg,#f6f8f5,#edf3ef); }
    .block-container { padding-top:1.3rem; max-width:1500px; }
    .eyebrow { color:var(--green); font:500 .75rem 'DM Mono',monospace; text-transform:uppercase; }
    .incident { background:white; border:1px solid #dce5df; border-left:5px solid var(--red); border-radius:6px; padding:1rem 1.2rem; margin:.5rem 0 1rem; }
    [data-testid="stMetric"] { background:white; border:1px solid #dce5df; border-radius:6px; padding:.7rem 1rem; }
    </style>
    """, unsafe_allow_html=True,
)


@st.cache_resource
def services():
    geo = create_geo_provider(DATA_DIR / "mock_facilities.json", DATA_DIR / "landmarks.json")
    store = IncidentStore(DB_PATH)
    store.cleanup_expired(RETENTION_DAYS)
    return (
        ResponseCoordinator(EmergencyClassifier(MODEL_DIR, ALLOW_MODEL_DOWNLOAD), geo, DATA_DIR / "landmarks.json"),
        store, geo, SemanticSimilarity(),
    )


def analyze_and_store(text: str, source: str, image: dict | None = None, voice: dict | None = None) -> bool:
    coordinator, store, _, semantic = services()
    incident = coordinator.analyze(text, source, image, voice)
    duplicate = find_duplicate(text, incident.location.lat, incident.location.lon, store.list_incidents(), semantic)
    if duplicate:
        prior = duplicate["incident"]
        count = prior.independent_reports + 1
        prior.priority, _ = priority_for(max(prior.severity.score, incident.severity.score), count)
        prior.independent_reports = count
        prior.evidence["duplicate_notice"] = f"Merged with this incident; {count} reports are now linked."
        store.merge_duplicate(prior.incident_id, count, prior.priority)
        st.session_state.current_incident = prior
    else:
        store.save_incident(incident, text)
        st.session_state.current_incident = incident
    return duplicate is not None


def show_incident(incident) -> None:
    color = {"Critical":"#ba3c32", "High":"#d98625", "Medium":"#b28a26", "Low":"#12665c"}.get(incident.severity.level, "#607276")
    st.markdown(
        f'<div class="incident" style="border-left-color:{color}"><div class="eyebrow">{incident.incident_id} · {incident.priority}</div>'
        f'<h3>{" / ".join(incident.incident_types) or "Unclassified report"}</h3>'
        f'<b>Severity {incident.severity.score}/100 · {incident.severity.level}</b> · {incident.severity.confidence:.0%} confidence<br>'
        f'{incident.location.text or "Location not resolved"} · {incident.victims.count if incident.victims.count is not None else "Count unknown"} people</div>',
        unsafe_allow_html=True,
    )
    if incident.follow_up_question:
        st.warning(f"One detail is needed before a team can be recommended: {incident.follow_up_question}")
    if needs_verification(incident.credibility):
        st.warning("Low credibility score: verify details with the reporter. This is not a finding that the report is false.")
    if incident.evidence.get("duplicate_notice"):
        st.info(incident.evidence["duplicate_notice"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Contributing factors**")
        for reason in incident.severity.reasons:
            st.write(f"- {reason}")
        st.markdown("**Recommended resources**")
        for item in incident.resources:
            st.write(f"- {item.qty} × {item.type}: {item.reason}")
    with right:
        st.markdown("**Team recommendation**")
        if incident.assigned_team.name:
            st.write(f"{incident.assigned_team.name} · ETA {incident.assigned_team.eta_min} min")
            st.caption(f"Route source: {incident.assigned_team.source or 'unavailable'}")
        else:
            st.write("Pending location and victim-count confirmation.")
        for facility in incident.evidence.get("nearby_facilities", []):
            details = []
            if facility.get("specialty"):
                details.append("specialty: " + ", ".join(facility["specialty"]))
            if facility.get("beds_available") is not None:
                details.append(f"beds available: {facility['beds_available']}")
            suffix = " · " + " · ".join(details) if details else ""
            st.write(f"- {facility['name']} ({facility['type']}) · ETA {facility['eta_min']} min{suffix}")
        for tip in incident.first_aid_tips:
            st.write(f"- {tip}")
        for source in incident.evidence.get("first_aid_sources", []):
            st.caption(source)
    with st.expander("Structured JSON"):
        st.json(incident.public_dict())
    with st.expander("Analysis details"):
        st.write({"classifier": incident.evidence.get("classifier_backend"), "credibility": incident.credibility, "source": incident.source})
        if incident.evidence.get("voice"):
            st.write(incident.evidence["voice"])
        if incident.evidence.get("image"):
            st.write(incident.evidence["image"])


def handle_audio(uploaded) -> None:
    from response_coordinator.voice import transcribe_audio
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".wav") as file:
            file.write(uploaded.getvalue())
            path = Path(file.name)
        with st.spinner("Transcribing with Whisper…"):
            result = transcribe_audio(path)
        st.session_state.voice_result = result
        st.session_state.voice_token = hashlib.sha256(uploaded.getvalue()).hexdigest()
    finally:
        if path and path.exists():
            path.unlink()


def handle_image(uploaded, description: str) -> None:
    from response_coordinator.vision import analyze_image
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".jpg") as file:
            file.write(uploaded.getvalue())
            path = Path(file.name)
        with st.spinner("Inspecting image with YOLO…"):
            result = analyze_image(path)
        st.session_state.image_result = result
        st.session_state.image_token = hashlib.sha256(uploaded.getvalue()).hexdigest()
    finally:
        if path and path.exists():
            path.unlink()


def render_intake() -> None:
    text_tab, voice_tab, image_tab = st.tabs(["Text", "Voice", "Image"])
    with text_tab:
        with st.form("report"):
            text = st.text_area("Emergency report", "Bike accident near college gate. Two people injured.", height=130)
            submitted = st.form_submit_button("Analyze report", type="primary")
        if submitted:
            if not text.strip():
                st.error("Enter a report first.")
            else:
                try:
                    duplicate = analyze_and_store(text, "text")
                    st.success("Merged with nearby report." if duplicate else "Added to queue.")
                    show_incident(st.session_state.current_incident)
                except Exception as error:
                    st.error(f"Analysis could not complete: {error}")
    with voice_tab:
        st.caption("The urgency indicator uses loudness and speaking-rate proxies; it is not a stress diagnosis.")
        microphone = st.audio_input("Record report") if hasattr(st, "audio_input") else None
        uploaded = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "ogg"], key="audio_file")
        audio = microphone or uploaded
        if audio and st.button("Transcribe audio", type="primary"):
            try:
                handle_audio(audio)
            except Exception as error:
                st.error(f"Audio processing failed. Verify Whisper and FFmpeg: {error}")
        if audio and st.session_state.get("voice_result") and st.session_state.get("voice_token") == hashlib.sha256(audio.getvalue()).hexdigest():
            result = st.session_state.voice_result
            st.text_area("Transcription", value=result["text"], key="voice_transcript", height=100)
            st.caption(f"Detected language: {result['language']} · urgency proxy: {result['urgency_indicator']}. Not a validated stress detector.")
            if st.button("Analyze transcription", type="primary", key="analyze_voice"):
                duplicate = analyze_and_store(result["text"], "voice", voice=result)
                st.success("Merged with nearby report." if duplicate else "Added to queue.")
                show_incident(st.session_state.current_incident)
    with image_tab:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        description = st.text_area("Optional incident description", key="image_text")
        if uploaded and st.button("Inspect image", type="primary"):
            try:
                handle_image(uploaded, description)
            except Exception as error:
                st.error(f"Image analysis failed. Verify Ultralytics/model weights: {error}")
        if uploaded and st.session_state.get("image_result") and st.session_state.get("image_token") == hashlib.sha256(uploaded.getvalue()).hexdigest():
            result = st.session_state.image_result
            st.write({key:value for key,value in result.items() if key != "gps"})
            st.caption("Generic YOLO identifies people and common vehicles only. Fire, smoke, and crash damage need custom weights.")
            if st.button("Analyze image report", type="primary"):
                duplicate = analyze_and_store(description.strip() or "Uploaded emergency image", "image", image=result)
                st.success("Merged with nearby report." if duplicate else "Added to queue.")
                show_incident(st.session_state.current_incident)


def render_map(records: list[dict], geo) -> None:
    import folium
    from streamlit_folium import st_folium
    items = [row["incident"] for row in records if row["incident"].location.lat is not None]
    center = (items[0].location.lat, items[0].location.lon) if items else (17.4126, 78.5265)
    view = folium.Map(location=center, zoom_start=12, control_scale=True)
    colors = {"Critical":"red", "High":"orange", "Medium":"beige", "Low":"green"}
    for incident in items:
        folium.Marker(
            [incident.location.lat, incident.location.lon],
            tooltip=f"{incident.incident_id} · {incident.priority}",
            popup=f"{', '.join(incident.incident_types)} · severity {incident.severity.score}",
            icon=folium.Icon(color=colors.get(incident.severity.level, "blue"), icon="warning-sign"),
        ).add_to(view)
    for team in geo.mock_data.get("response_teams", []):
        folium.CircleMarker([team["lat"], team["lon"]], radius=5, color="#12665c", fill=True, tooltip=team["name"]).add_to(view)
    st_folium(view, height=460, returned_objects=[])


def render_queue() -> None:
    _, store, geo, _ = services()
    records = store.list_incidents()
    st.subheader("Live priority queue")
    if not records:
        st.info("No reports yet.")
        render_map([], geo)
        return
    rank = {"Immediate":0, "High":1, "Elevated":2, "Routine":3}
    records.sort(key=lambda row:(rank.get(row["incident"].priority,4), row["incident"].created_at))
    rows = []
    for record in records:
        item = record["incident"]
        remaining = max(0, round((item.created_at.timestamp() + item.sla_minutes * 60 - datetime.now(timezone.utc).timestamp()) / 60))
        rows.append({"ID":item.incident_id, "Priority":item.priority, "Type":", ".join(item.incident_types) or "Other", "Severity":item.severity.score, "Location":item.location.text or "Needs location", "Victims":item.victims.count if item.victims.count is not None else "Unknown", "Team":item.assigned_team.name or "Pending", "SLA min":remaining, "Status":item.status})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    render_map(records, geo)
    st.subheader("Dispatcher controls")
    for record in records:
        item = record["incident"]
        with st.expander(f"{item.incident_id} · {item.priority} · {item.status}"):
            states = ["Reported", "Dispatched", "En route", "Resolved"]
            state = st.selectbox("Status", states, index=states.index(item.status), key=f"state_{item.incident_id}")
            if st.button("Save status", key=f"save_{item.incident_id}"):
                store.update_status(item.incident_id, state)
                st.rerun()
            allowed = item.follow_up_question is None and item.assigned_team.name is not None
            if st.button("Approve recommendation (demo only)", key=f"approve_{item.incident_id}", disabled=not allowed):
                store.record_override(item.incident_id, "dispatcher", "dispatch_approved_demo_only", {"team":item.assigned_team.name})
                store.update_status(item.incident_id, "Dispatched")
                notice = send_notification(item)
                st.info(f"Approval logged. Notification mode: {notice['mode']}; no real dispatch was sent.")
                st.rerun()
            priorities = ["Immediate", "High", "Elevated", "Routine"]
            chosen = st.selectbox("Priority override", priorities, index=priorities.index(item.priority), key=f"priority_{item.incident_id}")
            if st.button("Apply override", key=f"override_{item.incident_id}"):
                store.set_priority(item.incident_id, chosen, actor="dispatcher")
                st.rerun()


def render_analytics() -> None:
    _, store, _, _ = services()
    records = store.list_incidents()
    if not records:
        st.info("Submit reports to populate analytics.")
        return
    import plotly.express as px
    import numpy as np
    from sklearn.cluster import DBSCAN
    incidents = [row["incident"] for row in records]
    types = [kind for item in incidents for kind in (item.incident_types or ["Other"])]
    counts = pd.Series(types).value_counts().rename_axis("Incident type").reset_index(name="Reports")
    st.plotly_chart(px.bar(counts, x="Incident type", y="Reports", color="Incident type"), width="stretch")
    audit = store.audit_events(limit=1000)
    response = []
    for item in incidents:
        events = [event for event in audit if event["incident_id"] == item.incident_id and event["action"] == "status_updated" and event["details"].get("status") in {"Dispatched", "En route"}]
        if events:
            elapsed = (datetime.fromisoformat(events[-1]["created_at"]) - item.created_at).total_seconds() / 60
            response.append({"Incident":item.incident_id, "Minutes to dispatch":max(0, elapsed)})
    if response:
        st.plotly_chart(px.bar(pd.DataFrame(response), x="Incident", y="Minutes to dispatch", title="Dispatcher status-change time"), width="stretch")
    else:
        st.caption("No dispatch status-change times recorded yet.")
    located = [item for item in incidents if item.location.lat is not None and item.location.lon is not None]
    if len(located) >= 2:
        coordinates = np.radians([[item.location.lat, item.location.lon] for item in located])
        labels = DBSCAN(eps=2 / 6371, min_samples=2, metric="haversine").fit_predict(coordinates)
        data = pd.DataFrame({"lat":[i.location.lat for i in located], "lon":[i.location.lon for i in located], "cluster":["Noise" if n < 0 else f"Hotspot {n+1}" for n in labels]})
        st.plotly_chart(px.scatter_map(data, lat="lat", lon="lon", color="cluster", zoom=10), width="stretch")
    else:
        st.caption("At least two geocoded incidents are needed to preview DBSCAN hotspots.")


def main() -> None:
    st.markdown('<div class="eyebrow">Human-in-the-loop emergency triage</div>', unsafe_allow_html=True)
    st.title("Response Coordinator")
    st.caption("Decision support only. For a real emergency, contact local emergency services directly.")
    try:
        _, store, geo, _ = services()
        map_name = type(geo).__name__.replace("Provider", "")
        map_status = "live services enabled" if geo.enabled else "using mock services (Google key missing/placeholder or lookups disabled)"
        st.caption(
            f"Reports: {len(store.list_incidents())} · retention: {RETENTION_DAYS} days · "
            f"map: {map_name} ({map_status}) · model downloads: {'enabled' if ALLOW_MODEL_DOWNLOAD else 'disabled'}"
        )
        intake_tab, queue_tab, analytics_tab, audit_tab = st.tabs(["Report intake", "Operations", "Analytics", "Audit"])
        with intake_tab:
            render_intake()
        with queue_tab:
            render_queue()
        with analytics_tab:
            render_analytics()
        with audit_tab:
            st.caption("Reporter phone numbers are masked before storage; records expire after the configured retention period.")
            st.dataframe(pd.DataFrame(services()[1].audit_events()), width="stretch", hide_index=True)
            if st.button("Apply retention cleanup"):
                removed = store.cleanup_expired(RETENTION_DAYS)
                st.success(f"Removed {removed} expired incident(s).")
    except Exception as error:
        st.error(f"Dashboard initialization failed: {error}")


if __name__ == "__main__":
    main()