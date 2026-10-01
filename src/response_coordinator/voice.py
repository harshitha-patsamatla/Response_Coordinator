"""Whisper transcription and a deliberately limited acoustic urgency proxy."""

from __future__ import annotations

from pathlib import Path


def transcribe_audio(audio_path: str | Path, model_name: str = "base") -> dict:
    import whisper
    import numpy as np

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), task="transcribe", fp16=False)
    waveform = whisper.load_audio(str(audio_path))
    duration_seconds = len(waveform) / whisper.audio.SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if len(waveform) else 0.0
    words_per_second = len(result.get("text", "").split()) / max(duration_seconds, 1.0)
    # This is a coarse signal-quality/urgency hint, not a validated stress detector.
    proxy_score = min(1.0, rms * 4 + max(0.0, words_per_second - 2.0) * 0.12)
    level = "High" if proxy_score >= 0.65 else "Elevated" if proxy_score >= 0.35 else "Unclear"
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
        "urgency_indicator": level,
        "urgency_score": round(proxy_score, 2),
        "urgency_note": "Audio loudness and speaking rate are weak proxies and are not validated for stress.",
        "duration_seconds": round(duration_seconds, 1),
    }