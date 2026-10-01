"""Multilingual transformer classifier with a transparent offline fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LABELS = [
    "Road Accident",
    "Fire",
    "Medical",
    "Crime",
    "Flood",
    "Building Collapse",
    "Other",
]

RULE_TERMS = {
    "Road Accident": ["accident", "crash", "collision", "bike", "రోడ్డు ప్రమాదం", "ప్రమాదం", "दुर्घटना"],
    "Fire": ["fire", "smoke", "burning", "మంట", "आग", "धुआँ"],
    "Medical": ["injured", "unconscious", "bleeding", "heart", "medical", "గాయం", "चोट"],
    "Crime": ["robbery", "assault", "attack", "gun", "దాడి", "हमला"],
    "Flood": ["flood", "water rising", "submerged", "వరద", "बाढ़"],
    "Building Collapse": ["collapsed", "collapse", "building fell", "కూలింది", "इमारत गिर"],
}


@dataclass
class Classification:
    labels: list[tuple[str, float]]
    backend: str

    @property
    def confidence(self) -> float:
        return self.labels[0][1] if self.labels else 0.25


class EmergencyClassifier:
    """Lazily use a fine-tuned XLM-R or multilingual zero-shot transformer."""

    def __init__(self, model_dir: Path, allow_download: bool = True) -> None:
        self.model_dir = model_dir
        self.allow_download = allow_download
        self._pipeline = None
        self._backend = "rules"

    def predict(self, text: str, threshold: float = 0.28) -> Classification:
        if not text.strip():
            return Classification([("Other", 0.2)], "rules")
        try:
            runner = self._transformer()
            result = runner.predict(text)
            labels = [(name, float(score)) for name, score in result if score >= threshold]
            if not labels and result:
                labels = [result[0]]
            return Classification(labels or [("Other", 0.2)], runner.backend)
        except (ImportError, OSError, RuntimeError, ValueError):
            return Classification(self._rule_predict(text), "multilingual-rules")

    def _rule_predict(self, text: str) -> list[tuple[str, float]]:
        normalized = text.casefold()
        matches = []
        for label, terms in RULE_TERMS.items():
            count = sum(term.casefold() in normalized for term in terms)
            if count:
                matches.append((label, min(0.58 + 0.12 * (count - 1), 0.88)))
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches or [("Other", 0.35)]

    def _transformer(self):
        if self._pipeline is None:
            self._pipeline = _TransformerRunner(self.model_dir, self.allow_download)
        return self._pipeline


class _TransformerRunner:
    def __init__(self, model_dir: Path, allow_download: bool) -> None:
        from transformers import pipeline

        if model_dir.exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
            self.multilabel = True
            self.backend = "fine-tuned-xlm-roberta"
        elif allow_download:
            self.pipe = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                device=-1,
            )
            self.multilabel = False
            self.backend = "multilingual-zero-shot"
        else:
            raise OSError("Model downloads are disabled and no trained checkpoint exists")

    def predict(self, text: str) -> list[tuple[str, float]]:
        if self.multilabel:
            output = self.pipe(text, function_to_apply="sigmoid", truncation=True)[0]
            return sorted(
                ((item["label"], float(item["score"])) for item in output),
                key=lambda pair: pair[1],
                reverse=True,
            )
        output = self.pipe(text, candidate_labels=LABELS, multi_label=True)
        return sorted(zip(output["labels"], output["scores"]), key=lambda pair: pair[1], reverse=True)