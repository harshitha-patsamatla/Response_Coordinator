"""Fine-tune multilingual XLM-R for multi-label incident classification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

LABELS = ["Road Accident", "Fire", "Medical", "Crime", "Flood", "Building Collapse", "Other"]


def read_dataset(path: Path) -> tuple[list[str], np.ndarray]:
    texts, targets = [], []
    with path.open(encoding="utf-8", newline="") as data_file:
        for row in csv.DictReader(data_file):
            selected = set(row["labels"].split("|"))
            texts.append(row["text"])
            targets.append([float(label in selected) for label in LABELS])
    return texts, np.asarray(targets, dtype=np.float32)


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_reports.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/emergency-xlm-roberta"))
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args()
    texts, targets = read_dataset(arguments.dataset)
    train_text, eval_text, train_targets, eval_targets = train_test_split(
        texts, targets, test_size=0.2, random_state=41
    )
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)

    class ReportDataset(torch.utils.data.Dataset):
        def __init__(self, examples, labels):
            self.examples = tokenizer(examples, truncation=True, max_length=192)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.examples.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
            return item

    id_to_label = dict(enumerate(LABELS))
    label_to_id = {label: index for index, label in id_to_label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        arguments.model_name,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
        problem_type="multi_label_classification",
    )

    def metrics(result):
        predicted = 1 / (1 + np.exp(-result.predictions)) >= 0.5
        expected = result.label_ids.astype(bool)
        return {
            "precision_macro": precision_score(expected, predicted, average="macro", zero_division=0),
            "recall_macro": recall_score(expected, predicted, average="macro", zero_division=0),
            "f1_macro": f1_score(expected, predicted, average="macro", zero_division=0),
            "f1_micro": f1_score(expected, predicted, average="micro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(arguments.output / "checkpoints"),
            num_train_epochs=arguments.epochs,
            per_device_train_batch_size=arguments.batch_size,
            per_device_eval_batch_size=arguments.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="none",
        ),
        train_dataset=ReportDataset(train_text, train_targets),
        eval_dataset=ReportDataset(eval_text, eval_targets),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    prediction = trainer.predict(ReportDataset(eval_text, eval_targets))
    predicted = 1 / (1 + np.exp(-prediction.predictions)) >= 0.5
    evaluation = {
        "metrics": metrics(prediction),
        "labels": LABELS,
        "multilabel_confusion_matrix": multilabel_confusion_matrix(eval_targets.astype(bool), predicted).tolist(),
        "note": "Synthetic-data metrics are not evidence of real-world accuracy.",
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(arguments.output)
    tokenizer.save_pretrained(arguments.output)
    (arguments.output / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()