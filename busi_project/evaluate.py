from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)


def evaluate_model(model, generator, output_dir: Path):
    generator.reset()
    probabilities = model.predict(generator, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    labels = generator.classes
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "mae": mean_absolute_error(labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "auc": roc_auc_score(labels, probabilities, multi_class="ovr"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.txt").write_text(
        "\n".join(f"{key}: {value:.6f}" for key, value in result.items()), encoding="utf-8"
    )
    matrix = confusion_matrix(labels, predictions)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=generator.class_indices, yticklabels=generator.class_indices)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()
    return result
