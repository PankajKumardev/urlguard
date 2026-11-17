"""Train three classic models and keep the one with the best validation F1.

Run this after download_data.py. It does not start the website.
The test set is scored once, after the model has already been chosen.
"""

import hashlib
import json
import os
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import FEATURE_INFO, FEATURE_NAMES, feature_vector, normalize_url
from modeling import NEAR_F1, NEAR_RECALL, RANDOM_STATE, score_predictions, select_best_model, train_candidates
from write_docs import write_all_docs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "urls.csv")
SOURCE_INFO_PATH = os.path.join(BASE_DIR, "data", "source_info.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

SAMPLE_SIZE = 30000
OFFICIAL_LABEL_SENTENCE = "Label 1 corresponds to a legitimate URL, label 0 to a phishing URL"


def clean_dataset(df):
    """Drop missing, invalid, conflicting, and duplicate URL rows.

    Conflicting labels are removed before duplicate rows. If duplicates were
    deleted first, one of the disagreeing labels could be kept by accident.
    """
    summary = {"original_rows": int(len(df))}
    df = df.copy()
    numeric_labels = pd.to_numeric(df["label"], errors="coerce")
    summary["file_label_1"] = int((numeric_labels == 1).sum())
    summary["file_label_0"] = int((numeric_labels == 0).sum())

    df["label"] = numeric_labels
    df["url"] = df["url"].astype("string").str.strip()
    missing = df["url"].isna() | (df["url"] == "") | df["label"].isna()
    summary["removed_missing"] = int(missing.sum())
    df = df.loc[~missing].copy()

    bad_label = ~df["label"].isin([0, 1])
    summary["removed_bad_label"] = int(bad_label.sum())
    df = df.loc[~bad_label].copy()
    df["label"] = df["label"].astype(int)
    df["url"] = df["url"].astype(str)

    normalized_urls = []
    for url in df["url"]:
        try:
            normalized_urls.append(normalize_url(url))
        except ValueError:
            normalized_urls.append(None)
    df["url"] = normalized_urls
    summary["removed_invalid"] = int(df["url"].isna().sum())
    df = df.dropna(subset=["url"])

    label_variety = df.groupby("url")["label"].nunique()
    conflict_urls = label_variety[label_variety > 1].index
    conflict_rows = df["url"].isin(conflict_urls)
    summary["conflict_url_count"] = int(len(conflict_urls))
    summary["removed_conflicts"] = int(conflict_rows.sum())
    df = df.loc[~conflict_rows].copy()

    summary["duplicate_count"] = int(df.duplicated(subset=["url"]).sum())
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    summary["removed_invalid_rows"] = (
        summary["removed_missing"] + summary["removed_bad_label"] + summary["removed_invalid"]
    )
    summary["rows_before_label_conversion"] = int(len(df))
    summary["min_url_length"] = int(df["url"].str.len().min())
    summary["max_url_length"] = int(df["url"].str.len().max())
    summary["urls_longer_than_2048"] = int((df["url"].str.len() > 2048).sum())
    return df, summary


def convert_labels(df, source_info):
    """Flip the UCI codes into 0 = SAFE and 1 = PHISHING."""
    sentence = str(source_info.get("official_label_sentence", ""))
    if sentence != OFFICIAL_LABEL_SENTENCE or not source_info.get("label_mapping_confirmed"):
        raise SystemExit(
            "Refusing to convert labels. The source file does not confirm that "
            "1 = legitimate and 0 = phishing."
        )
    if int(source_info.get("downloaded_label_1", -1)) != 134850:
        raise SystemExit("Label 1 count does not match the published legitimate count (134,850).")
    if int(source_info.get("downloaded_label_0", -1)) != 100945:
        raise SystemExit("Label 0 count does not match the published phishing count (100,945).")

    converted = df.copy()
    converted["label"] = 1 - converted["label"].astype(int)
    return converted


def stratified_sample(df, sample_size, random_state):
    """Keep about sample_size real rows, with the same class mix as the cleaned data."""
    if len(df) <= sample_size:
        shuffled = df.sample(frac=1.0, random_state=random_state)
        return shuffled.reset_index(drop=True)

    labels = sorted(int(label) for label in df["label"].unique())
    proportions = df["label"].value_counts(normalize=True)
    raw = {label: float(proportions[label]) * sample_size for label in labels}
    allocation = {label: int(np.floor(raw[label])) for label in labels}
    leftover = sample_size - sum(allocation.values())
    order = sorted(labels, key=lambda label: (raw[label] - allocation[label], label), reverse=True)
    for label in order:
        if leftover == 0:
            break
        allocation[label] += 1
        leftover -= 1

    parts = []
    for label in labels:
        group = df[df["label"] == label]
        take = min(allocation[label], len(group))
        parts.append(group.sample(n=take, random_state=random_state))
    sampled = pd.concat(parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def make_feature_matrix(urls):
    rows = [feature_vector(url) for url in urls]
    matrix = np.asarray(rows, dtype=float)
    if np.isnan(matrix).any():
        raise SystemExit("Feature matrix contains missing values.")
    return matrix


def phishing_probability_from_model(model, matrix):
    """Read the probability column that belongs to class 1."""
    probabilities = model.predict_proba(matrix)
    classes = [int(label) for label in model.classes_]
    if 1 not in classes:
        raise SystemExit("The trained model does not contain the phishing class.")
    return probabilities[:, classes.index(1)]


def mask_phishing_url(url):
    """Hide a phishing URL before it is written into the report."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "redacted"
    if len(host) <= 4:
        masked_host = "xxxx"
    else:
        masked_host = host[:2] + "****" + host[-2:]
    return parsed.scheme + "://" + masked_host + "/[redacted-path]"


def choose_example(urls, y_true, y_pred, model, matrix):
    """Pick one real held-out row and hide the URL when it is phishing."""
    chosen = None
    fallback = None
    for index in range(len(y_true)):
        if int(y_true[index]) == 1 and fallback is None:
            fallback = index
        if int(y_true[index]) == 1 and int(y_pred[index]) == 1:
            chosen = index
            break
    if chosen is None:
        chosen = 0 if fallback is None else fallback

    true_label = int(y_true[chosen])
    predicted_label = int(y_pred[chosen])
    probability = float(phishing_probability_from_model(model, matrix[chosen : chosen + 1])[0])
    full_probabilities = model.predict_proba(matrix[chosen : chosen + 1])[0]
    classes = [int(label) for label in model.classes_]
    safe_probability = float(full_probabilities[classes.index(0)])
    if predicted_label == 1 and probability < safe_probability:
        raise SystemExit("Phishing probability does not match the predicted class.")
    if predicted_label == 0 and probability > safe_probability:
        raise SystemExit("Safe prediction does not match the probability columns.")

    raw_url = urls[chosen]
    if true_label == 1:
        shown = mask_phishing_url(raw_url)
        redacted = True
    elif len(raw_url) > 100:
        shown = raw_url[:80] + "..."
        redacted = False
    else:
        shown = raw_url
        redacted = False

    names = {0: "SAFE", 1: "PHISHING"}
    return {
        "source": "held-out test set",
        "url_display": shown,
        "url_was_redacted": redacted,
        "display_was_shortened": (not redacted) and shown != raw_url,
        "features": {
            name: int(matrix[chosen, feature_index])
            for feature_index, name in enumerate(FEATURE_NAMES)
        },
        "feature_vector": [int(value) for value in matrix[chosen].tolist()],
        "true_label": names[true_label],
        "predicted_label": names[predicted_label],
        "phishing_probability": round(probability, 6),
        "correct": true_label == predicted_label,
        "note": (
            "The original phishing URL is not printed. Features were calculated on the full URL before redaction."
            if redacted
            else "This safe URL comes from the held-out test split of the real dataset."
        ),
    }


def save_confusion_matrix(cm, model_name, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted SAFE", "Predicted PHISHING"])
    ax.set_yticklabels(["Actual SAFE", "Actual PHISHING"])
    cell_names = [["True negative", "False positive"], ["False negative", "True positive"]]
    darkest = cm.max() if cm.max() > 0 else 1
    for row in range(2):
        for column in range(2):
            color = "white" if cm[row, column] > darkest / 2 else "#1A1916"
            ax.text(
                column,
                row,
                str(int(cm[row, column])) + "\n" + cell_names[row][column],
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
    ax.set_title("Test confusion matrix — " + model_name)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_comparison_chart(rows, path):
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]
    colors = ["#1E4E8C", "#4F7CAC", "#C47B4A", "#2F6F4E"]
    x = np.arange(len(rows))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for index, (key, label) in enumerate(metric_keys):
        offset = (index - 1.5) * width
        values = [row[key] for row in rows]
        ax.bar(x + offset, values, width, label=label, color=colors[index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([row["model"] for row in rows])
    ax.set_title("Validation scores used for model selection")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def class_count(labels):
    values = np.asarray(labels)
    return int((values == 0).sum()), int((values == 1).sum())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_table(rows):
    print("")
    print(f"{'Model':<24} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for row in rows:
        print(
            f"{row['model']:<24} "
            f"{row['accuracy']:>10.4f} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f}"
        )
    print("")


def print_cleaning(summary):
    print("Original rows:", summary["original_rows"])
    print("Removed invalid rows:", summary["removed_invalid_rows"])
    print("  missing URL or label:", summary["removed_missing"])
    print("  bad label values:", summary["removed_bad_label"])
    print("  malformed URLs:", summary["removed_invalid"])
    print("Duplicate count:", summary["duplicate_count"])
    print("Removed conflicting-label rows:", summary["removed_conflicts"])
    print("Final rows:", summary["final_rows"])
    print("Legitimate URL count:", summary["safe_count"])
    print("Phishing URL count:", summary["phishing_count"])


def main():
    if not os.path.exists(DATA_PATH) or not os.path.exists(SOURCE_INFO_PATH):
        raise SystemExit("Dataset file not found. Run python download_data.py first.")

    with open(SOURCE_INFO_PATH, encoding="utf-8") as handle:
        source_info = json.load(handle)

    print("Loading", DATA_PATH)
    raw = pd.read_csv(DATA_PATH)
    cleaned, summary = clean_dataset(raw)
    cleaned = convert_labels(cleaned, source_info)
    summary["final_rows"] = int(len(cleaned))
    summary["safe_count"] = int((cleaned["label"] == 0).sum())
    summary["phishing_count"] = int((cleaned["label"] == 1).sum())
    print_cleaning(summary)
    print("Label conversion: phishing_label = 1 - original_label")
    print("Project labels: 0 = SAFE, 1 = PHISHING")

    sampled = stratified_sample(cleaned, SAMPLE_SIZE, RANDOM_STATE)
    print("Records selected for modeling:", len(sampled))
    print("This is a real stratified subset, not the full dataset, and not synthetic data.")

    urls = sampled["url"].tolist()
    # X holds the feature values. y holds the project labels: 0 = SAFE, 1 = PHISHING.
    y = sampled["label"].to_numpy(dtype=int)
    X = make_feature_matrix(urls)

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        url_train,
        url_temp,
    ) = train_test_split(
        X,
        y,
        urls,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    (
        X_val,
        X_test,
        y_val,
        y_test,
        url_val,
        url_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        url_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    overlap = len(set(url_train) & set(url_val))
    overlap += len(set(url_train) & set(url_test))
    overlap += len(set(url_val) & set(url_test))
    if overlap != 0:
        raise SystemExit("The same URL appeared in more than one split. Stopping.")
    if len(url_train) + len(url_val) + len(url_test) != len(sampled):
        raise SystemExit("Split sizes do not add up to the modeling sample.")

    train_safe, train_phishing = class_count(y_train)
    val_safe, val_phishing = class_count(y_val)
    test_safe, test_phishing = class_count(y_test)
    sample_safe, sample_phishing = class_count(y)
    print("Training rows:", len(y_train), "| SAFE", train_safe, "| PHISHING", train_phishing)
    print("Validation rows:", len(y_val), "| SAFE", val_safe, "| PHISHING", val_phishing)
    print("Test rows:", len(y_test), "| SAFE", test_safe, "| PHISHING", test_phishing)

    chosen_model, chosen_name, reason, chosen_scores, validation_rows, validation_reports = train_candidates(
        X_train, y_train, X_val, y_val
    )
    print_table(validation_rows)

    # The test set is used once, after selection. It is not used to choose again.
    print("Evaluating the selected model once on the held-out test set...")
    test_predictions = chosen_model.predict(X_test)
    test_scores = score_predictions(y_test, test_predictions)
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    true_negative = int(cm[0, 0])
    false_positive = int(cm[0, 1])
    false_negative = int(cm[1, 0])
    true_positive = int(cm[1, 1])
    test_report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["SAFE", "PHISHING"],
        digits=4,
        zero_division=0,
    )
    print("Test accuracy:", format(test_scores["accuracy"], ".4f"))
    print("Test precision:", format(test_scores["precision"], ".4f"))
    print("Test recall:", format(test_scores["recall"], ".4f"))
    print("Test F1:", format(test_scores["f1"], ".4f"))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    example = choose_example(url_test, y_test, test_predictions, chosen_model, X_test)
    print("Held-out example URL:", example["url_display"])
    print(
        "True:",
        example["true_label"],
        "| Predicted:",
        example["predicted_label"],
        "| Phishing probability:",
        example["phishing_probability"],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    uses_scaler = hasattr(chosen_model, "named_steps") and "scaler" in getattr(chosen_model, "named_steps", {})
    bundle = {
        "model": chosen_model,
        "model_name": chosen_name,
        "feature_names": list(FEATURE_NAMES),
        "positive_class": 1,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "uses_internal_scaler": bool(uses_scaler),
    }
    joblib.dump(bundle, MODEL_PATH)
    loaded = joblib.load(MODEL_PATH)
    reloaded = loaded["model"].predict(X_test[:5])
    if not np.array_equal(reloaded, test_predictions[:5]):
        raise SystemExit("The saved model did not reload with the same predictions.")
    if loaded["feature_names"] != list(FEATURE_NAMES):
        raise SystemExit("Saved feature order does not match features.py.")
    print("Saved and reloaded model:", MODEL_PATH)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    comparison_frame = pd.DataFrame(validation_rows)[["model", *metric_names]].round(6)
    rounded_validation = comparison_frame.to_dict("records")
    comparison_frame.columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_frame.to_csv(comparison_path, index=False)

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    save_confusion_matrix(cm, chosen_name, cm_path)
    save_comparison_chart(rounded_validation, chart_path)

    hash_rows = []
    for split_name, split_urls in (
        ("train", url_train),
        ("validation", url_val),
        ("test", url_test),
    ):
        for url in split_urls:
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
            hash_rows.append({"split": split_name, "url_sha256": digest})
    hash_frame = pd.DataFrame(hash_rows)
    if hash_frame["url_sha256"].duplicated().any():
        raise SystemExit("URL hash overlap found while saving the split check.")
    hash_frame.to_csv(os.path.join(RESULTS_DIR, "split_assignments.csv"), index=False)

    report_text = []
    report_text.append("URLGuard cleaning summary")
    report_text.append("Original rows: " + str(summary["original_rows"]))
    report_text.append("Removed invalid rows: " + str(summary["removed_invalid_rows"]))
    report_text.append("Duplicate count: " + str(summary["duplicate_count"]))
    report_text.append("Removed conflicting-label rows: " + str(summary["removed_conflicts"]))
    report_text.append("Final rows: " + str(summary["final_rows"]))
    report_text.append("Legitimate URL count: " + str(summary["safe_count"]))
    report_text.append("Phishing URL count: " + str(summary["phishing_count"]))
    report_text.append("Modeling rows: " + str(len(sampled)))
    report_text.append("Training rows: " + str(len(y_train)))
    report_text.append("Validation rows: " + str(len(y_val)))
    report_text.append("Test rows: " + str(len(y_test)))
    with open(os.path.join(RESULTS_DIR, "cleaning_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_text) + "\n")

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(validation_reports))
        handle.write("\n\nTEST — " + chosen_name + "\n")
        handle.write(test_report)

    hyperparameters = {
        "Logistic Regression": {
            "scaler": "StandardScaler inside a Pipeline, fit on training data only",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
        "K-Nearest Neighbors": {
            "scaler": "StandardScaler inside a Pipeline, fit on training data only",
            "n_neighbors": 5,
        },
        "Random Forest": {
            "n_estimators": 100,
            "max_depth": 20,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "scaler": "None. Trees do not need scaled features.",
        },
    }

    metrics = {
        "project": "URLGuard — Machine Learning Based Phishing URL Detection System",
        "college": "Jaypee Institute of Information Technology, Noida",
        "department": "Computer Applications",
        "course": "Machine Learning Lab",
        "program": "MCA, 1st Year, 1st Semester",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "sample_size_target": SAMPLE_SIZE,
        "near_tie_f1": NEAR_F1,
        "near_tie_recall": NEAR_RECALL,
        "simplicity_order": ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"],
        "positive_class": {"name": "PHISHING", "value": 1},
        "negative_class": {"name": "SAFE", "value": 0},
        "model_name": chosen_name,
        "selection_reason": reason,
        "uses_internal_scaler": bool(uses_scaler),
        "feature_names": list(FEATURE_NAMES),
        "features": FEATURE_INFO,
        "hyperparameters": hyperparameters,
        "dataset": {
            "name": source_info.get("dataset_name", "PhiUSIIL Phishing URL Dataset"),
            "uci_id": source_info.get("uci_id", 967),
            "official_url": source_info.get("official_url"),
            "citation": source_info.get("citation"),
            "paper": source_info.get("paper"),
            "authors": source_info.get("authors"),
            "license": source_info.get("license"),
            "license_url": source_info.get("license_url"),
            "doi": source_info.get("doi"),
            "confirmation_method": source_info.get("confirmation_method"),
            "official_label_sentence": source_info.get("official_label_sentence"),
            "published_rows": source_info.get("published_rows"),
            "published_legitimate": source_info.get("published_legitimate"),
            "published_phishing": source_info.get("published_phishing"),
            "file_sha256": sha256_file(DATA_PATH),
            "download_method": source_info.get("download_method"),
        },
        "label_mapping": {
            "original": "1 = legitimate, 0 = phishing",
            "project": "0 = SAFE, 1 = PHISHING",
            "conversion": "phishing_label = 1 - original_label",
        },
        "cleaning": summary,
        "counts": {
            "modeling_rows": int(len(sampled)),
            "modeling_safe": sample_safe,
            "modeling_phishing": sample_phishing,
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_safe": train_safe,
            "train_phishing": train_phishing,
            "validation_safe": val_safe,
            "validation_phishing": val_phishing,
            "test_safe": test_safe,
            "test_phishing": test_phishing,
            "overlap_count": int(overlap),
        },
        "validation_comparison": rounded_validation,
        "selected_validation": {
            "accuracy": round(float(chosen_scores["accuracy"]), 6),
            "precision": round(float(chosen_scores["precision"]), 6),
            "recall": round(float(chosen_scores["recall"]), 6),
            "f1": round(float(chosen_scores["f1"]), 6),
        },
        "test_metrics": {
            "accuracy": round(float(test_scores["accuracy"]), 6),
            "precision": round(float(test_scores["precision"]), 6),
            "recall": round(float(test_scores["recall"]), 6),
            "f1": round(float(test_scores["f1"]), 6),
            "confusion_matrix": [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "prediction_example": example,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "method": "stratified"},
        "warnings": [
            "Scores describe a stratified subset of about 30,000 real records, not the full 235,795-row dataset.",
            "A random split of URLs can still place related domains in both training and testing.",
            "The phishing probability is a model score, not a calibrated guarantee of real-world risk.",
        ],
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("Saved", metrics_path)

    write_all_docs(metrics)
    print("Wrote MODEL_REPORT.md, DATASET.md, VIVA.md, and README.md")
    print("Training finished. Start the website with: python app.py")


if __name__ == "__main__":
    main()
