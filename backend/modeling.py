"""Fit three classifiers on training rows and choose one using validation rows."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
NEAR_F1 = 0.005
NEAR_RECALL = 0.01

# Lower number means a simpler model. Used only when scores are nearly the same.
SIMPLICITY = {
    "Logistic Regression": 0,
    "K-Nearest Neighbors": 1,
    "Random Forest": 2,
}


def score_predictions(y_true, y_pred):
    """Score phishing as the positive class (label 1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def select_best_model(rows):
    """Choose a model from validation scores only.

    1. The main score is the phishing-class F1.
    2. If F1 scores are within 0.005, use phishing recall as the tie-breaker.
    3. If recall is also within 0.01, choose the simpler model.
    """
    best_f1 = max(row["f1"] for row in rows)
    contenders = [row for row in rows if best_f1 - row["f1"] <= NEAR_F1]
    if len(contenders) == 1:
        winner = contenders[0]
        reason = f"{winner['model']} had the highest phishing validation F1 ({best_f1:.4f})."
    else:
        best_recall = max(row["recall"] for row in contenders)
        finalists = [row for row in contenders if best_recall - row["recall"] <= NEAR_RECALL]
        winner = min(finalists, key=lambda row: SIMPLICITY[row["model"]])
        if len(finalists) == 1:
            reason = f"{winner['model']} won on phishing recall after a near tie on F1."
        else:
            reason = f"{winner['model']} is the simpler model among those close on F1 and recall."
    return winner["model"], reason, winner


def candidate_models():
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    neighbors = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ])
    forest = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    return [
        ("Logistic Regression", logistic),
        ("K-Nearest Neighbors", neighbors),
        ("Random Forest", forest),
    ]


def train_candidates(X_train, y_train, X_val, y_val):
    fitted = {}
    validation_rows = []
    validation_reports = []
    for name, model in candidate_models():
        print("Training", name + "...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        scores = score_predictions(y_val, predictions)
        scores["model"] = name
        validation_rows.append(scores)
        fitted[name] = model
        report = classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["SAFE", "PHISHING"],
            digits=4,
            zero_division=0,
        )
        validation_reports.append("VALIDATION — " + name + "\n" + report)
        print(name, "validation F1:", format(scores["f1"], ".4f"))

    print("Validation comparison (these numbers choose the model):")

    chosen_name, reason, chosen_scores = select_best_model(validation_rows)
    chosen_model = fitted[chosen_name]
    print("Selected model:", chosen_name)
    print("Reason:", reason)
    return chosen_model, chosen_name, reason, chosen_scores, validation_rows, validation_reports
