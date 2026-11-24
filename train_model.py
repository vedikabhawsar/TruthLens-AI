
import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "DATASET"
MODEL_DIR = BASE_DIR / "MODEL"
STATIC_DIR = BASE_DIR / "static"

MODEL_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 3. LOAD DATASETS
# ==========================================

print("Loading datasets...")

fake = pd.read_csv(DATASET_DIR / "Fake.csv")
true = pd.read_csv(DATASET_DIR / "True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

original_dataset_size = len(data)

print("Original records:", original_dataset_size)


# ==========================================
# 4. PREPARE NEWS CONTENT
# ==========================================

data["content"] = (
    data["title"].fillna("").astype(str)
    + " "
    + data["text"].fillna("").astype(str)
)

data["content"] = data["content"].apply(clean_text)

data = data[data["content"].str.len() > 0].copy()


# ==========================================
# 5. HANDLE CONFLICTING LABELS
# ==========================================

label_counts = data.groupby("content")["label"].nunique()

conflicting_contents = set(
    label_counts[label_counts > 1].index
)

ambiguous_records_removed = int(
    data["content"].isin(conflicting_contents).sum()
)

data = data[
    ~data["content"].isin(conflicting_contents)
].copy()


# ==========================================
# 6. REMOVE EXACT DUPLICATES
# ==========================================

records_before_deduplication = len(data)

data = data.drop_duplicates(
    subset=["content"],
    keep="first"
).copy()

duplicates_removed = (
    records_before_deduplication - len(data)
)

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nDATASET CLEANING REPORT")
print("--------------------------------")

print("Original records:", original_dataset_size)

print(
    "Ambiguous records removed:",
    ambiguous_records_removed
)

print("Duplicate records removed:", duplicates_removed)

print("Final unique records:", len(data))


# ==========================================
# 7. PREPARE FEATURES AND LABELS
# ==========================================

X = data["content"]
y = data["label"].astype(int)


# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. MACHINE LEARNING MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                stop_words="english"
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]),

    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                stop_words="english"
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

}


# ==========================================
# 10. TRAIN AND EVALUATE
# ==========================================

results = {}
predictions = {}

print("\nTRAINING MODELS")
print("--------------------------------")

for name, pipeline in models.items():

    print("\nTraining:", name)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = float(accuracy)

    predictions[name] = y_pred

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

best_name = max(results, key=results.get)

best_pipeline = models[best_name]

best_predictions = predictions[best_name]

print("\nSelected Model:", best_name)


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    output_dict=True,
    zero_division=0
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)


# ==========================================
# 14. ROC-AUC ANALYSIS
# ==========================================

y_probability = best_pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("TruthLens AI - ROC Curve")

plt.legend()
plt.tight_layout()

plt.savefig(
    STATIC_DIR / "roc_curve.png",
    dpi=150
)

plt.close()


# ==========================================
# 15. SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    best_pipeline,
    MODEL_DIR / "fake_news_pipeline.pkl"
)


# ==========================================
# 16. SAVE EVALUATION METRICS
# ==========================================

metrics = {

    "best_model": best_name,

    "accuracy": float(results[best_name]),

    "roc_auc": float(roc_auc),

    "original_dataset_size": int(original_dataset_size),

    "dataset_size": int(len(data)),

    "duplicates_removed": int(duplicates_removed),

    "ambiguous_records_removed": int(
        ambiguous_records_removed
    ),

    "test_size": int(len(y_test)),

    "confusion_matrix": [
        [int(value) for value in row]
        for row in cm
    ],

    "classification_report": report,

    "model_comparison": results

}


with open(
    MODEL_DIR / "metrics.json",
    "w"
) as file:

    json.dump(metrics, file, indent=4)


# ==========================================
# 17. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 45)

print("TRAINING COMPLETED SUCCESSFULLY!")

print("=" * 45)

print("Selected Model:", best_name)

print(
    "Accuracy:",
    round(results[best_name] * 100, 2),
    "%"
)

print("ROC-AUC:", round(roc_auc, 4))

print("Unique Dataset Records:", len(data))

print("Duplicate Records Removed:", duplicates_removed)

print("Model saved successfully!")

print("Evaluation metrics saved successfully!")

print("ROC Curve generated successfully!")

print("=" * 45)
