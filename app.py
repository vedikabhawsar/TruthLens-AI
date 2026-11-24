
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import json
import re

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "MODEL"

# Load trained ML model
pipeline = joblib.load(MODEL_DIR / "fake_news_pipeline.pkl")

# Load model evaluation metrics
with open(MODEL_DIR / "metrics.json", "r") as file:
    metrics = json.load(file)

app = Flask(__name__)

# Maximum request size
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


# Text cleaning (same preprocessing used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Homepage
@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


# News analysis
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True) or {}
    news_text = data.get("text", "")

    if not isinstance(news_text, str):
        return jsonify({
            "success": False,
            "message": "Please enter valid news text."
        }), 400

    news_text = news_text.strip()

    if len(news_text) > 10000:
        return jsonify({
            "success": False,
            "message": "Text is too long. Please keep it under 10,000 characters."
        }), 400

    word_count = len(news_text.split())

    if word_count < 20:
        return jsonify({
            "success": False,
            "message": "Please enter at least 20 words for analysis."
        }), 400

    cleaned_text = clean_text(news_text)

    if not cleaned_text:
        return jsonify({
            "success": False,
            "message": "The text does not contain enough readable content."
        }), 400

    # Generate prediction
    prediction = int(pipeline.predict([cleaned_text])[0])

    # Calculate model confidence
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    confidence = float(max(probabilities)) * 100

    label = "REAL" if prediction == 1 else "FAKE"

    return jsonify({
        "success": True,
        "prediction": label,
        "confidence": round(confidence, 2),
        "word_count": word_count,
        "disclaimer": (
            "This is a machine-learning prediction, not an independent "
            "verification of the news. Always check reliable sources."
        )
    })


# Handle oversized requests
@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "success": False,
        "message": "The submitted text is too large."
    }), 413


if __name__ == "__main__":
    app.run(debug=False)
