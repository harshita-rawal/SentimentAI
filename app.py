"""
app.py — Flask backend for Sentiment Analysis
=============================================
Start the server with:
    python app.py        (local dev)
    gunicorn app:app     (production)
"""

import os
import csv
import pickle
import subprocess
import numpy as np

from flask import Flask, render_template, request, jsonify

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join("model", "tokenizer.pkl")
TOKENIZER_JSON = os.path.join("model", "tokenizer.json")
MAX_LEN        = 50
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


# --------------------------------------------------------------------------- #
#  Auto-train model at startup if files are missing                            #
# --------------------------------------------------------------------------- #
def ensure_model():
    if not os.path.exists(MODEL_PATH) or (not os.path.exists(TOKENIZER_PATH) and not os.path.exists(TOKENIZER_JSON)):
        print("[INFO] Model files not found — running train.py ...")
        result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] Training failed:")
            print(result.stderr)
            raise RuntimeError("Model training failed. Check train.py and dataset.")
        print("[INFO] Training complete.")


# --------------------------------------------------------------------------- #
#  Load model & tokenizer once at startup                                      #
# --------------------------------------------------------------------------- #
ensure_model()

print("[INFO] Loading model and tokenizer ...")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

model = load_model(MODEL_PATH)
if os.path.exists(TOKENIZER_JSON):
    with open(TOKENIZER_JSON, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
else:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

# Warm up the computation graph so the first user request doesn't incur compilation lag
try:
    dummy = pad_sequences([[0]], maxlen=MAX_LEN, padding="pre")
    _ = model(dummy, training=False)
    print("[INFO] Model warmed up and ready!")
except Exception as e:
    print(f"[WARN] Warmup skipped: {e}")
    print("[INFO] Ready!")


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #
def predict_sentiment(text: str) -> dict:
    """Return label, confidence, and a friendly message."""
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="post")
    # Direct callable is significantly faster than model.predict for single samples
    score  = float(model(padded, training=False).numpy()[0][0])

    if score >= 0.5:
        label      = "Positive"
        confidence = score
        emoji      = "😊"
        message    = "This review has a positive sentiment!"
    else:
        label      = "Negative"
        confidence = 1.0 - score
        emoji      = "😞"
        message    = "This review has a negative sentiment."

    return {
        "label"     : label,
        "confidence": round(confidence * 100, 2),
        "emoji"     : emoji,
        "message"   : message,
        "raw_score" : round(score, 4),
    }


# --------------------------------------------------------------------------- #
#  Routes                                                                      #
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    if len(text) < 3:
        return jsonify({"error": "Text is too short. Please enter a full sentence."}), 400

    result = predict_sentiment(text)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
