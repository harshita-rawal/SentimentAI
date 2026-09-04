"""
app.py — Flask backend for Sentiment Analysis
=============================================
Start the server with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import pickle
import numpy as np

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join("model", "tokenizer.pkl")
MAX_LEN        = 100
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# --------------------------------------------------------------------------- #
#  Load model & tokenizer once at startup                                      #
# --------------------------------------------------------------------------- #
print("[INFO] Loading model and tokenizer …")

if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        "Model files not found. Please run 'python train.py' first."
    )

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("[INFO] Ready!")


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #
def predict_sentiment(text: str) -> dict:
    """Return label, confidence, and a friendly message."""
    seq     = tokenizer.texts_to_sequences([text])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    score   = float(model.predict(padded, verbose=0)[0][0])

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
    app.run(debug=True)
