"""
train.py — Train an LSTM-based sentiment classifier
=====================================================
Run this script to train the model:
    python train.py
It will:
  1. Load data/dataset.csv
  2. Tokenize & pad the text (pre-padding)
  3. Train a Bidirectional LSTM + GlobalMaxPooling model
  4. Save model     → model/sentiment_model.keras
  5. Save tokenizer → model/tokenizer.pkl & model/tokenizer.json
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH       = "data/dataset.csv"
MODEL_DIR       = "model"
MODEL_PATH      = os.path.join(MODEL_DIR, "sentiment_model.keras")
TOKENIZER_PATH  = os.path.join(MODEL_DIR, "tokenizer.pkl")
TOKENIZER_JSON  = os.path.join(MODEL_DIR, "tokenizer.json")

VOCAB_SIZE      = 3000      # max number of unique words to keep
MAX_LEN         = 50        # max sequence length (pad/truncate)
EMBED_DIM       = 64        # word-embedding dimensions
LSTM_UNITS      = 32
DROPOUT         = 0.2
BATCH_SIZE      = 16
EPOCHS          = 25        # EarlyStopping will stop earlier if needed
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: str):
    df = pd.read_csv(
        path,
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
        engine="python",
        encoding="utf-8",
    )
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["review", "sentiment"])
    df["review"]    = df["review"].astype(str).str.strip()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").dropna().astype(int)
    df = df.dropna(subset=["sentiment"])
    print(f"[INFO] Dataset loaded — {len(df)} samples "
          f"({int(df['sentiment'].sum())} positive, "
          f"{int((df['sentiment'] == 0).sum())} negative)")
    return df["review"].tolist(), df["sentiment"].tolist()


def build_tokenizer(texts, vocab_size):
    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    print(f"[INFO] Vocabulary size: {len(tok.word_index)} unique tokens")
    return tok


def preprocess(texts, tokenizer, max_len):
    seqs = tokenizer.texts_to_sequences(texts)
    # pre-padding ensures words are close to the output layer for optimal gradient flow
    return pad_sequences(seqs, maxlen=max_len, padding="pre", truncating="post")


def build_model(vocab_size, embed_dim, max_len, lstm_units, dropout):
    model = Sequential([
        Embedding(vocab_size, embed_dim),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(32, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),      # binary output
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load & split
    texts, labels = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    # 2. Tokenize on all texts to capture full vocabulary
    tokenizer = build_tokenizer(texts, VOCAB_SIZE)

    # 3. Pad / truncate
    X_train_pad = preprocess(X_train, tokenizer, MAX_LEN)
    X_val_pad   = preprocess(X_val,   tokenizer, MAX_LEN)
    y_train_arr = np.array(y_train)
    y_val_arr   = np.array(y_val)

    # 4. Build & train
    actual_vocab_size = min(len(tokenizer.word_index) + 2, VOCAB_SIZE)
    model = build_model(actual_vocab_size, EMBED_DIM, MAX_LEN, LSTM_UNITS, DROPOUT)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    print("\n[INFO] Starting training …")
    history = model.fit(
        X_train_pad, y_train_arr,
        validation_data=(X_val_pad, y_val_arr),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    # 5. Evaluate
    _, val_acc = model.evaluate(X_val_pad, y_val_arr, verbose=0)
    print(f"\n[INFO] Validation accuracy: {val_acc * 100:.2f}%")

    # 6. Save
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved -> {MODEL_PATH}")

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] Tokenizer saved -> {TOKENIZER_PATH}")

    with open(TOKENIZER_JSON, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    print(f"[INFO] Tokenizer JSON saved -> {TOKENIZER_JSON}")

    print("\n[DONE] Training complete! You can now run: python app.py")


if __name__ == "__main__":
    main()
