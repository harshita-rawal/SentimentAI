# 🧠 SentimentAI — LSTM Sentiment Analysis Web App

A beginner-friendly **sentiment analysis** web app built with:

| Layer     | Technology                    |
|-----------|-------------------------------|
| Frontend  | HTML + Vanilla CSS + JS       |
| Backend   | Flask (Python)                |
| ML Model  | TensorFlow / Keras LSTM (RNN) |

---

## 📁 Project Structure

```
sentientdetection/
├── app.py               # Flask server + /predict API
├── train.py             # Model training script (run once)
├── requirements.txt     # Python dependencies
├── README.md
│
├── data/
│   └── dataset.csv      # 100-sample labelled dataset
│
├── model/               # ← created by train.py
│   ├── sentiment_model.keras
│   └── tokenizer.pkl
│
├── templates/
│   └── index.html       # Jinja2 template (UI)
│
└── static/
    ├── css/style.css    # Glassmorphism dark theme
    └── js/main.js       # Fetch API + result rendering
```

---

## ⚙️ Setup & Run

### 1 — Create & activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Train the model *(do this once)*

```bash
python train.py
```

This will:
- Read `data/dataset.csv`
- Tokenise and pad text sequences
- Train an Embedding → LSTM → Dense model
- Save `model/sentiment_model.keras` and `model/tokenizer.pkl`

### 4 — Start the Flask server

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🤖 How the Model Works

```
Input text
   │
   ▼
Tokenizer  →  integer sequence  →  pad to length 100
   │
   ▼
Embedding layer  (maps each word to a 64-dim vector)
   │
   ▼
SpatialDropout1D (regularisation)
   │
   ▼
LSTM layer (64 units)  ← learns sequential patterns
   │
   ▼
Dense(32, relu)  →  Dropout  →  Dense(1, sigmoid)
   │
   ▼
Output: probability 0–1
  ≥ 0.5  →  Positive 😊
  < 0.5  →  Negative 😞
```

---

## 📊 Dataset

`data/dataset.csv` contains **100 labelled reviews**:

| Column     | Description           |
|------------|-----------------------|
| `review`   | Raw text sentence     |
| `sentiment`| `1` = Positive, `0` = Negative |

---

## 🛠 Tip — Improve Accuracy

- Add more rows to `dataset.csv` (aim for 1 000+)
- Re-run `python train.py` after adding data
- Tune `VOCAB_SIZE`, `MAX_LEN`, `LSTM_UNITS` in `train.py`
