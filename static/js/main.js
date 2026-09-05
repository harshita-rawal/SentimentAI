/**
 * main.js — SentimentAI frontend logic
 * Handles: textarea counter, sample chips, API call, result rendering
 */

const textarea      = document.getElementById("review-input");
const charCount     = document.getElementById("char-count");
const analyzeBtn    = document.getElementById("analyze-btn");
const clearBtn      = document.getElementById("clear-btn");
const resultSection = document.getElementById("result-section");
const errorCard     = document.getElementById("error-card");
const resultEmoji   = document.getElementById("result-emoji");
const resultLabel   = document.getElementById("result-label");
const resultMessage = document.getElementById("result-message");
const confidencePct = document.getElementById("confidence-pct");
const progressBar   = document.getElementById("progress-bar");
const rawScore      = document.getElementById("raw-score");
const chips         = document.querySelectorAll(".chip");

// ── Character counter ────────────────────────────────────────
textarea.addEventListener("input", () => {
  charCount.textContent = textarea.value.length;
});

// ── Sample chips ─────────────────────────────────────────────
chips.forEach(chip => {
  chip.addEventListener("click", () => {
    textarea.value = chip.dataset.text;
    charCount.textContent = textarea.value.length;
    textarea.focus();
    // animate the chip briefly
    chip.style.transform = "scale(0.92)";
    setTimeout(() => (chip.style.transform = ""), 150);
  });
});

// ── Clear button ─────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  textarea.value = "";
  charCount.textContent = "0";
  hideResult();
  hideError();
  textarea.focus();
});

// ── Analyze button ───────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);

textarea.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runAnalysis();
});

async function runAnalysis() {
  const text = textarea.value.trim();
  if (!text) {
    showError("⚠️  Please enter some text before analyzing.");
    return;
  }
  if (text.length < 3) {
    showError("⚠️  Text is too short. Please enter a full sentence.");
    return;
  }

  setLoading(true);
  hideResult();
  hideError();

  try {
    const response = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    if (!response.ok) {
      if (response.status === 502 || response.status === 503 || response.status === 504) {
        showError("⏳ The server is warming up from sleep. Please wait 15–20 seconds and click Analyze again.");
      } else {
        let errMessage = "Server error. Please try again.";
        try {
          const errData = await response.json();
          if (errData && errData.error) errMessage = errData.error;
        } catch (_) {}
        showError("⚠️  " + errMessage);
      }
      return;
    }

    const data = await response.json();
    showResult(data);
  } catch (err) {
    showError("⚠️  Could not reach the server. If testing locally, ensure Flask is running. If on Render, please wait ~20s for cold start.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}

// ── Helpers ───────────────────────────────────────────────────
function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.classList.toggle("loading", isLoading);
}

function showResult(data) {
  const isPositive = data.label === "Positive";

  // sentiment class for coloring
  resultSection.className = "result-card " + (isPositive ? "positive" : "negative");

  resultEmoji.textContent   = data.emoji;
  resultLabel.textContent   = data.label;
  resultMessage.textContent = data.message;
  confidencePct.textContent = data.confidence + "%";
  rawScore.textContent      = data.raw_score;

  // animate progress bar after a short delay so the animation is visible
  progressBar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      progressBar.style.width = data.confidence + "%";
    });
  });

  resultSection.classList.remove("hidden");
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideResult() {
  resultSection.classList.add("hidden");
  resultSection.className = "result-card hidden";
}

function showError(msg) {
  errorCard.textContent = msg;
  errorCard.classList.remove("hidden");
}

function hideError() {
  errorCard.classList.add("hidden");
}
