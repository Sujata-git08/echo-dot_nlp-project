"""
08_lstm_sentiment.py
Non-transformer LSTM sentiment classifier for Echo Dot reviews

Inputs (in priority order):
- preprocessed_reviews.csv    (expects: lemmas_str, review_raw, rating?)
- echo_reviews_translated.csv (expects: review_en, rating?)
- echo_dot_reviews.csv        (expects: review, rating?)

Labels:
- Primary: derived from 'rating' (>=4 positive, 3 neutral, <=2 negative)
- Fallback: VADER lexicon (compound > 0.2 pos, < -0.2 neg, else neutral)

Outputs:
- lstm_sentiments.csv         (review text + y_true + y_pred + probs)
- lstm_model.keras            (saved model)
- lstm_tokenizer.pkl          (saved tokenizer)
"""

import os, re, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---- try both sources gracefully ----
CANDIDATE_INPUTS = [
    "preprocessed_reviews.csv",
    "echo_reviews_translated.csv",
    "echo_spacy_preprocessed.csv",
    "echo_dot_reviews.csv",
]

# ----------------- VADER fallback -----------------
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
SIA = SentimentIntensityAnalyzer()

def vader_label(text: str) -> int:
    c = SIA.polarity_scores(str(text))["compound"]
    if c > 0.2:
        return 2   # positive
    elif c < -0.2:
        return 0   # negative
    else:
        return 1   # neutral

# ------------- pick input + text column -------------
df = None
for path in CANDIDATE_INPUTS:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"✅ Loaded: {path}")
            break
        except Exception as e:
            pass
if df is None:
    raise FileNotFoundError("No input file found. Expected one of: " + ", ".join(CANDIDATE_INPUTS))

# choose best text column available (translated/raw; NOT lemmas for LSTM)
TEXT_COLS = ["review_en", "review_raw", "review"]
for c in TEXT_COLS:
    if c in df.columns:
        text_col = c
        break
else:
    # last resort: use lemmas_str if nothing else exists
    text_col = "lemmas_str" if "lemmas_str" in df.columns else None

if text_col is None:
    raise ValueError("No usable text column found (looked for review_en/review_raw/review/lemmas_str).")

texts = df[text_col].astype(str)

# ----------------- labels from ratings (primary) -----------------
def parse_rating(x):
    """
    Accepts:
      - numeric 1..5
      - strings like '5.0 out of 5 stars'
    Returns float or np.nan
    """
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"([0-9]+(\.[0-9]+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except:
            return np.nan
    return np.nan

rating_num = None
if "rating" in df.columns:
    rating_num = df["rating"].apply(parse_rating)

labels = None
if rating_num is not None:
    def map_by_star(r):
        if np.isnan(r):
            return np.nan
        if r >= 4.0:
            return 2   # positive
        elif r <= 2.0:
            return 0   # negative
        else:
            return 1   # neutral
    labels = rating_num.apply(map_by_star)

# fallback to VADER if no labels
if labels is None or labels.isna().all():
    print("⚠️ Rating labels unavailable — using VADER silver labels.")
    labels = texts.apply(vader_label)

# if partially missing (some NaN), fill those with VADER
mask_nan = labels.isna()
if mask_nan.any():
    print(f"ℹ️ Filling {mask_nan.sum()} missing labels with VADER.")
    labels.loc[mask_nan] = texts.loc[mask_nan].apply(vader_label)

labels = labels.astype(int)

# ----------------- train/val split -----------------
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    texts, labels, np.arange(len(texts)), test_size=0.2, random_state=42, stratify=labels
)

# ----------------- tokenize & pad -----------------
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 12000
MAX_LEN    = 120

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train.tolist())

Xtr = tokenizer.texts_to_sequences(X_train.tolist())
Xva = tokenizer.texts_to_sequences(X_val.tolist())

Xtr = pad_sequences(Xtr, maxlen=MAX_LEN, padding="post", truncating="post")
Xva = pad_sequences(Xva, maxlen=MAX_LEN, padding="post", truncating="post")

num_classes = len(np.unique(labels))

# ----------------- model -----------------
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)

model = models.Sequential([
    layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=100, input_length=MAX_LEN, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# class weights to mitigate imbalance
classes = np.array([0,1,2])  # neg, neu, pos
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {i: w for i, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

history = model.fit(
    Xtr, y_train,
    validation_data=(Xva, y_val),
    epochs=12,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1,
)

# ----------------- evaluation -----------------
y_pred = model.predict(Xva, verbose=0).argmax(axis=1)
print("\nClassification Report (val):")
print(classification_report(y_val, y_pred, target_names=["negative","neutral","positive"]))

print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))

# ----------------- save artifacts -----------------
model.save("lstm_model.keras")
with open("lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ----------------- score all reviews & export -----------------
X_all = tokenizer.texts_to_sequences(texts.tolist())
X_all = pad_sequences(X_all, maxlen=MAX_LEN, padding="post", truncating="post")
probs  = model.predict(X_all, verbose=0)
preds  = probs.argmax(axis=1)

out = pd.DataFrame({
    "text": texts,
    "y_true": labels,
    "y_pred": preds,
    "p_neg": probs[:,0] if num_classes==3 else np.nan,
    "p_neu": probs[:,1] if num_classes==3 else np.nan,
    "p_pos": probs[:,2] if num_classes==3 else np.nan,
})

out.to_csv("lstm_sentiments.csv", index=False, encoding="utf-8-sig")
print("\n✅ Saved predictions to lstm_sentiments.csv")
print("✅ Saved model to lstm_model.keras and tokenizer to lstm_tokenizer.pkl")
