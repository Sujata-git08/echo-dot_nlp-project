"""
01_spacy_preprocess.py
Input : echo_dot_reviews.csv  (columns: name,rating,review,date,colour,title)
Output: echo_spacy_preprocessed.csv
What it does (mostly spaCy):
- lightweight normalization (lowercase, url/email/punct cleanup)
- spaCy tokenization
- spaCy stopword removal
- spaCy lemmatization
- preserve emojis if present (can carry sentiment)
"""

import re
import pandas as pd
import spacy
from pathlib import Path
import emoji

# ---------- config ----------
INPUT_CSV  = "echo_dot_reviews.csv"
OUTPUT_CSV = "echo_spacy_preprocessed.csv"
MIN_TOKENS = 3  # drop ultra-short reviews after cleaning
# ----------------------------

# load spaCy  (for faster performance I have disbled parser/NER)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # fast: we only need tok/pos/lemma for now

# use spaCy's stopword list
spacy_stop = nlp.Defaults.stop_words

# Basic Normalization 

# Using regex, removes URLs, Emails, Etra spaces, Punctuations but kept emojies
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MULTI_WS = re.compile(r"\s+")
PUNCT_EXCEPT_EMOJI = re.compile(r"[^0-9A-Za-z\s" + re.escape("".join(emoji.EMOJI_DATA.keys())) + r"]")

def normalize(text: str) -> str:
    # basic cleaning
    text = text.strip().lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)

    # --- Hinglish & contraction fixes ---
    replacements = {
        "doesn t": "does not",
        "doesnt": "does not",
        "don t": "do not",
        "dont": "do not",
        "cant": "can not",
        "can t": "can not",
        "won t": "will not",
        "wont": "will not",
        "didnt": "did not",
        "didn t": "did not",
        "haven t": "have not",
        "havent": "have not",
        "hadn t": "had not",
        "isn t": "is not",
        "isnt": "is not",

        # Hinglish sentiment words
        "achha": "good",
        "acha": "good",
        "achha tha": "was good",
        "accha": "good",
        "bahut": "very",
        "thoda": "little",
        "thodi": "little",
        "jyada": "more",
        "kam": "less",
        "nahi": "not",
        "nahiin": "not",
        "bohot": "very",
        "sahi": "good",
        "bekar": "bad",
        "paisa vasool": "value for money",
        "mast": "awesome"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # remove punctuation but keep emojis
    text = PUNCT_EXCEPT_EMOJI.sub(" ", text)
    text = MULTI_WS.sub(" ", text).strip()

    return text


def spacy_process(doc):
    
    toks = []
    lemmas = []
    for t in doc:
        if t.is_space:
            continue
        txt = t.text
      
        is_emoji = txt in emoji.EMOJI_DATA
        if not is_emoji:
            if (t.is_punct or t.is_quote or t.is_currency or t.like_url or t.like_email):
                continue
            if (t.is_stop or t.like_num and len(txt) <= 1):
                # drop common stopwords & single-character numerics
                continue
            if not (t.is_alpha or t.like_num):
                continue
        toks.append(txt)
        # lemma for words; keep emoji untouched
        lemmas.append(txt if is_emoji else (t.lemma_ if t.lemma_ != "-PRON-" else t.text))
    return toks, lemmas

def main():
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Could not find {INPUT_CSV}. Make sure the scraper saved it.")

    df = pd.read_csv(INPUT_CSV)
    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column")

    # normalize
    df["review_raw"] = df["review"].astype(str)
    df["review_norm"] = df["review_raw"].apply(normalize)

    # run spaCy in batches (fast)
    tokens_list = []
    lemmas_list = []
    lengths = []

    for doc in nlp.pipe(df["review_norm"].tolist(), batch_size=200):
        toks, lems = spacy_process(doc)
        tokens_list.append(toks)
        lemmas_list.append(lems)
        lengths.append(len(toks))

    df["tokens"] = tokens_list
    df["lemmas"] = lemmas_list
    df["len_tokens"] = lengths

    # drop super-short after cleaning (often noise)
    df = df[df["len_tokens"] >= MIN_TOKENS].reset_index(drop=True)

    # also store joined strings for downstream vectorizers
    df["tokens_str"] = df["tokens"].apply(lambda xs: " ".join(xs))
    df["lemmas_str"] = df["lemmas"].apply(lambda xs: " ".join(xs))

    # save
    keep_cols = ["name","rating","date","title","colour","review_raw","review_norm","tokens","lemmas","tokens_str","lemmas_str","len_tokens"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUTPUT_CSV, index=False)
    print(f" Preprocessing complete. Saved {len(df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
