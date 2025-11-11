"""
03_lang_detect_translate.py
- Detect language of reviews (rule/stat based; non-transformer)
- Translate non-English (e.g., Hindi) to English
- Re-run spaCy normalization on translated text (optional)
"""

import pandas as pd
from langdetect import detect
from googletrans import Translator

INPUT = "echo_dot_reviews.csv"
OUTPUT = "echo_reviews_translated.csv"

translator = Translator()

def safe_detect(text):
    try:
        return detect(text)
    except:
        return "unknown"

def safe_translate(text):
    try:
        return translator.translate(text, dest="en").text
    except:
        return text  # fallback

def main():
    df = pd.read_csv(INPUT)
    df["lang"] = df["review"].astype(str).apply(safe_detect)
    df["review_en"] = df.apply(
        lambda r: safe_translate(r["review"]) if r["lang"] != "en" and r["lang"] != "unknown" else r["review"],
        axis=1
    )
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ Translation pass done. Saved to {OUTPUT}")
    print(df[["review","lang","review_en"]].head(10))

if __name__ == "__main__":
    main()
