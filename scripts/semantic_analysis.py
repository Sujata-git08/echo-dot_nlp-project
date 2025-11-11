"""
07_sentiment_analysis.py
Sentiment using VADER (traditional lexicon-based approach)
"""

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

INPUT = "echo_spacy_preprocessed.csv"
OUTPUT = "echo_sentiments.csv"

df = pd.read_csv(INPUT)

sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["review_raw"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

df["sentiment_label"] = df["sentiment"].apply(
    lambda s: "positive" if s > 0.2 else ("negative" if s < -0.2 else "neutral")
)

df.to_csv(OUTPUT, index=False)

print("✅ Sentiment analysis done & saved to", OUTPUT)
print(df[["review_raw","sentiment","sentiment_label"]].head(10))
print(df["sentiment_label"].value_counts())
