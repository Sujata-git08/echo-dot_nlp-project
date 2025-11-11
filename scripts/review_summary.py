"""
08_review_summary.py
Summarize reviews by cosine similarity clustering
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

INPUT = "echo_spacy_preprocessed.csv"
N_SUMMARY = 5  # number of summary reviews to extract

df = pd.read_csv(INPUT)
texts = df["review_raw"].astype(str).tolist()

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

# Cosine similarity among reviews
sim_matrix = cosine_similarity(X)

# Average similarity per review = "centrality"
avg_sim = sim_matrix.mean(axis=1)

# Pick most central reviews (summary representatives)
top_idx = np.argsort(avg_sim)[-N_SUMMARY:]

summary_reviews = df.iloc[top_idx]["review_raw"].tolist()

print("\n📝 SUMMARY — Most representative customer reviews:\n")
for i, review in enumerate(summary_reviews, 1):
    print(f"{i}. {review}\n")

# Save
pd.DataFrame({"summary_reviews": summary_reviews}).to_csv("echo_review_summary.csv", index=False)
print("✅ Summary saved to echo_review_summary.csv")
