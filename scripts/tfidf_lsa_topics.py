"""
05_tfidf_lsa_topics.py
- Extract key themes
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

INPUT = "echo_spacy_preprocessed.csv"

df = pd.read_csv(INPUT)

texts = df["lemmas_str"].astype(str).tolist()

# TF-IDF Vectorizer (keep traditional constraints)
tfidf = TfidfVectorizer(
    max_features=1500,
    min_df=2,
    stop_words='english'
)

X = tfidf.fit_transform(texts)

# LSA (Latent Semantic Analysis)
lsa = TruncatedSVD(n_components=5, random_state=42)
lsa_fit = lsa.fit_transform(X)

terms = tfidf.get_feature_names_out()

print("\n Top 5 topics from customer reviews:\n")

for idx, component in enumerate(lsa.components_):
    print(f"Topic {idx+1}: ", end="")
    terms_sorted = component.argsort()[-10:]
    print([terms[i] for i in terms_sorted])
