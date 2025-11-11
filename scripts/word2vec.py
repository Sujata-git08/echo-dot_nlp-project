"""
06_word2vec_semantics.py
Train Word2Vec on review corpus + find similar words
"""

import pandas as pd
from gensim.models import Word2Vec

INPUT = "echo_spacy_preprocessed.csv"

df = pd.read_csv(INPUT)

# use token list (not joined text!)
sentences = df["tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()

# Train Word2Vec (traditional method, no transformer)
model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1   # skip-gram (better for rare words)
)

model.save("echo_word2vec.model")

keywords = ["alexa", "sound", "music", "voice", "connect", "slow"]

print("\n🔍 Word2Vec Semantic Neighbors:")
for word in keywords:
    if word in model.wv:
        print(f"\n📌 Similar words to '{word}':")
        print(model.wv.most_similar(word, topn=5))
    else:
        print(f"\n⚠️ '{word}' not found in vocabulary")
