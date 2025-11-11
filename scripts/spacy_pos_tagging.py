import spacy
import pandas as pd
from collections import Counter

df = pd.read_csv("echo_reviews_translated.csv")

nlp = spacy.load("en_core_web_sm")

pos_counts = Counter()
adj_list = []
verb_list = []

# Use translated raw English text for POS
text_col = "review_en" if "review_en" in df.columns else "review"
texts = df[text_col].astype(str).tolist()

for text in texts:
    doc = nlp(text)
    
    for token in doc:
        if token.pos_ != "SPACE":
            pos_counts[token.pos_] += 1
            
        if token.pos_ == "ADJ":
            adj_list.append(token.text.lower())
        if token.pos_ == "VERB":
            verb_list.append(token.text.lower())

print("\n POS Tag Counts:")
for tag, count in pos_counts.most_common():
    print(f"{tag}: {count}")

print("\n Top 20 Adjectives (Customer Opinions):")
print(Counter(adj_list).most_common(20))

print("\n Top 20 Verbs (Actions/Usage):")
print(Counter(verb_list).most_common(20))
