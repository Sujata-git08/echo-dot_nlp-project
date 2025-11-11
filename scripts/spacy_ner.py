"""
04_spacy_ner.py

"""

import pandas as pd
import spacy
from collections import Counter

INPUT = "echo_reviews_translated.csv"
OUTPUT = "echo_spacy_ner_output.csv"


nlp = spacy.load("en_core_web_sm", disable=["parser"])

df = pd.read_csv(INPUT)

# Choose column with cleaned text (lemmas_str best for NER context)
texts = df["review_en"].astype(str).tolist()



entities_list = []
entity_counter = Counter()

print(" Running spaCy NER...\n")

for doc in nlp.pipe(texts, batch_size=50):
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    entities_list.append(ents)
    for ent in ents:
       
        if ent[1] in ["PRODUCT","ORG","PERSON","DATE","TIME","GPE","QUANTITY","CARDINAL"]:
            entity_counter[ent] += 1

        skip_terms = {"doesn", "aur", "tha", "nahi", "hota", "thoda", "jyada", "achha"}
        text = ent[0].lower()
        label = ent[1]

        if text in skip_terms:
          continue
  
    

df["entities"] = entities_list

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f" NER saved to {OUTPUT}")

print("\n Top detected entities (product-context relevant):")
for ent, count in entity_counter.most_common(20):
    print(f"{ent[0]:20s} {ent[1]:10s} {count}")
