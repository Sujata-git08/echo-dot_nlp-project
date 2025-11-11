# echo-dot_nlp-project

# Customer Review NLP (Echo Dot 5th Gen) — Classical NLP Pipeline

End-to-end traditional NLP project (no transformers) to extract, clean, analyze, and interpret e-commerce product reviews.  
Covers scraping → translation → POS/NER → TF-IDF/LSA → Word2Vec → Sentiment (VADER + LSTM) → Similarity summary → QA.

## 📦 Environment

```bash
python -m venv .venv
source .venv/bin/activate         # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
