# Echo Dot Customer Review NLP Analysis

End-to-End Classical NLP Pipeline for Amazon Echo Dot Reviews — Scraping, Preprocessing, Analysis & Insights

# Project Overview

This project performs complete NLP pipeline analysis on Amazon Echo Dot (5th Gen) customer reviews, using only classical NLP approaches (no transformers). The aim is to help e-commerce platforms and product teams extract insights from customer feedback using rule-based, statistical, and deep learning methods.
  
""" 
Scraping ➜ Language Detection ➜ Translation ➜ Cleaning ➜ Tokenization ➜ POS Tagging ➜ NER ➜ TF-IDF ➜ LSA ➜ Word2Vec ➜ Sentiment Analysis (VADER + LSTM) ➜ Clustering + Summary ➜ QA Generation
"""


# Repository Structure

echo-dot-nlp-project/
│── data/
│   ├── raw/                # Scraped reviews
│   ├── interim/            # Translated / preprocessed
│   └── processed/          # Final results CSVs
│
│── scripts/                # Modular processing scripts
│   ├── scraping.py
│   ├── lang_detect_translate.py
│   ├── nlp_preprocessing.py
│   ├── spacy_pos_tagging.py
│   ├── spacy_ner.py
│   ├── tfidf_lsa_topics.py
│   ├── semantic_analysis.py
│   ├── word2vec.py
│   ├── lstm_sentiment.py
│   └── QA_from_reviews.py
│
│── models/                 # Saved ML models
│── reports/                
├── .gitignore
└── README.md
