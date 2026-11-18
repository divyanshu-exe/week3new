"""
Global Configuration and Hyperparameters for Week 3 NLP Project.
"""

import os

# Root directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# File paths
REVIEWS_CSV_PATH = os.path.join(DATA_DIR, "reviews_dataset.csv")
NER_JSON_PATH = os.path.join(DATA_DIR, "ner_ground_truth.json")
NER_CSV_PATH = os.path.join(DATA_DIR, "ner_tokens_benchmark.csv")
DOCX_REPORT_PATH = os.path.join(REPORTS_DIR, "Week_3_NLP_Sentiment_Analysis_and_NER_Report.docx")

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Sentiment Labels
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

# VADER Thresholds
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# TextBlob Thresholds
TEXTBLOB_POS_THRESHOLD = 0.1
TEXTBLOB_NEG_THRESHOLD = -0.1

# TF-IDF Parameters
TFIDF_MAX_FEATURES = 2500
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

# NER Entity Types
NER_ENTITY_TYPES = ["ORG", "PER", "LOC", "PROD", "DATE", "MONEY"]

# IOB Tags
IOB_TAGS = [
    "O",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-PROD", "I-PROD",
    "B-DATE", "I-DATE",
    "B-MONEY", "I-MONEY"
]
