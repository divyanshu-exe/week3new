# Week 3 NLP Task: Implementing Sentiment Analysis and Named Entity Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NLP Tasks](https://img.shields.io/badge/NLP-Sentiment%20Analysis%20%7C%20NER-orange.svg)]()
[![Documentation](https://img.shields.io/badge/DOC%20Deliverable-Complete-brightgreen.svg)]()

> **Internship Technical Practicum — Week 3**  
> **Author:** Divyanshu ([@divyanshu-exe](https://github.com/divyanshu-exe))  
> **Target Repository:** [https://github.com/divyanshu-exe/week3new](https://github.com/divyanshu-exe/week3new)  
> **Workload Simulation:** 30–35 Hours of Applied NLP & Machine Learning Engineering

---

## 📌 Executive Overview

This repository houses a comprehensive, industrial-grade Natural Language Processing (NLP) framework developed to bridge theoretical foundations and practical machine learning implementation. The project addresses two core text analytics tasks within a single unified framework:

1. **Mini-Project 1: Sentiment Analysis (Affective Computing)**
   - **Lexicon-Based Approaches:** VADER (Valence Aware Dictionary and sEntiment Reasoner) with heuristic valence boosters, and TextBlob (polarity & subjectivity).
   - **Supervised Machine Learning Approaches:** High-dimensional TF-IDF unigram & bigram representations with **Multinomial Naive Bayes (MNB)**, **L2-Regularized Logistic Regression (MaxEnt)**, **Linear Support Vector Classifier (LinearSVC)**, and **Random Forest**.
2. **Mini-Project 2: Named Entity Recognition (Information Extraction)**
   - **Rule-Based & Domain Gazetteer Matcher:** High-precision regex pattern compilers for structured entities (`MONEY`, `DATE`) and curated gazetteers for domain entities (`ORG`, `PER`, `LOC`, `PROD`) utilizing greedy longest-match-first disambiguation.
   - **NLTK Statistical MaxEnt Syntax Chunker:** Hierarchical syntactic parse tree chunking (`ne_chunk`) mapped to standardized IOB sequences.
   - **Supervised Token-Level Sequence Classifier:** A 26-dimensional morphological, orthographic, syntactic, and contextual window feature extractor coupled with multi-class `LinearSVC`.
3. **Unified Analytical Pipeline: Aspect-Based & Entity-Centric Sentiment Mining**
   - Synthesizes both mini-projects to extract named entities and bind sentence-level affective valence to specific corporate brands and hardware models.
4. **Primary Submission Deliverable:**
   - A 20+ page equivalent professional Word document: [`reports/Week_3_NLP_Sentiment_Analysis_and_NER_Report.docx`](reports/Week_3_NLP_Sentiment_Analysis_and_NER_Report.docx), complete with executive summary, methodology, mathematical formulations, diagnostic heatmaps, error analyses, and future deep learning roadmap.

---

## 🏗️ System Architecture & Repository Structure

```
week3New/
│
├── data/                                # Curated NLP datasets
│   ├── reviews_dataset.csv              # Multi-domain consumer review corpus (100 reviews across 11 categories)
│   ├── ner_ground_truth.json            # 40 manually annotated benchmark sentences in IOB2 format
│   └── ner_tokens_benchmark.csv         # 609 flattened token-level annotations with POS & NER tags
│
├── notebooks/                           # Interactive Jupyter walkthrough
│   └── NLP_Sentiment_and_NER_Comprehensive_Walkthrough.ipynb
│
├── reports/                             # Formal documentation & figures
│   ├── Week_3_NLP_Sentiment_Analysis_and_NER_Report.docx  # Full 35-Hour Internship DOC report
│   ├── benchmark_metrics.json          # Cached JSON metrics from cross-validation & holdout evaluation
│   └── figures/                         # High-resolution diagnostic plots (300 DPI)
│       ├── sentiment_models_comparison.png
│       ├── sentiment_confusion_matrices.png
│       ├── sentiment_feature_importance.png
│       ├── ner_models_comparison.png
│       ├── ner_entity_distribution.png
│       └── entity_sentiment_distribution.png
│
├── scripts/                             # Executable CLI orchestration
│   ├── prepare_dataset.py               # Dataset generation & annotation verification
│   ├── run_all.py                       # Master benchmark script (trains, evaluates, generates plots)
│   ├── generate_report_doc.py           # Programmatic Word document builder
│   └── create_notebook.py               # Automated notebook generator
│
├── src/                                 # Modular production source code
│   ├── __init__.py
│   ├── config.py                        # Constants, thresholds, and hyperparameters
│   ├── preprocessing.py                 # Multi-stage text cleaning, contractions, negation scoping
│   ├── pipeline.py                      # Unified NLP Analyzer for aspect-level sentiment
│   ├── sentiment/                       # Mini-Project 1: Sentiment Analysis
│   │   ├── __init__.py
│   │   ├── lexicon_analyzer.py          # VADER and TextBlob implementations
│   │   ├── ml_models.py                 # TF-IDF, MNB, Logistic Regression, LinearSVC, RF
│   │   └── evaluator.py                 # Sentiment metrics, confusion matrices, ROC/F1 plots
│   └── ner/                             # Mini-Project 2: Named Entity Recognition
│       ├── __init__.py
│       ├── rule_based_ner.py            # Gazetteers, regex patterns, greedy chunking
│       ├── nltk_chunker_ner.py          # NLTK MaxEnt tree chunking parser
│       ├── statistical_ner.py           # CRF-inspired token feature sequence classifier
│       └── evaluator.py                 # Token-level & exact span-level NER metrics
│
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 📊 Benchmark Results

### 1. Mini-Project 1: Sentiment Analysis (Held-Out Test Set)

| Model Paradigm | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VADER (Lexicon)** | **0.800** | 0.627 | 0.667 | 0.591 | 0.735 | *N/A (Unsupervised)* |
| **TextBlob (Rule)** | 0.680 | 0.611 | 0.628 | 0.597 | 0.681 | *N/A (Unsupervised)* |
| **Multinomial Naive Bayes** | 0.720 | 0.767 | 0.733 | 0.741 | 0.722 | 0.773 (±0.068) |
| **Logistic Regression (L2)** | 0.680 | 0.711 | 0.672 | 0.676 | 0.682 | 0.773 (±0.053) |
| **Linear Support Vector (SVC)** | 0.760 | 0.783 | 0.783 | 0.780 | 0.768 | **0.787 (±0.027)** |
| **Random Forest (100 Trees)** | **0.800** | **0.817** | **0.800** | **0.791** | **0.800** | 0.773 (±0.068) |

> **Key Sentiment Finding:** While VADER achieved strong zero-shot accuracy (80.0%) due to its heuristic valence booster mechanics, **Random Forest** and **Linear SVC** demonstrated superior balance across all sentiment classes, reaching Macro F1-scores of **0.791** and **0.780**, effectively capturing domain-specific terminology ('fanless', 'thermal throttling').

---

### 2. Mini-Project 2: Named Entity Recognition (Exact Span-Level)

| NER Architecture | Span Precision | Span Recall | Exact Span F1-Score | Primary Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Rule-Based & Gazetteers** | **0.958** | **0.972** | **0.965** | Near-perfect precision on curated tech hardware brands and models |
| **NLTK MaxEnt Chunker** | 0.259 | 0.197 | 0.224 | General-domain syntax chunking; limited product entity recall |
| **Statistical Sequence (LinearSVC)** | 0.868 | 0.930 | **0.898** | Strong generalization to unseen tokens via 26 linguistic window features |

> **Key NER Finding:** The curated **Rule-Based engine** dominated span-level precision on established entities (0.965 F1), while the **Feature-Based Statistical Classifier** proved highly capable of learning contextual entity boundaries from token window representations (0.898 F1), generalizing to novel morphological variations without requiring dictionary updates.

---

## 🔬 Core Preprocessing & Feature Engineering Highlights

1. **Contraction Expansion:** Normalizes 20+ English contractions (`didn't` -> `did not`, `can't` -> `cannot`), preventing orphaned punctuation and preserving negative semantics.
2. **Negation Retention in Stopword Filtering:** Preserves critical sentiment-shifting particles (`no`, `not`, `never`, `hardly`, `scarcely`, `without`, `cannot`), avoiding the classical pitfall where `not good` collapses into `good`.
3. **Negation Scoping (Scope Inversion):** Implements the Das & Chen (2001) algorithm, appending `_NOT` to all words within a negation particle's clause (`not enjoy` -> `not enjoy_NOT`).
4. **POS-Guided Lemmatization:** Converts Penn Treebank tags into WordNet constants (`ADJ`, `VERB`, `NOUN`, `ADV`), ensuring grammatically accurate base-form reduction.
5. **NER Window Feature Extractor:** Extracts 26 morphological and orthographic properties per token:
   - Word shape (`iPhone` -> `xXxxxx`, `2024` -> `dddd`)
   - Affixes (Prefix-2, Prefix-3, Suffix-2, Suffix-3)
   - Orthography (`isupper`, `istitle`, `has_hyphen`, `has_digit`)
   - Syntactic POS tags
   - Context window (`w-1`, `pos-1`, `w+1`, `pos+1`, `BOS`, `EOS`)

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone the Repository
```bash
git clone https://github.com/divyanshu-exe/week3new.git
cd week3new
```

### 2. Set Up Python Environment & Dependencies
```bash
# Recommended: Python 3.10 to 3.14
python -m pip install -r requirements.txt
```

### 3. Run the Master Benchmark & Visualizations
```bash
python scripts/run_all.py
```
This single command executes:
- Dataset verification
- Preprocessing pipelines
- 5-Fold Stratified Cross-Validation on all ML models
- Holdout test evaluation for Sentiment & NER
- Unified aspect sentiment pipeline
- Generation of all diagnostic plots in `reports/figures/`
- JSON metric serialization in `reports/benchmark_metrics.json`

### 4. Regenerate the Word Document Report
```bash
python scripts/generate_report_doc.py
```
Outputs the complete, styled Word document to `reports/Week_3_NLP_Sentiment_Analysis_and_NER_Report.docx`.

### 5. Launch the Interactive Walkthrough Notebook
```bash
jupyter notebook notebooks/NLP_Sentiment_and_NER_Comprehensive_Walkthrough.ipynb
```

---

## 🛠️ Code Usage Examples

### Example 1: Unified Aspect-Based Sentiment Mining
```python
from src.pipeline import UnifiedNLPAnalyzer
from src.sentiment.ml_models import SentimentMLTrainer

# Initialize analyzer
analyzer = UnifiedNLPAnalyzer()

text = (
    "Tim Cook announced the Apple iPhone 15 Pro in Cupertino for $999. "
    "The camera quality and titanium chassis are magnificent! "
    "However, Dell customer support in Seattle was completely awful and unhelpful."
)

result = analyzer.analyze_document(text)

# Extracted Entities
for ent in result["entities_found"]:
    print(f"[{ent['type']}] {ent['text']}")

# Aspect-Level Sentiments
for aspect in result["entity_aspect_sentiments"]:
    print(f"Entity: {aspect['entity_text']:<15} | Sentiment: {aspect['sentiment']} (Compound: {aspect['compound_score']:+.2f})")
```

**Output:**
```
[PER] Tim Cook
[ORG] Apple
[PROD] iPhone 15 Pro
[LOC] Cupertino
[MONEY] $999
[ORG] Dell
[LOC] Seattle

Entity: Tim Cook        | Sentiment: positive (Compound: +0.20)
Entity: Apple           | Sentiment: positive (Compound: +0.20)
Entity: iPhone 15 Pro   | Sentiment: positive (Compound: +0.20)
Entity: Cupertino       | Sentiment: positive (Compound: +0.20)
Entity: $999            | Sentiment: positive (Compound: +0.20)
Entity: Dell            | Sentiment: negative (Compound: -0.47)
Entity: Seattle         | Sentiment: negative (Compound: -0.47)
```

---

## 📖 Comprehensive DOC Deliverable Details

The submission deliverable (`reports/Week_3_NLP_Sentiment_Analysis_and_NER_Report.docx`) includes:
- **Title Page & Executive Summary** with metadata tables and effort accounting (35 hours).
- **Section 1: Introduction & Problem Formulation** — Mathematical framing of Sentiment and NER.
- **Section 2: Dataset Selection, Justification & EDA** — Multi-domain review corpus details and entity distribution.
- **Section 3: Preprocessing Pipeline & Feature Engineering** — Contraction expansion, negation retention, `_NOT` tagging, POS lemmatization, and TF-IDF.
- **Section 4: Mini-Project 1: Sentiment Analysis** — Detailed algorithmic breakdown of VADER, TextBlob, MNB, Logistic Regression, LinearSVC, and RF with 5-Fold CV.
- **Section 5: Mini-Project 2: Named Entity Recognition** — Rule-based regex/gazetteer matching, NLTK tree chunking, and supervised LinearSVC sequence modeling.
- **Section 6: Unified End-to-End Pipeline** — Sentence-level aspect extraction and brand reputation scoring.
- **Section 7: Experimental Results & Benchmarking** — 6 formatted comparison tables and 6 embedded 300-DPI diagnostic plots.
- **Section 8: Technical Challenges & Engineering Solutions** — Negation inversion, entity collision, sparse matrix index compatibility, and domain vocabulary shift.
- **Section 9: Scalability & Deep Learning Extensions** — BERT/RoBERTa fine-tuning, BiLSTM-CRF, LLM zero-shot aspect extraction, and ONNX deployment.
- **Section 10: Conclusion & Academic References** — Complete bibliography citing Hutto & Gilbert (2014), Pedregosa et al. (2011), Das & Chen (2001), Devlin et al. (2019), and others.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
