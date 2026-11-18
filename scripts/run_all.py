"""
Master Execution and Benchmarking Script for Week 3 NLP Task.
Executes Sentiment Analysis and NER pipelines, saves plots, and caches evaluation metrics.
"""

import os
import sys

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    REVIEWS_CSV_PATH,
    NER_JSON_PATH,
    RANDOM_SEED
)
from src.preprocessing import TextPreprocessor
from src.sentiment.lexicon_analyzer import VaderSentimentAnalyzer, TextBlobSentimentAnalyzer
from src.sentiment.ml_models import SentimentMLTrainer
from src.sentiment.evaluator import SentimentEvaluator
from src.ner.rule_based_ner import RuleBasedNER
from src.ner.nltk_chunker_ner import NLTKChunkerNER
from src.ner.statistical_ner import FeatureBasedStatisticalNER
from src.ner.evaluator import NEREvaluator
from src.pipeline import UnifiedNLPAnalyzer


def main():
    print("=" * 70)
    print("WEEK 3 NLP EXPERIMENTAL BENCHMARK: SENTIMENT ANALYSIS & NER")
    print("=" * 70)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------
    print("\n[Step 1] Loading Datasets...")
    if not os.path.exists(REVIEWS_CSV_PATH):
        raise FileNotFoundError(f"Reviews dataset not found at {REVIEWS_CSV_PATH}. Run prepare_dataset.py first.")

    df_reviews = pd.read_csv(REVIEWS_CSV_PATH)
    print(f"Loaded {len(df_reviews)} review records across categories: {df_reviews['category'].unique().tolist()}")

    with open(NER_JSON_PATH, "r", encoding="utf-8") as f:
        ner_benchmark_data = json.load(f)
    print(f"Loaded {len(ner_benchmark_data)} manually annotated NER benchmark sentences.")

    # -------------------------------------------------------------
    # 2. Preprocessing & Data Preparation for Sentiment
    # -------------------------------------------------------------
    print("\n[Step 2] Executing Preprocessing Pipeline...")
    preprocessor = TextPreprocessor()
    df_reviews["cleaned_ml_text"] = df_reviews["review_text"].apply(
        lambda t: preprocessor.preprocess_for_sentiment_ml(t, apply_negation=True)
    )

    X_raw = df_reviews["review_text"].tolist()
    X_clean = df_reviews["cleaned_ml_text"].tolist()
    y_sentiment = df_reviews["ground_truth_sentiment"].tolist()

    # Stratified Split (80% Train, 20% Test)
    X_train_clean, X_test_clean, y_train_s, y_test_s, X_train_raw, X_test_raw = train_test_split(
        X_clean, y_sentiment, X_raw, test_size=0.25, random_state=RANDOM_SEED, stratify=y_sentiment
    )
    print(f"Sentiment Split: {len(X_train_clean)} train samples, {len(X_test_clean)} test samples.")

    # -------------------------------------------------------------
    # 3. Mini-Project 1: Sentiment Analysis Evaluation
    # -------------------------------------------------------------
    print("\n[Step 3] Evaluating Sentiment Analysis Models...")
    sent_eval = SentimentEvaluator()
    sentiment_results = {}

    # 3a. Lexicon Approach 1: VADER (evaluated on test set raw text)
    vader = VaderSentimentAnalyzer()
    vader_df = vader.predict_batch(X_test_raw)
    sentiment_results["VADER (Lexicon)"] = sent_eval.evaluate_predictions(
        y_test_s, vader_df["predicted_sentiment"].tolist(), model_name="VADER (Lexicon)"
    )

    # 3b. Lexicon Approach 2: TextBlob (evaluated on test set raw text)
    textblob = TextBlobSentimentAnalyzer()
    tb_df = textblob.predict_batch(X_test_raw)
    sentiment_results["TextBlob (Rule)"] = sent_eval.evaluate_predictions(
        y_test_s, tb_df["predicted_sentiment"].tolist(), model_name="TextBlob (Rule)"
    )

    # 3c. Machine Learning Classifiers
    ml_trainer = SentimentMLTrainer()
    print("Running 5-Fold Stratified Cross Validation on Training Set...")
    cv_scores = ml_trainer.cross_validate_all(X_train_clean, y_train_s, cv_folds=5)
    for model_name, scores in cv_scores.items():
        print(f"  > {model_name:25s} | CV Acc: {scores['cv_accuracy_mean']:.3f} (+/- {scores['cv_accuracy_std']:.3f}) | CV F1: {scores['cv_f1_macro_mean']:.3f}")

    ml_trainer.fit_all(X_train_clean, y_train_s)

    for name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVC", "Random Forest"]:
        y_pred = ml_trainer.predict(name, X_test_clean)
        sentiment_results[name] = sent_eval.evaluate_predictions(
            y_test_s, y_pred.tolist(), model_name=name
        )

    # Plot Sentiment Results
    cm_plot_path = sent_eval.plot_confusion_matrices(sentiment_results, "sentiment_confusion_matrices.png")
    comp_plot_path = sent_eval.plot_model_comparison(sentiment_results, "sentiment_models_comparison.png")

    feature_importance = ml_trainer.get_feature_importance("Logistic Regression", top_n=10)
    feat_plot_path = sent_eval.plot_feature_importance(feature_importance, "sentiment_feature_importance.png")
    print(f"Saved sentiment diagnostic plots to {FIGURES_DIR}")

    # -------------------------------------------------------------
    # 4. Mini-Project 2: Named Entity Recognition Evaluation
    # -------------------------------------------------------------
    print("\n[Step 4] Evaluating Named Entity Recognition Models...")
    ner_eval = NEREvaluator()

    sentences_tokens = [s["tokens"] for s in ner_benchmark_data]
    sentences_pos = [s["pos_tags"] for s in ner_benchmark_data]
    sentences_true_tags = [s["ner_tags"] for s in ner_benchmark_data]

    # Split NER benchmark for statistical model evaluation (70% Train, 30% Test)
    idx_train, idx_test = train_test_split(range(len(sentences_tokens)), test_size=0.35, random_state=RANDOM_SEED)

    train_toks = [sentences_tokens[i] for i in idx_train]
    train_pos = [sentences_pos[i] for i in idx_train]
    train_tags = [sentences_true_tags[i] for i in idx_train]

    test_toks = [sentences_tokens[i] for i in idx_test]
    test_pos = [sentences_pos[i] for i in idx_test]
    test_tags = [sentences_true_tags[i] for i in idx_test]

    # Model 1: Rule-Based NER
    rule_ner = RuleBasedNER()
    pred_tags_rule = [rule_ner.predict_iob_tags(toks) for toks in test_toks]

    # Model 2: NLTK Chunker NER
    nltk_ner = NLTKChunkerNER()
    pred_tags_nltk = [nltk_ner.predict_iob_tags(toks, pos) for toks, pos in zip(test_toks, test_pos)]

    # Model 3: Statistical Sequence Classifier
    stat_ner = FeatureBasedStatisticalNER()
    stat_ner.fit(train_toks, train_pos, train_tags)
    pred_tags_stat = stat_ner.predict_all_iob(test_toks, test_pos)

    # Evaluate Span-Level & Token-Level
    flat_true_tags = [t for s in test_tags for t in s]
    flat_rule_tags = [t for s in pred_tags_rule for t in s]
    flat_nltk_tags = [t for s in pred_tags_nltk for t in s]
    flat_stat_tags = [t for s in pred_tags_stat for t in s]

    ner_token_results = {
        "Rule-Based NER": ner_eval.evaluate_token_level(flat_true_tags, flat_rule_tags, "Rule-Based NER"),
        "NLTK MaxEnt Chunker": ner_eval.evaluate_token_level(flat_true_tags, flat_nltk_tags, "NLTK MaxEnt Chunker"),
        "Statistical LinearSVC NER": ner_eval.evaluate_token_level(flat_true_tags, flat_stat_tags, "Statistical LinearSVC NER")
    }

    ner_span_results = {
        "Rule-Based NER": ner_eval.evaluate_span_level(test_tags, pred_tags_rule, "Rule-Based NER"),
        "NLTK MaxEnt Chunker": ner_eval.evaluate_span_level(test_tags, pred_tags_nltk, "NLTK MaxEnt Chunker"),
        "Statistical LinearSVC NER": ner_eval.evaluate_span_level(test_tags, pred_tags_stat, "Statistical LinearSVC NER")
    }

    # Plot NER Results
    ner_comp_plot = ner_eval.plot_ner_comparison(ner_span_results, "ner_models_comparison.png")

    # Plot Entity Distribution across benchmark
    all_entity_types = {}
    for sent in ner_benchmark_data:
        for tag in sent["ner_tags"]:
            if tag.startswith("B-"):
                etype = tag.split("-")[1]
                all_entity_types[etype] = all_entity_types.get(etype, 0) + 1
    ner_dist_plot = ner_eval.plot_entity_distribution(all_entity_types, "ner_entity_distribution.png")

    # -------------------------------------------------------------
    # 5. Unified Pipeline & Entity-Centric Sentiment Intelligence
    # -------------------------------------------------------------
    print("\n[Step 5] Running Unified End-to-End Pipeline on Corpus...")
    unified_analyzer = UnifiedNLPAnalyzer(ml_trainer=ml_trainer)
    corpus_df = unified_analyzer.process_corpus(X_raw, ids=df_reviews["review_id"].tolist())

    # Calculate Brand-Level Sentiment Profile
    brand_sentiments = []
    for text in X_raw:
        res = unified_analyzer.analyze_document(text, preferred_ml_model="Linear SVC")
        for aspect in res["entity_aspect_sentiments"]:
            if aspect["entity_type"] == "ORG":
                brand_sentiments.append({
                    "brand": aspect["entity_text"],
                    "sentiment": aspect["sentiment"],
                    "compound": aspect["compound_score"]
                })

    df_brand_sents = pd.DataFrame(brand_sentiments)
    if not df_brand_sents.empty:
        brand_summary = df_brand_sents.groupby("brand")["compound"].agg(["count", "mean"]).reset_index()
        brand_summary = brand_summary.sort_values(by="count", ascending=False).head(8)

        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        colors = ["#2E7D32" if x > 0.05 else ("#C62828" if x < -0.05 else "#F57C00") for x in brand_summary["mean"]]
        ax = sns.barplot(data=brand_summary, x="brand", y="mean", palette=colors)
        plt.title("Entity-Centric Sentiment Intelligence: Mean Compound Score by Brand", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("Mean VADER Compound Score", fontsize=10)
        plt.xlabel("Brand Entity", fontsize=10)
        plt.ylim(-1.0, 1.0)
        plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)

        for p in ax.patches:
            val = p.get_height()
            ax.annotate(f"{val:.2f}",
                        (p.get_x() + p.get_width() / 2., val),
                        ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=9, color="black",
                        xytext=(0, 3 if val >= 0 else -10),
                        textcoords="offset points")

        plt.tight_layout()
        ent_sent_plot = os.path.join(FIGURES_DIR, "entity_sentiment_distribution.png")
        plt.savefig(ent_sent_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved entity-centric sentiment intelligence plot to {ent_sent_plot}")

    # -------------------------------------------------------------
    # 6. Save Cached Metrics to JSON
    # -------------------------------------------------------------
    serializable_metrics = {
        "dataset_stats": {
            "total_reviews": len(df_reviews),
            "categories": df_reviews["category"].value_counts().to_dict(),
            "sentiment_counts": df_reviews["ground_truth_sentiment"].value_counts().to_dict(),
            "ner_sentences": len(ner_benchmark_data),
            "ner_tokens": len(flat_true_tags) + len([t for s in train_tags for t in s]),
            "entity_type_counts": all_entity_types
        },
        "sentiment_cv_scores": cv_scores,
        "sentiment_results": {
            name: {
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"]
            }
            for name, m in sentiment_results.items()
        },
        "ner_token_results": {
            name: {
                "token_accuracy": m["token_accuracy"],
                "token_precision_macro": m["token_precision_macro"],
                "token_recall_macro": m["token_recall_macro"],
                "token_f1_macro": m["token_f1_macro"],
                "token_f1_weighted": m["token_f1_weighted"]
            }
            for name, m in ner_token_results.items()
        },
        "ner_span_results": {
            name: {
                "span_precision": m["span_precision"],
                "span_recall": m["span_recall"],
                "span_f1": m["span_f1"],
                "per_type_metrics": m["per_type_metrics"]
            }
            for name, m in ner_span_results.items()
        }
    }

    metrics_path = os.path.join(REPORTS_DIR, "benchmark_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"\nSuccessfully cached benchmark metrics to: {metrics_path}")

    # -------------------------------------------------------------
    # 7. Print Terminal Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY TABLE: SENTIMENT ANALYSIS")
    print("=" * 70)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 70)
    for name, m in sentiment_results.items():
        print(f"{name:<28} | {m['accuracy']:<10.3f} | {m['f1_macro']:<10.3f} | {m['f1_weighted']:<12.3f}")

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY TABLE: NAMED ENTITY RECOGNITION (SPAN-LEVEL)")
    print("=" * 70)
    print(f"{'Model Name':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for name, m in ner_span_results.items():
        print(f"{name:<28} | {m['span_precision']:<10.3f} | {m['span_recall']:<10.3f} | {m['span_f1']:<10.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
