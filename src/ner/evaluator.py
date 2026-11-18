"""
Comprehensive Named Entity Recognition Evaluator.
Computes token-level and span-level entity metrics (Precision, Recall, F1),
and produces diagnostic visualization plots.
"""

import os
from typing import List, Dict, Tuple, Set, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.config import FIGURES_DIR, NER_ENTITY_TYPES


def iob_to_spans(tags: List[str]) -> Set[Tuple[str, int, int]]:
    """
    Converts a sequence of IOB tags into a set of entity spans: (entity_type, start_idx, end_idx).
    End index is inclusive.
    """
    spans = set()
    current_type = None
    start_idx = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                spans.add((current_type, start_idx, i - 1))
            current_type = tag.split("-")[1]
            start_idx = i
        elif tag.startswith("I-"):
            ent_type = tag.split("-")[1]
            if current_type == ent_type:
                # continuation
                pass
            else:
                if current_type is not None:
                    spans.add((current_type, start_idx, i - 1))
                current_type = ent_type
                start_idx = i
        else:  # tag == 'O'
            if current_type is not None:
                spans.add((current_type, start_idx, i - 1))
                current_type = None
                start_idx = None

    if current_type is not None:
        spans.add((current_type, start_idx, len(tags) - 1))

    return spans


class NEREvaluator:
    """
    Dual-level NER evaluation:
    1. Token-level classification metrics
    2. Span-level exact-match entity extraction metrics per category
    """

    def __init__(self, entity_types: List[str] = NER_ENTITY_TYPES):
        self.entity_types = entity_types
        os.makedirs(FIGURES_DIR, exist_ok=True)

    def evaluate_token_level(self, y_true_flat: List[str], y_pred_flat: List[str], model_name: str = "NER Model") -> Dict[str, Any]:
        """Calculates token-level Accuracy, Precision, Recall, and F1."""
        acc = accuracy_score(y_true_flat, y_pred_flat)
        labels = sorted(list(set(y_true_flat) | set(y_pred_flat)))
        prec_macro = precision_score(y_true_flat, y_pred_flat, labels=labels, average="macro", zero_division=0)
        rec_macro = recall_score(y_true_flat, y_pred_flat, labels=labels, average="macro", zero_division=0)
        f1_macro = f1_score(y_true_flat, y_pred_flat, labels=labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true_flat, y_pred_flat, labels=labels, average="weighted", zero_division=0)

        report = classification_report(y_true_flat, y_pred_flat, labels=labels, output_dict=True, zero_division=0)

        return {
            "model_name": model_name,
            "token_accuracy": float(acc),
            "token_precision_macro": float(prec_macro),
            "token_recall_macro": float(rec_macro),
            "token_f1_macro": float(f1_macro),
            "token_f1_weighted": float(f1_weighted),
            "token_report": report
        }

    def evaluate_span_level(self, y_true_sentences: List[List[str]], y_pred_sentences: List[List[str]], model_name: str = "NER Model") -> Dict[str, Any]:
        """
        Calculates exact span-level Precision, Recall, and F1 per entity type and macro overall.
        """
        type_counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in self.entity_types}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for true_tags, pred_tags in zip(y_true_sentences, y_pred_sentences):
            true_spans = iob_to_spans(true_tags)
            pred_spans = iob_to_spans(pred_tags)

            # Match spans
            tp_spans = true_spans & pred_spans
            fp_spans = pred_spans - true_spans
            fn_spans = true_spans - pred_spans

            total_tp += len(tp_spans)
            total_fp += len(fp_spans)
            total_fn += len(fn_spans)

            for ent_type, _, _ in tp_spans:
                if ent_type in type_counts:
                    type_counts[ent_type]["tp"] += 1
            for ent_type, _, _ in fp_spans:
                if ent_type in type_counts:
                    type_counts[ent_type]["fp"] += 1
            for ent_type, _, _ in fn_spans:
                if ent_type in type_counts:
                    type_counts[ent_type]["fn"] += 1

        # Calculate per-type metrics
        per_type_metrics = {}
        for ent_type, counts in type_counts.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_type_metrics[ent_type] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": tp + fn
            }

        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

        return {
            "model_name": model_name,
            "span_precision": float(overall_prec),
            "span_recall": float(overall_rec),
            "span_f1": float(overall_f1),
            "per_type_metrics": per_type_metrics
        }

    def plot_ner_comparison(self, model_metrics: Dict[str, Dict[str, float]], filename: str = "ner_models_comparison.png") -> str:
        """
        Plots comparative bar charts showing Span Precision, Recall, and F1 across NER models.
        """
        rows = []
        for name, m in model_metrics.items():
            rows.append({
                "Model": name,
                "Precision": m["span_precision"],
                "Recall": m["span_recall"],
                "F1-Score": m["span_f1"]
            })
        df_comp = pd.DataFrame(rows)
        df_melted = df_comp.melt(id_vars="Model", var_name="Metric", value_name="Score")

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        palette = ["#1B365D", "#4A777A", "#E06D53"]
        ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=palette)

        plt.title("Comparative Evaluation Across Named Entity Recognition Approaches", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
        plt.xlabel("NER Paradigm", fontsize=11)
        plt.ylim(0, 1.1)
        plt.legend(loc="lower right", frameon=True)

        for p in ax.patches:
            height = p.get_height()
            if height > 0.01:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom",
                            fontsize=9, color="black",
                            xytext=(0, 2),
                            textcoords="offset points")

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_entity_distribution(self, type_counts: Dict[str, int], filename: str = "ner_entity_distribution.png") -> str:
        """
        Plots the distribution of entity occurrences across categories.
        """
        df_counts = pd.DataFrame(list(type_counts.items()), columns=["Entity Type", "Frequency"])
        df_counts = df_counts.sort_values(by="Frequency", ascending=False)

        plt.figure(figsize=(9, 5))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df_counts, x="Entity Type", y="Frequency", palette="viridis")

        plt.title("Entity Frequency Distribution in Benchmark Corpus", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("Count of Entity Mentions", fontsize=11)
        plt.xlabel("Named Entity Category", fontsize=11)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom",
                        fontsize=10, color="black",
                        xytext=(0, 3),
                        textcoords="offset points")

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
