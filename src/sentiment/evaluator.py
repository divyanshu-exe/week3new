"""
Evaluation and Visualization Module for Sentiment Analysis.
Computes multi-class classification metrics and generates publication-quality plots.
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.config import SENTIMENT_LABELS, FIGURES_DIR


class SentimentEvaluator:
    """
    Computes standard NLP classification metrics and produces graphical diagnostic plots.
    """

    def __init__(self, labels: List[str] = SENTIMENT_LABELS):
        self.labels = labels
        os.makedirs(FIGURES_DIR, exist_ok=True)

    def evaluate_predictions(self, y_true: List[str], y_pred: List[str], model_name: str = "Model") -> Dict[str, Any]:
        """
        Calculates Accuracy, Precision, Recall, and F1-score across macro and weighted averages.
        """
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=self.labels, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        report = classification_report(y_true, y_pred, labels=self.labels, output_dict=True, zero_division=0)

        return {
            "model_name": model_name,
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm,
            "classification_report": report
        }

    def plot_confusion_matrices(self, results: Dict[str, Dict[str, Any]], filename: str = "sentiment_confusion_matrices.png") -> str:
        """
        Plots a grid of confusion matrices for multiple models and saves the figure.
        """
        n_models = len(results)
        cols = 2
        rows = (n_models + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (name, metrics) in enumerate(results.items()):
            cm = metrics["confusion_matrix"]
            ax = axes[idx]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.labels,
                yticklabels=self.labels,
                ax=ax,
                cbar=False
            )
            ax.set_title(f"{name} (Acc: {metrics['accuracy']:.2%}, F1: {metrics['f1_macro']:.2f})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label", fontsize=10)

        # Hide any unused subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    def plot_model_comparison(self, results: Dict[str, Dict[str, Any]], filename: str = "sentiment_models_comparison.png") -> str:
        """
        Plots comparative bar charts showing Accuracy, Macro F1, and Weighted F1 across models.
        """
        data = []
        for name, metrics in results.items():
            data.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["f1_macro"],
                "Weighted F1": metrics["f1_weighted"]
            })
        df_comp = pd.DataFrame(data)

        # Melt for seaborn grouped barplot
        df_melted = df_comp.melt(id_vars="Model", var_name="Metric", value_name="Score")

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        palette = ["#1B365D", "#4A777A", "#D97724"]
        ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=palette)

        plt.title("Comparative Performance Across Sentiment Analysis Approaches", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
        plt.xlabel("Model Paradigm", fontsize=11)
        plt.ylim(0, 1.1)
        plt.legend(loc="lower right", frameon=True)
        plt.xticks(rotation=15, ha="right")

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

    def plot_feature_importance(self, feature_dict: Dict[str, List[Tuple[str, float]]], filename: str = "sentiment_feature_importance.png") -> str:
        """
        Plots the top predictive features for positive and negative sentiment classes.
        """
        if not feature_dict:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if "positive" in feature_dict and feature_dict["positive"]:
            pos_feats, pos_weights = zip(*feature_dict["positive"][:10])
            y_pos = np.arange(len(pos_feats))
            ax1.barh(y_pos, pos_weights, color="#2E7D32", align="center")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(pos_feats, fontsize=10)
            ax1.invert_yaxis()
            ax1.set_xlabel("Coefficient Weight (Log-Odds)", fontsize=10)
            ax1.set_title("Top 10 Most Predictive Features: POSITIVE", fontsize=12, fontweight="bold")
            ax1.grid(axis="x", linestyle="--", alpha=0.7)

        if "negative" in feature_dict and feature_dict["negative"]:
            neg_feats, neg_weights = zip(*feature_dict["negative"][:10])
            y_neg = np.arange(len(neg_feats))
            ax2.barh(y_neg, neg_weights, color="#C62828", align="center")
            ax2.set_yticks(y_neg)
            ax2.set_yticklabels(neg_feats, fontsize=10)
            ax2.invert_yaxis()
            ax2.set_xlabel("Coefficient Weight (Log-Odds)", fontsize=10)
            ax2.set_title("Top 10 Most Predictive Features: NEGATIVE", fontsize=12, fontweight="bold")
            ax2.grid(axis="x", linestyle="--", alpha=0.7)

        plt.suptitle("Sentiment Model Interpretability: Informative N-Gram Coefficients", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
