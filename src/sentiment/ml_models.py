"""
Machine Learning Sentiment Analysis Module.
Trains, tunes, and evaluates TF-IDF based classification models:
- Multinomial Naive Bayes (MultinomialNB)
- Logistic Regression (MaxEnt)
- Linear Support Vector Classifier (LinearSVC)
- Random Forest Classifier (RandomForest)
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

from src.config import (
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TFIDF_SUBLINEAR_TF
)


class SentimentMLTrainer:
    """
    Manages TF-IDF vectorization and training of supervised machine learning classifiers.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF
        )

        self.models: Dict[str, Any] = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
            "Logistic Regression": LogisticRegression(
                C=1.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            "Linear SVC": LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                dual="auto"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=RANDOM_SEED
            )
        }

        self.trained_pipelines: Dict[str, Pipeline] = {}

    def build_pipeline(self, model_name: str) -> Pipeline:
        """Constructs an sklearn Pipeline with TF-IDF Vectorizer and model."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.models.keys())}")
        
        clf = self.models[model_name]
        return Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", clf)
        ])

    def cross_validate_all(self, X: List[str], y: List[str], cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Runs Stratified K-Fold Cross Validation across all candidate ML models.
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        cv_results = {}

        for name, clf in self.models.items():
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=TFIDF_MIN_DF,
                    sublinear_tf=TFIDF_SUBLINEAR_TF
                )),
                ("clf", clf)
            ])
            scores_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
            scores_f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")

            cv_results[name] = {
                "cv_accuracy_mean": float(np.mean(scores_acc)),
                "cv_accuracy_std": float(np.std(scores_acc)),
                "cv_f1_macro_mean": float(np.mean(scores_f1)),
                "cv_f1_macro_std": float(np.std(scores_f1))
            }

        return cv_results

    def fit_all(self, X_train: List[str], y_train: List[str]):
        """Fits all models on the training corpus."""
        for name in self.models.keys():
            pipe = self.build_pipeline(name)
            pipe.fit(X_train, y_train)
            self.trained_pipelines[name] = pipe

    def predict(self, model_name: str, X_test: List[str]) -> np.ndarray:
        """Generates predictions for a specific model."""
        if model_name not in self.trained_pipelines:
            raise RuntimeError(f"Model '{model_name}' has not been trained yet. Call fit_all first.")
        return self.trained_pipelines[model_name].predict(X_test)

    def get_feature_importance(self, model_name: str = "Logistic Regression", top_n: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extracts the top positive, neutral, and negative n-gram features from linear models.
        """
        if model_name not in self.trained_pipelines:
            raise RuntimeError(f"Model '{model_name}' not fitted.")

        pipe = self.trained_pipelines[model_name]
        vectorizer = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]

        if not hasattr(clf, "coef_"):
            return {}

        feature_names = np.array(vectorizer.get_feature_names_out())
        classes = clf.classes_
        importance_dict = {}

        for idx, cls in enumerate(classes):
            coefs = clf.coef_[idx]
            top_indices = np.argsort(coefs)[-top_n:][::-1]
            top_features = [(feature_names[i], float(coefs[i])) for i in top_indices]
            importance_dict[cls] = top_features

        return importance_dict

    def save_model(self, model_name: str, filepath: str):
        """Persists trained pipeline to disk."""
        if model_name not in self.trained_pipelines:
            raise ValueError(f"Model {model_name} not found.")
        joblib.dump(self.trained_pipelines[model_name], filepath)

    def load_model(self, model_name: str, filepath: str):
        """Loads trained pipeline from disk."""
        self.trained_pipelines[model_name] = joblib.load(filepath)
