"""
Statistical Machine Learning Sequence Classification for Named Entity Recognition.
Implements token-level feature extraction (context windows, word shapes, POS, affixes)
with a Linear Support Vector Classifier / Logistic Regression classifier.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

from src.config import RANDOM_SEED


class FeatureBasedStatisticalNER:
    """
    Sequence labeling classifier trained on rich linguistic and morphological token features:
    - Contextual window features (w-1, w+1, pos-1, pos+1)
    - Orthographic features (capitalization, digits, hyphens, word shape)
    - Affix features (prefixes and suffixes of length 2 and 3)
    - Syntactic POS tags
    """

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.classifier = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            dual="auto",
            max_iter=3000
        )
        self.is_fitted = False

    @staticmethod
    def get_word_shape(word: str) -> str:
        """Computes word shape representation (e.g., iPhone -> xXxxxx, 2024 -> dddd)."""
        shape = re.sub(r"[A-Z]", "X", word)
        shape = re.sub(r"[a-z]", "x", shape)
        shape = re.sub(r"[0-9]", "d", shape)
        return shape

    def extract_token_features(self, tokens: List[str], pos_tags: List[str], i: int) -> Dict[str, Any]:
        """
        Extracts a feature dictionary for the i-th token in a sentence.
        """
        word = tokens[i]
        pos = pos_tags[i]

        features = {
            "bias": 1.0,
            "word.lower": word.lower(),
            "word.isupper": word.isupper(),
            "word.istitle": word.istitle(),
            "word.isdigit": word.isdigit(),
            "word.has_digit": any(c.isdigit() for c in word),
            "word.has_hyphen": "-" in word,
            "word.len": len(word),
            "word.shape": self.get_word_shape(word),
            "word.prefix_2": word[:2],
            "word.prefix_3": word[:3],
            "word.suffix_2": word[-2:],
            "word.suffix_3": word[-3:],
            "pos": pos,
            "pos[:2]": pos[:2]
        }

        # Previous token features
        if i > 0:
            prev_word = tokens[i - 1]
            prev_pos = pos_tags[i - 1]
            features.update({
                "-1:word.lower": prev_word.lower(),
                "-1:word.istitle": prev_word.istitle(),
                "-1:word.isupper": prev_word.isupper(),
                "-1:pos": prev_pos,
                "-1:pos[:2]": prev_pos[:2]
            })
        else:
            features["BOS"] = True  # Beginning of sentence

        # Next token features
        if i < len(tokens) - 1:
            next_word = tokens[i + 1]
            next_pos = pos_tags[i + 1]
            features.update({
                "+1:word.lower": next_word.lower(),
                "+1:word.istitle": next_word.istitle(),
                "+1:word.isupper": next_word.isupper(),
                "+1:pos": next_pos,
                "+1:pos[:2]": next_pos[:2]
            })
        else:
            features["EOS"] = True  # End of sentence

        return features

    def sentence_to_features(self, tokens: List[str], pos_tags: List[str]) -> List[Dict[str, Any]]:
        """Converts an entire sentence of tokens and pos_tags into a list of feature dictionaries."""
        return [self.extract_token_features(tokens, pos_tags, i) for i in range(len(tokens))]

    def fit(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]], sentences_tags: List[List[str]]):
        """
        Fits the DictVectorizer and LinearSVC classifier on token sequences.
        """
        X_feats = []
        y_all = []

        for tokens, pos, tags in zip(sentences_tokens, sentences_pos, sentences_tags):
            sent_feats = self.sentence_to_features(tokens, pos)
            X_feats.extend(sent_feats)
            y_all.extend(tags)

        X_transformed = self.vectorizer.fit_transform(X_feats)
        self.classifier.fit(X_transformed, y_all)
        self.is_fitted = True

    def predict_sentence_iob(self, tokens: List[str], pos_tags: List[str]) -> List[str]:
        """Predicts IOB tags for a single tokenized sentence."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call fit first.")

        sent_feats = self.sentence_to_features(tokens, pos_tags)
        X_transformed = self.vectorizer.transform(sent_feats)
        predictions = self.classifier.predict(X_transformed)
        return list(predictions)

    def predict_all_iob(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]]) -> List[List[str]]:
        """Predicts IOB tags for multiple sentences."""
        return [self.predict_sentence_iob(toks, pos) for toks, pos in zip(sentences_tokens, sentences_pos)]
