"""
Lexicon-Based Sentiment Analysis Module.
Implements VADER (Valence Aware Dictionary and sEntiment Reasoner) and TextBlob.
"""

from typing import Dict, List, Any
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from src.config import (
    VADER_POS_THRESHOLD,
    VADER_NEG_THRESHOLD,
    TEXTBLOB_POS_THRESHOLD,
    TEXTBLOB_NEG_THRESHOLD
)


class VaderSentimentAnalyzer:
    """
    Lexicon- and rule-based sentiment analysis using NLTK's VADER.
    VADER takes into account:
    - Word valence scores from a human-curated lexicon
    - Intensifiers and degree modifiers ("very", "extremely", "slightly")
    - Capitalization ("GREAT" vs "great")
    - Punctuation emphasis ("amazing!!!")
    - Contrastive conjunctions ("The screen is good, but the battery is terrible")
    - Negation scoping ("not bad")
    """

    def __init__(self, pos_threshold: float = VADER_POS_THRESHOLD, neg_threshold: float = VADER_NEG_THRESHOLD):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single text string and returns valence scores and categorical label.
        """
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= self.pos_threshold:
            sentiment = "positive"
        elif compound <= self.neg_threshold:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "compound": compound,
            "predicted_sentiment": sentiment
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Runs VADER sentiment analysis over a list of texts and returns a DataFrame.
        """
        records = [self.analyze_text(t) for t in texts]
        return pd.DataFrame(records)


class TextBlobSentimentAnalyzer:
    """
    Rule-based sentiment analyzer using TextBlob.
    Calculates:
    - Polarity: Float within range [-1.0, 1.0], where -1 is intensely negative and +1 is intensely positive.
    - Subjectivity: Float within range [0.0, 1.0], where 0.0 is very objective and 1.0 is very subjective.
    """

    def __init__(self, pos_threshold: float = TEXTBLOB_POS_THRESHOLD, neg_threshold: float = TEXTBLOB_NEG_THRESHOLD):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single text string and returns polarity, subjectivity, and categorical label.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity >= self.pos_threshold:
            sentiment = "positive"
        elif polarity <= self.neg_threshold:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "predicted_sentiment": sentiment
        }

    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Runs TextBlob sentiment analysis over a list of texts and returns a DataFrame.
        """
        records = [self.analyze_text(t) for t in texts]
        return pd.DataFrame(records)
