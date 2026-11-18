"""
Unified NLP Pipeline: Sentiment Analysis and Named Entity Recognition.
Integrates Mini-Project 1 and Mini-Project 2 into a single unified analytical framework
to perform Entity-Centric / Aspect-Based Sentiment Mining on unstructured text.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from nltk import sent_tokenize

from src.preprocessing import TextPreprocessor
from src.sentiment.lexicon_analyzer import VaderSentimentAnalyzer, TextBlobSentimentAnalyzer
from src.sentiment.ml_models import SentimentMLTrainer
from src.ner.rule_based_ner import RuleBasedNER
from src.ner.nltk_chunker_ner import NLTKChunkerNER
from src.ner.statistical_ner import FeatureBasedStatisticalNER


class UnifiedNLPAnalyzer:
    """
    Unified NLP Pipeline orchestrating:
    1. Preprocessing (tokenization, lemmatization, normalization)
    2. Named Entity Extraction (Rule-Based, NLTK Chunker, Statistical)
    3. Sentiment Scoring (VADER, TextBlob, Trained ML Classifier)
    4. Entity-Sentiment Association (sentence-level aspect attribution)
    """

    def __init__(self, ml_trainer: Optional[SentimentMLTrainer] = None):
        self.preprocessor = TextPreprocessor()
        self.vader = VaderSentimentAnalyzer()
        self.textblob = TextBlobSentimentAnalyzer()
        self.ml_trainer = ml_trainer
        self.rule_ner = RuleBasedNER()
        self.nltk_ner = NLTKChunkerNER()

    def analyze_document(self, text: str, preferred_ml_model: str = "Linear SVC") -> Dict[str, Any]:
        """
        Runs comprehensive analysis on a single document or review text.
        Returns document-level sentiment, extracted entities, and entity-specific sentiment associations.
        """
        # 1. Document-level sentiment scoring
        vader_res = self.vader.analyze_text(text)
        tb_res = self.textblob.analyze_text(text)

        ml_pred = None
        if self.ml_trainer and preferred_ml_model in self.ml_trainer.trained_pipelines:
            clean_ml_text = self.preprocessor.preprocess_for_sentiment_ml(text)
            ml_pred = self.ml_trainer.predict(preferred_ml_model, [clean_ml_text])[0]

        # 2. Extract Entities
        entities = self.rule_ner.extract_entities_from_text(text)

        # 3. Sentence-level Entity-Sentiment Mining
        sentences = sent_tokenize(text)
        entity_sentiments = []

        for sent in sentences:
            sent_entities = self.rule_ner.extract_entities_from_text(sent)
            if sent_entities:
                sent_vader = self.vader.analyze_text(sent)
                for ent in sent_entities:
                    entity_sentiments.append({
                        "entity_text": ent["text"],
                        "entity_type": ent["type"],
                        "sentence": sent,
                        "sentiment": sent_vader["predicted_sentiment"],
                        "compound_score": sent_vader["compound"]
                    })

        return {
            "raw_text": text,
            "document_sentiment": {
                "vader_sentiment": vader_res["predicted_sentiment"],
                "vader_compound": vader_res["compound"],
                "textblob_polarity": tb_res["polarity"],
                "textblob_subjectivity": tb_res["subjectivity"],
                "ml_model_used": preferred_ml_model,
                "ml_prediction": ml_pred
            },
            "entities_found": entities,
            "entity_aspect_sentiments": entity_sentiments
        }

    def process_corpus(self, texts: List[str], ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Processes a collection of texts and compiles tabular entity-sentiment intelligence.
        """
        if ids is None:
            ids = [f"DOC_{i+1:03d}" for i in range(len(texts))]

        rows = []
        for doc_id, text in zip(ids, texts):
            res = self.analyze_document(text)
            doc_sent = res["document_sentiment"]
            entities = res["entities_found"]

            brands = [e["text"] for e in entities if e["type"] == "ORG"]
            products = [e["text"] for e in entities if e["type"] == "PROD"]
            persons = [e["text"] for e in entities if e["type"] == "PER"]
            locations = [e["text"] for e in entities if e["type"] == "LOC"]
            prices = [e["text"] for e in entities if e["type"] == "MONEY"]

            rows.append({
                "doc_id": doc_id,
                "vader_sentiment": doc_sent["vader_sentiment"],
                "vader_compound": doc_sent["vader_compound"],
                "textblob_polarity": doc_sent["textblob_polarity"],
                "ml_prediction": doc_sent["ml_prediction"],
                "brands_detected": ", ".join(set(brands)) if brands else "None",
                "products_detected": ", ".join(set(products)) if products else "None",
                "persons_detected": ", ".join(set(persons)) if persons else "None",
                "locations_detected": ", ".join(set(locations)) if locations else "None",
                "prices_detected": ", ".join(set(prices)) if prices else "None",
                "entity_count": len(entities)
            })

        return pd.DataFrame(rows)
