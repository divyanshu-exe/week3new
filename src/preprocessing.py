"""
Text Preprocessing Module for NLP Sentiment Analysis and Named Entity Recognition.
Provides multi-stage preprocessing pipelines tailored for both tasks.
"""

import re
import html
import unicodedata
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# Contractions mapping
CONTRACTIONS_DICT = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not"
}

NEGATION_WORDS = {
    "no", "not", "never", "none", "neither", "nor", "nowhere",
    "hardly", "scarcely", "barely", "cannot", "without", "didn't",
    "don't", "wasn't", "isn't", "aren't", "won't"
}

PUNCTUATION_CLAUSE_END = {".", "!", "?", ";", ":", ","}


class TextPreprocessor:
    """
    Modular text preprocessing pipeline supporting:
    - Dual modes: 'sentiment' (cleaning, lemmatization, stopword removal) and 'ner' (casing & structure preservation)
    - Negation scoping
    - Contextual POS-guided Lemmatization
    """

    def __init__(self, language: str = "english"):
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords and retain negation words for sentiment integrity
        base_stopwords = set(stopwords.words(language))
        self.sentiment_stopwords = base_stopwords - NEGATION_WORDS

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expands common English contractions."""
        expanded = text
        for contraction, replacement in CONTRACTIONS_DICT.items():
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            expanded = pattern.sub(replacement, expanded)
        return expanded

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalizes unicode characters and unescapes HTML entities."""
        text = html.unescape(text)
        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def clean_text_basic(text: str) -> str:
        """Standard cleaning: strip extra whitespace and normalize line breaks."""
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def clean_for_ml(text: str) -> str:
        """
        Removes special symbols and URLs while retaining alphanumeric characters and spaces.
        Suitable for bag-of-words / TF-IDF representations.
        """
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^\w\s\$\.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def get_wordnet_pos(treebank_tag: str) -> str:
        """Maps Penn Treebank POS tags to WordNet POS tags."""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into a list of word tokens using NLTK."""
        return word_tokenize(text)

    def tag_pos(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Computes Part-of-Speech tags for a token list."""
        return pos_tag(tokens)

    def lemmatize_tokens(self, tokens: List[str], pos_tags: Optional[List[Tuple[str, str]]] = None) -> List[str]:
        """
        Lemmatizes tokens using POS context. If pos_tags is not provided, computes them.
        """
        if pos_tags is None:
            pos_tags = self.tag_pos(tokens)

        lemmas = []
        for token, tag in pos_tags:
            wn_tag = self.get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(token.lower(), pos=wn_tag)
            lemmas.append(lemma)
        return lemmas

    def remove_stopwords(self, tokens: List[str], preserve_negation: bool = True) -> List[str]:
        """
        Removes stopwords from a token list. Negations are preserved by default.
        """
        stop_set = self.sentiment_stopwords if preserve_negation else set(stopwords.words("english"))
        return [t for t in tokens if t.lower() not in stop_set and t.isalnum()]

    @staticmethod
    def apply_negation_marking(tokens: List[str]) -> List[str]:
        """
        Appends '_NOT' suffix to words occurring within the scope of a negation particle
        until the next punctuation mark.
        Example: ['not', 'very', 'good', '.'] -> ['not', 'very_NOT', 'good_NOT', '.']
        """
        transformed = []
        negation_active = False

        for token in tokens:
            if token.lower() in NEGATION_WORDS:
                negation_active = True
                transformed.append(token)
            elif token in PUNCTUATION_CLAUSE_END:
                negation_active = False
                transformed.append(token)
            else:
                if negation_active and token.isalnum():
                    transformed.append(f"{token}_NOT")
                else:
                    transformed.append(token)

        return transformed

    def preprocess_for_sentiment_ml(self, text: str, apply_negation: bool = True) -> str:
        """
        Full end-to-end preprocessing pipeline for Machine Learning Sentiment Classifiers.
        1. Normalization & contraction expansion
        2. Tokenization
        3. POS tagging & POS-aware lemmatization
        4. Stopword removal (preserving negations)
        5. Optional negation tagging
        6. Reassembly into a clean normalized string
        """
        expanded = self.expand_contractions(text)
        normalized = self.normalize_unicode(expanded)
        tokens = self.tokenize(normalized)
        
        # Lemmatize
        tagged = self.tag_pos(tokens)
        lemmas = []
        for word, tag in tagged:
            if word in PUNCTUATION_CLAUSE_END:
                lemmas.append(word)
            elif word.isalnum():
                wn_tag = self.get_wordnet_pos(tag)
                lemmas.append(self.lemmatizer.lemmatize(word.lower(), pos=wn_tag))

        # Filter stopwords
        filtered = []
        for t in lemmas:
            if t in PUNCTUATION_CLAUSE_END or (t not in self.sentiment_stopwords):
                filtered.append(t)

        if apply_negation:
            filtered = self.apply_negation_marking(filtered)

        # Remove lone punctuation before rejoining for ML
        final_tokens = [t for t in filtered if t not in PUNCTUATION_CLAUSE_END]
        return " ".join(final_tokens)

    def preprocess_for_ner(self, text: str) -> List[Tuple[str, str]]:
        """
        Preprocessing pipeline for Named Entity Recognition.
        Maintains case sensitivity, punctuation tokens, and produces (token, POS) tuples.
        """
        normalized = self.normalize_unicode(text)
        cleaned = self.clean_text_basic(normalized)
        tokens = self.tokenize(cleaned)
        tagged = self.tag_pos(tokens)
        return tagged
