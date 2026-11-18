"""
NLTK Statistical Named Entity Recognition Module.
Wraps NLTK's Maximum Entropy Tree Chunker (nltk.ne_chunk) and maps chunk trees to standard IOB sequences.
"""

from typing import List, Tuple, Dict, Any
import nltk
from nltk import ne_chunk, tree2conlltags


class NLTKChunkerNER:
    """
    Standard statistical tree chunker from NLTK.
    Trained on the ACE / CoNLL corpora using Maximum Entropy Markov Models (MEMMs).
    """

    NLTK_TYPE_MAP = {
        "PERSON": "PER",
        "ORGANIZATION": "ORG",
        "GPE": "LOC",
        "LOCATION": "LOC",
        "FACILITY": "ORG"
    }

    def __init__(self):
        pass

    def parse_chunk_tree(self, pos_tagged_tokens: List[Tuple[str, str]]) -> nltk.Tree:
        """Runs ne_chunk on POS-tagged tokens to produce a syntax/entity tree."""
        return ne_chunk(pos_tagged_tokens, binary=False)

    def predict_iob_tags(self, tokens: List[str], pos_tags: List[str]) -> List[str]:
        """
        Parses tokens and pos_tags through ne_chunk and returns aligned IOB tags.
        """
        tagged_pairs = list(zip(tokens, pos_tags))
        tree = self.parse_chunk_tree(tagged_pairs)
        conll_tags = tree2conlltags(tree)

        iob_result = []
        for word, pos, tag in conll_tags:
            if tag == "O":
                iob_result.append("O")
            else:
                # tag format: B-PERSON, I-ORGANIZATION
                prefix, ent_type = tag.split("-", 1)
                standard_type = self.NLTK_TYPE_MAP.get(ent_type, ent_type)
                iob_result.append(f"{prefix}-{standard_type}")

        return iob_result

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entities from raw text using NLTK word_tokenize, pos_tag, and ne_chunk.
        """
        tokens = nltk.word_tokenize(text)
        pos_tagged = nltk.pos_tag(tokens)
        tree = self.parse_chunk_tree(pos_tagged)

        entities = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                ent_type = self.NLTK_TYPE_MAP.get(subtree.label(), subtree.label())
                ent_text = " ".join([word for word, pos in subtree.leaves()])
                entities.append({
                    "text": ent_text,
                    "type": ent_type
                })
        return entities
