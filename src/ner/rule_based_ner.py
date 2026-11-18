"""
Rule-Based and Gazetteer Named Entity Recognition Module.
Combines regular expression patterns, contextual heuristics, and domain-specific gazetteers.
"""

import re
from typing import List, Dict, Tuple, Any

# Curated domain gazetteers
ORGANIZATIONS = {
    "Apple", "Samsung", "Sony", "Bose", "Google", "Microsoft", "Dell",
    "Anker", "Asus", "Lenovo", "HP", "Amazon", "Beats", "Logitech",
    "Sennheiser", "Fitbit", "OnePlus", "Ring", "Intel", "AMD", "Nvidia"
}

PERSONS = {
    "Tim Cook", "Sundar Pichai", "Satya Nadella", "Jeff Bezos", "Andy Jassy",
    "Dr. Elena Rostova", "Elena Rostova", "Marcus Sterling", "Priya Sharma",
    "Alexander Wright", "Chloe Dupont", "Michael Chang", "Emily Davis",
    "Liam O'Connor", "Carlos Gomez", "Kenji Sato", "Jonathan Reed",
    "Sophia Lorenzi", "Lucas Weber", "Hassan Al-Mansoor", "Jessica Taylor",
    "Vikram Patel", "Daniel Craig", "Nia Robinson", "Felix Lindqvist",
    "Brenda Scott", "Tariq Mansour", "Rachel Green", "Dr. Hans Gruber",
    "Hans Gruber", "Aarav Mehta", "Oliver Queen", "Dmitri Volkov",
    "Klaus Schmidt", "Anita Roy", "Arthur Pendelton", "Sarah Connor",
    "David Miller", "Leila Zahra", "Gabriel Silva", "Claire Bennet",
    "Chen Wei", "Nadia Petrova", "Mateo Rossi", "Arjun Nair"
}

LOCATIONS = {
    "Tokyo", "London", "San Francisco", "Cupertino", "Mountain View",
    "Seattle", "Bangalore", "New York", "Paris", "Austin", "Dublin",
    "Madrid", "Osaka", "Chicago", "Milan", "Berlin", "Dubai", "Toronto",
    "Mumbai", "Los Angeles", "Atlanta", "Stockholm", "Denver", "Doha",
    "Frankfurt", "Moscow", "Edinburgh", "Munich", "Kolkata", "Boston",
    "Seoul", "Sydney", "Singapore", "Zurich", "Hong Kong", "San Jose",
    "Melbourne", "San Diego", "Montreal", "Star City"
}

PRODUCTS = {
    "iPhone 15 Pro", "iPhone 15", "iPhone",
    "Galaxy S24 Ultra", "Galaxy S24", "Galaxy Z Fold 5", "Galaxy Watch 6",
    "Galaxy Tab S9 Ultra", "Galaxy S23 FE", "Galaxy A54",
    "WH-1000XM5", "WF-1000XM5", "LinkBuds S", "A7 IV", "PlayStation 5 Slim",
    "MacBook Air M3", "MacBook Pro M3 Max", "AirPods Max", "iPad Pro 12.9",
    "QuietComfort Ultra", "Smart Soundbar 900", "Frames Tenor",
    "XPS 15", "Alienware m16", "Inspiron 14", "UltraSharp 27 4K",
    "Pixel 8 Pro", "Pixel 7a", "Pixel Watch 2", "Pixel Fold", "Nest Hub Max",
    "Surface Pro 9", "Surface Laptop 5", "Xbox Series X",
    "Soundcore Space Q45", "Soundcore Liberty 4", "737 Power Bank",
    "ROG Zephyrus G14", "Zenbook 14 OLED",
    "ThinkPad X1 Carbon", "Legion Pro 7i",
    "Spectre x360 14", "Pavilion 15",
    "Fire Max 11", "Kindle Paperwhite", "Echo Show 10", "Video Doorbell Pro",
    "Studio Pro", "G Pro X Superlight", "Momentum 4", "Open"
}

# Regex patterns
MONTHS_REGEX = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_PATTERNS = [
    re.compile(rf"\b{MONTHS_REGEX}\s+\d{{1,2}}(?:,\s*\d{{4}})?\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b{MONTHS_REGEX}\s+\d{{4}}\b", re.IGNORECASE)
]

MONEY_PATTERN = re.compile(r"(?:\$|EUR\s*|GBP\s*|USD\s*)\s*\d+(?:,\d{3})*(?:\.\d+)?", re.IGNORECASE)


class RuleBasedNER:
    """
    Rule-based and gazetteer-driven Named Entity Recognizer.
    Identifies entities: ORG, PER, LOC, PROD, DATE, MONEY.
    Supports both raw text span extraction and token-sequence IOB tagging.
    """

    def __init__(self):
        # Sort gazetteers by length descending to match longest phrases first (e.g., "iPhone 15 Pro" before "iPhone")
        self.orgs_sorted = sorted(ORGANIZATIONS, key=len, reverse=True)
        self.persons_sorted = sorted(PERSONS, key=len, reverse=True)
        self.locs_sorted = sorted(LOCATIONS, key=len, reverse=True)
        self.prods_sorted = sorted(PRODUCTS, key=len, reverse=True)

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entity spans from raw text string.
        Returns a list of dicts: {'text': str, 'type': str, 'start': int, 'end': int}
        """
        entities = []
        occupied_spans = set()

        def is_overlapping(start: int, end: int) -> bool:
            return any(start < o_end and end > o_start for o_start, o_end in occupied_spans)

        def add_entity(entity_text: str, ent_type: str, start: int, end: int):
            if not is_overlapping(start, end):
                entities.append({
                    "text": entity_text,
                    "type": ent_type,
                    "start": start,
                    "end": end
                })
                occupied_spans.add((start, end))

        # 1. Match Money entities
        for match in MONEY_PATTERN.finditer(text):
            add_entity(match.group(0), "MONEY", match.start(), match.end())

        # 2. Match Date entities
        for pat in DATE_PATTERNS:
            for match in pat.finditer(text):
                add_entity(match.group(0), "DATE", match.start(), match.end())

        # 3. Match Products (longest first)
        for prod in self.prods_sorted:
            pattern = re.compile(rf"\b{re.escape(prod)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                add_entity(match.group(0), "PROD", match.start(), match.end())

        # 4. Match Persons (longest first)
        for per in self.persons_sorted:
            pattern = re.compile(rf"\b{re.escape(per)}\b")
            for match in pattern.finditer(text):
                add_entity(match.group(0), "PER", match.start(), match.end())

        # 5. Match Organizations
        for org in self.orgs_sorted:
            pattern = re.compile(rf"\b{re.escape(org)}\b")
            for match in pattern.finditer(text):
                add_entity(match.group(0), "ORG", match.start(), match.end())

        # 6. Match Locations
        for loc in self.locs_sorted:
            pattern = re.compile(rf"\b{re.escape(loc)}\b")
            for match in pattern.finditer(text):
                add_entity(match.group(0), "LOC", match.start(), match.end())

        # Sort entities by start index
        entities.sort(key=lambda x: x["start"])
        return entities

    def predict_iob_tags(self, tokens: List[str]) -> List[str]:
        """
        Maps token list to IOB tags using gazetteers and regex pattern heuristics.
        """
        n = len(tokens)
        tags = ["O"] * n
        i = 0

        while i < n:
            token = tokens[i]

            # Check Money (e.g. '$' followed by digits, or 'EUR' followed by digits)
            if token in ["$", "EUR", "USD", "GBP"] and i + 1 < n and tokens[i+1].isdigit():
                tags[i] = "B-MONEY"
                tags[i+1] = "I-MONEY"
                i += 2
                continue
            elif token.startswith("$") and len(token) > 1 and token[1:].isdigit():
                tags[i] = "B-MONEY"
                i += 1
                continue

            # Check Date expressions (e.g. "January 17 , 2024" or "September 12")
            if token in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]:
                tags[i] = "B-DATE"
                j = i + 1
                while j < n and (tokens[j].isdigit() or tokens[j] in [","]):
                    tags[j] = "I-DATE"
                    j += 1
                i = j
                continue

            # Check Multi-token Gazetteers: Products (up to 4 tokens), Persons (up to 3 tokens), Locations, Orgs
            matched = False
            for span_len in [4, 3, 2, 1]:
                if i + span_len <= n:
                    candidate = " ".join(tokens[i:i+span_len])

                    # Product match
                    if any(candidate.lower() == p.lower() for p in PRODUCTS):
                        tags[i] = "B-PROD"
                        for k in range(1, span_len):
                            tags[i+k] = "I-PROD"
                        i += span_len
                        matched = True
                        break

                    # Person match
                    if any(candidate.lower() == p.lower() for p in PERSONS):
                        tags[i] = "B-PER"
                        for k in range(1, span_len):
                            tags[i+k] = "I-PER"
                        i += span_len
                        matched = True
                        break

                    # Location match
                    if any(candidate.lower() == l.lower() for l in LOCATIONS):
                        tags[i] = "B-LOC"
                        for k in range(1, span_len):
                            tags[i+k] = "I-LOC"
                        i += span_len
                        matched = True
                        break

                    # Org match
                    if any(candidate.lower() == o.lower() for o in ORGANIZATIONS):
                        tags[i] = "B-ORG"
                        for k in range(1, span_len):
                            tags[i+k] = "I-ORG"
                        i += span_len
                        matched = True
                        break

            if not matched:
                i += 1

        return tags
