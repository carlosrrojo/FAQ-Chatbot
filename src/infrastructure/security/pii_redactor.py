# src/infrastructure/safety/pii_redactor.py
import re
import logging
import spacy
from src.domain.ports import ISafetyGuard

logger = logging.getLogger(__name__)

# Compile patterns once. Ordering: more specific first.
_PATTERNS = [
    # Credit Card: 13 to 19 digits, possibly separated by spaces or hyphens. Validated by Luhn below.
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # Spanish DNI: 8 digits followed by a letter (excluding I, O, U)
    ("DNI", re.compile(r"\b\d{8}[A-HJ-NP-TV-Z]\b", re.I)),
    # Spanish NIE: X, Y, Z followed by 7 digits and a letter
    ("NIE", re.compile(r"\b[XYZ]\d{7}[A-HJ-NP-TV-Z]\b", re.I)),
    # Spanish Social Security Number (NUSS): PP-NNNNNNNNN-CC or PP/NNNNNNNNN/CC or PP NNNNNNNNN CC or PPNNNNNNNNNCC
    ("SSN", re.compile(r"\b\d{2}[ /.-]?\d{7,8}[ /.-]?\d{2}\b")),
    # Email: basic robust pattern
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # IBAN: General IBAN matching, supports ES + 22 digits (spaced/hyphenated/continuous)
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?\d){12,30}\b", re.I)),
    # Spanish Postal Code: 5 digits (01000 - 52999)
    ("ADDRESS", re.compile(r"\b(?:0[1-9]|[1-4]\d|5[0-2])\d{3}\b")),
    # Phone number: Spanish and generic phone patterns
    ("PHONE", re.compile(r"(?:\+?\d{1,3}[ .-]?)?(?:\d[ .-]?){9}\b")),
]

# Load spaCy model for NER
try:
    nlp = spacy.load("es_core_news_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
except Exception as e:
    logger.warning("Failed to load es_core_news_sm with disables: %s. Trying standard load.", e)
    try:
        nlp = spacy.load("es_core_news_sm")
    except Exception as ex:
        logger.error("Failed to load spaCy es_core_news_sm model: %s. NER will be disabled.", ex)
        nlp = None

def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    chk = digits[-1]
    body = digits[-2::-1]
    total = sum(
        d * 2 - 9 if (i % 2 == 0 and d * 2 > 9) else (d * 2 if i % 2 == 0 else d)
        for i, d in enumerate(body)
    )
    return (total + chk) % 10 == 0

def redact(text: str) -> tuple[str, dict[str, str]]:
    """
    Scans the text, redacting PII using both deterministic regex patterns and spaCy NER.
    Returns the redacted text and a mapping of placeholders to original values.
    """
    if not text:
        return text, {}

    mapping: dict[str, str] = {}

    # 1. Regex pass
    for label, pat in _PATTERNS:
        label_counter = 1
        def _sub(m):
            nonlocal label_counter
            tok = m.group(0)
            if label == "CARD" and not _luhn_ok(tok):
                return tok
            placeholder = f"[{label}_REDACTED_{label_counter}]"
            label_counter += 1
            mapping[placeholder] = tok
            return placeholder
        text = pat.sub(_sub, text)

    # 2. NER pass
    if nlp is not None:
        doc = nlp(text)
        spans_to_redact = []
        per_counter = 1
        loc_counter = 1
        for ent in doc.ents:
            if ent.label_ in ("PER", "LOC"):
                if "_REDACTED" in ent.text or "[" in ent.text or "]" in ent.text:
                    continue
                label = ent.label_
                if label == "PER":
                    placeholder = f"[{label}_REDACTED_{per_counter}]"
                    per_counter += 1
                else:
                    placeholder = f"[{label}_REDACTED_{loc_counter}]"
                    loc_counter += 1
                spans_to_redact.append((ent.start_char, ent.end_char, placeholder, ent.text))

        # Sort backwards to replace without index shifts
        spans_to_redact.sort(key=lambda x: x[0], reverse=True)
        for start, end, placeholder, original_text in spans_to_redact:
            text = text[:start] + placeholder + text[end:]
            mapping[placeholder] = original_text

    return text, mapping


class PIISafetyGuard(ISafetyGuard):
    """
    Concrete safety guard implementing ISafetyGuard.
    Uses regexes and spaCy NER (es_core_news_sm) to sanitize inbound and outbound text offline.
    """
    def sanitize_inbound(self, text: str) -> tuple[str, dict[str, str]]:
        return redact(text)

    def screen_outbound(self, text: str, mapping: dict[str, str]) -> str:
        if not text or not mapping:
            return text

        # 1. Replace exact indexed placeholders
        for placeholder, original in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(placeholder, original)

        # 2. Fallback for generic placeholders
        for placeholder, original in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            if "_" in placeholder:
                parts = placeholder.rstrip("]").split("_")
                if len(parts) >= 2:
                    generic_placeholder = "_".join(parts[:-1]) + "]"
                    text = text.replace(generic_placeholder, original)

        return text