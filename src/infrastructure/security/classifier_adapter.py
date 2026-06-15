# src/infrastructure/security/classifier_adapter.py
from src.domain.ports import IClassifier
from src.domain.models import Classification, Category
from src.infrastructure.security.classifier import classify_message

class LlmClassifier(IClassifier):
    def classify(self, text: str) -> Classification:
        raw = classify_message(text)                 # infra-native object
        return Classification(                        # → domain DTO
            category=Category(raw.category),
            sub_category=raw.sub_category,
            section=raw.section,
            language=raw.language,
            reasoning=raw.reasoning,
        )