# src/tests/test_classifier.py
from unittest.mock import patch, MagicMock
import pytest
from pydantic import ValidationError
from src.infrastructure.security.classifier import (
    classify_message,
    MessageClassification,
    get_sections,
    _vectorstore
)
from src.infrastructure.security.classifier_adapter import LlmClassifier
from src.domain.models import Classification, Category

def test_message_classification_model_validation_success():
    """Verify that MessageClassification can be instantiated with valid attributes."""
    valid_sections = list(get_sections(_vectorstore)) + [""]
    for sec in valid_sections:
        mc = MessageClassification(
            category="greeting",
            sub_category="none",
            section=sec,
            language="es",
            reasoning="Simple test greeting"
        )
        assert mc.category == "greeting"
        assert mc.sub_category == "none"
        assert mc.section == sec
        assert mc.language == "es"
        assert mc.reasoning == "Simple test greeting"

def test_message_classification_model_validation_failures():
    """Verify that MessageClassification raises ValidationError for invalid attributes."""
    # Test invalid category
    with pytest.raises(ValidationError) as excinfo:
        MessageClassification(
            category="invalid_category_here",
            sub_category="none",
            section="",
            language="es",
            reasoning="Reason"
        )
    assert "Input should be 'greeting'" in str(excinfo.value)

    # Test invalid sub_category
    with pytest.raises(ValidationError) as excinfo:
        MessageClassification(
            category="privacy_rights",
            sub_category="invalid_sub_category_here",
            section="",
            language="es",
            reasoning="Reason"
        )
    assert "Input should be 'forget'" in str(excinfo.value)

    # Test invalid language
    with pytest.raises(ValidationError) as excinfo:
        MessageClassification(
            category="greeting",
            sub_category="none",
            section="",
            language="invalid_language_here",
            reasoning="Reason"
        )
    assert "Input should be 'es'" in str(excinfo.value)

    # Test invalid section
    with pytest.raises(ValidationError) as excinfo:
        MessageClassification(
            category="greeting",
            sub_category="none",
            section="invalid_section_name_not_in_vectorstore",
            language="es",
            reasoning="Reason"
        )
    assert "literal_error" in str(excinfo.value)

@patch("src.infrastructure.security.classifier._structured_classifier")
def test_classify_message_success(mock_classifier):
    """Test successful classification paths for all categories."""
    valid_sections = list(get_sections(_vectorstore))
    section_to_use = valid_sections[0] if valid_sections else ""

    categories_to_test = [
        ("greeting", "none", "", "es"),
        ("faq", "none", section_to_use, "en"),
        ("privacy_rights", "forget", "", "fr"),
        ("privacy_rights", "access", "", "de"),
        ("privacy_rights", "retention", "", "es"),
        ("privacy_rights", "minors", "", "es"),
        ("privacy_rights", "third_party", "", "es"),
        ("privacy_rights", "general_policy", "", "es"),
        ("booking_payment", "none", section_to_use, "en"),
        ("pii_disclosure", "none", "", "es"),
        ("injection", "none", "", "es"),
        ("unsupported", "none", "", "es")
    ]

    for cat, subcat, sec, lang in categories_to_test:
        expected_response = MessageClassification(
            category=cat,
            sub_category=subcat,
            section=sec,
            language=lang,
            reasoning=f"Reason for {cat}"
        )
        mock_classifier.invoke.return_value = expected_response

        res = classify_message("Dummy text input")
        assert res.category == cat
        assert res.sub_category == subcat
        assert res.section == sec
        assert res.language == lang
        assert res.reasoning == f"Reason for {cat}"

@patch("src.infrastructure.security.classifier._structured_classifier")
def test_classify_message_invalid_type_fallback(mock_classifier):
    """Test fallback logic when the LLM returns an invalid object type."""
    mock_classifier.invoke.return_value = "Not a MessageClassification object"

    res = classify_message("Dummy text input")
    assert res.category == "faq"
    assert res.sub_category == "none"
    assert res.section == ""
    assert res.language == "es"
    assert "Invalid structured output type" in res.reasoning

@patch("src.infrastructure.security.classifier._structured_classifier")
def test_classify_message_exception_fallback(mock_classifier):
    """Test fallback logic when LLM invocation throws an exception (e.g. timeout, API error)."""
    mock_classifier.invoke.side_effect = Exception("LLM connection timed out")

    res = classify_message("Dummy text input")
    assert res.category == "faq"
    assert res.sub_category == "none"
    assert res.section == ""
    assert res.language == "es"
    assert "Exception encountered" in res.reasoning

@patch("src.infrastructure.security.classifier._structured_classifier")
def test_llm_classifier_adapter(mock_classifier):
    """Test LlmClassifier adapter successfully maps internal models to domain DTOs."""
    mock_classifier.invoke.return_value = MessageClassification(
        category="privacy_rights",
        sub_category="access",
        section="",
        language="fr",
        reasoning="Requesting personal data information"
    )

    adapter = LlmClassifier()
    classification = adapter.classify("Quelles données avez-vous?")

    assert isinstance(classification, Classification)
    assert classification.category == Category.PRIVACY_RIGHTS
    assert classification.sub_category == "access"
    assert classification.section == ""
    assert classification.language == "fr"
    assert classification.reasoning == "Requesting personal data information"

@patch("src.infrastructure.security.classifier._structured_classifier")
def test_llm_classifier_adapter_all_categories(mock_classifier):
    """Verify that all classification category literals map perfectly to Category Enum."""
    mapping_checks = {
        "greeting": Category.GREETING,
        "faq": Category.FAQ,
        "privacy_rights": Category.PRIVACY_RIGHTS,
        "booking_payment": Category.BOOKING_PAYMENT,
        "pii_disclosure": Category.PII_DISCLOSURE,
        "injection": Category.INJECTION,
        "unsupported": Category.UNSUPPORTED
    }

    adapter = LlmClassifier()
    for raw_cat, expected_enum in mapping_checks.items():
        mock_classifier.invoke.return_value = MessageClassification(
            category=raw_cat,
            sub_category="none",
            section="",
            language="es",
            reasoning="Testing enum mapping"
        )
        classification = adapter.classify("test")
        assert classification.category == expected_enum

@patch("src.infrastructure.security.classifier._structured_classifier")
def test_classify_message_includes_all_sections_in_system_prompt(mock_classifier):
    """Verify that all sections retrieved from the vector store are in the system prompt."""
    expected_response = MessageClassification(
        category="greeting",
        sub_category="none",
        section="",
        language="es",
        reasoning="Simple test greeting"
    )
    mock_classifier.invoke.return_value = expected_response

    classify_message("Hello")

    # Inspect the system message passed to the classifier LLM
    call_args = mock_classifier.invoke.call_args
    assert call_args is not None
    messages_passed = call_args[0][0]
    
    # The first message should be the system message containing the formatted sections
    system_msg = messages_passed[0]
    assert system_msg["role"] == "system"
    
    # Verify that get_sections names are indeed included in the prompt
    active_sections = get_sections(_vectorstore)
    for section in active_sections:
        assert section in system_msg["content"]
