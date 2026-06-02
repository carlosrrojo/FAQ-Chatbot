import pytest
from src.infrastructure.security.pii_redactor import PIISafetyGuard, _luhn_ok

def test_luhn_validator():
    assert _luhn_ok("499273987160000") is True
    assert _luhn_ok("499273987160001") is False

def test_pii_safety_guard_regex_redactions():
    guard = PIISafetyGuard()

    # DNI & NIE
    raw = "Mi DNI es 12345678Z."
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[DNI_REDACTED_1]" in sanitized
    assert mapping["[DNI_REDACTED_1]"] == "12345678Z"
    assert guard.screen_outbound(sanitized, mapping) == raw

    raw = "Mi NIE es X1234567Z."
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[NIE_REDACTED_1]" in sanitized
    assert mapping["[NIE_REDACTED_1]"] == "X1234567Z"
    assert guard.screen_outbound(sanitized, mapping) == raw

    # Email
    raw = "Contacto: carlos@example.com"
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[EMAIL_REDACTED_1]" in sanitized
    assert mapping["[EMAIL_REDACTED_1]"] == "carlos@example.com"
    assert guard.screen_outbound(sanitized, mapping) == raw

    # Phone
    raw = "Llama al +34 600 000 000"
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[PHONE_REDACTED_1]" in sanitized
    assert mapping["[PHONE_REDACTED_1]"] == "+34 600 000 000"
    assert guard.screen_outbound(sanitized, mapping) == raw

    raw = "Llama al 981123456"
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[PHONE_REDACTED_1]" in sanitized
    assert mapping["[PHONE_REDACTED_1]"] == "981123456"
    assert guard.screen_outbound(sanitized, mapping) == raw

    # IBAN
    raw = "Mi cuenta es ES12 3456 7890 1234 5678 9012"
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[IBAN_REDACTED_1]" in sanitized
    assert mapping["[IBAN_REDACTED_1]"] == "ES12 3456 7890 1234 5678 9012"
    assert guard.screen_outbound(sanitized, mapping) == raw

    # SSN
    raw = "NUSS: 12-34567890-12"
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[SSN_REDACTED_1]" in sanitized
    assert mapping["[SSN_REDACTED_1]"] == "12-34567890-12"
    assert guard.screen_outbound(sanitized, mapping) == raw

    # Postal code / Address
    raw = "Vivo en el código postal 28001"
    sanitized, mapping = guard.sanitize_inbound(raw)
    assert "[ADDRESS_REDACTED_1]" in sanitized
    assert mapping["[ADDRESS_REDACTED_1]"] == "28001"
    assert guard.screen_outbound(sanitized, mapping) == raw

def test_pii_safety_guard_ner_redactions():
    guard = PIISafetyGuard()
    # "Juan" is a Spanish name (PER)
    # "Madrid" is a Spanish city (LOC)
    text = "Hola Juan Perez, nos vemos en Madrid."
    sanitized, mapping = guard.sanitize_inbound(text)
    
    assert "[PER_REDACTED_1]" in sanitized
    assert "[LOC_REDACTED_1]" in sanitized
    # Placeholder should not be nested
    assert "[[PER_REDACTED]_REDACTED]" not in sanitized
    assert guard.screen_outbound(sanitized, mapping) == text

def test_pii_safety_guard_egress_reconstruction():
    guard = PIISafetyGuard()
    raw = "Mi nombre es Juan, mi email es juan@gmail.com, y mi telefono es 600000000."
    sanitized, mapping = guard.sanitize_inbound(raw)
    
    # Simulate LLM repeating the redacted placeholders in its output
    llm_output = f"Hola [PER_REDACTED_1], te contactaremos a [EMAIL_REDACTED_1] o al [PHONE_REDACTED_1]."
    reconstructed = guard.screen_outbound(llm_output, mapping)
    
    assert reconstructed == "Hola Juan, te contactaremos a juan@gmail.com o al 600000000."
