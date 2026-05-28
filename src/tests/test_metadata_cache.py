"""
test_metadata_cache.py
----------------------
Tests for the persistent SQLite metadata cache and its integration with
MetadataExtractor.

All tests use ``tmp_path`` for the SQLite file and mock the LLM — no running
Ollama instance is required.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import string
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.cache.metadata_cache import MetadataCache
from src.rag.metadata_extractor import (
    EXTRACTION_PROMPT,
    MetadataExtractor,
    EntityRecord,
    ExtractedMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_metadata() -> dict:
    """Return a random metadata dict to simulate non-deterministic LLM output."""
    rng = random.Random()
    return {
        "entity_id": "test-id",
        "entity_type": rng.choice(["servicio", "alojamiento", "entorno"]),
        "metadata": {
            "section": "",
            "subsection": "",
            "content_type": rng.choice(["descripcion", "precio", "normas"]),
            "accommodation_type": rng.choice(["villa", "apartamento", None]),
            "capacity": [f"{rng.randint(1, 12)}_personas"],
            "features": ["".join(rng.choices(string.ascii_lowercase, k=6))],
            "services": ["wifi", "piscina"],
            "location": ["playa_razo"],
            "environment": ["playa"],
            "activities": ["surf"],
            "target_audience": ["familias"],
        },
    }


def _build_extraction_version(prompt: str, model: str) -> str:
    return hashlib.sha256((prompt + model).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 1. Miss then hit
# ---------------------------------------------------------------------------

class TestMissThenHit:
    """``get`` on an absent key returns None; after ``set``, ``get`` returns
    an equal dict."""

    def test_miss_returns_none(self, tmp_path):
        db = tmp_path / "cache.sqlite"
        with MetadataCache(str(db)) as cache:
            assert cache.get("nonexistent_key") is None

    def test_set_then_get_returns_equal(self, tmp_path):
        db = tmp_path / "cache.sqlite"
        meta = {"entity_type": "servicio", "metadata": {"content_type": "precio"}}
        with MetadataCache(str(db)) as cache:
            cache.set("key_1", meta, "llama3.1")
            result = cache.get("key_1")
        assert result == meta


# ---------------------------------------------------------------------------
# 2. Round-trip fidelity
# ---------------------------------------------------------------------------

class TestRoundTripFidelity:
    """A metadata dict containing nested lists and unicode (Spanish accents)
    survives ``set`` → ``get`` unchanged."""

    def test_unicode_and_nested_lists(self, tmp_path):
        db = tmp_path / "cache.sqlite"
        meta = {
            "entity_type": "alojamiento",
            "metadata": {
                "content_type": "descripción",
                "features": ["piscina_climatizada", "terraza_exterior"],
                "location": ["playa_de_razo", "laguna_de_baldaío"],
                "environment": ["naturaleza", "costa"],
                "target_audience": ["familias", "parejas"],
                "accommodation_type": "cabaña",
                "activities": ["senderismo", "surf"],
                "services": ["wifi_de_alta_velocidad"],
                "capacity": ["6_personas"],
            },
        }
        with MetadataCache(str(db)) as cache:
            cache.set("unicode_key", meta, "llama3.1")
            result = cache.get("unicode_key")

        assert result == meta
        # Extra check: verify nested list identity
        assert result["metadata"]["features"] == meta["metadata"]["features"]
        assert result["metadata"]["location"] == meta["metadata"]["location"]


# ---------------------------------------------------------------------------
# 3. Memoisation — LLM called at most once
# ---------------------------------------------------------------------------

class TestMemoisation:
    """Extracting the same content twice (a) invokes ``_call_llm`` exactly once
    and (b) returns identical metadata both times."""

    @patch("src.rag.metadata_extractor.ChatOllama")
    def test_llm_called_once_for_same_content(self, mock_ollama_cls, tmp_path):
        # Build a fake LLM response (valid JSON the extractor can parse)
        fake_response = json.dumps({
            "entity_type": "servicio",
            "content_type": "descripcion",
            "accommodation_type": None,
            "capacity": [],
            "features": ["wifi"],
            "services": [],
            "location": [],
            "environment": ["playa"],
            "activities": [],
            "target_audience": ["familias"],
        })

        # Mock the LLM so the constructor doesn't need a real Ollama
        mock_llm_instance = MagicMock()
        mock_ollama_cls.return_value = mock_llm_instance

        db = tmp_path / "cache.sqlite"
        cache = MetadataCache(str(db))

        # Construct the extractor (ChatOllama is mocked)
        extractor = MetadataExtractor("llama3.1", cache=cache)

        # Replace the chain with a mock that returns our fake JSON
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = fake_response
        extractor._chain = mock_chain

        content = "La Gran Villa es una espaciosa villa de lujo."
        section = "Alojamientos"
        parent = "Espazo Nature"

        # First extraction — should invoke the LLM
        result_1 = extractor.extract(
            content, entity_id="e1", section=section, parent_section=parent,
        )
        assert mock_chain.invoke.call_count == 1

        # Second extraction — should hit the cache, no extra LLM call
        result_2 = extractor.extract(
            content, entity_id="e2", section=section, parent_section=parent,
        )
        assert mock_chain.invoke.call_count == 1  # still 1 — no new call

        # Both results share the same entity_type and metadata content_type
        assert result_1.entity_type == result_2.entity_type
        assert (
            result_1.metadata.content_type == result_2.metadata.content_type
        )

        cache.close()


# ---------------------------------------------------------------------------
# 4. Version invalidation
# ---------------------------------------------------------------------------

class TestVersionInvalidation:
    """Changing the EXTRACTION_PROMPT produces a different ``cache_key`` for the
    same content, forcing a fresh extraction."""

    def test_different_prompt_yields_different_key(self):
        content = "Texto de ejemplo para extracción."
        section = "Servicios"
        parent = "Espazo Nature"

        version_a = _build_extraction_version(EXTRACTION_PROMPT, "llama3.1")
        version_b = _build_extraction_version(
            EXTRACTION_PROMPT + " EXTRA INSTRUCTION", "llama3.1"
        )

        key_a = MetadataExtractor._build_cache_key(
            content, section, parent, version_a,
        )
        key_b = MetadataExtractor._build_cache_key(
            content, section, parent, version_b,
        )

        assert key_a != key_b, (
            "Cache keys must differ when extraction_version changes"
        )

    def test_different_model_yields_different_key(self):
        content = "Texto de ejemplo para extracción."
        section = "Servicios"
        parent = "Espazo Nature"

        version_a = _build_extraction_version(EXTRACTION_PROMPT, "llama3.1")
        version_b = _build_extraction_version(EXTRACTION_PROMPT, "llama3.2")

        key_a = MetadataExtractor._build_cache_key(
            content, section, parent, version_a,
        )
        key_b = MetadataExtractor._build_cache_key(
            content, section, parent, version_b,
        )

        assert key_a != key_b, (
            "Cache keys must differ when the model name changes"
        )

    def test_same_inputs_yield_same_key(self):
        content = "Texto idéntico."
        section = "Sec"
        parent = "Par"
        version = _build_extraction_version(EXTRACTION_PROMPT, "llama3.1")

        key_1 = MetadataExtractor._build_cache_key(content, section, parent, version)
        key_2 = MetadataExtractor._build_cache_key(content, section, parent, version)

        assert key_1 == key_2


# ---------------------------------------------------------------------------
# 5. Persistence across reopens
# ---------------------------------------------------------------------------

class TestPersistence:
    """Data written survives closing and reopening the cache."""

    def test_data_survives_reopen(self, tmp_path):
        db = tmp_path / "cache.sqlite"
        meta = {"entity_type": "actividad", "metadata": {"activities": ["kayak"]}}

        with MetadataCache(str(db)) as cache:
            cache.set("persist_key", meta, "llama3.1")

        # Reopen — data should still be there
        with MetadataCache(str(db)) as cache:
            result = cache.get("persist_key")

        assert result == meta
