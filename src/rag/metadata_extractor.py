"""
metadata_extractor.py
---------------------
Extracts structured metadata from raw Spanish-language text chunks
during the ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import difflib
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.infrastructure.cache.metadata_cache import MetadataCache

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL   = "llama3"
TEMPERATURE = 0.0        # deterministic — extraction is not creative


# ---------------------------------------------------------------------------
# Schema dataclasses — for type safety and easy serialisation
# ---------------------------------------------------------------------------

@dataclass
class ExtractedMetadata:
    section:            str             = ""
    subsection:         str             = ""
    content_type:       str             = "descripcion"
    accommodation_type: Optional[str]   = None
    capacity:           list[str]       = field(default_factory=list)
    features:           list[str]       = field(default_factory=list)
    services:           list[str]       = field(default_factory=list)
    location:           list[str]       = field(default_factory=list)
    environment:        list[str]       = field(default_factory=list)
    activities:         list[str]       = field(default_factory=list)
    target_audience:    list[str]       = field(default_factory=list)


@dataclass
class EntityRecord:
    entity_id:   str
    entity_type: str
    metadata:    ExtractedMetadata

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None values from metadata to keep Chroma metadata clean
        d["metadata"] = {k: v for k, v in d["metadata"].items() if v is not None}
        # Chroma requires list fields to be stored as comma-separated strings
        d["metadata"] = _flatten_for_chroma(d["metadata"])
        return d

    def to_langchain_metadata(self) -> dict:
        """
        Returns a flat dict suitable for LangChain Document.metadata.
        Lists are stored as comma-separated strings (Chroma requirement).
        entity_id and entity_type are hoisted to the top level.
        """
        flat = _flatten_for_chroma(asdict(self.metadata))
        flat["entity_id"]   = self.entity_id
        flat["entity_type"] = self.entity_type
        return {k: v for k, v in flat.items() if v}   # drop empty strings


def _flatten_for_chroma(d: dict) -> dict:
    """Convert list values → comma-separated strings for Chroma storage."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = ", ".join(v) if v else ""
        elif v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Eres un extractor de metadatos para una plataforma de turismo rural en español.
Tu tarea es analizar el texto de un chunk y devolver un JSON con los metadatos indicados.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con JSON válido. Sin explicaciones, sin markdown, sin comillas extras.
2. Todos los valores de tipo string van en minúsculas con guiones bajos en lugar de espacios.
3. Los campos de lista deben ser arrays JSON aunque tengan un solo elemento.
4. Si un campo no aplica o no se menciona, usa "" para strings y [] para listas.
5. accommodation_type solo aplica si el texto describe un alojamiento específico.
6. Normaliza los nombres de lugares: "Playa de Razo" → "playa_razo".

ESQUEMA JSON A DEVOLVER:
{{
  "entity_type":         "<servicio|alojamiento|entorno|actividad|general>",
  "content_type":        "<descripcion|precio|normas|servicios|ubicacion|faqs>",
  "accommodation_type":  "<villa|apartamento|casa|cabaña|glamping|null>",
  "capacity":            ["<N_personas>"],
  "features":            ["<característica_física>"],
  "services":            ["<servicio_incluido>"],
  "location":            ["<lugar_o_zona>"],
  "environment":         ["<tipo_entorno>"],
  "activities":          ["<actividad_disponible>"],
  "target_audience":     ["<publico_objetivo>"]
}}

VOCABULARIO DE REFERENCIA (úsalo para normalizar, no es exhaustivo):
- entity_type:        servicio, alojamiento, entorno, actividad, general
- content_type:       descripcion, precio, normas, servicios, ubicacion, faqs
- accommodation_type: villa, apartamento, casa, cabaña, glamping
- environment:        playa, naturaleza, rural, montaña, bosque, costa, ria
- target_audience:    familias, parejas, grupos, adultos, niños, mascotas, surf

TEXTO A ANALIZAR:
{text}

JSON:"""


# ---------------------------------------------------------------------------
# Reference Vocabulary for Validation
# ---------------------------------------------------------------------------

VOCABULARY = {
    "entity_type":        ["servicio", "alojamiento", "entorno", "actividad", "general"],
    "content_type":       ["descripcion", "precio", "normas", "servicios", "ubicacion", "faqs"],
    "accommodation_type": ["villa", "apartamento", "casa", "cabaña", "glamping"],
    "environment":        ["playa", "naturaleza", "rural", "montaña", "bosque", "costa", "ria"],
    "target_audience":    ["familias", "parejas", "grupos", "adultos", "niños", "mascotas", "surf"]
}


# ---------------------------------------------------------------------------
# MetadataExtractor
# ---------------------------------------------------------------------------

class MetadataExtractor:
    """
    Extracts structured metadata from raw Spanish text using ChatOllama.

    The LLM is instructed to return strict JSON with a fixed schema.
    All outputs are validated and sanitised before returning — malformed
    LLM responses fall back to safe defaults rather than crashing.

    Parameters
    ----------
    llm_model   : Ollama model name (default: llama3)
    temperature : LLM temperature — keep at 0.0 for deterministic extraction
    cache       : optional MetadataCache for persistent memoisation
    """

    def __init__(
        self,
        llm_model:   str   = LLM_MODEL,
        temperature: float = TEMPERATURE,
        cache: "MetadataCache | None" = None,
    ):
        self.llm = ChatOllama(model=llm_model, temperature=temperature)
        self._chain = (
            ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        self._cache = cache
        # Extraction version: hash of prompt + model so cache auto-invalidates
        # when either changes.
        version_input = EXTRACTION_PROMPT + llm_model
        self._extraction_version = hashlib.sha256(
            version_input.encode()
        ).hexdigest()[:12]
        logger.debug(
            "MetadataExtractor initialised (model=%s, extraction_version=%s, cache=%s)",
            llm_model, self._extraction_version, "enabled" if cache else "disabled",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cache_key(
        content: str,
        section: str,
        parent_section: str,
        extraction_version: str,
    ) -> str:
        """Compute a deterministic cache key from chunk content and structural metadata."""
        payload = "\x00".join([content, section, parent_section, extraction_version])
        return hashlib.sha256(payload.encode()).hexdigest()

    def extract(
        self,
        text:      str,
        entity_id: Optional[str] = None,
        section: str = "",
        parent_section: str = "",
    ) -> EntityRecord:
        """
        Extract metadata from a single text chunk.

        Parameters
        ----------
        text             : raw chunk text (Spanish)
        entity_id        : optional ID — auto-generated UUID if not provided
        section          : section heading (used for cache key)
        parent_section   : parent section heading (used for cache key)

        Returns
        -------
        EntityRecord with entity_id, entity_type, and ExtractedMetadata
        """
        eid = entity_id or str(uuid.uuid4())

        # --- Cache lookup (if a cache is available) ---
        if self._cache is not None:
            cache_key = self._build_cache_key(
                text, section, parent_section, self._extraction_version,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._parse_from_dict(cached, eid)

        # --- Cache miss or no cache: invoke the LLM ---
        raw = self._chain.invoke({"text": text})
        record = self._parse(raw, eid)

        # --- Write back to cache ---
        if self._cache is not None:
            from src.config import MODEL_NAME
            self._cache.set(cache_key, record.to_dict(), MODEL_NAME)

        return record

    def extract_batch(
        self,
        chunks: list[dict],
    ) -> list[EntityRecord]:
        """
        Extract metadata for multiple chunks.

        Parameters
        ----------
        chunks : list of {"text": str, "entity_id": str (optional)}

        Returns
        -------
        list[EntityRecord] in the same order as input
        """
        results = []
        for chunk in chunks:
            record = self.extract(
                text      = chunk["text"],
                entity_id = chunk.get("entity_id"),
            )
            results.append(record)
        return results

    def extract_to_langchain_metadata(
        self,
        text:      str,
        entity_id: Optional[str] = None,
        section: str = "",
        parent_section: str = "",
    ) -> dict:
        """
        Convenience wrapper — returns a flat dict ready to set as
        LangChain Document.metadata. Lists are comma-joined for Chroma.
        """
        return self.extract(text, entity_id, section, parent_section).to_langchain_metadata()

    # ------------------------------------------------------------------
    # Parsing + validation
    # ------------------------------------------------------------------

    def _parse(self, raw: str, entity_id: str) -> EntityRecord:
        """
        Parse raw LLM output into an EntityRecord.
        Strips markdown fences, fixes common JSON issues, and validates
        every field. Falls back to safe defaults on any parse error.
        """
        cleaned = self._clean_json(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {}   # full fallback — all fields → defaults

        entity_type = self._str(data.get("entity_type"), "general")
        entity_type = self._fuzzy_match(entity_type, VOCABULARY["entity_type"]) or "general"

        content_type = self._str(data.get("content_type"), "descripcion")
        content_type = self._fuzzy_match(content_type, VOCABULARY["content_type"]) or "descripcion"

        accommodation_type = self._nullable_str(data.get("accommodation_type"))
        if accommodation_type:
            accommodation_type = self._fuzzy_match(accommodation_type, VOCABULARY["accommodation_type"])

        environment = self._list(data.get("environment"))
        environment = self._fuzzy_match_list(environment, VOCABULARY["environment"])

        target_audience = self._list(data.get("target_audience"))
        target_audience = self._fuzzy_match_list(target_audience, VOCABULARY["target_audience"])

        meta = ExtractedMetadata(
            content_type       = content_type,
            accommodation_type = accommodation_type,
            capacity           = self._list(data.get("capacity")),
            features           = self._list(data.get("features")),
            services           = self._list(data.get("services")),
            location           = self._list(data.get("location")),
            environment        = environment,
            activities         = self._list(data.get("activities")),
            target_audience    = target_audience,
        )
        return EntityRecord(
            entity_id   = entity_id,
            entity_type = entity_type,
            metadata    = meta,
        )

    @staticmethod
    def _parse_from_dict(data: dict, entity_id: str) -> EntityRecord:
        """Reconstruct an EntityRecord from a previously cached dict.

        The cached dict is the output of ``EntityRecord.to_dict()`` and has
        already been validated; this avoids re-running fuzzy matching.
        """
        meta_data = data.get("metadata", {})
        # Lists may be stored as comma-separated strings from Chroma flattening
        def _to_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val:
                return [v.strip() for v in val.split(",") if v.strip()]
            return []

        meta = ExtractedMetadata(
            section            = meta_data.get("section", ""),
            subsection         = meta_data.get("subsection", ""),
            content_type       = meta_data.get("content_type", "descripcion"),
            accommodation_type = meta_data.get("accommodation_type") or None,
            capacity           = _to_list(meta_data.get("capacity", [])),
            features           = _to_list(meta_data.get("features", [])),
            services           = _to_list(meta_data.get("services", [])),
            location           = _to_list(meta_data.get("location", [])),
            environment        = _to_list(meta_data.get("environment", [])),
            activities         = _to_list(meta_data.get("activities", [])),
            target_audience    = _to_list(meta_data.get("target_audience", [])),
        )
        return EntityRecord(
            entity_id   = entity_id,
            entity_type = data.get("entity_type", "general"),
            metadata    = meta,
        )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences and isolate the first JSON object."""
        # Remove ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Extract first { ... } block in case the LLM adds extra text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        return match.group(0) if match else cleaned

    @staticmethod
    def _str(value, default: str = "") -> str:
        """Coerce to lowercase_underscore string."""
        if not value or value in ("null", "none", "n/a", ""):
            return default
        return str(value).lower().strip().replace(" ", "_")

    @staticmethod
    def _nullable_str(value) -> Optional[str]:
        """Return None if the value is null-like, otherwise normalise."""
        if not value or str(value).lower() in ("null", "none", "n/a", ""):
            return None
        return str(value).lower().strip().replace(" ", "_")

    @staticmethod
    def _list(value) -> list[str]:
        """
        Coerce to a list of normalised strings.
        Handles: JSON array, comma-separated string, single string, null.
        """
        if not value:
            return []
        if isinstance(value, list):
            return [
                str(v).lower().strip().replace(" ", "_")
                for v in value if v
            ]
        if isinstance(value, str):
            return [
                v.strip().lower().replace(" ", "_")
                for v in value.split(",") if v.strip()
            ]
        return []

    @staticmethod
    def _fuzzy_match(value: str | None, possibilities: list[str], cutoff: float = 0.6) -> str | None:
        """Fuzzy match a single string against a list of possibilities."""
        if not value:
            return None
        matches = difflib.get_close_matches(value, possibilities, n=1, cutoff=cutoff)
        return matches[0] if matches else value

    @staticmethod
    def _fuzzy_match_list(values: list[str], possibilities: list[str], cutoff: float = 0.6) -> list[str]:
        """Fuzzy match a list of strings against possibilities."""
        result = []
        for v in values:
            matches = difflib.get_close_matches(v, possibilities, n=1, cutoff=cutoff)
            result.append(matches[0] if matches else v)
        # Deduplicate after matching while preserving order
        seen = set()
        return [x for x in result if not (x in seen or seen.add(x))]



# ---------------------------------------------------------------------------
# Integration helper — drop-in for the ingestion pipeline
# ---------------------------------------------------------------------------

def enrich_document(
    doc,                          # LangChain Document
    extractor: MetadataExtractor,
    entity_id: Optional[str] = None,
    extras: Optional[list[str]] = None,
) -> None:
    """
    Enrich a LangChain Document in-place with extracted metadata.
    And redundance.
    Existing metadata keys are preserved; extracted fields are merged in.
    """
    extracted = extractor.extract_to_langchain_metadata(
        text      = doc.page_content,
        entity_id = entity_id or doc.metadata.get("entity_id"),
        section   = doc.metadata.get("section", ""),
        parent_section = doc.metadata.get("parent_section", ""),
    )
    hijos = "\n".join(extras)
    if len(extras) > 1:
        doc.page_content = f"""
        {doc.page_content}
        [{doc.metadata["section"]}]:
        {hijos}
        """
    doc.metadata.update(extracted)


# ---------------------------------------------------------------------------
# Label validation helper
# ---------------------------------------------------------------------------

def find_valid_labels(finding: str, chroma_snapshot: dict, logger=None) -> tuple[str | None, str | None, float]:
    """
    Ensure the extracted finding actually exists in the collection as a section or subsection.
    Uses difflib for fuzzy matching. Returns a tuple (canonical_label, field_type, match_score)
    where field_type is 'section', 'subsection', or 'both', and match_score is the similarity [0.0, 1.0].
    Returns (None, None, 0.0) if not found.
    """

    valid_labels: dict[str, set[str]] = {}
    for meta_dict in chroma_snapshot.get("metadatas", []):
        if meta_dict:
            for field in ["section", "subsection"]:
                if field in meta_dict:
                    val = meta_dict[field]
                    if val not in valid_labels:
                        valid_labels[val] = set()
                    valid_labels[val].add(field)
                
    label_map = {str(lbl).lower(): lbl for lbl in valid_labels}
    finding_lower = finding.lower()
    
    canonical = None
    score = 0.0
    
    if finding_lower in label_map:
        canonical = label_map[finding_lower]
        score = 1.0
    else:
        best_match = None
        best_score = 0.0
        for label in label_map.keys():
            s = difflib.SequenceMatcher(None, finding_lower, label).ratio()
            if s > best_score:
                best_score = s
                best_match = label
        
        if best_match:
            canonical = label_map[best_match]
            score = best_score
            
    if canonical:
        fields = valid_labels[canonical]
        field_type = "both" if len(fields) > 1 else next(iter(fields))
        return canonical, field_type, score
    
    if logger:
        logger.warning("Metadata finding '%s' not present in collection labels.", finding)
    return None, None, 0.0


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """
    La Gran Villa es una espaciosa villa de lujo con capacidad para 6 personas,
    situada a pocos metros de la Playa de Razo y con vistas directas a la Laguna de Baldaio.
    Dispone de piscina privada, terraza exterior, cocina totalmente equipada y WiFi de alta velocidad.
    Ideal para familias y parejas que buscan descanso en plena naturaleza.
    Cerca encontrarás rutas de senderismo y escuelas de surf.
    """

    extractor = MetadataExtractor()
    record    = extractor.extract(text=sample, entity_id="2312")

    logger.info("=== EntityRecord ===")
    logger.info(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))

    logger.info("\n=== LangChain metadata (flat, Chroma-ready) ===")
    logger.info(json.dumps(record.to_langchain_metadata(), ensure_ascii=False, indent=2))