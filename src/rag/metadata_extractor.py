"""
metadata_extractor.py
---------------------
Extracts structured metadata from raw Spanish-language text chunks
during the ingestion pipeline.

Schema
------
{
    "entity_id":   str,           # auto-generated UUID or passed in
    "entity_type": str,           # servicio | alojamiento | entorno | actividad | general
    "metadata": {
        "section":             str,        # top-level section (e.g. "alojamientos")
        "subsection":          str,        # specific entity name (e.g. "gran_villa")
        "content_type":        str,        # descripcion | precio | normas | servicios | ubicacion
        "accommodation_type":  str | None, # villa | apartamento | casa | cabaña | None
        "capacity":            list[str],  # ["4_personas", "6_personas"]
        "features":            list[str],  # ["piscina_privada", "terraza", ...]
        "services":            list[str],  # ["wifi", "cocina_equipada", ...]
        "location":            list[str],  # ["playa_razo", "laguna_baldaio", ...]
        "environment":         list[str],  # ["naturaleza", "playa", "rural", ...]
        "activities":          list[str],  # ["surf", "senderismo", ...]
        "target_audience":     list[str],  # ["familias", "parejas", "grupos", ...]
    }
}

Usage
-----
    from metadata_extractor import MetadataExtractor

    extractor = MetadataExtractor()
    result    = extractor.extract(text="...", entity_id="2312")
    print(result)

    # Batch (ingestion pipeline)
    records = extractor.extract_batch(chunks)   # list of {"text": str, "entity_id": str}
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

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
    """

    def __init__(
        self,
        llm_model:   str   = LLM_MODEL,
        temperature: float = TEMPERATURE,
    ):
        self.llm = ChatOllama(model=llm_model, temperature=temperature)
        self._chain = (
            ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text:      str,
        entity_id: Optional[str] = None,
    ) -> EntityRecord:
        """
        Extract metadata from a single text chunk.

        Parameters
        ----------
        text      : raw chunk text (Spanish)
        entity_id : optional ID — auto-generated UUID if not provided

        Returns
        -------
        EntityRecord with entity_id, entity_type, and ExtractedMetadata
        """
        eid = entity_id or str(uuid.uuid4())
        raw = self._chain.invoke({"text": text})
        return self._parse(raw, eid)

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
    ) -> dict:
        """
        Convenience wrapper — returns a flat dict ready to set as
        LangChain Document.metadata. Lists are comma-joined for Chroma.
        """
        return self.extract(text, entity_id).to_langchain_metadata()

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
        meta = ExtractedMetadata(
            content_type       = self._str(data.get("content_type"), "descripcion"),
            accommodation_type = self._nullable_str(data.get("accommodation_type")),
            capacity           = self._list(data.get("capacity")),
            features           = self._list(data.get("features")),
            services           = self._list(data.get("services")),
            location           = self._list(data.get("location")),
            environment        = self._list(data.get("environment")),
            activities         = self._list(data.get("activities")),
            target_audience    = self._list(data.get("target_audience")),
        )
        return EntityRecord(
            entity_id   = entity_id,
            entity_type = entity_type,
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
    )
    siblings = "\n".join(extras)
    if len(extras) > 1:
        doc.page_content = f"""
        {doc.page_content}
        [{doc.metadata["section"]}]:
        {siblings}
        """
    doc.metadata.update(extracted)


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

    print("=== EntityRecord ===")
    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))

    print("\n=== LangChain metadata (flat, Chroma-ready) ===")
    print(json.dumps(record.to_langchain_metadata(), ensure_ascii=False, indent=2))