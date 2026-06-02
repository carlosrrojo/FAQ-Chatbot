# src/domain/orchestrator.py - FR-AGT-01..04
from src.domain.models import Category
from src.domain.ports import IResponsePolicy
from src.domain.ports import IClassifier
import logging
from langchain_core.messages import HumanMessage
from src.domain.models import ChatRequest, ChatResponse
from src.domain.ports import IMemoryStore, ISafetyGuard
from src.telemetry import timed

logger = logging.getLogger(__name__)

# Out-of-scope fallback messages, keyed by detected language
_OOS_RESPONSES = {
    "es": (
        "Lo siento, esa pregunta está fuera del ámbito de información que puedo "
        "proporcionar sobre Espazo Nature. Para consultas más específicas, puedes "
        "contactarnos directamente. ¿Puedo ayudarte con algo relacionado con nuestros "
        "alojamientos, servicios o entorno?"
    ),
    "en": (
        "I'm sorry, that question falls outside the scope of information I can provide "
        "about Espazo Nature. For more specific enquiries, please contact us directly. "
        "Can I help you with something related to our accommodation, services, or surroundings?"
    ),
    "fr": (
        "Désolé, cette question sort du cadre des informations que je peux fournir sur Espazo Nature. "
        "Pour des demandes plus spécifiques, veuillez nous contacter directement. "
        "Puis-je vous aider pour quelque chose concernant nos hébergements, nos services ou notre environnement ?"
    ),
    "de": (
        "Es tut mir leid, diese Frage liegt außerhalb des Rahmens der Informationen, die ich über Espazo Nature bereitstellen kann. "
        "Für spezifischere Anfragen wenden Sie sich bitte direkt an uns. "
        "Kann ich Ihnen bei Fragen zu unseren Unterkünften, Dienstleistungen oder der Umgebung helfen?"
    ),
}


class RAGOrchestrator:
    """
    Domain facade over the LangGraph agentic graph.
    Responsibilities:
    - Accept ChatRequest from transport layer
    - Invoke the compiled LangGraph agent
    - Apply history truncation policy (FR-MEM-03)
    - Apply out-of-scope detection (FR-AGT-04)
    - Return ChatResponse
    """

    def __init__(
        self,
        memory_store: IMemoryStore,
        classifier: IClassifier,
        response_policy: IResponsePolicy,
        system_prompt_template: str | None = None,
        safety_guard: ISafetyGuard | None = None,
        agent = None,
    ) -> None:
        self._memory = memory_store
        self._classifier = classifier
        self._response_policy = response_policy
        self._system_prompt_template = system_prompt_template
        self._safety_guard = safety_guard
        self._agent = agent

    def _clean_query_text(self, text: str) -> str:
        """
        Cleans user queries of structural placeholders and sensitive terms
        before passing to the RAG agent retrieval, preventing out-of-scope
        triggers and model reflection.

        Strategy: split on sentence-level boundaries (`.` `;` `?` `!`) only,
        preserving comma-separated clauses within a sentence.  Within each
        sentence, remove placeholder tokens and sensitive keyword phrases
        **inline** rather than dropping the entire sentence — this keeps
        legitimate FAQ content that co-occurs with PII (e.g. dietary needs
        mentioned alongside a social security number).

        Only drop a sentence entirely if, after inline scrubbing, its
        remaining text is purely whitespace / punctuation.
        """
        import re

        placeholder_re = re.compile(r'\[[A-Z0-9_]+_REDACTED_\d+\]')

        # Sensitive phrases to strip inline (order: longer first)
        _sensitive_phrases = [
            "número de la seguridad social", "numero de la seguridad social",
            "número de afiliación", "numero de afiliacion",
            "seguridad social", "social security",
            "tarjeta de crédito", "tarjeta de credito",
            "credit card", "credit-card",
            "por si acaso",  # filler phrase often wrapping PII offers
        ]

        # Split on sentence-level separators, keeping the separator
        parts = re.split(r'([.;?!]+)', text)

        cleaned_sentences: list[str] = []
        i = 0
        while i < len(parts):
            part = parts[i]
            # If this element is just a separator, attach to previous
            if re.match(r'^[.;?!]+$', part):
                if cleaned_sentences:
                    cleaned_sentences[-1] += part
                i += 1
                continue

            # --- Inline scrubbing of the sentence fragment ---
            sentence = part

            # 1. Remove redaction placeholders
            sentence = placeholder_re.sub('', sentence)

            # 2. Remove sensitive keyword phrases (case-insensitive)
            for phrase in _sensitive_phrases:
                sentence = re.sub(
                    re.escape(phrase), '', sentence, flags=re.IGNORECASE
                )

            # 3. Collapse whitespace and strip leading/trailing junk
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            sentence = sentence.strip(" ,;:–—-")

            # Keep the sentence only if substantive text remains
            if sentence and not re.match(r'^[\s,;:.?!–—-]*$', sentence):
                cleaned_sentences.append(sentence)

            i += 1

        cleaned_text = ' '.join(cleaned_sentences).strip()

        # Final cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'([.;,?!])\s*([.;,?!])+', r'\1', cleaned_text)
        cleaned_text = cleaned_text.strip(" .;,?!")

        # If everything was stripped, fall back to the original minus
        # placeholders and keywords
        if not cleaned_text:
            fallback = placeholder_re.sub('', text)
            for phrase in _sensitive_phrases:
                fallback = re.sub(
                    re.escape(phrase), '', fallback, flags=re.IGNORECASE
                )
            cleaned_text = re.sub(r'\s+', ' ', fallback).strip(" .;,?!")

        return cleaned_text

    def _run_agent(self, text: str, classification, sender_id: str) -> str:
        config = {
            "configurable": {
                "thread_id": sender_id,
                "section": classification.section
            }
        }
        cleaned_text = self._clean_query_text(text)
        logger.info("Original sanitized query: '%s' -> Cleaned for RAG retrieval: '%s'", text, cleaned_text)
        input_state = {"messages": [HumanMessage(content=cleaned_text)]}
        result = self._agent.invoke(input_state, config=config)
        return result["messages"][-1].content

    def _oos_response(self, request: ChatRequest, lang: str) -> ChatResponse:
        reply_text = _OOS_RESPONSES.get(lang, _OOS_RESPONSES["es"])
        return ChatResponse(
            text=reply_text,
            sender_id=request.sender_id,
            platform=request.platform,
        )

    @timed("orchestrator.generate_reply")
    def generate_reply(self, request: ChatRequest) -> ChatResponse:
        mapping = {}
        text = request.text
        if self._safety_guard is not None:
            try:
                text, mapping = self._safety_guard.sanitize_inbound(text)
            except Exception as e:
                logger.error("Safety guard inbound sanitisation failed: %s", e)
                #return self._oos_response(request, lang="es")

        # Run explicit routing classification
        classification = self._classifier.classify(text)
        logger.info("Classification for '%s': %s", request.sender_id, classification)

        if classification.category == Category.FAQ:
            reply_text = self._run_agent(text, classification, request.sender_id)
        else:
            reply_text = self._response_policy.get_response(classification)

        if self._safety_guard is not None:
            try:
                reply_text = self._safety_guard.screen_outbound(reply_text, mapping)
            except Exception as e:
                logger.error("Safety guard outbound screening failed: %s", e)
                return self._oos_response(request, lang=classification.language)
        
        #self._safe_touch(request.sender_id)

        if self._memory is not None:
            try:
                self._memory.touch_session(request.sender_id)
            except Exception as e:
                logger.error("Failed to touch session for %s: %s", request.sender_id, e)

        return ChatResponse(
            text=reply_text,
            sender_id=request.sender_id,
            platform=request.platform,
        )