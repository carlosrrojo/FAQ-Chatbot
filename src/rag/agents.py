"""
agents.py
---------
Multi-agent system with a Supervisor LLM routing between:
  - TriageAgent       : classifies intent, enriches query
  - RAGAgent          : customer service Q&A (wraps RAGCore)
  - WeatherAgent      : live weather via Open-Meteo (no API key needed)

Communication pattern: Supervisor LLM → decides which agent handles
each turn → result flows back through Supervisor → user.

Usage:
    from src.rag.agents import SupervisorAgent

    supervisor = SupervisorAgent(session_id="user_42")
    result = supervisor.run("Will it rain in Madrid tomorrow?")
    print(result["answer"])
    print(result["agent_used"])
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag.rag_core import RAGCore, MetadataFilter


# ---------------------------------------------------------------------------
# Shared LLM (all agents can share one instance or have their own)
# ---------------------------------------------------------------------------

def make_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model="llama3.1", temperature=temperature)


# ---------------------------------------------------------------------------
# AgentResult — standard envelope returned by every agent
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer:     str
    agent_used: str
    metadata:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer":     self.answer,
            "agent_used": self.agent_used,
            "metadata":   self.metadata,
        }


# ---------------------------------------------------------------------------
# TriageAgent
# ---------------------------------------------------------------------------

TRIAGE_PROMPT = """You are a triage assistant. Given a user message, return a JSON object with:
  - "intent": one of ["customer_service", "weather", "unclear"]
  - "location": city name if the message is about weather, else null
  - "enriched_query": a cleaner rewrite of the user's question (fix typos, expand abbreviations)

Respond ONLY with valid JSON. No explanation, no markdown fences.

User message: {query}"""


class TriageAgent:
    """
    Classifies intent and enriches the query.
    Returns: intent, optional location, cleaned query.
    """

    def __init__(self):
        self.llm = make_llm(temperature=0.0)

    def run(self, query: str) -> dict:
        """
        Returns dict with keys: intent, location, enriched_query
        Falls back gracefully if LLM returns malformed JSON.
        """
        prompt  = ChatPromptTemplate.from_template(TRIAGE_PROMPT)
        chain   = prompt | self.llm | StrOutputParser()
        raw     = chain.invoke({"query": query})

        # Strip accidental markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: treat as customer service if we can't parse
            result = {
                "intent":         "customer_service",
                "location":       None,
                "enriched_query": query,
            }

        result.setdefault("intent",         "customer_service")
        result.setdefault("location",       None)
        result.setdefault("enriched_query", query)
        return result


# ---------------------------------------------------------------------------
# RAGAgent
# ---------------------------------------------------------------------------

class RAGAgent:
    """
    Wraps RAGCore. One instance per session to preserve memory.
    Accepts an optional MetadataFilter passed down from the Supervisor.
    """

    def __init__(self, session_id: str = "default"):
        self.core = RAGCore(session_id=session_id)

    def run(
        self,
        query:  str,
        filter: Optional[MetadataFilter] = None,
    ) -> AgentResult:
        result = self.core.run(query, filter=filter)
        return AgentResult(
            answer     = result["answer"],
            agent_used = "RAGAgent",
            metadata   = {
                "sources":          result["sources"],
                "chunks_retrieved": len(result["chunks"]),
                "session_id":       self.core.session_id,
            },
        )

    def get_memory_summary(self) -> str:
        return self.core.get_history_summary()


# ---------------------------------------------------------------------------
# WeatherAgent — uses Open-Meteo (free, no API key)
# ---------------------------------------------------------------------------

WEATHER_ANSWER_PROMPT = """You are a helpful weather assistant.
Given the following weather data, answer the user's question in a friendly, concise way.
Mention temperature, conditions, and any relevant details.

Weather data (JSON):
{weather_json}

User question: {query}"""

GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherAgent:
    """
    Fetches weather from Open-Meteo (no API key required) and
    uses the LLM to compose a natural-language answer.
    """

    def __init__(self):
        self.llm = make_llm(temperature=0.3)

    def run(self, query: str, location: Optional[str] = None) -> AgentResult:
        if not location:
            return AgentResult(
                answer     = "I need a location to look up the weather. Could you specify a city?",
                agent_used = "WeatherAgent",
                metadata   = {"error": "no_location"},
            )

        try:
            weather_data = self._fetch_weather(location)
        except Exception as e:
            return AgentResult(
                answer     = f"Sorry, I couldn't fetch weather data for '{location}'. Please try again.",
                agent_used = "WeatherAgent",
                metadata   = {"error": str(e)},
            )

        prompt = ChatPromptTemplate.from_template(WEATHER_ANSWER_PROMPT)
        chain  = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "weather_json": json.dumps(weather_data, indent=2),
            "query":        query,
        })

        return AgentResult(
            answer     = answer,
            agent_used = "WeatherAgent",
            metadata   = {
                "location":    location,
                "weather_raw": weather_data,
            },
        )

    def _fetch_weather(self, location: str) -> dict:
        # Step 1: Geocode
        geo_resp = requests.get(
            GEOCODE_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        )
        geo_resp.raise_for_status()
        geo = geo_resp.json()

        if not geo.get("results"):
            raise ValueError(f"Location '{location}' not found.")

        place = geo["results"][0]
        lat, lon = place["latitude"], place["longitude"]
        name     = place.get("name", location)

        # Step 2: Forecast
        fc_resp = requests.get(
            FORECAST_URL,
            params={
                "latitude":      lat,
                "longitude":     lon,
                "current":       "temperature_2m,weathercode,windspeed_10m,precipitation",
                "daily":         "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                "timezone":      "auto",
                "forecast_days": 3,
            },
            timeout=5,
        )
        fc_resp.raise_for_status()
        fc = fc_resp.json()

        return {
            "location": name,
            "current":  fc.get("current", {}),
            "daily":    fc.get("daily", {}),
            "units":    fc.get("current_units", {}),
        }


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are the supervisor of a multi-agent customer service system.
You have already received a triage decision. Now decide the best course of action.

Triage result:
- Intent:         {intent}
- Location:       {location}
- Enriched query: {enriched_query}
- Original query: {original_query}

Based on intent:
- "customer_service" → route to RAGAgent
- "weather"          → route to WeatherAgent
- "unclear"          → ask user for clarification

Respond with ONLY a JSON object:
{{
  "route":    "rag" | "weather" | "clarify",
  "reason":   "one sentence explanation",
  "clarification_question": "question to ask user if route is clarify, else null"
}}"""


class SupervisorAgent:
    """
    Orchestrates the full multi-agent pipeline:
      1. Triage classifies intent
      2. Supervisor LLM decides routing
      3. Appropriate agent handles the query
      4. Result returned with full metadata

    One SupervisorAgent per user session (preserves RAGAgent memory).
    """

    def __init__(self, session_id: str = "default"):
        self.session_id  = session_id
        self.llm         = make_llm(temperature=0.0)
        self.triage      = TriageAgent()
        self.rag_agent   = RAGAgent(session_id=session_id)
        self.weather     = WeatherAgent()

    def run(self, query: str) -> dict:
        """
        Full pipeline: triage → supervisor routing → agent → result.

        Returns
        -------
        {
            "answer":      str,
            "agent_used":  str,
            "route":       str,
            "triage":      dict,
            "metadata":    dict,
        }
        """
        # Step 1: Triage
        triage = self.triage.run(query)

        # Step 2: Supervisor routing decision
        route_decision = self._route(triage, query)
        route = route_decision.get("route", "rag")

        # Step 3: Dispatch
        if route == "weather":
            result = self.weather.run(
                query    = triage["enriched_query"],
                location = triage.get("location"),
            )
        elif route == "clarify":
            result = AgentResult(
                answer     = route_decision.get("clarification_question", "Could you clarify your question?"),
                agent_used = "Supervisor",
                metadata   = {"route_reason": route_decision.get("reason")},
            )
        else:
            # Default: RAG
            result = self.rag_agent.run(query=triage["enriched_query"])

        return {
            "answer":     result.answer,
            "agent_used": result.agent_used,
            "route":      route,
            "triage":     triage,
            "metadata":   result.metadata,
        }

    def _route(self, triage: dict, original_query: str) -> dict:
        prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
        chain  = prompt | self.llm | StrOutputParser()
        raw    = chain.invoke({
            "intent":         triage["intent"],
            "location":       triage.get("location") or "N/A",
            "enriched_query": triage["enriched_query"],
            "original_query": original_query,
        })

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"route": "rag", "reason": "fallback", "clarification_question": None}
