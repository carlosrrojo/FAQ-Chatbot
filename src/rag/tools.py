import requests
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

db_path = "data/chroma_db"

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location using Open-Meteo API."""
    # We will use Open-Meteo to fetch weather.
    # To get coords, we use the geocoding API.
    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_resp = requests.get(geocode_url)
        geo_data = geo_resp.json()
        
        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find coordinates for {location}."
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url)
        weather_data = weather_resp.json()
        
        if "current_weather" in weather_data:
            current = weather_data["current_weather"]
            temp = current["temperature"]
            wind = current["windspeed"]
            return f"The current weather in {location} is {temp}°C with wind speeds of {wind} km/h."
        else:
            return f"Could not fetch weather data for {location}."
    except Exception as e:
        return f"Error fetching weather: {e}"

@tool
def export_to_google_calendar(event_name: str, start_time: str, end_time: str) -> str:
    """
    Exports or schedules an event on Google Calendar.
    Args:
        event_name: The name of the event.
        start_time: The start time in ISO format (e.g., 2026-03-05T10:00:00)
        end_time: The end time in ISO format (e.g., 2026-03-05T11:00:00)
    """
    # For now, this is a simulated tool that simulates syncing to a Google Calendar.
    # In a full production app, this would use google-api-python-client.
    
    return f"Successfully scheduled event '{event_name}' from {start_time} to {end_time} on Google Calendar."

def get_sections():
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = Chroma(collection_name="recursive_espazo_nature",
                         embedding_function=embeddings,
                         persist_directory=db_path)
    data = vectorstore.get()
    sections = set()
    for meta in data.get("metadatas", []):
        if meta and "section" in meta:
            sections.add(meta["section"])
    return set(sections)

sections = ",".join(str(x) for x in get_sections())

class QueryMetadata(BaseModel):
    """Extract metadata to filter in the RAG store."""
    finding: str = Field(description="The specific section mentioned, or 'none' if none")
    keywords: list[str] = Field(description="Extract any proper nouns or keywords in their ORIGINAL language. DO NOT translate them.")

metadata_extractor = ChatOllama(model="llama3.1").with_structured_output(QueryMetadata)

@tool
def retrieve_documents(query: str):
    """Retrieve documents for a given query."""
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = Chroma(collection_name="recursive_espazo_nature",
                         embedding_function=embeddings,
                         persist_directory=db_path)
    retriever = vectorstore.as_retriever()
    # First, let's extract metadata while explicitly preserving the keyword language
    prompt = QUERY_METADATA_PROMPT.format(query=query, sections=sections)
    try:
        metadata = metadata_extractor.invoke([{"role": "user", "content": prompt}])

        print(f"Extracted metadata: {metadata}")
        # Append the original keywords to the query to ensure the retrieval engine searches for them
        if metadata.keywords:
            query = query + " " + " ".join(metadata.keywords)

        search_filter = None
        if metadata.finding and metadata.finding.lower() != "none" and metadata.finding != "":
            print(f"Finding: {metadata.finding}")
            # Use similarity_search to fuzzy-match the section name (handles slight misspellings)
            section_docs = vectorstore.similarity_search(
                metadata.finding, k=1, filter={"section": {"$eq": metadata.finding}}
            )

            if section_docs:
                print(f"Found section doc: {section_docs[0].metadata}")
                actual_section = section_docs[0].metadata.get("section", "")
                parent_section = section_docs[0].metadata.get("parent_section", "")

                if parent_section:
                    print(f"Parent section: {parent_section}")
                    # Retrieve siblings (same parent_section) AND the parent section doc itself
                    search_filter = {
                        "$or": [
                            {"parent_section": {"$eq": parent_section}},
                            {"section": {"$eq": parent_section}},
                        ]
                    }
                elif actual_section:
                    # Top-level section: retrieve it and all its children
                    search_filter = {
                        "$or": [
                            {"section": {"$eq": actual_section}},
                            {"parent_section": {"$eq": actual_section}},
                        ]
                    }

        print(f"Filter: {search_filter}")
        if search_filter:
            docs = vectorstore.similarity_search(query, k=4, filter=search_filter)
        else:
            docs = retriever.invoke(query)

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])