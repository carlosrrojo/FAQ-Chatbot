import json
import fitz
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from config import MODEL_NAME
from langchain_core.prompts import ChatPromptTemplate
from keybert import KeyBERT
import re
import ast

def _clean_text(text: str) -> str:
    """Normalize whitespace in a heading string."""
    return " ".join(text.split())

def _parse_to_list(metadata: str) -> list[str]:
    candidates = re.findall(r'\[.*?\]', metadata, flags=re.DOTALL)
    valid_lists = []
    for c in candidates:
        try:
            parsed = ast.literal_eval(c)
            if isinstance(parsed, list):
                valid_lists.append(parsed)
        except:
            pass
    
    return valid_lists[0] if valid_lists else []

def add_metadata(chunks: list[Document]) -> list[Document]:
    llm = ChatOllama(model=MODEL_NAME)
    prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in metadata extraction.
            Extract important keywords from the following content.
            {content}
            Dont include empty or null fields.
            Return the keywords in a list format separated by commas and between [].
            Each keyword should be sourrounded by ""
            """
        )
    chain = prompt | llm
    docs_with_metadata = []
    for doc in chunks:
        metadata = chain.invoke({"content": doc.page_content})
        parsed_metadata = _parse_to_list(metadata.content)
        doc.page_content = f"""
        [KEYWORDS]
        {" , ".join(parsed_metadata)}
        [CONTENT]
        {doc.page_content}
        """
        doc.metadata.update({"keywords": json.dumps(parsed_metadata)})
        docs_with_metadata.append(doc)

def add_metadata_keyBERT(
    chunks: list[Document],
    top_n: int = 10,
    keyphrase_ngram_range: tuple[int, int] = (1, 2),
    stop_words: str = "spanish",
) -> list[Document]:
    """Extract keywords from each chunk using KeyBERT (no LLM, no network calls).

    Args:
        chunks: List of Document objects to enrich with keyword metadata.
        top_n: Maximum number of keywords to extract per document.
        keyphrase_ngram_range: Min/max n-gram size for keyphrases.
        stop_words: Language for stop-word filtering ('english', 'spanish', None).

    Returns:
        The same list of Documents with updated page_content and metadata.
    """
    kw_model = KeyBERT()
    docs_with_metadata = []

    for doc in chunks:
        raw_keywords = kw_model.extract_keywords(
            doc.page_content,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            top_n=top_n,
        )
        # extract_keywords returns [(keyword, score), ...] — keep only the strings
        keywords = [kw for kw, _score in raw_keywords]

        doc.page_content = f"""
        [KEYWORDS]
        {" , ".join(keywords)}
        [CONTENT]
        {doc.page_content}
        """
        doc.metadata.update({"keywords": json.dumps(keywords)})
        docs_with_metadata.append(doc)

    return docs_with_metadata

def docs_into_sections(document_path: str) -> list[Document]:
    """Open a PDF by path with fitz and split it into section-level Documents.

    Heading detection uses font size, bold flag, and ALL-CAPS heuristics from
    the raw PDF glyph data — preserving the original rich heading logic.
    """
    new_docs = []

    try:
        fitz_doc = fitz.open(document_path)
    except Exception as e:
        print(f"Error opening {document_path} with PyMuPDF: {e}")
        return new_docs

    current_section = {"parent_section": "", "section": "Espazo Nature", "text": "", "page_start": 0}
    font_sizes = []  # stack: {"section_name": ..., "size": ...}

    for page_num, page in enumerate(fitz_doc):
        blocks = page.get_text("dict")["blocks"]
        # Sort blocks by vertical position to maintain reading order
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        for block in blocks:
            # Only process text blocks
            if block.get("type", 0) != 0:
                continue
            if "lines" not in block:
                continue

            text = ""
            max_font_size = 0
            is_bold = False

            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"]
                    max_font_size = max(max_font_size, span["size"])
                    if span["flags"] & 2:  # bold flag
                        is_bold = True
                    text += "\n"

            text = text.strip()
            if not text:
                continue

            is_short = len(text) < 120
            is_large_font = max_font_size > 15  # adjust threshold as needed
            is_upper = text.isupper()

            if (is_bold or is_large_font or is_upper) and is_short:
                if current_section["text"].strip():
                    # Save the previous section as a LangChain Document
                    metadata = {
                        "parent_section": current_section["parent_section"],
                        "section": current_section["section"],
                        "page": current_section["page_start"],
                    }
                    new_docs.append(Document(page_content=current_section["text"].strip(), metadata=metadata))

                # Find the nearest ancestor heading (smallest size still larger than current)
                mayores = [f for f in font_sizes if f["size"] > max_font_size]
                parent_section_name = min(mayores, key=lambda x: x["size"])["section_name"] if mayores else ""

                text = _clean_text(text)
                # Keep only headings with larger font (maintain hierarchy stack)
                font_sizes = [f for f in font_sizes if f["size"] > max_font_size]
                font_sizes.append({"section_name": text, "size": max_font_size})

                current_section = {
                    "parent_section": parent_section_name,
                    "section": text,
                    "text": "",
                    "page_start": page_num,
                }
            else:
                current_section["text"] += text + "\n\n"

    # Append the final section
    if current_section["text"].strip():
        metadata = {
            "parent_section": current_section["parent_section"],
            "section": current_section["section"],
            "page": current_section["page_start"],
        }
        new_docs.append(Document(page_content=current_section["text"].strip(), metadata=metadata))

    fitz_doc.close()
    return new_docs