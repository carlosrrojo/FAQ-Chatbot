
from langchain_core.documents import Document
import fitz

def parse_doc(document_path: str) -> list[Document]:
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
        
    # Divide the document into sections
    new_docs = divide_into_sections(fitz_doc, new_docs)
    fitz_doc.close() 
    
    return new_docs

def _clean_text(text: str) -> str:
    """Clean heading text (strips whitespace control chars and colons)."""
    text = text.strip()
    text = text.replace("\n", "")
    text = text.replace(":", "")
    text = text.replace("\t", "")
    text = text.replace("\r", "")
    return text

_BULLET_CHARS = (
    "\u25cf",  # ●  BLACK CIRCLE
    "\u2022",  # •  BULLET
    "\u25e6",  # ◦  WHITE BULLET
    "\u2023",  # ‣  TRIANGULAR BULLET
    "\u2043",  # ⁃  HYPHEN BULLET
    "\uf0b7",  # private-use bullet (common in PDF exports)
)

def _clean_content(text: str) -> str:
    """Normalise body text extracted from a PDF.

    - Removes bullet / list marker characters.
    - Replaces non-breaking and zero-width spaces with regular spaces.
    - Collapses runs of blank lines into a single blank line.
    - Strips leading/trailing whitespace from each line.
    """
    import re

    for ch in _BULLET_CHARS:
        text = text.replace(ch, "")

    # Non-breaking space, soft hyphen, zero-width space / joiner / non-joiner
    for ch in ("\u00a0", "\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(ch, " ")

    # Strip each line and drop lines that are now pure whitespace
    lines = [line.strip() for line in text.splitlines()]
    # Collapse 2+ consecutive blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip()

def divide_into_sections(fitz_doc: fitz.Document, docs: list[Document]) -> list[Document]:
    current_section = {"section":"Espazo Nature","parent_section": "", "text": "", "page_start": 0}
    font_sizes = []  # stack: {"section_name": ..., "size": ...}

    # Find sections by heading fontsize and bold flag
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
                    metadata = {}
                    metadata.update({"section":current_section["section"],"parent_section": current_section["parent_section"], "page": current_section["page_start"]})
                    docs.append(Document(page_content=current_section["text"].strip(), metadata=metadata))

                # Find the nearest ancestor heading (smallest size still larger than current)
                mayores = [f for f in font_sizes if f["size"] > max_font_size]
                if mayores:
                    padre = min(mayores, key=lambda x: x["size"])
                    parent_section_name = padre["section_name"]
                else:
                    parent_section_name = ""

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
                current_section["text"] += _clean_content(text) + "\n\n"

    # Append the final section
    if current_section["text"].strip():
        metadata = {}
        metadata.update({"section":current_section["section"],"parent_section": current_section["parent_section"], "page": current_section["page_start"]})
        docs.append(Document(page_content=current_section["text"].strip(), metadata=metadata))
    return docs

def get_children(section : str, chunks: list[Document]) -> list[str]:
    children = set()
    for c in chunks:
        if section == c.metadata["parent_section"]:
            children.add(c.metadata["section"])
    return list(children)


if __name__ == "__main__":
    doc1 = "data/documents/espazo_nature.pdf"
    #doc2 = "/home/carlos/Documents/personal/UDC/TFG/a-practical-guide-to-building-agents.pdf"
    docs = parse_doc(doc1)
    for doc in docs:
        print("======================")
        print(doc.metadata)
        print("*******************")
        print(get_siblings(doc.metadata["section"], docs))
        print("===================\n")
        #print(doc.page_content)