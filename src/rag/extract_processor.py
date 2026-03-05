from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import json
import regex
from langchain_core.documents import Document
import fitz
import re
import ast


class ExtractProcessor:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
    
    def extract_metadata(self, content: str) -> dict:
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
        chain = prompt | self.llm
        metadata = chain.invoke({"content": content})
        parsed_metadata = self.parse_to_list(metadata.content)
        return parsed_metadata
    
    def parse_to_list(self, metadata: str) -> list:
        # We can use re.DOTALL in case the list spans multiple lines
        candidates = re.findall(r'\[.*?\]', metadata, flags=re.DOTALL)
        valid_lists = []
        for c in candidates:
            try:
                parsed = ast.literal_eval(c)
                if isinstance(parsed, list):
                    valid_lists.append(parsed) # <--- Ahora guardamos 'parsed', que es una list de Python real
            except:
                pass
        
        return valid_lists[0] if valid_lists else []
    
    def parse_to_dict(self, metadata: str) -> dict:
        metadata = regex.search(pattern=r"\{(?:[^{}]|(?R))*\}", string=metadata).group()
        return json.loads(metadata)
    
    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace("\n", "")
        text = text.replace(":", "")
        text = text.replace("\t", "")
        text = text.replace("\r", "")
        return text
    
    def process_document(self, document_path: str) -> list[Document]:
        section_docs = []

        fitz_doc = fitz.open(document_path)
        current_section = {"parent_section":"","section": "Espazo Nature", "text": "", "page_start": 0}
        font_sizes = [] # Lista para almacenar el tamaño superior {"section_name": ..., "size": ...}
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

                #is_numbered = re.match(r"^\d+(\.\d+)*\s+", text)
                is_short = len(text) < 120
                is_large_font = max_font_size > 15  # adjust threshold
                is_upper = text.isupper()

                if (is_bold or is_large_font or is_upper) and is_short:
                    if current_section["text"].strip():
                        # Save previous section as a Langchain Document
                        #metadata = fitz_doc.metadata.copy()
                        #self.extract_metadata(current_section["text"])
                        metadata = {}
                        metadata.update({"parent_section":current_section["parent_section"],"section": current_section["section"], "page": current_section["page_start"]})
                        section_docs.append(Document(page_content=current_section["text"].strip(), metadata=metadata))

                    # Buscar el elemento con el menor tamaño dentro de los mayores
                    mayores = [f for f in font_sizes if f["size"] > max_font_size]
                    if mayores:
                        padre = min(mayores, key=lambda x: x["size"])
                        parent_section_name = padre["section_name"]
                    else:
                        parent_section_name = ""
                    
                    text = self.clean_text(text)
                    # Actualizamos la lista eliminando tamaños menores o iguales,
                    # para que la jerarquía siga siendo válida (funciona como una pila)
                    font_sizes = [f for f in font_sizes if f["size"] > max_font_size]
                    font_sizes.append({"section_name": text, "size": max_font_size})

                    # Start new section
                    current_section = {
                        "parent_section": parent_section_name,
                        "section": text,
                        "text": "",
                        "page_start": page_num
                    }
                else:
                    current_section["text"] += text + "\n\n"

        # Append the final section
        if current_section["text"].strip():
            #metadata = fitz_doc.metadata.copy()
            metadata = {}
            metadata.update({"parent_section":current_section["parent_section"],"section": current_section["section"], "page": current_section["page_start"]})
            section_docs.append(Document(page_content=current_section["text"].strip(), metadata=metadata))
                    
        try:
            fitz_doc.close()
        except Exception as e:
            print(f"Error processing {document_path} with PyMuPDF: {e}")
            # Fallback: Just append original doc if there's an error
            section_docs.append(doc)

        return section_docs


if __name__ == "__main__":
    doc1 = "data/documents/espazo_nature.pdf"
    doc2 = "/home/carlos/Documents/personal/UDC/TFG/a-practical-guide-to-building-agents.pdf"
    processor = ExtractProcessor(ChatOllama(model="llama3.1"))
    docs = processor.process_document(doc1)
    for doc in docs:
        print(doc.metadata)