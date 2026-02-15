# FAQ Chatbot with RAG & Social Media Integration

This project is an intelligent FAQ Chatbot designed to answer user queries based on a knowledge base of documents. It utilizes Retrieval-Augmented Generation (RAG) to provide accurate/context-aware answers and integrates with Instagram and WhatsApp for multi-channel support.

## 🚀 Features

*   **RAG Engine**:
    *   **Document Ingestion**: Supports `.txt` and `.pdf` files.
    *   **Intelligent Chunking**:
        *   **Structural Splitting**: Uses Markdown headers for text files to preserve context.
        *   **Semantic Splitting**: Uses embedding similarity to chunk PDFs based on meaning.
    *   **Metadata Extraction**: Automatically extracts summaries and keywords for each chunk using an LLM to improve retrieval.
    *   **Vector Database**: Stores embeddings in **ChromaDB** for fast similarity search.
*   **LLM Integration**: powered by **Llama 3** (via Ollama) for natural language understanding and generation.
*   **API Integrations**:
    *   **Instagram Graph API**: Responds to Direct Messages.
    *   **WhatsApp Cloud API**: Responds to text messages.
    *   **Webhook Support**: Verifies and processes standard Meta webhooks.
*   **Automation**:
    *   **Use `watcher.py`** to automatically ingest new documents added to the data folder.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **Framework**: FastAPI (Web Server)
*   **LLM & Embeddings**: Ollama (Llama 3)
*   **Orchestration**: LangChain
*   **Database**: ChromaDB (Vector Store)

## 📋 Prerequisites

1.  **Python 3.10+** installed.
2.  **Ollama** installed and running with Llama 3:
    ```bash
    ollama run llama3
    ```
3.  **Meta Developer Account** (for Instagram/WhatsApp integration).

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd FAQ-Chatbot
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration**:
    Create a `.env` file in the root directory (use `.env.example` as a template):
    ```env
    INSTAGRAM_ACCESS_TOKEN=your_instagram_token
    WHATSAPP_API_TOKEN=your_whatsapp_token
    WHATSAPP_PHONE_NUMBER_ID=your_phone_id
    VERIFY_TOKEN=your_custom_verify_token
    ```

## 📖 Usage

### 1. Ingest Documents
Place your FAQ documents (text or PDF) in `data/documents/` and run:

```bash
python src/rag/ingest.py
```
This will process the files, extract metadata, and store embeddings in ChromaDB.

### 2. Chat via CLI
To test the bot directly in your terminal:

```bash
python src/rag/chatbot.py
```

### 3. Run the API Server
Start the FastAPI server to handle webhooks and API requests:

```bash
uvicorn src.api.main:app --reload
```
*   **Chat Endpoint**: `POST /chat`
*   **Instagram Webhook**: `GET/POST /instagram/webhook`
*   **WhatsApp Webhook**: `GET/POST /whatsapp/webhook`

### 4. Automatic Ingestion (Optional)
Run the watcher to automatically ingest files when they are added or modified:

```bash
python src/rag/watcher.py
```

## 📂 Project Structure

```
src/
├── api/             # FastAPI routes and external integrations
│   ├── instagram.py # Instagram Graph API logic
│   ├── whatsapp.py  # WhatsApp Cloud API logic
│   ├── main.py      # Entry point for the web server
│   └── utils.py     # Helper functions (verification, etc.)
├── rag/             # Retrieval-Augmented Generation logic
│   ├── ingest.py    # Document processing & vectorization
│   ├── chatbot.py   # RAG pipeline & CLI interface
│   └── watcher.py   # File system watcher
data/
├── documents/       # Place source files here
└── chroma_db/       # Vector database storage
```