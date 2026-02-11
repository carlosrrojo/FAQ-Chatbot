
# ChromaDB Auto-Reload Feature & Robust Ingestion

I have implemented an automated system to reload the Chroma vector database whenever files in the `data/documents` directory change. Additionally, I have hardened the ingestion process to handle large files and failures gracefully.

## Changes Made

1.  **`src/rag/ingest.py`**:
    *   **Refactored Loading**: Files are now loaded individually. If one file is corrupted or fails to load, it enters the "skipped" list, allowing other files to be ingested successfully.
    *   **Batched Insertion**: Embeddings are generated and inserted into ChromaDB in batches of 100 chunks. This prevents timeouts and memory issues when processing large PDFs.
    *   **Safe Reset**: Uses `chromadb` client to safely clear the 'langchain' collection without deleting the physical directory, fixing "readonly database" errors.

2.  **`src/rag/watcher.py`**:
    *   Monitors `data/documents` for `created`, `modified`, `deleted`, and `moved` events on `.txt` and `.pdf` files.
    *   Triggers a database reload with a debounce delay.

3.  **`src/api/main.py`**:
    *   Starts the watcher automatically with the API server.

## Verification

To verify the features:

1.  **Start the API Server**:
    ```bash
    uvicorn src.api.main:app --reload
    ```
    Logs should show: `INFO:src.api.main:Starting document watcher...`

2.  **Add Documents**:
    *   Add a mix of small and large files to `data/documents`.
    *   Check logs to see them being loaded individually:
        ```
        Loading large_document.pdf...
          Loaded 150 document(s) from large_document.pdf
        Loading small_note.txt...
          Loaded 1 document(s) from small_note.txt
        ```

3.  **Check for Failures**:
    *   If a file is corrupted, you will see: `Error loading file ...: [Error Details]`.
    *   The process will continue and ingest the valid files.

## Troubleshooting

*   **"Attempt to write a readonly database"**: Check for other processes holding locks.
*   **Module Errors**: Run scripts using `python -m src.rag.ingest`.
