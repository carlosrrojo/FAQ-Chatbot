
import time
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.rag.ingest import ingest_docs

# Debounce time in seconds
DEBOUNCE_DELAY = 2.0

class IngestHandler(FileSystemEventHandler):


    def __init__(self):
        self.timer = None
        self.lock = threading.Lock()
        self.ingest_lock = threading.Lock()

    def _process_event(self, event):
        if event.is_directory:
            return
        
        valid_extensions = ('.txt', '.pdf')
        if not event.src_path.endswith(valid_extensions):
            return

        print(f"Detected change in {event.src_path} ({event.event_type}). Scheduling reload...")
        self.debounce_ingest()

    def on_created(self, event):
        self._process_event(event)

    def on_deleted(self, event):
        self._process_event(event)

    def on_modified(self, event):
        self._process_event(event)

    def on_moved(self, event):
        self._process_event(event)

    def debounce_ingest(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(DEBOUNCE_DELAY, self.trigger_ingest)
            self.timer.start()

    def trigger_ingest(self):
        with self.ingest_lock:
            print("\nChange detected in documents. Reloading database...")
            try:
                ingest_docs(clear_db=True)
                print("Database reload complete.\n")
            except Exception as e:
                print(f"Error reloading database: {e}")

def start_watcher(path):
    event_handler = IngestHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"Started watching {path} for changes...")
    return observer
