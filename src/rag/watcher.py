import time
import os
import logging

logger = logging.getLogger(__name__)
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import sys

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

        logger.info("Detected change in %s (%s). Scheduling reload...", event.src_path, event.event_type)
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
            logger.info("Change detected in documents. Reloading database...")
            try:
                subprocess.run([sys.executable, "-m", "src.rag.ingest"], check=True)
                logger.info("Database reload complete.")
            except Exception as e:
                logger.error("Error reloading database: %s", e)

def start_watcher(on_reingest_callback, path):
    event_handler = IngestHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    logger.info("Started watching %s for changes...", path)
    return observer
