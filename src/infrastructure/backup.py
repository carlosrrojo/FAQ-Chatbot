# src/infrastructure/backup.py
import os
import time
import tarfile
import sqlite3
import datetime
import logging
from src.config import MEMORY_DB_PATH, DB_PATH, BACKUP_DIR, BACKUP_RETENTION_DAYS

logger = logging.getLogger(__name__)

def perform_sqlite_backup(source_path: str, target_path: str) -> None:
    """Safely copies a live SQLite database without locking active transactions."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"SQLite source database not found at '{source_path}'")
    
    logger.info("Initiating hot backup from '%s' to '%s'", source_path, target_path)
    src_conn = sqlite3.connect(source_path)
    dst_conn = sqlite3.connect(target_path)
    try:
        with dst_conn:
            src_conn.backup(dst_conn, pages=100)
        logger.info("SQLite hot backup completed successfully.")
    except Exception as e:
        logger.error("SQLite hot backup failed: %s", e)
        raise e
    finally:
        dst_conn.close()
        src_conn.close()

def archive_chromadb(source_dir: str, target_tar_path: str) -> None:
    """Archives the ChromaDB directory into a compressed tarball."""
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"ChromaDB directory not found at '{source_dir}'")
        
    logger.info("Archiving ChromaDB directory from '%s' to '%s'", source_dir, target_tar_path)
    try:
        with tarfile.open(target_tar_path, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        logger.info("ChromaDB archive completed successfully.")
    except Exception as e:
        logger.error("ChromaDB archiving failed: %s", e)
        raise e

def upload_to_s3(file_path: str, bucket_name: str) -> bool:
    """Uploads a backup file to an AWS S3 bucket if credentials are set up."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        logger.warning("boto3 package not installed. Skipping remote S3 upload.")
        return False

    s3 = boto3.client("s3")
    object_name = os.path.basename(file_path)
    try:
        logger.info("Uploading '%s' to S3 bucket '%s'", object_name, bucket_name)
        s3.upload_file(file_path, bucket_name, object_name)
        logger.info("S3 upload successful.")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found. Skipping S3 upload.")
        return False
    except Exception as e:
        logger.error("S3 upload failed: %s", e)
        return False

def prune_old_backups(backup_dir: str, retention_days: int) -> None:
    """Deletes local backup files older than the retention window."""
    if not os.path.exists(backup_dir):
        return

    now = time.time()
    cutoff = now - (retention_days * 86400)
    logger.info("Scanning for backups older than %d days in '%s'", retention_days, backup_dir)
    
    for filename in os.listdir(backup_dir):
        filepath = os.path.join(backup_dir, filename)
        if os.path.isfile(filepath):
            file_mtime = os.path.getmtime(filepath)
            if file_mtime < cutoff:
                try:
                    os.remove(filepath)
                    logger.info("Pruned old backup file: '%s'", filename)
                except Exception as e:
                    logger.error("Failed to delete '%s': %s", filepath, e)

def run_backup_job() -> dict:
    """Orchestrates the entire backup job."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sqlite_backup_filename = f"memory_backup_{timestamp}.sqlite"
    sqlite_backup_path = os.path.join(BACKUP_DIR, sqlite_backup_filename)
    
    chroma_backup_filename = f"chroma_backup_{timestamp}.tar.gz"
    chroma_backup_path = os.path.join(BACKUP_DIR, chroma_backup_filename)

    results = {
        "sqlite": {"status": "skipped", "path": None},
        "chromadb": {"status": "skipped", "path": None},
        "s3_sqlite": "skipped",
        "s3_chromadb": "skipped",
    }

    # 1. SQLite Backup
    try:
        perform_sqlite_backup(MEMORY_DB_PATH, sqlite_backup_path)
        results["sqlite"] = {"status": "success", "path": sqlite_backup_path}
    except Exception as e:
        results["sqlite"] = {"status": "failed", "error": str(e)}

    # 2. ChromaDB Backup
    try:
        archive_chromadb(DB_PATH, chroma_backup_path)
        results["chromadb"] = {"status": "success", "path": chroma_backup_path}
    except Exception as e:
        results["chromadb"] = {"status": "failed", "error": str(e)}

    # 3. Optional S3 Upload
    s3_bucket = os.getenv("BACKUP_AWS_S3_BUCKET")
    if s3_bucket:
        if results["sqlite"]["status"] == "success":
            s3_sqlite_ok = upload_to_s3(sqlite_backup_path, s3_bucket)
            results["s3_sqlite"] = "success" if s3_sqlite_ok else "failed"
        if results["chromadb"]["status"] == "success":
            s3_chroma_ok = upload_to_s3(chroma_backup_path, s3_bucket)
            results["s3_chromadb"] = "success" if s3_chroma_ok else "failed"

    # 4. Local Pruning
    prune_old_backups(BACKUP_DIR, BACKUP_RETENTION_DAYS)

    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting manual backup task execution...")
    job_results = run_backup_job()
    logger.info("Backup job results: %s", job_results)
