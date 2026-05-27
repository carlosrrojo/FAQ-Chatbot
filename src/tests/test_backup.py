# src/tests/test_backup.py
import os
import time
import shutil
import sqlite3
import tarfile
import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.backup import (
    perform_sqlite_backup,
    archive_chromadb,
    prune_old_backups,
    run_backup_job,
    upload_to_s3
)

@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary workspace directory for test files."""
    d = tmp_path / "test_backup_workspace"
    d.mkdir()
    return str(d)

def test_perform_sqlite_backup(temp_dir):
    source_db = os.path.join(temp_dir, "source.sqlite")
    target_db = os.path.join(temp_dir, "target.sqlite")

    # Set up source database with a test table and a row
    conn = sqlite3.connect(source_db)
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT);")
    conn.execute("INSERT INTO test (val) VALUES ('hello');")
    conn.commit()
    conn.close()

    # Perform backup
    perform_sqlite_backup(source_db, target_db)

    # Verify target database contents
    assert os.path.exists(target_db)
    conn_target = sqlite3.connect(target_db)
    cursor = conn_target.execute("SELECT val FROM test;")
    rows = cursor.fetchall()
    conn_target.close()

    assert len(rows) == 1
    assert rows[0][0] == "hello"

def test_perform_sqlite_backup_missing_source():
    with pytest.raises(FileNotFoundError):
        perform_sqlite_backup("nonexistent_db.sqlite", "target.sqlite")

def test_archive_chromadb(temp_dir):
    source_chroma = os.path.join(temp_dir, "chroma_db")
    os.makedirs(source_chroma)
    
    # Create some dummy files inside the chroma_db folder
    with open(os.path.join(source_chroma, "index.bin"), "w") as f:
        f.write("mock index data")
    with open(os.path.join(source_chroma, "data.parquet"), "w") as f:
        f.write("mock parquet data")

    target_tar = os.path.join(temp_dir, "chroma_backup.tar.gz")

    # Archive
    archive_chromadb(source_chroma, target_tar)

    assert os.path.exists(target_tar)

    # Verify tarball contents
    with tarfile.open(target_tar, "r:gz") as tar:
        members = tar.getnames()
        assert any("index.bin" in m for m in members)
        assert any("data.parquet" in m for m in members)

def test_archive_chromadb_missing_source():
    with pytest.raises(FileNotFoundError):
        archive_chromadb("nonexistent_chroma", "target.tar.gz")

def test_prune_old_backups(temp_dir):
    backup_folder = os.path.join(temp_dir, "backups")
    os.makedirs(backup_folder)

    # Create two files
    new_file = os.path.join(backup_folder, "new_backup.sqlite")
    old_file = os.path.join(backup_folder, "old_backup.sqlite")

    with open(new_file, "w") as f:
        f.write("new")
    with open(old_file, "w") as f:
        f.write("old")

    # Modify access/modification time of the old file to be 10 days ago
    ten_days_ago = time.time() - (10 * 86400)
    os.utime(old_file, (ten_days_ago, ten_days_ago))

    # Prune with 7 days retention
    prune_old_backups(backup_folder, retention_days=7)

    # Verify old file was deleted, and new file was kept
    assert os.path.exists(new_file)
    assert not os.path.exists(old_file)

@patch("src.infrastructure.backup.upload_to_s3")
def test_run_backup_job(mock_upload, temp_dir):
    # Setup temporary source database and directory
    source_db = os.path.join(temp_dir, "memory.sqlite")
    source_chroma = os.path.join(temp_dir, "chroma_db")
    backup_dir = os.path.join(temp_dir, "backups")

    os.makedirs(source_chroma)
    with open(os.path.join(source_chroma, "dummy"), "w") as f:
        f.write("data")

    conn = sqlite3.connect(source_db)
    conn.execute("CREATE TABLE t (id INT);")
    conn.commit()
    conn.close()

    # Patch the config variables and env variables to point to our test folders
    with patch("src.infrastructure.backup.MEMORY_DB_PATH", source_db), \
         patch("src.infrastructure.backup.DB_PATH", source_chroma), \
         patch("src.infrastructure.backup.BACKUP_DIR", backup_dir), \
         patch("src.infrastructure.backup.BACKUP_RETENTION_DAYS", 7), \
         patch.dict(os.environ, {"BACKUP_AWS_S3_BUCKET": "my-test-bucket"}):

        mock_upload.return_value = True

        results = run_backup_job()

        # Assert correct keys in result
        assert results["sqlite"]["status"] == "success"
        assert results["chromadb"]["status"] == "success"
        assert results["s3_sqlite"] == "success"
        assert results["s3_chromadb"] == "success"

        # Assert backup files were created on disk
        assert os.path.exists(results["sqlite"]["path"])
        assert os.path.exists(results["chromadb"]["path"])
        assert mock_upload.call_count == 2

@patch("src.infrastructure.backup.logging")
def test_upload_to_s3_missing_boto3(mock_logging, temp_dir):
    # Test upload behavior when boto3 is not installed or import fails
    test_file = os.path.join(temp_dir, "dummy_backup.sqlite")
    with open(test_file, "w") as f:
        f.write("data")

    # Force boto3 import failure
    with patch.dict("sys.modules", {"boto3": None}):
        result = upload_to_s3(test_file, "my-bucket")
        assert result is False
