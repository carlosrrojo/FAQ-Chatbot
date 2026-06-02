# src/tests/test_hexagonal_architecture.py
import os
import subprocess
import sys

def test_hexagonal_architecture():
    """
    Ensures adapters, ports, and domain logic adhere to the hexagonal
    architecture boundaries using import-linter.
    Specifically, domain logic (src.domain) must not import or depend on
    infrastructure (src.infrastructure) or transport (src.transport).
    """
    # Locate the repository root (where .importlinter is defined)
    current_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = current_dir
    while root_dir != os.path.dirname(root_dir):
        if os.path.exists(os.path.join(root_dir, ".importlinter")):
            break
        root_dir = os.path.dirname(root_dir)
    else:
        root_dir = os.getcwd()

    # Find the lint-imports executable in the current python environment
    bindir = os.path.dirname(sys.executable)
    lint_imports_path = os.path.join(bindir, "lint-imports")
    if not os.path.exists(lint_imports_path):
        # Fallback to PATH resolution if not in current virtualenv bin
        lint_imports_path = "lint-imports"

    # Execute import linter in the repository root directory
    result = subprocess.run(
        [lint_imports_path],
        capture_output=True,
        text=True,
        cwd=root_dir
    )
    
    assert result.returncode == 0, (
        f"Hexagonal architecture contract check failed!\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
