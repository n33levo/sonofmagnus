"""
No-search compliance test (Gate G5).

Ensures that chess.engine is not imported in the play/ module,
preventing accidental search usage during inference.
"""

import ast
import sys
from pathlib import Path

import pytest


def check_file_for_engine_imports(file_path: Path) -> list[str]:
    """
    Check a Python file for chess.engine imports.

    Returns:
        List of error messages (empty if no violations)
    """
    errors = []

    try:
        with open(file_path) as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in ast.walk(tree):
            # Check for: import chess.engine
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if 'chess.engine' in alias.name:
                        errors.append(
                            f"{file_path}:{node.lineno}: "
                            f"Forbidden import: {alias.name}"
                        )

            # Check for: from chess import engine
            elif isinstance(node, ast.ImportFrom):
                if node.module == 'chess' and any(
                    alias.name == 'engine' for alias in node.names
                ):
                    errors.append(
                        f"{file_path}:{node.lineno}: "
                        f"Forbidden import: from chess import engine"
                    )
                elif node.module == 'chess.engine':
                    errors.append(
                        f"{file_path}:{node.lineno}: "
                        f"Forbidden import: from chess.engine"
                    )

    except SyntaxError as e:
        errors.append(f"{file_path}: Syntax error - {e}")

    return errors


def test_play_module_no_search():
    """Test that play/ module doesn't import chess.engine."""
    print("Testing play/ module for forbidden chess.engine imports...")

    project_root = Path(__file__).parent.parent
    play_dir = project_root / "play"

    if not play_dir.exists():
        pytest.fail(f"play/ directory not found at {play_dir}")

    all_errors = []

    # Check all Python files in play/
    for py_file in play_dir.rglob("*.py"):
        errors = check_file_for_engine_imports(py_file)
        all_errors.extend(errors)

    if all_errors:
        error_list = "\n".join(all_errors)
        pytest.fail(f"Forbidden chess.engine imports detected:\n{error_list}")

    print(f"✓ No chess.engine imports found in play/ module")


def test_train_module_allows_search():
    """Verify that train/ module CAN use chess.engine (for distillation)."""
    print("Verifying train/ module can use chess.engine (for offline labeling)...")

    project_root = Path(__file__).parent.parent
    distill_file = project_root / "train" / "distill_labeler.py"

    if not distill_file.exists():
        pytest.skip(f"{distill_file} not found; skipping chess.engine allowance check")

    with open(distill_file) as f:
        content = f.read()

    # Should have chess.engine import
    has_engine = "chess.engine" in content or "import chess.engine" in content

    assert has_engine, (
        "train/distill_labeler.py should import chess.engine for offline labeling. "
        "Add `import chess.engine` or equivalent."
    )

    print("✓ train/distill_labeler.py correctly uses chess.engine for offline labeling")


def test_io_model_no_search():
    """Test that io/ and model/ modules don't import chess.engine."""
    print("Testing io/ and model/ modules for forbidden imports...")

    project_root = Path(__file__).parent.parent
    all_errors = []

    for module_dir in ["io", "model"]:
        module_path = project_root / module_dir
        if not module_path.exists():
            continue

        for py_file in module_path.rglob("*.py"):
            errors = check_file_for_engine_imports(py_file)
            all_errors.extend(errors)

    if all_errors:
        error_list = "\n".join(all_errors)
        pytest.fail(f"Forbidden chess.engine imports detected:\n{error_list}")

    print(f"✓ No chess.engine imports found in io/ and model/")


def run_all_tests():
    """Run all no-search compliance tests."""
    print("="*60)
    print("NO-SEARCH COMPLIANCE TESTS (Gate G5)")
    print("="*60)
    print()

    results = []

    results.append(test_play_module_no_search())
    print()

    results.append(test_train_module_allows_search())
    print()

    results.append(test_io_model_no_search())
    print()

    print("="*60)
    if all(results):
        print("✓ ALL NO-SEARCH COMPLIANCE TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME COMPLIANCE TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
