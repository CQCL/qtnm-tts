"""Test that all example notebooks can be executed without error."""

import os
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


MAX_RETRIES = 5


def execute_notebook(notebook_path: str, retries: int = MAX_RETRIES):
    """Execute a Jupyter notebook and return any errors."""
    for attempt in range(1, retries + 1):
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)  # type: ignore
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})  # type: ignore
            return None  # Success, no error
        except Exception as e:
            error_message = str(e)
            # Kernel died, try again, not sure how to fix this
            if "Kernel died" in error_message and attempt < retries:
                continue  # Retry
            return error_message


directory = "./examples/circuits/"
ipynb_files = glob.glob(os.path.join(directory, "*.ipynb"))


@pytest.mark.parametrize("notebook_path", ipynb_files)
def test_notebook_execution(notebook_path: str):
    """Test that the notebook can be executed without error."""
    error = execute_notebook(notebook_path)
    assert error is None, f"Error executing {notebook_path}: {error}"
