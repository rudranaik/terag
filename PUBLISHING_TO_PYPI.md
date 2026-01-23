# Publishing TERAG to PyPI

This guide provides step-by-step instructions for publishing the `terag` package to PyPI.

## Prerequisites

1.  **PyPI Account**: Create an account on [pypi.org](https://pypi.org).
2.  **Build Tools**: Ensure you have `build` and `twine` installed:
    ```bash
    pip install build twine
    ```

## Steps to Publish

### 1. Update Version

Open `terag/terag/__init__.py` and update the `__version__` variable:

```python
__version__ = "0.1.0"  # Update this for each new release
```

Also update `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

### 2. Clean Previous Builds

Remove any existing `dist/` directory to avoid confusion:

```bash
rm -rf dist/
```

### 3. Build the Package

Run the build command from the root directory:

```bash
python3 -m build
```

This will create a `dist/` directory containing `.tar.gz` and `.whl` files.

### 4. Upload to PyPI

Use `twine` to upload the package. You will be prompted for your PyPI username and password (or API token).

**TestPyPI (Optional but Recommended for First Time)**:
```bash
python3 -m twine upload --repository testpypi dist/*
```

**Production PyPI**:
```bash
python3 -m twine upload dist/*
```

### 5. Verify Installation

Create a new virtual environment and try installing your package:

```bash
pip install terag
```

## Troubleshooting

-   **Name Conflict**: If the name `terag` is already taken on PyPI, you will need to choose a different name in `pyproject.toml`.
-   **Authentication Error**: Ensure your API token is correct and has the necessary permissions.
