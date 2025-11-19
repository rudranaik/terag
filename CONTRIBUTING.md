# Contributing to TERAG

Thank you for your interest in contributing to TERAG! We welcome contributions from the community to help make graph-based RAG more efficient and accessible.

## How to Contribute

### Reporting Bugs
If you find a bug, please open an issue on GitHub describing:
1. Steps to reproduce the issue
2. Expected behavior
3. Actual behavior
4. Your environment (OS, Python version, etc.)

### Suggesting Enhancements
We love ideas for new features! Please open an issue describing your idea and how it would benefit the project.

### Pull Requests
1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure everything works (see below)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rudranaik/terag.git
   cd terag
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Running Tests

We use `pytest` for testing. To run the tests:

```bash
pip install pytest
pytest tests/
```

## Style Guide

- Follow PEP 8 guidelines for Python code.
- Add docstrings to classes and functions.
- Keep code clean and readable.

## Publishing (Maintainers Only)

To publish a new version to PyPI:

1. Update version in `terag/terag/__init__.py`
2. Build the package:
   ```bash
   python3 -m build
   ```
3. Upload to PyPI:
   ```bash
   python3 -m twine upload dist/*
   ```
