# Contributing to Path-Constrained Retrieval

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/Godskid89/path-constrained-retrieval.git`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install in development mode: `pip install -e .`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings to all classes and functions
- Keep functions focused and modular

## Testing

Run tests with:
```bash
pytest tests/ -v
```

Add tests for new features in the `tests/` directory.

## Submitting Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes
3. Add tests if applicable
4. Ensure all tests pass
5. Commit with descriptive messages
6. Push to your fork
7. Create a Pull Request

## Questions?

Open an issue for questions or discussions about the project.

