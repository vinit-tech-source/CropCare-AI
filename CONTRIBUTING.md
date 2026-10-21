# Contributing to CropCare AI

Thank you for your interest in contributing to CropCare AI! This document
provides guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/CropCare-AI.git
   cd CropCare-AI
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Linux/Mac
   pip install -r requirements-dev.txt
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Code Style

- **Formatting**: [Black](https://github.com/psf/black) (line length 100)
- **Linting**: [Ruff](https://github.com/astral-sh/ruff)
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for all public functions and classes (Google style)
- Run `make lint` before committing

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `test:` — Adding or updating tests
- `chore:` — Maintenance (deps, CI, etc.)
- `refactor:` — Code restructuring without behaviour change

## Testing

- Write tests for new features in `tests/`
- Run `make test` to verify all tests pass
- Aim for ≥ 70% coverage on core modules

## Pull Request Process

1. Ensure `make lint` and `make test` pass
2. Update documentation if needed
3. Fill out the PR template
4. Request a review

## Reporting Issues

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) to report
issues. Include reproduction steps, expected vs actual behaviour, and your
environment details.
