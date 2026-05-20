# Development Guide

## Local setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip in the virtual environment:

```bash
python3 -m pip install --upgrade pip
```

Install development dependencies (editable install + dev extras):

```bash
python3 -m pip install -r requirements-dev.txt
```

Equivalent direct command:

```bash
python3 -m pip install -e ".[dev]"
```

### Troubleshooting: externally-managed-environment (macOS/Homebrew)

If you see this error while running pip, your shell is using a system-managed
Python interpreter. Install dependencies only inside the activated virtual
environment.

Quick fix:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
```

## Run tests

Run all tests:

```bash
python3 -m pytest
```

Run with coverage:

```bash
python3 -m pytest --cov=stylometry --cov-report=term-missing
```

## Lint and format

```bash
ruff check .
black .
```

Auto-fix lint issues when possible:

```bash
ruff check . --fix
```

## Continuous integration

The GitHub Actions workflow runs on every push and pull request and executes:

- `ruff check .`
- `black --check .`
- `pytest --cov=stylometry --cov-report=term-missing`

Workflow file: `.github/workflows/ci.yml`

## Releases and versioning

Version is computed from Conventional Commits and released with semantic-release.
Published package version is still derived from Git tags via setuptools-scm.
Release workflow runs only after CI succeeds on `main`/`master`.

- Tag format: `vMAJOR.MINOR.PATCH` (example: `v0.2.0`)
- Optional prereleases: `v0.2.0rc1`, `v0.2.0b1`, `v0.2.0a1`

Commit types used for bumps:

- `fix:` -> PATCH
- `feat:` -> MINOR
- `BREAKING CHANGE:` (or `!`) -> MAJOR

Release example:

```bash
git commit -m "feat: add multilingual function-word presets"
git push origin main
```

Workflow file: `.github/workflows/release.yml`

## Publish to PyPI

Publishing is automated by GitHub Actions after a successful `Release` workflow
run.

- Build + check workflow: `.github/workflows/publish.yml`
- Trigger: successful `Release` workflow on `main`/`master`

Version behavior:

- On tag `v1.2.3` -> published version is `1.2.3`
- On commits after a tag -> local/dev builds can be `1.2.4.devN`
- Next release tag is computed from commit messages (semantic-release)

Required one-time setup on PyPI:

1. Create the project on PyPI (first release).
2. Configure a Trusted Publisher for this GitHub repository.
3. Authorize workflow file: `.github/workflows/publish.yml`.

After setup, publishing a tag automatically uploads both sdist and wheel to PyPI.
