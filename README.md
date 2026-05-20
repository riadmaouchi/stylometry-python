# stylometry-python

**Authorship attribution and stylometric analysis in Python.**

A lightweight, dependency-minimal library for measuring writing style,
attributing authorship, and detecting stylistic shifts introduced by LLMs.

```bash
pip install stylometry-python
```

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/riadmaouchi/stylometry-python/actions/workflows/ci.yml/badge.svg)](https://github.com/riadmaouchi/stylometry-python/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/riadmaouchi/stylometry-python/graph/badge.svg?token=GP1274n1wW)](https://codecov.io/github/riadmaouchi/stylometry-python)

---

## What is stylometry?

Stylometry is the statistical analysis of writing style.
Every author has unconscious stylistic habits — frequency of function words,
sentence length patterns, punctuation choices — that form a measurable **fingerprint**.

Mosteller & Wallace used it to resolve the Federalist Papers authorship debate in 1964.
Patrick Juola used it to identify JK Rowling behind the pseudonym Robert Galbraith in 2013.

This library makes those techniques accessible in 5 lines of Python.

---

## Quickstart

```python
from stylometry import StyleAnalyzer

sa = StyleAnalyzer()

# Fit on known texts
sa.fit(zola_texts, label="Zola")
sa.fit(maupassant_texts, label="Maupassant")

# Attribute an unknown text
predicted, distances = sa.predict(unknown_text)
print(f"Predicted author: {predicted}")
# → Predicted author: Zola

# Measure stylistic shift (original vs LLM rewrite)
shift = sa.shift(original_text, gpt_rewrite)
print(f"Stylistic shift: {shift:.4f}")
# → Stylistic shift: 0.2409
```

---

## Installation

```bash
pip install stylometry-python
```

**Dependencies:** numpy, matplotlib, scikit-learn — nothing else.
Works 100% offline. No API keys. No GPU.

### Development setup

On macOS (Homebrew Python), use a virtual environment to avoid
`externally-managed-environment` errors:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
```

Equivalent direct command:

```bash
python3 -m pip install -e ".[dev]"
```

Run tests:

```bash
python3 -m pytest
```

For a full local workflow (venv, tests, coverage, lint, format), see
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

### Continuous Integration

GitHub Actions runs lint + tests on each push and pull request:

- `ruff check .`
- `black --check .`
- `pytest --cov=stylometry --cov-report=term-missing`

Workflow file: `.github/workflows/ci.yml`

### Publishing

Package publication is automated on semantic tags (`v*`) via GitHub Actions.
Tags are generated from Conventional Commits by semantic-release.

Release workflow: `.github/workflows/release.yml`
Workflow file: `.github/workflows/publish.yml`

---

## Core API

### `StyleAnalyzer(function_words=None, language='fr', min_words=50)`

The main class. Handles vectorization, attribution, and visualization.

```python
from stylometry import StyleAnalyzer

# French (default) — 41 function words
sa = StyleAnalyzer()

# Custom vocabulary
sa = StyleAnalyzer(function_words=['the', 'of', 'and', 'to', 'a', 'in'])

# English preset
sa = StyleAnalyzer(language='en')
```

---

### `vectorize(text) → np.ndarray`

Convert a text to a style vector (L2-normalized function word frequencies).

```python
v = sa.vectorize("Il pleuvait a verse. La nuit etait noire...")
print(v.shape)  # (41,)
print(v.sum())  # ≈ 1.0 after normalization
```

---

### `fit(texts, label) → self`

Compute a centroid from a list of texts. Chainable.

```python
sa.fit(zola_corpus, "Zola").fit(maupassant_corpus, "Maupassant")
```

---

### `predict(text) → (label, distances)`

Attribute a text to the nearest centroid.

```python
predicted, distances = sa.predict(unknown)

print(predicted)           # "Zola"
print(distances)           # {"Zola": 0.12, "Maupassant": 0.43}
print(sa.confidence(distances))  # "HIGH" / "MEDIUM" / "LOW"
```

---

### `shift(original, rewrite) → float`

Measure the cosine distance between two texts in style space.
Use this to quantify how much an LLM changed the style of a text.

```python
shift = sa.shift(original, gpt4_rewrite)
# 0.00 = style unchanged
# 0.24 = significant shift (typical GPT-4)
# 1.00 = maximally different
```

---

### `cosine_distance(text_a, text_b) → float`

Direct cosine distance between two texts.

```python
d = sa.cosine_distance(text_a, text_b)
```

---

## Visualization

### `plot_fingerprint(texts_dict, top_n=15)`

Bar chart comparing function word frequencies across groups.

```python
fig = sa.plot_fingerprint(
    texts_dict={
        "Zola": zola_corpus,
        "Maupassant": maupassant_corpus,
        "GPT-4": gpt4_corpus,
    },
    top_n=12,
    title="Writing fingerprints",
)
fig.savefig("fingerprints.png", dpi=150)
```

### `plot_clusters(texts_groups, labels)`

PCA scatter plot — visualize stylistic distances between groups.

```python
fig = sa.plot_clusters(
    texts_groups=[zola_corpus, maupassant_corpus, gpt4_corpus],
    labels=["Zola", "Maupassant", "GPT-4"],
    title="Do LLMs form a distinct stylistic cluster?",
)
```

### `plot_shift_distribution(originals, rewrites_dict)`

Box plot of cosine shifts per model.

```python
fig = sa.plot_shift_distribution(
    originals=original_texts,
    rewrites_dict={
        "GPT-4": gpt4_rewrites,
        "Claude 3": claude_rewrites,
    },
)
```

---

## Code Stylometry

Apply stylometry to source code. Measure developer fingerprints.

```python
from stylometry.code import CodeAnalyzer

ca = CodeAnalyzer()

# Fit on known code samples
ca.fit(alice_code_files, label="Alice")
ca.fit(bob_code_files, label="Bob")

# Attribute an unknown file
predicted, distances = ca.predict(unknown_file)
print(f"Predicted author: {predicted}")

# Detect Copilot patterns
copilot_score = ca.copilot_score(code_file)
print(f"Copilot likelihood: {copilot_score:.2f}")
```

**Code features measured:**

| Feature | Description |
|---------|-------------|
| `camelCase_ratio` | Fraction of identifiers in camelCase |
| `snake_case_ratio` | Fraction of identifiers in snake_case |
| `comment_density` | Comment lines / total non-empty lines |
| `docstring_density` | Docstring occurrences / non-empty lines |
| `type_hint_usage` | Type annotations per line |
| `list_comp_usage` | List comprehensions per line |
| `avg_line_length` | Average line length (normalized) |
| `blank_line_ratio` | Blank lines / total lines |

---

## Examples

See the `examples/` directory:

- `examples/rowling.py` — Reproduce the Rowling identification experiment
- `examples/llm_shift.py` — Measure GPT-4 stylistic shift on your own texts
- `examples/code_attribution.py` — Attribute code files to developers

```bash
cd examples
python rowling.py
# → Most likely author: Rowling (distance: 0.18)
# → Second closest:     Rendell (distance: 0.31)
```

---

## Limitations

Stylometry provides **probabilistic signals**, not forensic proof.

- Minimum ~100 words per text for reliable results
- Function word analysis is language-dependent
- Cross-domain generalization degrades significantly
- LLM detection is prompt-dependent and model-dependent

See [LIMITATIONS.md](LIMITATIONS.md) for a full discussion.

---

## References

- Mosteller & Wallace (1964). *Inference and Disputed Authorship: The Federalist.*
- Juola (2015). *The Rowling Case.* DSH, Oxford.
- Stamatatos (2009). *A Survey of Modern Authorship Attribution Methods.* JASIST.
- Kestemont et al. (2020). *PAN @ CLEF 2020 Authorship Verification.*
- Caliskan et al. (2015). *De-anonymizing Programmers via Code Stylometry.* USENIX.

---

## License

MIT
