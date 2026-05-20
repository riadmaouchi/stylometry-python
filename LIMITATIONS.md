# Limitations

Stylometry provides probabilistic signals, not definitive proof.

## Core constraints

- Reliable attribution generally requires at least 100 words per text sample.
- Short texts produce unstable vectors and lower confidence.
- Results depend on language-specific function-word lists.
- Cross-domain transfer is limited (for example: novels -> emails, code comments -> docs).

## Authorship attribution caveats

- Nearest-centroid attribution can be biased when training corpora are small.
- Similar genre or topic can reduce separability between authors.
- Heavy editing by third parties may distort the original stylistic fingerprint.

## LLM shift / detection caveats

- Stylistic shift scores are model-dependent and prompt-dependent.
- A low shift does not guarantee human-only writing.
- A high shift does not prove machine generation.
- Post-editing of LLM output can mask model signatures.

## Code stylometry caveats

- Very short snippets may not contain enough stylistic signals.
- Team conventions (formatter, linter, templates) can flatten individual style.
- Boilerplate-heavy repositories reduce author-specific variation.

## Visualization caveats

- PCA projections are lossy and should be interpreted as exploratory visuals.
- Visual cluster overlap does not always imply attribution failure.

## Best practices

- Use multiple samples per author.
- Keep evaluation and reference corpora in comparable domains.
- Report confidence and distance values, not just predicted labels.
- Combine stylometry with external evidence in real-world investigations.
