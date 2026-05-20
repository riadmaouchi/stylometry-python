"""Example: generate stylometry visualizations as PNG files."""

from pathlib import Path

from stylometry import StyleAnalyzer


def main() -> None:
    analyzer = StyleAnalyzer(language="en", min_words=20)

    author_a = [
        (
            "The rain fell softly over the old street and the windows stayed dark. "
            "I waited near the station gate and counted each passing shadow in silence."
        ),
        (
            "She opened the letter slowly, then folded it twice and put it away. "
            "No one spoke while the train lights moved across the platform wall."
        ),
    ]

    author_b = [
        (
            "In the morning the office filled quickly and everyone checked the board. "
            "We reviewed the numbers again and updated the plan before lunch."
        ),
        (
            "They discussed the project timeline in short clear sentences. "
            "By evening the final version was approved and ready for release."
        ),
    ]

    rewrites = [
        (
            "The rain arrived gently above the old avenue while most windows remained dark. "
            "I stood beside the station entrance and observed movement in the distance."
        ),
        (
            "She read the note in silence, folded it, and kept it in her coat pocket. "
            "Across the platform, lights traced brief patterns over the concrete wall."
        ),
    ]

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint_fig = analyzer.plot_fingerprint(
        texts_dict={
            "Author A": author_a,
            "Author B": author_b,
            "Rewrite": rewrites,
        },
        top_n=10,
        title="Function-Word Fingerprints",
    )
    fingerprint_path = output_dir / "fingerprints.png"
    fingerprint_fig.savefig(fingerprint_path, dpi=150)

    clusters_fig = analyzer.plot_clusters(
        texts_groups=[author_a, author_b, rewrites],
        labels=["Author A", "Author B", "Rewrite"],
        title="Stylometric Clusters",
    )
    clusters_path = output_dir / "clusters.png"
    clusters_fig.savefig(clusters_path, dpi=150)

    print(f"Saved: {fingerprint_path}")
    print(f"Saved: {clusters_path}")


if __name__ == "__main__":
    main()
