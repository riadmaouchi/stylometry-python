"""Example: measure stylistic shift between originals and rewrites."""

from stylometry import StyleAnalyzer


def main() -> None:
    analyzer = StyleAnalyzer(language="en", min_words=30)

    original = (
        "I arrived just before dawn, when the station was almost empty and the lights buzzed "
        "with that tired, metallic hum. I bought a coffee I did not need, sat near the last "
        "platform, and watched travelers drift in small uncertain groups toward departures."
    )

    rewrite = (
        "I reached the station at first light while most platforms remained quiet and only a "
        "few passengers moved through the hall. I purchased coffee, found a seat near the end, "
        "and observed people gather in loose groups as departure boards changed above them."
    )

    shift = analyzer.shift(original, rewrite)
    print(f"Stylistic shift: {shift:.4f}")


if __name__ == "__main__":
    main()
