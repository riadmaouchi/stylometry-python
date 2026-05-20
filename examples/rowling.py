"""Example: simple authorship attribution inspired by the Rowling case."""

from stylometry import StyleAnalyzer


def main() -> None:
    analyzer = StyleAnalyzer(language="en", min_words=30)

    rowling_texts = [
        (
            "He walked the narrow street in silence, counting windows and doors, "
            "watching shadows move behind curtains while rain tapped softly on stone. "
            "Every sound seemed deliberate, as if the city itself had chosen to whisper "
            "secrets only to those who waited long enough to listen."
        ),
        (
            "She paused at the gate, then smiled at the old sign above the shop. "
            "Inside, dust and warm light settled over shelves of forgotten things, "
            "and with each step she felt memory and fear intertwine like threads in "
            "a story that refused to end."
        ),
    ]

    rendell_texts = [
        (
            "The room was too bright for the hour, and yet no one moved to close the blind. "
            "They spoke in clipped sentences, careful not to name what everyone already knew, "
            "while the clock marked each second like a warning no one wished to hear."
        ),
        (
            "He folded the letter twice before placing it in his pocket, then stood by the "
            "window and watched the road empty itself of people. The quiet that followed was "
            "not peace, but a pause before consequence."
        ),
    ]

    unknown_text = (
        "She moved through the narrow street in silence, watching light and shadow shift "
        "behind the windows while rain tapped against stone. Every detail seemed deliberate, "
        "and the city felt like a story unfolding in fragments she could almost read."
    )

    analyzer.fit(rowling_texts, "Rowling").fit(rendell_texts, "Rendell")
    predicted, distances = analyzer.predict(unknown_text)

    ranking = sorted(distances.items(), key=lambda item: item[1])
    best_label, best_distance = ranking[0]
    second_label, second_distance = ranking[1]

    print(f"Most likely author: {predicted} (distance: {best_distance:.2f})")
    print(f"Second closest:     {second_label} (distance: {second_distance:.2f})")


if __name__ == "__main__":
    main()
