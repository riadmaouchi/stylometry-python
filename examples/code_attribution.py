"""Example: attribute an unknown code snippet to a developer profile."""

from stylometry.code import CodeAnalyzer


def main() -> None:
    analyzer = CodeAnalyzer(min_lines=4)

    alice_samples = [
        """
def sortItems(valuesList):
    if valuesList == None:
        return []
    sortedList = sorted(valuesList)
    return sortedList
""",
        """
def mergeItems(itemListA, itemListB):
    mergedList = itemListA + itemListB
    uniqueItems = list(dict.fromkeys(mergedList))
    return uniqueItems
""",
    ]

    bob_samples = [
        """
def sort_items(values, reverse=False):
    if not values:
        return []
    sorted_values = sorted(values, reverse=reverse)
    return sorted_values
""",
        """
def merge_items(items_a, items_b):
    merged_items = items_a + items_b
    unique_items = list(dict.fromkeys(merged_items))
    return unique_items
""",
    ]

    unknown = """
def mergeItems(leftItems, rightItems):
    allItems = leftItems + rightItems
    cleanItems = list(dict.fromkeys(allItems))
    return cleanItems
"""

    analyzer.fit(alice_samples, "Alice")
    analyzer.fit(bob_samples, "Bob")

    predicted, distances = analyzer.predict(unknown)

    print(f"Predicted author: {predicted}")
    for label, distance in sorted(distances.items(), key=lambda item: item[1]):
        print(f"  - {label}: {distance:.4f}")


if __name__ == "__main__":
    main()
