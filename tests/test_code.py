"""
Tests for CodeAnalyzer.
Run: pytest tests/ -v
"""

import numpy as np
from stylometry.code import CodeAnalyzer, CODE_FEATURES

DEV_A = """
# Tri des elements d une liste
def sortElements(inputList, reverseOrder):
    if inputList == None or len(inputList) == 0:
        return []
    resultList = []
    for i in range(len(inputList)):
        resultList.append(inputList[i])
    for i in range(len(resultList)):
        for j in range(i + 1, len(resultList)):
            if resultList[i] > resultList[j]:
                tempValue = resultList[i]
                resultList[i] = resultList[j]
                resultList[j] = tempValue
    return resultList
"""

DEV_B = """
def sort_elements(lst, reverse=False):
    if not lst:
        return []
    return sorted(lst, reverse=reverse)
"""

COPILOT = '''
def sort_elements(
        elements: list,
        reverse: bool = False
) -> list:
    """
    Sort a list of elements.
    Args:
        elements: The list to sort.
        reverse: Descending if True.
    Returns:
        Sorted list, or empty list if input is empty.
    """
    if not elements:
        return []
    return sorted(elements, reverse=reverse)
'''


class TestCodeAnalyzer:

    def test_extract_features_returns_dict(self):
        ca = CodeAnalyzer()
        f = ca.extract_features(DEV_A)
        assert isinstance(f, dict)

    def test_extract_features_all_keys(self):
        ca = CodeAnalyzer()
        f = ca.extract_features(DEV_A)
        assert set(f.keys()) == set(CODE_FEATURES)

    def test_features_in_range(self):
        ca = CodeAnalyzer()
        for code in [DEV_A, DEV_B, COPILOT]:
            f = ca.extract_features(code)
            for k, v in f.items():
                assert 0.0 <= v <= 1.0, f"{k} = {v} out of range"

    def test_camelcase_higher_in_dev_a(self):
        ca = CodeAnalyzer()
        fa = ca.extract_features(DEV_A)
        fb = ca.extract_features(DEV_B)
        assert fa["camelCase_ratio"] > fb["camelCase_ratio"]

    def test_type_hints_higher_in_copilot(self):
        ca = CodeAnalyzer()
        fa = ca.extract_features(DEV_A)
        fc = ca.extract_features(COPILOT)
        assert fc["type_hint_usage"] > fa["type_hint_usage"]

    def test_docstrings_higher_in_copilot(self):
        ca = CodeAnalyzer()
        fa = ca.extract_features(DEV_A)
        fc = ca.extract_features(COPILOT)
        assert fc["docstring_density"] > fa["docstring_density"]

    def test_copilot_score_higher_for_copilot(self):
        ca = CodeAnalyzer()
        score_a = ca.copilot_score(DEV_A)
        score_c = ca.copilot_score(COPILOT)
        assert score_c > score_a

    def test_copilot_score_in_range(self):
        ca = CodeAnalyzer()
        for code in [DEV_A, DEV_B, COPILOT]:
            s = ca.copilot_score(code)
            assert 0.0 <= s <= 1.0

    def test_predict_correct_author(self):
        ca = CodeAnalyzer()
        ca.fit([DEV_A] * 3, "Dev A")
        ca.fit([DEV_B] * 3, "Dev B")
        predicted, _ = ca.predict(DEV_A)
        assert predicted == "Dev A"

    def test_vectorize_normalized(self):
        ca = CodeAnalyzer()
        v = ca.vectorize(DEV_A)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6
