"""
Tests for CodeAnalyzer and StyleProfile.
Run: pytest tests/ -v
"""

import numpy as np
import pytest
from stylometry.code import (
    CodeAnalyzer,
    StyleProfile,
    CODE_FEATURES,
    SUPPORTED_LANGUAGES,
    copilot_score,
    extract_features,
)

# ---------------------------------------------------------------------------
# Sample code fixtures
# ---------------------------------------------------------------------------

PYTHON_HUMAN = """
# Sort elements
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

PYTHON_LLM = '''
def sort_elements(
        elements: list,
        reverse: bool = False
) -> list:
    """
    Sort a list of elements in ascending or descending order.

    Args:
        elements: The list to sort.
        reverse: If True, sort in descending order.

    Returns:
        Sorted list, or empty list if input is None or empty.
    """
    if not elements:
        return []
    try:
        return sorted(elements, reverse=reverse)
    except TypeError as error:
        raise ValueError(f"Elements must be comparable: {error}") from error
'''

TYPESCRIPT_HUMAN = """
function fetchUser(id) {
    return db.find(id);
}

const processItems = (lst) => {
    let res = [];
    for (let i of lst) {
        if (i > 0) res.push(i * 2);
    }
    return res;
}
"""

TYPESCRIPT_LLM = """
/**
 * Fetches a user from the database by their unique identifier.
 * @param userId - The unique identifier of the user to retrieve.
 * @returns Promise resolving to the User object, or null if not found.
 */
async function fetchUserById(userId: string): Promise<User | null> {
    try {
        const user = await database.findById(userId);
        return user ?? null;
    } catch (error: unknown) {
        logger.error(`Failed to fetch user ${userId}:`, error);
        throw new DatabaseError(`User lookup failed: ${error}`);
    }
}
"""

GO_HUMAN = """
func getUser(id int) (*User, error) {
    u, err := db.Find(id)
    if err != nil {
        return nil, err
    }
    return u, nil
}

func process(items []int) []int {
    result := []int{}
    for _, v := range items {
        if v > 0 {
            result = append(result, v*2)
        }
    }
    return result
}
"""


# ---------------------------------------------------------------------------
# CodeAnalyzer — basic contract
# ---------------------------------------------------------------------------

class TestCodeAnalyzerContract:

    def test_extract_features_returns_all_keys(self):
        ca = CodeAnalyzer()
        f = ca.extract_features(PYTHON_HUMAN)
        assert set(f.keys()) == set(CODE_FEATURES)

    def test_all_features_in_unit_range(self):
        ca = CodeAnalyzer()
        for code in [PYTHON_HUMAN, PYTHON_LLM]:
            f = ca.extract_features(code)
            for k, v in f.items():
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"

    def test_vectorize_is_normalized(self):
        ca = CodeAnalyzer()
        v = ca.vectorize(PYTHON_HUMAN)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_vector_dimension_matches_features(self):
        ca = CodeAnalyzer()
        v = ca.vectorize(PYTHON_HUMAN)
        assert len(v) == len(CODE_FEATURES)

    def test_min_lines_raises(self):
        ca = CodeAnalyzer(min_lines=100)
        with pytest.raises(ValueError, match="too short"):
            ca.extract_features(PYTHON_HUMAN)

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            CodeAnalyzer(language="cobol")

    def test_repr(self):
        ca = CodeAnalyzer(language="go")
        assert "go" in repr(ca)


# ---------------------------------------------------------------------------
# Feature discrimination — Python
# ---------------------------------------------------------------------------

class TestPythonFeatures:

    def test_camelcase_higher_in_human(self):
        ca = CodeAnalyzer()
        assert ca.extract_features(PYTHON_HUMAN)["camelCase_ratio"] > \
               ca.extract_features(PYTHON_LLM)["camelCase_ratio"]

    def test_type_hints_higher_in_llm(self):
        ca = CodeAnalyzer()
        assert ca.extract_features(PYTHON_LLM)["type_hint_usage"] > \
               ca.extract_features(PYTHON_HUMAN)["type_hint_usage"]

    def test_docstring_completeness_higher_in_llm(self):
        ca = CodeAnalyzer()
        assert ca.extract_features(PYTHON_LLM)["docstring_completeness"] > \
               ca.extract_features(PYTHON_HUMAN)["docstring_completeness"]

    def test_error_handling_higher_in_llm(self):
        ca = CodeAnalyzer()
        assert ca.extract_features(PYTHON_LLM)["error_handling_density"] > \
               ca.extract_features(PYTHON_HUMAN)["error_handling_density"]

    def test_identifier_verbosity_in_range(self):
        # Verbosity = avg identifier length / 20; just verify range, not direction
        # (camelCase human names can be as long as descriptive LLM snake_case names)
        ca = CodeAnalyzer()
        for code in [PYTHON_HUMAN, PYTHON_LLM]:
            v = ca.extract_features(code)["identifier_verbosity"]
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Multi-language support
# ---------------------------------------------------------------------------

class TestMultiLanguage:

    def test_typescript_extracts_features(self):
        ca = CodeAnalyzer(language="typescript")
        f = ca.extract_features(TYPESCRIPT_LLM)
        assert set(f.keys()) == set(CODE_FEATURES)

    def test_typescript_all_features_in_range(self):
        ca = CodeAnalyzer(language="typescript")
        for code in [TYPESCRIPT_HUMAN, TYPESCRIPT_LLM]:
            f = ca.extract_features(code)
            for k, v in f.items():
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_typescript_docstring_higher_in_llm(self):
        ca = CodeAnalyzer(language="typescript")
        assert ca.extract_features(TYPESCRIPT_LLM)["docstring_completeness"] > \
               ca.extract_features(TYPESCRIPT_HUMAN)["docstring_completeness"]

    def test_typescript_error_handling_higher_in_llm(self):
        ca = CodeAnalyzer(language="typescript")
        assert ca.extract_features(TYPESCRIPT_LLM)["error_handling_density"] > \
               ca.extract_features(TYPESCRIPT_HUMAN)["error_handling_density"]

    def test_go_extracts_features(self):
        ca = CodeAnalyzer(language="go")
        f = ca.extract_features(GO_HUMAN)
        assert set(f.keys()) == set(CODE_FEATURES)

    def test_all_supported_languages_instantiate(self):
        for lang in SUPPORTED_LANGUAGES:
            ca = CodeAnalyzer(language=lang)
            assert ca.language == lang

    def test_from_extension_python(self, tmp_path):
        f = tmp_path / "foo.py"
        f.write_text(PYTHON_HUMAN)
        ca = CodeAnalyzer.from_extension(f)
        assert ca.language == "python"

    def test_from_extension_typescript(self, tmp_path):
        f = tmp_path / "foo.ts"
        f.write_text(TYPESCRIPT_LLM)
        ca = CodeAnalyzer.from_extension(f)
        assert ca.language == "typescript"

    def test_detect_language(self):
        assert CodeAnalyzer.detect_language("main.go") == "go"
        assert CodeAnalyzer.detect_language("app.rs") == "rust"
        assert CodeAnalyzer.detect_language("unknown.xyz") == "python"


# ---------------------------------------------------------------------------
# copilot_score
# ---------------------------------------------------------------------------

class TestCopilotScore:

    def test_score_in_range(self):
        ca = CodeAnalyzer()
        for code in [PYTHON_HUMAN, PYTHON_LLM]:
            assert 0.0 <= ca.copilot_score(code) <= 1.0

    def test_llm_code_scores_higher(self):
        ca = CodeAnalyzer()
        assert ca.copilot_score(PYTHON_LLM) > ca.copilot_score(PYTHON_HUMAN)

    def test_typescript_llm_scores_higher(self):
        ca = CodeAnalyzer(language="typescript")
        assert ca.copilot_score(TYPESCRIPT_LLM) > ca.copilot_score(TYPESCRIPT_HUMAN)

    def test_too_short_returns_zero(self):
        ca = CodeAnalyzer(min_lines=100)
        assert ca.copilot_score("x = 1") == 0.0

    def test_module_level_copilot_score(self):
        s = copilot_score(PYTHON_LLM)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

class TestAttribution:

    def test_predict_correct_author(self):
        ca = CodeAnalyzer()
        ca.fit([PYTHON_HUMAN] * 3, "human")
        ca.fit([PYTHON_LLM] * 3, "llm")
        predicted, _ = ca.predict(PYTHON_HUMAN)
        assert predicted == "human"

    def test_predict_no_centroids_raises(self):
        ca = CodeAnalyzer()
        with pytest.raises(RuntimeError, match="No centroids"):
            ca.predict(PYTHON_HUMAN)

    def test_feature_table_returns_string(self):
        ca = CodeAnalyzer()
        table = ca.feature_table({"human": [PYTHON_HUMAN], "llm": [PYTHON_LLM]})
        assert isinstance(table, str)
        assert "comment_density" in table


# ---------------------------------------------------------------------------
# StyleProfile
# ---------------------------------------------------------------------------

class TestStyleProfile:

    def test_add_and_features(self):
        p = StyleProfile("dev", language="python")
        p.add(PYTHON_HUMAN)
        assert p.n_files == 1
        assert p.features is not None
        assert set(p.features.keys()) == set(CODE_FEATURES)

    def test_copilot_score_higher_for_llm_profile(self):
        human_p = StyleProfile("human").add_files([PYTHON_HUMAN] * 3)
        llm_p = StyleProfile("llm").add_files([PYTHON_LLM] * 3)
        assert llm_p.copilot_score > human_p.copilot_score

    def test_empty_profile_returns_none(self):
        p = StyleProfile("empty")
        assert p.centroid is None
        assert p.features is None
        assert p.copilot_score == 0.0

    def test_language_param_propagates(self):
        p = StyleProfile("dev", language="typescript")
        assert p._analyzer.language == "typescript"

    def test_repr(self):
        p = StyleProfile("alice", language="go")
        assert "alice" in repr(p)
        assert "go" in repr(p)
