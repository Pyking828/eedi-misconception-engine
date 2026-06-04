"""Unit tests: listwise parsing"""

from src.eedi.reranker.listwise import build_listwise_prompt, parse_listwise_output


def test_parse_perfect():
    assert parse_listwise_output("A, B, C, D", 4) == [0, 1, 2, 3]


def test_parse_reversed():
    assert parse_listwise_output("D, C, B, A", 4) == [3, 2, 1, 0]


def test_parse_partial():
    result = parse_listwise_output("B, A", 4)
    assert result[0] == 1
    assert result[1] == 0
    assert set(result) == {0, 1, 2, 3}  # pad missing ranks


def test_parse_with_noise():
    result = parse_listwise_output("The answer is: A, C, B", 3)
    assert result[0] == 0
    assert result[1] == 2


def test_build_prompt_contains_candidates():
    prompt = build_listwise_prompt(
        query="Question about math",
        candidates=["Misconception A", "Misconception B"],
    )
    assert "Misconception A" in prompt
    assert "Misconception B" in prompt
