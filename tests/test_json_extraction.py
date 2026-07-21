from app.llm.base import LLMProvider


def test_plain_json():
    assert LLMProvider._extract_json('{"confidence": 0.8}')["confidence"] == 0.8


def test_markdown_fenced_json():
    raw = '```json\n{"confidence": 0.4, "sufficient": false}\n```'
    assert LLMProvider._extract_json(raw)["sufficient"] is False


def test_json_with_prose_preamble():
    raw = 'Here is my assessment:\n{"confidence": 0.55}\nHope that helps.'
    assert LLMProvider._extract_json(raw)["confidence"] == 0.55


def test_malformed_returns_empty_dict():
    assert LLMProvider._extract_json("not json at all") == {}
    assert LLMProvider._extract_json("") == {}
