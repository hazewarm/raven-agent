from raven_agent.memory2.tokenizer import extract_terms


def test_extract_terms_segments_chinese_query() -> None:
    terms = extract_terms("我今天路过了上海市长江大桥")

    assert terms
    assert "上海市" in terms or "长江大桥" in terms or "长江" in terms

def test_extract_terms_segments_mixed_query() -> None:
    terms = extract_terms("我在用 sqlite vec 和 Fitbit Charge 6")
    lowered = {term.lower() for term in terms}

    assert "sqlite" in lowered
    assert "fitbit" in lowered

def test_extract_terms_respects_limit() -> None:
    terms = extract_terms("回答风格 简洁 详细 Fitbit Charge sqlite memory", limit=3)

    assert len(terms) <= 3