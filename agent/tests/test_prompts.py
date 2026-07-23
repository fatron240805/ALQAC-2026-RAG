from app import prompts


def test_prompt_version_and_source():
    assert "table8" in prompts.PROMPT_VERSION
    assert "2604.10470" in prompts.PAPER_SOURCE


def test_element_prompt_has_table8_fields():
    text = prompts.ELEMENT_SYSTEM
    for field in (
        "entities",
        "events",
        "relationships",
        "user_claims",
        "key_facts",
        "legal_questions",
    ):
        assert field in text
    assert "chuyên gia trích xuất yếu tố pháp lý" in text


def test_draft_is_label_not_prose_only():
    text = prompts.DRAFT_SYSTEM
    assert "A_WIN" in text
    assert "B_WIN" in text
    assert "PARTIAL_A_WIN" in text
    assert "PARTIAL_B_WIN" in text
    assert "KHÔNG ĐƯỢC tạo identifier" in text or "MUST NOT create" in text


def test_manager_route_order():
    text = prompts.MANAGER_SYSTEM
    assert "FormatCheckAgent" in text
    assert "LawSearchAgent" in text
    assert "Pass" in text
    assert "rồi LawSearchAgent" in text or "FormatCheckAgent" in text


def test_format_check_no_meaning_change():
    assert "không thay đổi ý nghĩa pháp lý" in prompts.FORMAT_CHECK_SYSTEM


def test_law_search_authoritative():
    assert "quy định pháp luật có thẩm quyền" in prompts.LAW_SEARCH_SYSTEM
    assert "law_id" in prompts.LAW_SEARCH_SYSTEM


def test_content_check_pass_fail_gate():
    text = prompts.CONTENT_CHECK_SYSTEM
    assert "pass" in text and "fail" in text
    assert "không được thêm" in text.lower() or "KHÔNG" in text


def test_all_prompts_have_provenance_rules():
    for name, text in prompts.all_prompt_texts().items():
        assert "QUY TẮC NGUỒN GỐC ALQAC" in text or "case_evidence" in text, name
