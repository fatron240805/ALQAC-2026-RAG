from app import prompts


def test_prompt_version():
    assert prompts.PROMPT_VERSION == "alqac-v3"


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
    assert "Không tự tạo" in text
    assert "dự thảo tư vấn pháp lý" in text


def test_manager_route_order():
    text = prompts.MANAGER_SYSTEM
    assert "format_check" in text
    assert "law_search" in text
    assert "pass" in text
    assert "format_check` trước, rồi `law_search" in text
    assert "hệ thống tư vấn pháp lý nhiều vai trò" in text


def test_format_check_no_meaning_change():
    assert "không thay đổi ý nghĩa pháp lý" in prompts.FORMAT_CHECK_SYSTEM


def test_law_search_authoritative():
    assert "quy định pháp luật" in prompts.LAW_SEARCH_SYSTEM
    assert "có thẩm quyền" in prompts.LAW_SEARCH_SYSTEM


def test_content_check_pass_fail_gate():
    text = prompts.CONTENT_CHECK_SYSTEM
    assert "pass" in text and "fail" in text
    assert "Không tự tạo" in text
    assert "ý kiến pháp lý lưu loát, chuyên nghiệp" in text


def test_evidence_prompts_have_provenance_rules():
    for name in ("DRAFT_SYSTEM", "FORMAT_CHECK_SYSTEM", "CONTENT_CHECK_SYSTEM"):
        assert "Ràng buộc bằng chứng" in prompts.all_prompt_texts()[name]
