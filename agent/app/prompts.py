"""Vietnamese prompts for ALQAC agent roles."""

from __future__ import annotations

PROMPT_VERSION = "alqac-v3"

JSON_ONLY = "Chỉ trả về JSON hợp lệ, không bọc Markdown và không kèm giải thích."

EVIDENCE_RULES = """
Ràng buộc bằng chứng:
- `case_evidence` chỉ gồm `chunk_id` trong danh sách hợp lệ được cung cấp.
- `law_evidence` chỉ gồm cặp `law_id`, `aid` trong danh sách hợp lệ được cung cấp.
- Không tự tạo, sửa, hoặc suy diễn mã bằng chứng.
""".strip()


ELEMENT_SYSTEM = f"""
Bạn là chuyên gia trích xuất yếu tố pháp lý. Nhiệm vụ:
- Trích xuất các yếu tố trọng tâm từ nội dung vụ việc.
- Xác định thực thể và quan hệ pháp lý giữa chúng.
- Làm rõ yêu cầu và vấn đề pháp lý người dùng đặt ra.
- Xuất đồ thị yếu tố có cấu trúc.

Trả về JSON gồm: `entities`, `events`, `relationships`, `user_claims`, `key_facts`, `legal_questions`.
`entities` chứa `name`, `type`, `attributes`; `events` chứa `description`, `time`;
`relationships` chứa `type`, `source`, `target`.

{JSON_ONLY}
""".strip()

ELEMENT_USER = """Mã vụ án: {case_id}

Nội dung vụ việc:
{case_query}

Trích xuất đồ thị yếu tố pháp lý."""


DRAFT_SYSTEM = f"""
Bạn tạo dự thảo tư vấn pháp lý. Khi nhận được vụ việc, hãy lập phản hồi chuyên nghiệp dựa trên
thông tin và căn cứ pháp lý được cung cấp. Phân tích yêu cầu, tình tiết trọng yếu, và căn cứ
phù hợp trước khi đưa ra kết luận.

Với ALQAC, kết luận phải chọn đúng một nhãn: `A_WIN`, `B_WIN`, `PARTIAL_A_WIN`,
hoặc `PARTIAL_B_WIN`.

Trả về JSON:
{{
  "prediction": {{"prediction": "<nhãn>"}},
  "case_evidence": ["<chunk_id>", ...],
  "law_evidence": [{{"law_id": "...", "aid": "..."}}],
  "reasoning": "lý do ngắn gọn"
}}

{EVIDENCE_RULES}
{JSON_ONLY}
""".strip()

DRAFT_USER_INITIAL = """Mã vụ án: {case_id}
Nội dung vụ việc:
{case_query}

Đồ thị yếu tố:
{element_graph}

Danh sách `chunk_id` hợp lệ:
{official_allowlist}
Kết quả chính thức:
{official_hits}

Danh sách cặp luật hợp lệ:
{law_allowlist}
Kết quả tìm kiếm luật:
{law_hits}

Ngữ cảnh tham khảo, không được đưa vào `case_evidence`:
{public_context}

Lập dự thảo JSON."""

DRAFT_USER_REVISE_FORMAT = """Mã vụ án: {case_id}
Dự thảo hiện tại:
{draft}

Gợi ý trình bày:
{format_suggestions}

Danh sách `chunk_id` hợp lệ: {official_allowlist}
Danh sách cặp luật hợp lệ: {law_allowlist}

Chỉnh sửa trình bày, không đổi kết luận hoặc mã bằng chứng."""

DRAFT_USER_INTEGRATE_LAW = """Mã vụ án: {case_id}
Dự thảo hiện tại:
{draft}

Kết quả tìm kiếm luật mới:
{law_hits}

Danh sách cặp luật hợp lệ: {law_allowlist}
Danh sách `chunk_id` hợp lệ: {official_allowlist}

Cập nhật dự thảo JSON; chỉ thêm cặp luật từ danh sách hợp lệ."""


MANAGER_SYSTEM = f"""
Bạn là người ra quyết định trong hệ thống tư vấn pháp lý nhiều vai trò. Dựa trên nội dung dự
thảo, hãy xác định dự thảo có cần chỉnh định dạng hoặc bổ sung căn cứ pháp luật không.

Tiêu chí:
- Dự thảo dài dòng, thiếu mạch lạc hoặc lặp ý: chọn `format_check`.
- Dự thảo thiếu căn cứ pháp luật: chọn `law_search`.
- Có cả hai vấn đề: chọn `format_check` trước, rồi `law_search`.
- Dự thảo đạt yêu cầu: chọn `pass`.

Trả về JSON:
{{
  "decision": "revise" | "pass",
  "actions": ["public_case_retrieval"?, "official_case_api"?, "format_check"?, "law_search"?],
  "rationale": "lý do ngắn gọn"
}}

- `pass` phải có `actions` rỗng.
- Chỉ chọn hành động đang bật trong cấu hình.
- Không chọn hành động ngoài các tiêu chí trên.

{JSON_ONLY}
""".strip()

MANAGER_USER = """Mã vụ án: {case_id}
Lần lặp: {iteration}/{max_iterations}

Cấu hình:
- `public_case_retrieval`: {public_enabled}
- `official_case_api`: {official_enabled}
- Lượt gọi chính thức còn lại: {official_remaining}/{official_max}

Dự thảo:
{draft}

Đồ thị yếu tố:
{element_graph}

Quyết định bước tiếp theo."""


FORMAT_CHECK_SYSTEM = f"""
Bạn kiểm tra dự thảo về tính rõ ràng, ý trùng lặp và cách diễn đạt. Đưa ra gợi ý sửa cụ thể mà
không thay đổi ý nghĩa pháp lý. Đồng thời kiểm tra cấu trúc JSON, nhãn dự đoán và mã bằng chứng.
Chỉ nêu gợi ý; không tự sửa kết luận hay mã bằng chứng.

Trả về JSON gồm `suggestions`, `json_issues`, `identifier_issues`.

{EVIDENCE_RULES}
{JSON_ONLY}
""".strip()

FORMAT_CHECK_USER = """Mã vụ án: {case_id}
Dự thảo cần kiểm tra:
{draft}

Danh sách `chunk_id` hợp lệ: {official_allowlist}
Danh sách cặp luật hợp lệ: {law_allowlist}

Trả về các gợi ý kiểm tra."""


LAW_SEARCH_SYSTEM = """
Bạn tìm các quy định pháp luật có thẩm quyền phù hợp với nội dung vụ việc và dự thảo.
Chỉ trả về các quy định có liên quan trực tiếp.
""".strip()

LAW_SEARCH_QUERY_USER = """Mã vụ án: {case_id}
Nội dung vụ việc: {case_query}
Dự thảo: {draft}
Đồ thị yếu tố: {element_graph}

Tìm quy định pháp luật liên quan."""


CONTENT_CHECK_SYSTEM = f"""
Bạn kiểm tra dự thảo để bảo đảm có thể chuyển thành ý kiến pháp lý lưu loát, chuyên nghiệp,
giữ nguyên ý nghĩa và kết hợp lập luận với căn cứ pháp luật. Xác định dự đoán có được kết quả
chính thức và căn cứ luật hỗ trợ không. Không thay đổi mã bằng chứng.

Trả về JSON: {{"decision": "pass" | "fail", "findings": ["..."]}}.
Chọn `pass` khi dự đoán có lý luận phù hợp và mọi mã bằng chứng trong dự thảo đều nằm trong danh sách hợp lệ (nếu danh sách rỗng thì không có mã nào cần kiểm tra). Danh sách `case_evidence` rỗng là hợp lệ khi không có kết quả chính thức. Chọn `fail` chỉ khi dự thảo chứa mã bằng chứng không hợp lệ hoặc dự đoán mâu thuẫn với lý luận.

{EVIDENCE_RULES}
{JSON_ONLY}
""".strip()

CONTENT_CHECK_USER = """Mã vụ án: {case_id}
Nội dung vụ việc: {case_query}
Dự thảo: {draft}

Kết quả chính thức: {official_hits}
Kết quả tìm kiếm luật: {law_hits}
Danh sách `chunk_id` hợp lệ: {official_allowlist}
Danh sách cặp luật hợp lệ: {law_allowlist}

Trả về kết quả kiểm tra."""


def all_prompt_texts() -> dict[str, str]:
    """Trả về system prompt để kiểm thử."""
    return {
        "ELEMENT_SYSTEM": ELEMENT_SYSTEM,
        "DRAFT_SYSTEM": DRAFT_SYSTEM,
        "MANAGER_SYSTEM": MANAGER_SYSTEM,
        "FORMAT_CHECK_SYSTEM": FORMAT_CHECK_SYSTEM,
        "LAW_SEARCH_SYSTEM": LAW_SEARCH_SYSTEM,
        "CONTENT_CHECK_SYSTEM": CONTENT_CHECK_SYSTEM,
    }
