"""Bootstrap data/local_validation_gold.json từ case_id THẬT trong ALQAC2026_public_test.json.

Vấn đề đã phát hiện: local_validation_gold.json hiện tại chỉ có 2 case dummy
("0001", "0002") không tồn tại trong public test thật (case_id dạng "case_XXXX")
-> evaluate_alqac_system() luôn báo THIẾU CASE_ID và Final Score = 0.0000.

Script này lấy mẫu N case_id THẬT bằng reservoir sampling (Algorithm R) — duyệt
qua ijson theo kiểu streaming, KHÔNG load toàn bộ public test (có thể hàng nghìn
case) vào RAM cùng lúc, xuất ra 1 file template để Thịnh/Tuấn Anh điền tay
label + law_evidence + case_evidence theo Plan.md Phase 0.

Usage:
    python -m evaluation.scaffold_gold_labels --n 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

try:
    import ijson  # streaming JSON parser — tránh load cả file public test vào RAM
except ImportError:
    ijson = None


def iter_public_test_cases(path: Path) -> Iterator[dict[str, Any]]:
    """Duyệt từng case một trong ALQAC2026_public_test.json theo kiểu streaming.

    Dùng ijson.items() nếu có (streaming thật, O(1) memory theo item), nếu
    không có ijson thì fallback về json.load() (load hết vào RAM — chấp nhận
    được cho quy mô public test hiện tại, nhưng nên cài ijson nếu file lớn).
    """
    if ijson is not None:
        with path.open("rb") as handle:
            # payload có thể là list thuần hoặc {"cases": [...]}; thử cả 2 prefix.
            found_any = False
            for prefix in ("item", "cases.item", "data.item"):
                handle.seek(0)
                try:
                    for case in ijson.items(handle, prefix):
                        found_any = True
                        yield case
                    if found_any:
                        return
                except ijson.JSONError:
                    continue
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else (payload.get("cases") or payload.get("data") or [])
    yield from items


def reservoir_sample(items: Iterator[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Algorithm R: lấy mẫu ngẫu nhiên k phần tử từ 1 stream có độ dài chưa biết
    trong 1 lần duyệt (O(k) memory, không cần biết trước tổng số case).
    """
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def extract_case_id_and_query(case: dict[str, Any]) -> tuple[str, str]:
    case_id = str(case.get("case_id") or case.get("id"))
    query = case.get("case_query") or case.get("query") or case.get("query_text") or ""
    return case_id, query


def build_template(case_id: str, query: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "_case_query_preview": query[:200] + ("..." if len(query) > 200 else ""),
        "prediction": "TODO_A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN",
        "law_evidence": [{"law_id": "TODO", "aid": 0}],
        # TODO(Thịnh/Tuấn Anh): schema case_evidence/n_segments chưa xác nhận —
        # đoán là segment ID của văn bản vụ án (case narrative), KHÔNG phải
        # law_evidence. Cần xác nhận corpus case-segment tương ứng trước khi
        # điền field này, nếu không Penalized_Case_Recall (20% điểm) sẽ sai.
        "case_evidence": ["TODO_seg_id"],
        "n_segments": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold local_validation_gold.json từ public test thật")
    parser.add_argument("--public-test", type=Path, default=Path("data/ALQAC2026_public_test.json"))
    parser.add_argument("--output", type=Path, default=Path("data/local_validation_gold.template.json"))
    parser.add_argument("--n", type=int, default=30, help="Số case lấy mẫu (Plan.md khuyến nghị 20-50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.public_test.exists():
        raise FileNotFoundError(f"Không tìm thấy {args.public_test}")

    cases = iter_public_test_cases(args.public_test)
    sampled = reservoir_sample(cases, k=args.n, seed=args.seed)

    if not sampled:
        raise RuntimeError(
            f"Không đọc được case nào từ {args.public_test} — kiểm tra lại schema "
            "(list thuần hay bọc trong key 'cases'/'data')."
        )

    templates = [build_template(*extract_case_id_and_query(case)) for case in sampled]
    args.output.write_text(json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Đã lấy mẫu {len(templates)} case_id THẬT -> {args.output}")
    print("Case_id mẫu:", [t["case_id"] for t in templates[:5]], "...")
    print(
        "\nBước tiếp theo: Thịnh/Tuấn Anh mở file này, điền tay 'prediction'/'law_evidence'\n"
        "sau khi đọc case_query + đối chiếu luật liên quan, rồi rename thành\n"
        "data/local_validation_gold.json (đè file dummy cũ)."
    )


if __name__ == "__main__":
    main()
