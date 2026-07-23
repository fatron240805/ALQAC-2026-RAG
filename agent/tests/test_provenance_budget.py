from app.schemas import (
    ALQAC_VALID_LABELS,
    AlqacLabel,
    AlqacPrediction,
    CaseDraft,
    LawEvidenceItem,
)
from app.validator import serialize_submission, validate_and_build_result, validate_draft_provenance


def test_ledger_max_2n():
    from app.schemas import OfficialCallLedger

    ledger = OfficialCallLedger(max_calls=4, no_gain_limit=1)
    assert ledger.can_call()
    for i in range(4):
        ledger.record(f"c{i}", is_duplicate=False, is_no_gain=False)
    assert ledger.used == 4
    assert not ledger.can_call()
    assert ledger.stopped_reason == "budget_exhausted"


def test_ledger_duplicate_stops():
    from app.schemas import OfficialCallLedger

    ledger = OfficialCallLedger(max_calls=10, no_gain_limit=1)
    ledger.record("chunk_a", is_duplicate=False, is_no_gain=False)
    ledger.record("chunk_a", is_duplicate=True, is_no_gain=True)
    assert not ledger.can_call()
    assert ledger.stopped_reason == "duplicate_chunk_id"


def test_reject_hallucinated_chunk_id():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
        case_evidence=["fake_chunk"],
        law_evidence=[],
    )
    errs = validate_draft_provenance(
        draft, official_allowlist=set(), law_allowlist=set()
    )
    assert any("fake_chunk" in e for e in errs)


def test_content_fail_never_serializes(tmp_path):
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.B_WIN),
        case_evidence=[],
        law_evidence=[],
    )
    result = validate_and_build_result(
        "case_1",
        draft,
        content_passed=False,
        official_allowlist=[],
        law_pairs=[],
        content_findings=["unsupported"],
    )
    assert result.status == "rejected"
    rows = serialize_submission([result], tmp_path / "out.json", write=True)
    assert rows == []


def test_pass_with_allowlisted_ids(tmp_path):
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
        case_evidence=["chunk_1"],
        law_evidence=[LawEvidenceItem(law_id="L1", aid="1")],
    )
    result = validate_and_build_result(
        "case_1",
        draft,
        content_passed=True,
        official_allowlist=["chunk_1"],
        law_pairs=[LawEvidenceItem(law_id="L1", aid="1")],
    )
    assert result.status == "ok"
    rows = serialize_submission([result], tmp_path / "out.json", write=True)
    assert len(rows) == 1
    assert rows[0]["case_evidence"] == ["chunk_1"]


def test_valid_alqac_labels():
    assert ALQAC_VALID_LABELS == {"A_WIN", "B_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN"}
    for label in AlqacLabel:
        assert label.value in ALQAC_VALID_LABELS


def test_reject_invalid_prediction_label():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
        case_evidence=[],
        law_evidence=[],
    )
    # Force an invalid label value by manipulating internal state
    draft.prediction.prediction = "INVALID_LABEL"  # type: ignore[assignment]
    errs = validate_draft_provenance(
        draft, official_allowlist=set(), law_allowlist=set()
    )
    assert any("invalid" in e.lower() for e in errs)


def test_reject_hallucinated_law_pair():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.PARTIAL_B_WIN),
        case_evidence=[],
        law_evidence=[LawEvidenceItem(law_id="FAKE_LAW", aid="999")],
    )
    errs = validate_draft_provenance(
        draft, official_allowlist=set(), law_allowlist=set()
    )
    assert any("FAKE_LAW" in e for e in errs)


def test_submission_preserves_order(tmp_path):
    from app.schemas import CaseResult

    results = [
        CaseResult(
            case_id="c1",
            status="ok",
            prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
            case_evidence=["ch1"],
            law_evidence=[LawEvidenceItem(law_id="L1", aid="1")],
        ),
        CaseResult(case_id="c2", status="rejected"),
        CaseResult(
            case_id="c3",
            status="ok",
            prediction=AlqacPrediction(prediction=AlqacLabel.B_WIN),
            case_evidence=[],
            law_evidence=[],
        ),
    ]
    rows = serialize_submission(results, tmp_path / "out.json", write=True)
    assert len(rows) == 2
    assert rows[0]["case_id"] == "c1"
    assert rows[1]["case_id"] == "c3"
    assert rows[0]["prediction"]["prediction"] == "A_WIN"
    assert rows[1]["prediction"]["prediction"] == "B_WIN"


# ---- Minimum-evidence enforcement tests ----


def test_reject_empty_law_evidence():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
        case_evidence=["chunk_1"],
        law_evidence=[],
    )
    errs = validate_draft_provenance(
        draft, official_allowlist={"chunk_1"}, law_allowlist={("L1", "1")}
    )
    assert any("law_evidence is empty" in e for e in errs)


def test_reject_empty_case_evidence_when_official_enabled():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.B_WIN),
        case_evidence=[],
        law_evidence=[LawEvidenceItem(law_id="L1", aid="1")],
    )
    errs = validate_draft_provenance(
        draft,
        official_allowlist=set(),
        law_allowlist={("L1", "1")},
        official_api_enabled=True,
    )
    assert any("case_evidence is empty" in e for e in errs)


def test_allow_empty_case_evidence_when_official_disabled():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.B_WIN),
        case_evidence=[],
        law_evidence=[LawEvidenceItem(law_id="L1", aid="1")],
    )
    errs = validate_draft_provenance(
        draft,
        official_allowlist=set(),
        law_allowlist={("L1", "1")},
        official_api_enabled=False,
    )
    assert not errs


def test_both_empty_evidence_rejected():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
        case_evidence=[],
        law_evidence=[],
    )
    errs = validate_draft_provenance(
        draft, official_allowlist=set(), law_allowlist=set()
    )
    assert any("law_evidence is empty" in e for e in errs)


def test_validate_and_build_rejects_zero_evidence():
    draft = CaseDraft(
        prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
        case_evidence=[],
        law_evidence=[],
    )
    result = validate_and_build_result(
        "case_x",
        draft,
        content_passed=True,
        official_allowlist=[],
        law_pairs=[],
    )
    assert result.status == "error"
    assert "Provenance" in result.error.message
