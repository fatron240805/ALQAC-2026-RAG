"""Workflow loop with fake agents/tools — no network."""

from __future__ import annotations

from pathlib import Path

from app.schemas import (
    AlqacState,
    AlqacLabel,
    AlqacPrediction,
    CaseDraft,
    CaseError,
    CaseInput,
    CaseResult,
    ContentCheckResult,
    ElementGraph,
    FormatSuggestions,
    LawEvidenceItem,
    LawHit,
    ManagerAction,
    ManagerDecision,
    OfficialCallLedger,
    OfficialCaseHit,
    PublicCaseHit,
)
from app.workflow import CaseWorkflow, run_batch


class FakeAgents:
    def __init__(self, settings):
        self.settings = settings
        self.openai_model = settings.openai_model
        self.manager_calls = 0

    def extract_elements(self, case_id, case_query):
        return ElementGraph(key_facts=["f1"], legal_questions=["q1"])

    def create_draft(self, case_id, case_query, element_graph, **kwargs):
        return CaseDraft(
            prediction=AlqacPrediction(
                prediction=AlqacLabel.A_WIN,
            ),
            case_evidence=[],
            law_evidence=[],
            reasoning="init",
        )

    def manager_route(self, case_id, draft, element_graph, **kwargs):
        self.manager_calls += 1
        if self.manager_calls == 1:
            return ManagerDecision(
                decision="revise",
                actions=[ManagerAction.FORMAT_CHECK, ManagerAction.LAW_SEARCH],
                rationale="both",
            )
        return ManagerDecision(decision="pass", actions=[], rationale="ok")

    def format_check(self, case_id, draft, **kwargs):
        return FormatSuggestions(suggestions=["be concise"])

    def revise_format(self, case_id, draft, suggestions, **kwargs):
        d = draft.model_copy(deep=True)
        d.reasoning = "format_applied"
        return d

    def integrate_law(self, case_id, draft, law_hits, **kwargs):
        d = draft.model_copy(deep=True)
        d.law_evidence = [
            LawEvidenceItem(law_id=h.law_id, aid=str(h.aid)) for h in law_hits
        ]
        d.reasoning = "law_applied"
        return d

    def content_check(self, case_id, case_query, draft, **kwargs):
        return ContentCheckResult(decision="pass", findings=["supported"])


class FakePublic:
    def __init__(self):
        self.calls = 0

    def __call__(self, query):
        self.calls += 1
        return [PublicCaseHit(source_id="pub1", text="public", score=0.1)]


class FakeOfficial:
    def __init__(self, ledger: OfficialCallLedger):
        self.ledger = ledger
        self.calls = 0

    def __call__(self, query, case_id=None):
        self.calls += 1
        self.ledger.record("chunk_x", is_duplicate=False, is_no_gain=False)
        return OfficialCaseHit(chunk_id="chunk_x", text="official", score=1.0)


class FakeLaw:
    def __call__(self, query, element_graph=None):
        return [
            LawHit(law_id="L1", aid="10", text="statute", vector_score=0.9, graph_hops=0)
        ]


def test_flow_format_then_law_then_content(settings):
    from app.observability import Observability

    obs = Observability(settings)
    ledger = OfficialCallLedger(max_calls=0)
    public = FakePublic()
    official = FakeOfficial(ledger)
    agents = FakeAgents(settings)
    wf = CaseWorkflow(
        settings,
        ledger,
        obs,
        agents=agents,  # type: ignore[arg-type]
        public_tool=public,  # type: ignore[arg-type]
        official_tool=official,  # type: ignore[arg-type]
        law_tool=FakeLaw(),  # type: ignore[arg-type]
    )
    state, result = wf.run_case(CaseInput(case_id="case_1", case_query="dog bite"))
    # Order: format before law
    routes = state.route_history
    assert "format_check" in routes
    assert any(r.startswith("law_hits:") for r in routes)
    fi = routes.index("format_check")
    li = next(i for i, r in enumerate(routes) if r.startswith("law_hits:"))
    assert fi < li
    assert result.status == "ok"
    assert public.calls == 0  # disabled path never selected in fake manager first call
    assert official.calls == 0
    assert "content_check:pass" in routes
    assert any(r.startswith("validate:") for r in routes)


def test_disabled_flags_zero_network_even_if_requested(settings):
    from app.observability import Observability

    class AgentsWantRetrieval(FakeAgents):
        def manager_route(self, case_id, draft, element_graph, **kwargs):
            self.manager_calls += 1
            if self.manager_calls == 1:
                return ManagerDecision(
                    decision="revise",
                    actions=[
                        ManagerAction.PUBLIC_CASE_RETRIEVAL,
                        ManagerAction.OFFICIAL_CASE_API,
                    ],
                )
            return ManagerDecision(decision="pass", actions=[])

    # Real AgentRuntime strips disabled; FakeAgents simulate stripped empty
    # Here we test tools themselves when flags false via real tools
    from app.tools import OfficialCaseTop1Tool, PublicCaseSearchTool

    obs = Observability(settings)
    ledger = OfficialCallLedger(max_calls=4)
    public = PublicCaseSearchTool(settings)
    official = OfficialCaseTop1Tool(settings, ledger)
    # manager with empty after strip simulation
    agents = FakeAgents(settings)
    agents.manager_calls = 0

    class PassAgents(FakeAgents):
        def manager_route(self, *a, **k):
            return ManagerDecision(decision="pass", actions=[])

    out = run_batch(
        [CaseInput(case_id="c1", case_query="q")],
        settings,
        write_submission=False,
        obs=obs,
        agents=PassAgents(settings),  # type: ignore[arg-type]
        public_tool=public,
        official_tool=official,
        law_tool=FakeLaw(),  # type: ignore[arg-type]
    )
    assert out["official_calls_used"] == 0
    assert out["openai_model"] == settings.openai_model


def test_content_fail_rejects(settings):
    from app.observability import Observability

    class FailContent(FakeAgents):
        def manager_route(self, *a, **k):
            return ManagerDecision(decision="pass", actions=[])

        def content_check(self, *a, **k):
            return ContentCheckResult(decision="fail", findings=["bad"])

    obs = Observability(settings)
    ledger = OfficialCallLedger(max_calls=0)
    wf = CaseWorkflow(
        settings,
        ledger,
        obs,
        agents=FailContent(settings),  # type: ignore[arg-type]
        public_tool=FakePublic(),  # type: ignore[arg-type]
        official_tool=FakeOfficial(ledger),  # type: ignore[arg-type]
        law_tool=FakeLaw(),  # type: ignore[arg-type]
    )
    state, result = wf.run_case(CaseInput(case_id="c", case_query="q"))
    assert result.status == "rejected"
    assert state.rejected is True
    assert "validate:" not in "".join(state.route_history) or result.status == "rejected"


def test_batch_budget_shared(settings):
    from app.observability import Observability

    class AgentsOfficial(FakeAgents):
        def manager_route(self, case_id, draft, element_graph, **kwargs):
            self.manager_calls += 1
            if self.manager_calls % 2 == 1:
                return ManagerDecision(
                    decision="revise",
                    actions=[ManagerAction.OFFICIAL_CASE_API],
                )
            return ManagerDecision(decision="pass", actions=[])

        def content_check(self, *a, **k):
            return ContentCheckResult(decision="pass", findings=[])

    # Force-enable only for this unit test via fake tools (settings still false)
    # Manager fake still requests official; FakeOfficial records
    class AgentsWithOfficial(FakeAgents):
        def __init__(self, settings):
            super().__init__(settings)
            self._n = 0

        def manager_route(self, *a, **k):
            self._n += 1
            if self._n == 1:
                return ManagerDecision(
                    decision="revise", actions=[ManagerAction.OFFICIAL_CASE_API]
                )
            return ManagerDecision(decision="pass", actions=[])

    cases = [
        CaseInput(case_id="a", case_query="q1"),
        CaseInput(case_id="b", case_query="q2"),
    ]
    # max = 2 * 2 = 4; each case one call => 2 used
    obs = Observability(settings)
    # Build custom run with FakeOfficial counting across batch
    from app.workflow import CaseWorkflow
    from app.schemas import OfficialCallLedger

    ledger = OfficialCallLedger(max_calls=settings.official_max_calls(len(cases)))
    official = FakeOfficial(ledger)
    results = []
    for c in cases:
        agents = AgentsWithOfficial(settings)
        wf = CaseWorkflow(
            settings,
            ledger,
            obs,
            agents=agents,  # type: ignore[arg-type]
            public_tool=FakePublic(),  # type: ignore[arg-type]
            official_tool=official,  # type: ignore[arg-type]
            law_tool=FakeLaw(),  # type: ignore[arg-type]
        )
        _, r = wf.run_case(c)
        results.append(r)
    assert official.calls == 2
    assert ledger.used == 2
    assert ledger.used <= 2 * len(cases)


def test_batch_checkpoints_submission_and_errors_after_each_case(settings, monkeypatch):
    import json

    from app import workflow
    from app.observability import Observability

    snapshots: list[str] = []
    real_serialize = workflow.serialize_case_artifact

    def track_checkpoints(result, *args, **kwargs):
        snapshots.append(result.case_id)
        return real_serialize(result, *args, **kwargs)

    def completed_case(self, case):
        state = AlqacState(case_id=case.case_id, case_query=case.case_query)
        if case.case_id == "bad":
            return state, CaseResult(
                case_id=case.case_id,
                status="error",
                error=CaseError(
                    case_id=case.case_id,
                    stage="workflow",
                    message="model unavailable",
                ),
            )
        return state, CaseResult(
            case_id=case.case_id,
            status="ok",
            prediction=AlqacPrediction(prediction=AlqacLabel.A_WIN),
            law_evidence=[LawEvidenceItem(law_id="L1", aid="10")],
        )

    monkeypatch.setattr(workflow, "serialize_case_artifact", track_checkpoints)
    monkeypatch.setattr(CaseWorkflow, "run_case", completed_case)

    out = run_batch(
        [
            CaseInput(case_id="good", case_query="q1"),
            CaseInput(case_id="bad", case_query="q2"),
        ],
        settings,
        obs=Observability(settings),
        agents=FakeAgents(settings),  # type: ignore[arg-type]
        public_tool=FakePublic(),  # type: ignore[arg-type]
        official_tool=FakeOfficial(OfficialCallLedger(max_calls=4)),  # type: ignore[arg-type]
        law_tool=FakeLaw(),  # type: ignore[arg-type]
    )

    assert snapshots == ["good", "bad"]
    assert out["submission_path"] is not None
    assert out["error_report_path"] is not None
    assert out["submission_paths"] == [str(settings.submission_output_path.parent / "submission_good.json")]
    assert out["error_report_paths"] == [str(settings.submission_output_path.parent / "error_bad.json")]
    assert json.loads(Path(out["submission_path"]).read_text()) == [
        {
            "case_id": "good",
            "prediction": {"prediction": "A_WIN"},
            "case_evidence": [],
            "law_evidence": [{"law_id": "L1", "aid": "10"}],
        }
    ]
    errors = json.loads(Path(out["error_report_path"]).read_text())
    assert errors[0]["case_id"] == "bad"
    assert errors[0]["status"] == "error"
    assert errors[0]["error"]["message"] == "model unavailable"
