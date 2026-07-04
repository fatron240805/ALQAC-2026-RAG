# ALQAC 2026 Label Definitions

Status: evidence-backed public-test definitions as of 2026-07-03.

This file records what Thinh's reasoning component may assume about ALQAC
2026 labels after auditing the downloaded public-test dataset in
`data/public/alqac2026/ALQAC2026_public_test.json`.

## Local Evidence Checked

- `Plan.md`: says ALQAC 2026 has four labels and that Thinh must inspect 20-50
  public samples to define them.
- `ALQAC_2026_Game_Rules_Strategy_Plan.md`: says the four labels are an
  outcome-classification expansion from two labels, but marks the meanings as
  unknown guesses.
- `ALQAC_2026_Literature_Search.md`: identifies label-definition accuracy as
  the highest-priority blocker.
- `2025.vlsp-1.23.pdf`: provides useful task patterns for Vietnamese legal QA,
  including citation usefulness, MCQ, and free-text QA. It does not define the
  four ALQAC 2026 outcome labels.
- `data/public/alqac2026/ALQAC2026_public_test.json`: 50 public-test civil
  judgments with `verdict_label`.
- `data/public/alqac2026/corpus_law_pub.json`: 18 law records for retrieval
  corpus context.

## Officially Known

- There are exactly four output labels.
- The task is civil court outcome classification inside an Agentic RAG pipeline.
- In the public-test file, `A_role` is always `Nguyên đơn` and `B_role` is
  always `Bị đơn`.
- `case_type` is always `Dân sự`; `court_level` is always `Sơ thẩm`.
- Final competition inference must use open-weight models under 10B parameters.
- ChatGPT, Claude, Gemini, and other proprietary systems must not be part of the
  competition inference pipeline.
- The reasoning agent should classify only after retrieval provides legally
  useful evidence.

## Official Labels

| Official label | Working meaning | Public-test count |
|---|---|---:|
| `A_WIN` | Plaintiff wins fully or substantially | 16 |
| `B_WIN` | Defendant wins fully or substantially | 10 |
| `PARTIAL_A_WIN` | Mixed result, plaintiff is the dominant winner | 19 |
| `PARTIAL_B_WIN` | Mixed result, defendant is the dominant winner | 5 |

## Label Definitions

### `A_WIN`

Select when the court accepts the plaintiff's main claim in full or
substantially, and any denied/withdrawn/procedural part is not central to the
case outcome.

Positive signals:

- `Chấp nhận yêu cầu khởi kiện của nguyên đơn`.
- `Chấp nhận toàn bộ yêu cầu khởi kiện`.
- Defendant must perform the main requested obligation: pay debt/damages,
  return land/property, continue/recognize a contract, or accept division of
  inheritance according to plaintiff's core request.

Do not select when the court expressly accepts only a minor part of the claim,
rejects the main claim, or defendant/counterclaim wins the core dispute.

Representative public-test cases:

- `case_8219`: court accepts all of plaintiff's loan claim.
- `case_2705`: service contract declared invalid and defendant must refund
  plaintiff.
- `case_4337`: appellate court keeps judgment accepting plaintiff's land/house
  return claim.
- `case_5226`: bank's debt recovery claim accepted.
- `case_6284`: plaintiff's land-transfer contract claim accepted and opposing
  invalidity request rejected.

### `B_WIN`

Select when the court rejects the plaintiff's claim fully or substantially and
the defendant keeps the disputed legal position, or the defendant's counterclaim
is accepted on the core issue.

Positive signals:

- `Không chấp nhận toàn bộ yêu cầu khởi kiện`.
- `Bác toàn bộ yêu cầu khởi kiện`.
- Plaintiff bears the core loss: no return of land/property, no damages/debt
  award, no invalidation of transaction, or defendant's transaction/right is
  recognized.

Do not select when the plaintiff receives meaningful relief on a core claim,
even if not full relief; use a partial label instead.

Representative public-test cases:

- `case_2978`: all plaintiff property-return claims rejected.
- `case_9089`: plaintiff land-return claim rejected for insufficient basis.
- `case_669`: plaintiff contract-cancellation claim rejected and defendant
  counterclaim recognized.
- `case_950`: all plaintiff land and crop-damage claims rejected.
- `case_4557`: plaintiff damages claim rejected.

### `PARTIAL_A_WIN`

Select when both sides partly win or some plaintiff claims are denied, but the
plaintiff receives meaningful relief on the core dispute and is the dominant
winner.

Positive signals:

- `Chấp nhận một phần yêu cầu khởi kiện` with substantial award to plaintiff.
- Plaintiff receives the main legal remedy in reduced amount/scope.
- Defendant's counterclaim is rejected or defendant is still ordered to perform
  the primary obligation.

Do not select when the plaintiff's award is minor compared with the rejected
claim, the main issue is resolved for defendant, or defendant receives the more
important property/right. Use `PARTIAL_B_WIN` in those cases.

Representative public-test cases:

- `case_4101`: plaintiff damages claim partly accepted.
- `case_1087`: plaintiff receives partial property-damage compensation while
  other claimed amounts are rejected.
- `case_3735`: plaintiff gets part of contributed capital returned and
  defendant's counterclaim is rejected.
- `case_8812`: plaintiff receives partial personal-injury compensation.
- `case_4834`: plaintiff receives part of requested debt amount.

### `PARTIAL_B_WIN`

Select when the result is mixed, but defendant is the dominant winner: the
plaintiff receives only limited/alternative relief, the plaintiff loses the
main or larger part of the requested relief, or defendant receives/keeps the
more important right or property.

Positive signals:

- Court says `Chấp nhận một phần` but rejects the main/larger part of
  plaintiff's request.
- Plaintiff gets a smaller monetary/property amount while most of the claim is
  rejected.
- Defendant's request or legal position is accepted on the core issue.

Do not select merely because the result is partial. If plaintiff's core claim
is substantially accepted, use `PARTIAL_A_WIN`.

Representative public-test cases:

- `case_5860`: plaintiff claimed 22,187,000 VND, receives 6,700,000 VND, and
  15,487,000 VND is rejected.
- `case_149`: plaintiff gets refund but damages claim is rejected and
  defendant's contract-invalidity position is accepted.
- `case_7467`: plaintiffs receive one inherited portion, but claims over the
  larger land area are rejected and defendant receives/keeps major property.
- `case_2238`: plaintiff's primary land-transfer claim is rejected; only an
  alternative refund follows after contract invalidity.
- `case_4861`: inheritance/land result is mixed; plaintiff-side requests are
  accepted only partly while defendants retain substantial land/use interests.

## Required Label-Audit Procedure

For future train/dev/test data:

1. Normalize records into rows with at least `case_id`, `question`,
   `provided_context` or retrieved passages if present, `gold_label`, and any
   answer/explanation fields.
2. Inspect 20-50 samples, balanced across all observed labels if possible.
3. For each label, verify:
   - positive definition: when this label should be selected;
   - negative definition: when this label should not be selected;
   - minimum evidence required;
   - common confusing neighboring labels;
   - 3-5 representative case IDs.
4. Compute label distribution on all local public samples.
5. Build a confusion-risk table before writing any classifier rules.
6. Update this file if official documentation overrides the public-test
   inference.

## Decision Rules

- Select exactly one of `A_WIN`, `B_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`.
- Treat `A` as `Nguyên đơn` and `B` as `Bị đơn` only after checking the record
  fields. The public-test file uses this mapping for all 50 samples.
- Decide dominance from the court's actual disposition, not only from the words
  `chấp nhận một phần`.
- Compare the plaintiff's requested relief in `case_query` against the court's
  relief in `court_verdict`.
- For partial outcomes, decide who wins the core or larger-value issue.
- Base the decision on retrieved legal evidence, not model memory.
- Prefer current, effective law over repealed or superseded law.
- Treat a passage as decisive only if it answers the actual legal issue, not
  merely the same topic or document name.
- If facts are date-sensitive, compare the event date against the statute's
  effective date.
- If retrieved evidence conflicts, record the conflict and lower confidence.
- There is no public-test evidence for an insufficient/unknown output label.
  Do not invent one.

## Evidence Requirements

A label decision should include:

- `case_id`: original case identifier.
- `label`: one of the official four labels.
- `confidence`: calibrated score from 0.0 to 1.0.
- `evidence_ids`: IDs of passages judged useful.
- `citation_judgments`: per-passage usefulness decisions.
- `justification`: concise legal reason grounded in cited evidence.

Useful evidence must satisfy all of the following:

- It addresses the same legal issue as the question.
- It contains a rule, condition, exception, authority, deadline, obligation, or
  sanction needed to decide the label.
- It is current or otherwise legally applicable to the case facts.
- It is specific enough to support the final label.

## Citation Usefulness Labels

These are internal retrieval-filter labels, not the final four ALQAC labels:

| Internal label | Definition |
|---|---|
| `useful` | The passage directly helps answer the legal question or decide the outcome. |
| `not_useful` | The passage is irrelevant, only topically related, outdated, or too generic. |
| `uncertain` | The passage may help but needs another provision, metadata check, or date check. |

## JSON Output Schema

The reasoning component should return valid JSON before final submission
formatting:

```json
{
  "case_id": "string",
  "label": "A_WIN|B_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN",
  "confidence": 0.0,
  "evidence_ids": ["string"],
  "citation_judgments": [
    {
      "evidence_id": "string",
      "judgment": "useful|not_useful|uncertain",
      "reason": "short evidence-usefulness reason"
    }
  ],
  "justification": "Concise evidence-grounded legal reason."
}
```

## Edge Cases Found

- `case_4588` is labeled `A_WIN` although the verdict says `Chấp nhận một phần`;
  this suggests the label can treat partial wording as an A win when plaintiff's
  core inheritance relief is substantially granted.
- `case_8219` is labeled `A_WIN` even though one withdrawn request is
  discontinued; discontinued/minor issues do not automatically make a partial
  label.
- `case_5860` is labeled `PARTIAL_B_WIN` because plaintiff gets only a small
  fraction of the requested amount.
- `case_149` is labeled `PARTIAL_B_WIN` because defendant's contract-invalidity
  position and rejection of damages outweigh plaintiff's refund.
- `case_2238` is labeled `PARTIAL_B_WIN` because the plaintiff's primary demand
  is rejected even though an alternative refund is ordered.
