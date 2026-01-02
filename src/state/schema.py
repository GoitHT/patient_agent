from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from utils import now_iso


def _as_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for c in chunks:
        citations.append(
            {
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
            }
        )
    return citations


def make_audit_entry(
    *,
    node_name: str,
    inputs_summary: dict[str, Any],
    outputs_summary: dict[str, Any],
    decision: str,
    chunks: list[dict[str, Any]] | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    """Create an audit entry that follows the project traceability spec.

    Notes:
    - inputs_summary / outputs_summary should avoid sensitive details.
    - chunks are the retrieved RAG chunks; only doc_id/chunk_id/score are stored as citations.
    """

    return {
        "ts": now_iso(),
        "node_name": node_name,
        "inputs_summary": inputs_summary,
        "outputs_summary": outputs_summary,
        "decision": decision,
        "citations": _as_citations(chunks or []),
        "flags": flags or [],
    }


class BaseState(BaseModel):
    """Common state shared across all departments and graph nodes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dept: Literal[
        "internal_medicine", "surgery", "orthopedics", "urology",
        "obstetrics_gynecology", "pediatrics", "neurology", "oncology",
        "infectious_disease", "dermatology_std", "ent_ophthalmology_stomatology",
        "psychiatry", "emergency", "rehabilitation_pain", "traditional_chinese_medicine"
    ]
    patient_profile: dict[str, Any] = Field(default_factory=dict)
    appointment: dict[str, Any] = Field(default_factory=dict)
    chief_complaint: str = ""
    history: dict[str, Any] = Field(default_factory=dict)
    exam_findings: dict[str, Any] = Field(default_factory=dict)
    dept_payload: dict[str, Any] = Field(default_factory=dict)
    specialty_summary: dict[str, Any] = Field(default_factory=dict)

    preliminary_assessment: dict[str, Any] = Field(default_factory=dict)
    need_aux_tests: bool = False
    ordered_tests: list[dict[str, Any]] = Field(default_factory=list)
    test_prep: list[dict[str, Any]] = Field(default_factory=list)
    test_results: list[dict[str, Any]] = Field(default_factory=list)

    diagnosis: dict[str, Any] = Field(default_factory=dict)
    treatment_plan: dict[str, Any] = Field(default_factory=dict)
    discharge_docs: list[dict[str, Any]] = Field(default_factory=list)
    followup_plan: dict[str, Any] = Field(default_factory=dict)

    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    
    # 多智能体系统新增字段
    agent_interactions: dict[str, Any] = Field(default_factory=dict)  # 医患护对话记录
    ground_truth: dict[str, Any] = Field(default_factory=dict)  # 标准答案（用于评估）
    case_data: dict[str, Any] = Field(default_factory=dict)  # 完整病例数据
    node_qa_counts: dict[str, int] = Field(default_factory=dict)  # 每个节点的问答轮数计数

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        # LangGraph may serialize Pydantic state with `exclude_unset=True`.
        # In-place mutations (e.g., list.append / dict.__setitem__) do not mark fields as "set" in Pydantic v2,
        # which can cause those fields to be omitted from the serialized state and effectively lost between nodes.
        #
        # We proactively mark all mutable containers as "set" to guarantee persistence across graph steps.
        fields = {
            "patient_profile",
            "appointment",
            "history",
            "exam_findings",
            "dept_payload",
            "specialty_summary",
            "preliminary_assessment",
            "ordered_tests",
            "test_prep",
            "test_results",
            "diagnosis",
            "treatment_plan",
            "discharge_docs",
            "followup_plan",
            "retrieved_chunks",
            "escalations",
            "audit_trail",
            "agent_interactions",
            "ground_truth",
            "case_data",
            "node_qa_counts",
        }
        try:
            self.__pydantic_fields_set__.update(fields)
        except Exception:
            # If internal field tracking changes, we prefer correctness at runtime over strict tracking.
            pass

    def add_retrieved_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self.retrieved_chunks = [*self.retrieved_chunks, *chunks]

    def add_audit(self, entry: dict[str, Any]) -> None:
        self.audit_trail = [*self.audit_trail, entry]
