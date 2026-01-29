from __future__ import annotations

import time
from typing import Any

from utils import now_iso


class DocumentService:
    def __init__(self) -> None:
        pass

    def generate_emr(self, *, state: Any) -> dict[str, Any]:
        return {
            "doc_type": "EMR",
            "doc_id": f"EMR-{int(time.time() * 1000) % 100000}",
            "generated_at": now_iso(),
            "content": {
                "chief_complaint": getattr(state, "chief_complaint", ""),
                "history": getattr(state, "history", {}),
                "exam_findings": getattr(state, "exam_findings", {}),
                "diagnosis": getattr(state, "diagnosis", {}),
                "treatment_plan": getattr(state, "treatment_plan", {}),
            },
        }

    def diagnosis_cert(self, *, state: Any) -> dict[str, Any]:
        dx = getattr(state, "diagnosis", {})
        return {
            "doc_type": "DIAGNOSIS_CERT",
            "doc_id": f"DX-{int(time.time() * 1000) % 100000}",
            "generated_at": now_iso(),
            "content": {"diagnosis": dx.get("final", dx.get("name", "待明确"))},
        }

    def sick_leave(self, *, state: Any) -> dict[str, Any]:
        days = 2  # 固定休息2天
        return {
            "doc_type": "SICK_LEAVE",
            "doc_id": f"SL-{int(time.time() * 1000) % 100000}",
            "generated_at": now_iso(),
            "content": {"days": days, "note": "如症状加重请及时就医"},
        }

    def education_sheet(self, *, state: Any) -> dict[str, Any]:
        return {
            "doc_type": "EDUCATION",
            "doc_id": f"EDU-{int(time.time() * 1000) % 100000}",
            "generated_at": now_iso(),
            "content": {"followup_plan": getattr(state, "followup_plan", {})},        }

