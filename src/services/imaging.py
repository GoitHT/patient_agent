from __future__ import annotations

from typing import Any

from utils import now_iso


class ImagingService:
    def __init__(self) -> None:
        pass

    def run_imaging(self, *, ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in ordered:
            name = item.get("name", "")
            modality = item.get("type", "imaging")
            abnormal = False  # 默认正常
            finding = "未见明显异常"

            if any(k in name for k in ("CT", "MRI", "CTA")):
                finding = "未见急性出血或占位征象"
            elif any(k in name for k in ("超声", "B超")):
                finding = "腹部超声未见明显异常"

            results.append(
                {
                    "type": modality,
                    "name": name,
                    "performed_at": now_iso(),
                    "abnormal": abnormal,
                    "summary": finding,
                    "findings": {"text": finding},
                    "suggestions": [
                        "结合临床与其他检查综合判断",
                        "必要时进一步影像或专科评估",
                    ],
                }
            )
        return results

