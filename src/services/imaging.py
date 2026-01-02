from __future__ import annotations

import random
from typing import Any

from utils import now_iso


class ImagingService:
    def __init__(self, *, rng: random.Random) -> None:
        self.rng = rng

    def run_imaging(self, *, ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in ordered:
            name = item.get("name", "")
            modality = item.get("type", "imaging")
            abnormal = False
            finding = "未见明显异常"

            if any(k in name for k in ("CT", "MRI", "CTA")):
                abnormal = self.rng.random() < 0.25
                finding = "提示可疑异常影像信号" if abnormal else "未见急性出血或占位征象"
            elif any(k in name for k in ("超声", "B超")):
                abnormal = self.rng.random() < 0.2
                finding = "提示胆囊结石/脂肪肝可能" if abnormal else "腹部超声未见明显异常"

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

