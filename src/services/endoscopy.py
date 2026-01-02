from __future__ import annotations

import random
from typing import Any

from utils import now_iso


class EndoscopyService:
    def __init__(self, *, rng: random.Random) -> None:
        self.rng = rng

    def schedule_endoscopy(self, *, procedure: str) -> dict[str, Any]:
        return {
            "procedure": procedure,
            "scheduled": True,
            "schedule_id": f"END-{self.rng.randint(10000, 99999)}",
            "scheduled_at": now_iso(),
        }

    def bowel_prep_checklist(self, *, procedure: str) -> list[dict[str, Any]]:
        if "结肠" in procedure:
            return [
                {"item": "检查前3天低渣饮食", "required": True},
                {"item": "检查前1天清流质饮食", "required": True},
                {"item": "按医嘱服用肠道清洁剂", "required": True},
                {"item": "抗凝/抗血小板药物需提前评估", "required": True},
                {"item": "糖尿病患者需调整降糖方案", "required": True},
            ]
        return [
            {"item": "检查前6-8小时禁食禁饮", "required": True},
            {"item": "如需镇静需家属陪同", "required": True},
        ]

    def generate_report(self, *, procedure: str) -> dict[str, Any]:
        abnormal = self.rng.random() < 0.3
        conclusion = "未见明显异常" if not abnormal else "发现黏膜炎症/糜烂改变"
        return {
            "type": "endoscopy",
            "name": procedure,
            "performed_at": now_iso(),
            "abnormal": abnormal,
            "summary": conclusion,
            "findings": {"conclusion": conclusion},
            "suggestions": [
                "结合病理/幽门螺杆菌检测结果（如有）",
                "遵医嘱用药并复查",
            ],
        }

