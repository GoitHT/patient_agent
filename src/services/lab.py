from __future__ import annotations

import random
from typing import Any

from utils import now_iso


class LabService:
    def __init__(self, *, rng: random.Random) -> None:
        self.rng = rng

    def run_tests(self, *, ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in ordered:
            name = item.get("name", "")
            if "幽门螺杆菌" in name or "hp" in name.lower():
                positive = self.rng.random() < 0.5
                results.append(
                    {
                        "type": "lab",
                        "name": name,
                        "performed_at": now_iso(),
                        "abnormal": bool(positive),
                        "summary": "Hp 阳性" if positive else "Hp 阴性",
                        "values": {"hp": "positive" if positive else "negative"},
                        "suggestions": [
                            "阳性：考虑根除治疗并按疗程复查" if positive else "阴性：结合症状考虑其他原因",
                            "如症状持续或存在红旗需进一步评估",
                        ],
                    }
                )
                continue
            if "肝" in name or "肝功" in name:
                alt = int(self.rng.normalvariate(35, 12))
                ast = int(self.rng.normalvariate(30, 10))
                abnormal = alt > 60 or ast > 60
                results.append(
                    {
                        "type": "lab",
                        "name": name,
                        "performed_at": now_iso(),
                        "abnormal": abnormal,
                        "summary": "肝功能检测" if not abnormal else "转氨酶升高",
                        "values": {"ALT": alt, "AST": ast},
                        "suggestions": [
                            "结合症状复查肝功/肝炎相关指标",
                            "避免饮酒与可疑肝毒性药物",
                        ],
                    }
                )
            elif "血常规" in name or "血" in name:
                wbc = round(self.rng.normalvariate(6.5, 2.0), 1)
                hb = int(self.rng.normalvariate(135, 18))
                abnormal = wbc > 11.0 or hb < 110
                results.append(
                    {
                        "type": "lab",
                        "name": name,
                        "performed_at": now_iso(),
                        "abnormal": abnormal,
                        "summary": "血常规" if not abnormal else "血常规异常",
                        "values": {"WBC": wbc, "Hb": hb},
                        "suggestions": [
                            "结合感染/出血线索进一步评估",
                            "必要时复查或加做相关项目",
                        ],
                    }
                )
            elif "便" in name:
                occult = self.rng.random() < 0.2
                abnormal = occult
                results.append(
                    {
                        "type": "lab",
                        "name": name,
                        "performed_at": now_iso(),
                        "abnormal": abnormal,
                        "summary": "便检" if not abnormal else "提示隐血阳性",
                        "values": {"occult_blood": "positive" if occult else "negative"},
                        "suggestions": [
                            "结合消化道症状评估出血风险",
                            "必要时进一步内镜检查",
                        ],
                    }
                )
            else:
                results.append(
                    {
                        "type": "lab",
                        "name": name,
                        "performed_at": now_iso(),
                        "abnormal": False,
                        "summary": "实验室检查完成",
                        "values": {},
                        "suggestions": ["结合临床综合判断"],
                    }
                )
        return results
