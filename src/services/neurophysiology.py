from __future__ import annotations

from typing import Any

from utils import now_iso


class NeuroPhysiologyService:
    def __init__(self) -> None:
        pass

    def run_eeg_emg_ncv(self, *, ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in ordered:
            name = item.get("name", "")
            abnormal = False  # 默认正常
            if "EEG" in name or "脑电" in name:
                summary = "未见癫痫样放电"
            elif any(k in name for k in ("EMG", "肌电", "NCV", "神经传导")):
                summary = "神经传导速度正常"
            else:
                summary = "电生理检查完成"
            results.append(
                {
                    "type": "neurophysiology",
                    "name": name,
                    "performed_at": now_iso(),
                    "abnormal": abnormal,
                    "summary": summary,
                    "findings": {"text": summary},
                    "suggestions": [
                        "结合临床与影像综合判断",
                        "必要时复查或进一步评估",
                    ],
                }
            )
        return results

