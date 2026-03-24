"""
Tick-based simulation clock for fair multi-patient time tracking.

核心思想：
- 全局 tick 计数器：统一推进模拟时间，用于共享资源（设备等）的调度。
- 每患者个人 tick 计数：只累计该患者自身动作消耗的 tick，
  保证多患者并发时每人的有效就诊时长相互独立。
- 动作锁定（busy_until_tick）：患者执行动作后锁定若干 tick，
  为后续统一调度循环提供基础。

默认 1 tick = 1 分钟。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional


DEFAULT_TICK_MINUTES: int = 1


@dataclass
class PatientTickState:
    """单个患者的 tick 状态。"""

    patient_id: str
    join_tick: int
    lane_tick: int = 0
    personal_elapsed_ticks: int = 0
    busy_until_tick: int = 0


class SimulationClock:
    """Tick-based 模拟时钟（最终版：患者/资源/系统三层时钟）。"""

    def __init__(
        self,
        start_time: datetime,
        tick_minutes: int = DEFAULT_TICK_MINUTES,
    ) -> None:
        self._lock = threading.RLock()
        self.start_datetime: datetime = start_time
        self.tick_minutes: int = max(1, tick_minutes)

        # 三层时钟：
        # 1) 患者个人时钟（按 patient_id 分 lane）
        # 2) 资源调度时钟（按 actor 分 lane，用于设备/队列/预约等）
        # 3) 系统时钟（系统级事件）
        # 最终全局时钟 = max(患者层, 资源层, 系统层)
        self._global_tick: int = 0
        self._resource_tick: int = 0
        self._system_tick: int = 0
        self._patients: Dict[str, PatientTickState] = {}
        self._resource_lanes: Dict[str, int] = {}

    @property
    def current_tick(self) -> int:
        """当前全局 tick 数。"""
        return self._global_tick

    @property
    def current_datetime(self) -> datetime:
        """当前全局模拟时间。"""
        return self.start_datetime + timedelta(
            minutes=self._global_tick * self.tick_minutes
        )

    @property
    def resource_datetime(self) -> datetime:
        """资源调度时钟（设备/队列）的当前时间。"""
        return self.start_datetime + timedelta(
            minutes=self._resource_tick * self.tick_minutes
        )

    @property
    def system_datetime(self) -> datetime:
        """系统级时钟的当前时间。"""
        return self.start_datetime + timedelta(
            minutes=self._system_tick * self.tick_minutes
        )

    def _recompute_global_tick(self) -> None:
        """
        叠加式全局时钟合成：

        - 患者层：累计所有患者的个人推进（体现总就诊负载）
        - 资源层：累计设备/队列/预约等资源调度推进
        - 系统层：累计系统级事件推进

        与 max 模式相比，该模式可反映多患者并发下的总体时间负载。
        """
        patient_sum = sum(p.personal_elapsed_ticks for p in self._patients.values())
        self._global_tick = patient_sum + self._resource_tick + self._system_tick

    def register_patient(self, patient_id: str) -> datetime:
        """注册或重置患者个人时钟。"""
        with self._lock:
            self._patients[patient_id] = PatientTickState(
                patient_id=patient_id,
                join_tick=self._global_tick,
                lane_tick=self._global_tick,
                personal_elapsed_ticks=0,
                busy_until_tick=self._global_tick,
            )
            return self.current_datetime

    def unregister_patient(self, patient_id: str) -> None:
        """移除患者个人时钟状态。"""
        with self._lock:
            self._patients.pop(patient_id, None)

    def advance(
        self,
        minutes: float,
        patient_id: Optional[str] = None,
        *,
        affect_resource: bool = False,
        affect_system: bool = False,
        resource_actor_id: Optional[str] = None,
    ) -> datetime:
        """
        推进模拟时间。

                规则：
                - patient_id：推进患者个人时间轴（就诊有效时长）。
                - affect_resource：推进资源调度时钟（设备/队列），按 actor lane 取最大值，
                    避免并发患者导致资源时间串行叠加。
                - affect_system：推进系统级时钟。
                - 若三者都未指定，默认按系统级事件推进。
        """
        ticks = max(1, round(minutes / self.tick_minutes))

        with self._lock:
            touched = False

            if patient_id is not None:
                patient_state = self._patients.get(patient_id)
                if patient_state is None:
                    patient_state = PatientTickState(
                        patient_id=patient_id,
                        join_tick=self._global_tick,
                        lane_tick=self._global_tick,
                        personal_elapsed_ticks=0,
                        busy_until_tick=self._global_tick,
                    )
                    self._patients[patient_id] = patient_state

                patient_state.lane_tick += ticks
                patient_state.personal_elapsed_ticks += ticks
                patient_state.busy_until_tick = patient_state.lane_tick
                touched = True

            if affect_resource:
                actor = resource_actor_id or patient_id or "system"
                # 每个资源 actor 有独立 lane；资源总时钟取所有 lane 的最大值
                lane_base = self._resource_lanes.get(actor, self._resource_tick)
                lane_new = lane_base + ticks
                self._resource_lanes[actor] = lane_new
                if lane_new > self._resource_tick:
                    self._resource_tick = lane_new
                touched = True

            if affect_system:
                self._system_tick += ticks
                touched = True

            if not touched:
                self._system_tick += ticks

            self._recompute_global_tick()

            return self.current_datetime

    def patient_elapsed_minutes(self, patient_id: str) -> float:
        """获取患者有效就诊时长（分钟）。"""
        with self._lock:
            patient_state = self._patients.get(patient_id)
            if patient_state is None:
                return 0.0
            return patient_state.personal_elapsed_ticks * self.tick_minutes

    def patient_current_datetime(self, patient_id: str) -> datetime:
        """获取患者个人当前时刻。"""
        with self._lock:
            patient_state = self._patients.get(patient_id)
            if patient_state is None:
                return self.current_datetime
            return self.start_datetime + timedelta(
                minutes=patient_state.lane_tick * self.tick_minutes
            )

    def is_patient_busy(self, patient_id: str) -> bool:
        """检查患者是否仍处于动作锁定中。"""
        with self._lock:
            patient_state = self._patients.get(patient_id)
            if patient_state is None:
                return False
            return patient_state.lane_tick < patient_state.busy_until_tick

    def summary(self) -> dict:
        """返回时钟状态摘要。"""
        with self._lock:
            return {
                "global_tick": self._global_tick,
                "global_datetime": self.current_datetime.strftime("%H:%M"),
                "resource_tick": self._resource_tick,
                "resource_datetime": self.resource_datetime.strftime("%H:%M"),
                "system_tick": self._system_tick,
                "system_datetime": self.system_datetime.strftime("%H:%M"),
                "tick_minutes": self.tick_minutes,
                "patients": {
                    patient_id: {
                        "elapsed_minutes": state.personal_elapsed_ticks * self.tick_minutes,
                        "lane_tick": state.lane_tick,
                        "busy_until_tick": state.busy_until_tick,
                        "personal_time": self.patient_current_datetime(patient_id).strftime("%H:%M"),
                    }
                    for patient_id, state in self._patients.items()
                },
            }