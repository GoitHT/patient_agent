"""
医院时间管理系统 - 事件驱动的时间推进机制
Time Management System - Event-driven time progression

功能：
1. 统一的时间推进机制 - 每个事件触发时间前进
2. 患者完整时间线记录 - 记录所有事件的时间戳
3. 资源时间槽管理 - 确保同一时间段内资源不被重复占用
4. 时间冲突检测 - 自动检测和防止时间冲突
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from utils import get_logger

logger = get_logger("hospital_agent.time_manager")


class EventType(Enum):
    """事件类型枚举"""
    # 患者流程事件
    PATIENT_ARRIVAL = "patient_arrival"              # 患者到达
    PATIENT_REGISTRATION = "patient_registration"    # 患者挂号
    PATIENT_TRIAGE = "patient_triage"                # 患者分诊
    PATIENT_WAITING = "patient_waiting"              # 患者候诊
    CONSULTATION_START = "consultation_start"        # 就诊开始
    CONSULTATION_END = "consultation_end"            # 就诊结束
    EXAM_ORDERED = "exam_ordered"                    # 检查开单
    EXAM_START = "exam_start"                        # 检查开始
    EXAM_END = "exam_end"                            # 检查结束
    LAB_TEST_ORDERED = "lab_test_ordered"            # 化验开单
    LAB_TEST_START = "lab_test_start"                # 化验开始
    LAB_TEST_END = "lab_test_end"                    # 化验结束
    PRESCRIPTION_ISSUED = "prescription_issued"      # 处方开具
    MEDICATION_DISPENSED = "medication_dispensed"    # 药品发放
    PATIENT_DISCHARGE = "patient_discharge"          # 患者离院
    
    # 资源事件
    RESOURCE_ALLOCATED = "resource_allocated"        # 资源分配
    RESOURCE_RELEASED = "resource_released"          # 资源释放
    RESOURCE_QUEUED = "resource_queued"              # 资源排队
    
    # 医生事件
    DOCTOR_START_SHIFT = "doctor_start_shift"        # 医生上班
    DOCTOR_END_SHIFT = "doctor_end_shift"            # 医生下班
    DOCTOR_BREAK_START = "doctor_break_start"        # 医生休息开始
    DOCTOR_BREAK_END = "doctor_break_end"            # 医生休息结束


@dataclass
class TimeEvent:
    """时间事件 - 记录单个事件的详细信息"""
    event_type: EventType
    timestamp: datetime
    patient_id: Optional[str] = None
    resource_id: Optional[str] = None  # 医生ID、设备ID等
    resource_type: Optional[str] = None  # doctor, equipment, room
    location: Optional[str] = None
    duration_minutes: int = 0  # 事件持续时间（分钟）
    metadata: Dict = field(default_factory=dict)  # 额外信息
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {self.event_type.value} - Patient: {self.patient_id}, Resource: {self.resource_id}"


@dataclass
class ResourceTimeSlot:
    """资源时间槽 - 用于跟踪资源占用情况"""
    resource_id: str
    resource_type: str  # doctor, equipment, room
    start_time: datetime
    end_time: datetime
    patient_id: str
    event_type: EventType
    
    def overlaps_with(self, other: ResourceTimeSlot) -> bool:
        """检查是否与另一个时间槽重叠"""
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def __str__(self):
        return f"{self.resource_id} [{self.start_time.strftime('%H:%M')}-{self.end_time.strftime('%H:%M')}] 占用者: {self.patient_id}"


class PatientTimeline:
    """患者时间线 - 记录患者的完整就诊流程"""
    
    def __init__(self, patient_id: str, arrival_time: datetime):
        self.patient_id = patient_id
        self.arrival_time = arrival_time
        self.events: List[TimeEvent] = []
        self._lock = threading.RLock()
        
        # 关键时间点
        self.registration_time: Optional[datetime] = None
        self.triage_time: Optional[datetime] = None
        self.consultation_start_time: Optional[datetime] = None
        self.consultation_end_time: Optional[datetime] = None
        self.discharge_time: Optional[datetime] = None
        
        # 统计信息
        self.total_waiting_time_minutes: int = 0
        self.total_consultation_time_minutes: int = 0
        self.total_exam_time_minutes: int = 0
        
        # 记录到达事件
        self.add_event(TimeEvent(
            event_type=EventType.PATIENT_ARRIVAL,
            timestamp=arrival_time,
            patient_id=patient_id
        ))
    
    def add_event(self, event: TimeEvent):
        """添加事件到时间线"""
        with self._lock:
            self.events.append(event)
            self._update_key_times(event)
            logger.debug(f"[P{self.patient_id}] 添加事件: {event.event_type.value} @ {event.timestamp.strftime('%H:%M:%S')}")
    
    def _update_key_times(self, event: TimeEvent):
        """更新关键时间点"""
        if event.event_type == EventType.PATIENT_REGISTRATION:
            self.registration_time = event.timestamp
        elif event.event_type == EventType.PATIENT_TRIAGE:
            self.triage_time = event.timestamp
        elif event.event_type == EventType.CONSULTATION_START:
            self.consultation_start_time = event.timestamp
        elif event.event_type == EventType.CONSULTATION_END:
            self.consultation_end_time = event.timestamp
            if self.consultation_start_time:
                self.total_consultation_time_minutes = int(
                    (event.timestamp - self.consultation_start_time).total_seconds() / 60
                )
        elif event.event_type == EventType.PATIENT_DISCHARGE:
            self.discharge_time = event.timestamp
    
    def get_total_duration(self) -> Optional[int]:
        """获取总就诊时长（分钟）"""
        if self.discharge_time:
            return int((self.discharge_time - self.arrival_time).total_seconds() / 60)
        return None
    
    def get_current_status(self) -> str:
        """获取当前状态"""
        if not self.events:
            return "未知"
        
        last_event = self.events[-1]
        status_map = {
            EventType.PATIENT_ARRIVAL: "已到达",
            EventType.PATIENT_REGISTRATION: "已挂号",
            EventType.PATIENT_TRIAGE: "已分诊",
            EventType.PATIENT_WAITING: "候诊中",
            EventType.CONSULTATION_START: "就诊中",
            EventType.CONSULTATION_END: "就诊完成",
            EventType.EXAM_START: "检查中",
            EventType.LAB_TEST_START: "化验中",
            EventType.PATIENT_DISCHARGE: "已离院"
        }
        return status_map.get(last_event.event_type, "处理中")
    
    def generate_report(self) -> str:
        """生成时间线报告"""
        lines = [
            f"\n{'='*60}",
            f"患者时间线报告 - {self.patient_id}",
            f"{'='*60}",
            f"到达时间: {self.arrival_time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if self.discharge_time:
            total_duration = self.get_total_duration()
            lines.append(f"离院时间: {self.discharge_time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"总就诊时长: {total_duration} 分钟")
        
        lines.append(f"\n当前状态: {self.get_current_status()}")
        lines.append(f"\n事件时间线:")
        lines.append(f"{'-'*60}")
        
        for i, event in enumerate(self.events, 1):
            time_str = event.timestamp.strftime('%H:%M:%S')
            duration_str = f" (持续{event.duration_minutes}分钟)" if event.duration_minutes > 0 else ""
            resource_str = f" - {event.resource_type}: {event.resource_id}" if event.resource_id else ""
            location_str = f" @ {event.location}" if event.location else ""
            
            lines.append(f"{i:2d}. [{time_str}] {event.event_type.value}{duration_str}{resource_str}{location_str}")
            
            if event.metadata:
                for key, value in event.metadata.items():
                    lines.append(f"     └─ {key}: {value}")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class TimeManager:
    """时间管理器 - 统一管理医院系统的时间推进和事件记录"""
    
    def __init__(self, start_time: datetime = None):
        """
        初始化时间管理器
        
        Args:
            start_time: 起始时间，默认为当天8:00
        """
        self.current_time = start_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        self._lock = threading.RLock()
        
        # 患者时间线记录
        self.patient_timelines: Dict[str, PatientTimeline] = {}
        
        # 资源时间槽管理
        self.resource_slots: Dict[str, List[ResourceTimeSlot]] = {}  # resource_id -> [slots]
        
        # 全局事件日志
        self.global_events: List[TimeEvent] = []
        
        # 不显示初始化提示，避免冗余
    
    def advance_time(self, minutes: int, reason: str = "") -> datetime:
        """
        推进时间
        
        Args:
            minutes: 推进的分钟数
            reason: 推进原因（用于日志）
            
        Returns:
            新的当前时间
        """
        with self._lock:
            old_time = self.current_time
            self.current_time += timedelta(minutes=minutes)
            
            if reason:
                logger.debug(f"时间推进: {old_time.strftime('%H:%M')} -> {self.current_time.strftime('%H:%M')} (+{minutes}分钟) - {reason}")
            
            return self.current_time
    
    def register_patient(self, patient_id: str, arrival_time: datetime = None) -> PatientTimeline:
        """
        注册患者并创建时间线
        
        Args:
            patient_id: 患者ID
            arrival_time: 到达时间，默认为当前时间
            
        Returns:
            患者时间线对象
        """
        with self._lock:
            if patient_id in self.patient_timelines:
                logger.warning(f"患者 {patient_id} 已存在时间线")
                return self.patient_timelines[patient_id]
            
            arrival_time = arrival_time or self.current_time
            timeline = PatientTimeline(patient_id, arrival_time)
            self.patient_timelines[patient_id] = timeline
            
            logger.info(f"患者 {patient_id} 注册 - 到达时间: {arrival_time.strftime('%H:%M:%S')}")
            return timeline
    
    def record_event(self, event: TimeEvent) -> bool:
        """
        记录事件
        
        Args:
            event: 时间事件
            
        Returns:
            是否成功记录
        """
        with self._lock:
            # 添加到全局事件日志
            self.global_events.append(event)
            
            # 如果有患者ID，添加到患者时间线
            if event.patient_id and event.patient_id in self.patient_timelines:
                self.patient_timelines[event.patient_id].add_event(event)
            
            # 如果涉及资源占用，记录资源时间槽
            if event.resource_id and event.duration_minutes > 0:
                self._record_resource_slot(event)
            
            return True
    
    def _record_resource_slot(self, event: TimeEvent):
        """记录资源时间槽"""
        if not event.resource_id or event.duration_minutes <= 0:
            return
        
        slot = ResourceTimeSlot(
            resource_id=event.resource_id,
            resource_type=event.resource_type or "unknown",
            start_time=event.timestamp,
            end_time=event.timestamp + timedelta(minutes=event.duration_minutes),
            patient_id=event.patient_id or "unknown",
            event_type=event.event_type
        )
        
        if event.resource_id not in self.resource_slots:
            self.resource_slots[event.resource_id] = []
        
        self.resource_slots[event.resource_id].append(slot)
    
    def check_resource_availability(self, resource_id: str, start_time: datetime, 
                                   duration_minutes: int) -> Tuple[bool, Optional[str]]:
        """
        检查资源在指定时间段是否可用
        
        Args:
            resource_id: 资源ID
            start_time: 开始时间
            duration_minutes: 持续时间
            
        Returns:
            (是否可用, 冲突原因)
        """
        with self._lock:
            if resource_id not in self.resource_slots:
                return True, None
            
            end_time = start_time + timedelta(minutes=duration_minutes)
            test_slot = ResourceTimeSlot(
                resource_id=resource_id,
                resource_type="test",
                start_time=start_time,
                end_time=end_time,
                patient_id="test",
                event_type=EventType.RESOURCE_ALLOCATED
            )
            
            for existing_slot in self.resource_slots[resource_id]:
                if test_slot.overlaps_with(existing_slot):
                    conflict_reason = (
                        f"资源 {resource_id} 在 {start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} "
                        f"期间已被占用 (占用者: {existing_slot.patient_id}, "
                        f"占用时间: {existing_slot.start_time.strftime('%H:%M')}-{existing_slot.end_time.strftime('%H:%M')})"
                    )
                    return False, conflict_reason
            
            return True, None
    
    def allocate_resource(self, patient_id: str, resource_id: str, resource_type: str,
                         duration_minutes: int, location: str = None,
                         event_type: EventType = EventType.RESOURCE_ALLOCATED,
                         metadata: Dict = None) -> Tuple[bool, Optional[str]]:
        """
        分配资源并记录事件
        
        Args:
            patient_id: 患者ID
            resource_id: 资源ID
            resource_type: 资源类型
            duration_minutes: 持续时间
            location: 位置
            event_type: 事件类型
            metadata: 额外信息
            
        Returns:
            (是否成功, 错误信息)
        """
        with self._lock:
            # 检查资源可用性
            available, conflict_reason = self.check_resource_availability(
                resource_id, self.current_time, duration_minutes
            )
            
            if not available:
                logger.warning(f"资源分配冲突: {conflict_reason}")
                return False, conflict_reason
            
            # 记录事件
            event = TimeEvent(
                event_type=event_type,
                timestamp=self.current_time,
                patient_id=patient_id,
                resource_id=resource_id,
                resource_type=resource_type,
                location=location,
                duration_minutes=duration_minutes,
                metadata=metadata or {}
            )
            
            self.record_event(event)
            
            logger.info(f"资源分配成功: 患者 {patient_id} -> {resource_type} {resource_id} "
                       f"({duration_minutes}分钟) @ {self.current_time.strftime('%H:%M')}")
            
            return True, None
    
    def release_resource(self, patient_id: str, resource_id: str, resource_type: str,
                        location: str = None) -> bool:
        """
        释放资源
        
        Args:
            patient_id: 患者ID
            resource_id: 资源ID
            resource_type: 资源类型
            location: 位置
            
        Returns:
            是否成功
        """
        event = TimeEvent(
            event_type=EventType.RESOURCE_RELEASED,
            timestamp=self.current_time,
            patient_id=patient_id,
            resource_id=resource_id,
            resource_type=resource_type,
            location=location
        )
        
        self.record_event(event)
        logger.info(f"资源释放: 患者 {patient_id} 释放 {resource_type} {resource_id}")
        
        return True
    
    def get_patient_timeline(self, patient_id: str) -> Optional[PatientTimeline]:
        """获取患者时间线"""
        return self.patient_timelines.get(patient_id)
    
    def get_resource_schedule(self, resource_id: str) -> List[ResourceTimeSlot]:
        """获取资源的完整日程"""
        return self.resource_slots.get(resource_id, [])
    
    def get_current_time(self) -> datetime:
        """获取当前时间"""
        return self.current_time
    
    def generate_summary_report(self) -> str:
        """生成汇总报告"""
        lines = [
            f"\n{'='*80}",
            f"医院时间管理系统 - 汇总报告",
            f"{'='*80}",
            f"当前时间: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"总患者数: {len(self.patient_timelines)}",
            f"总事件数: {len(self.global_events)}",
            f"总资源数: {len(self.resource_slots)}",
            f"\n患者状态统计:",
            f"{'-'*80}"
        ]
        
        status_count = {}
        for timeline in self.patient_timelines.values():
            status = timeline.get_current_status()
            status_count[status] = status_count.get(status, 0) + 1
        
        for status, count in sorted(status_count.items()):
            lines.append(f"{status}: {count} 人")
        
        lines.append(f"\n资源使用统计:")
        lines.append(f"{'-'*80}")
        for resource_id, slots in sorted(self.resource_slots.items()):
            total_minutes = sum((slot.end_time - slot.start_time).total_seconds() / 60 
                              for slot in slots)
            lines.append(f"{resource_id}: 使用 {len(slots)} 次, 总计 {int(total_minutes)} 分钟")
        
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)
