"""
医院物理环境模拟系统 - 基于 ScienceWorld 思想
实现真实的物理空间、时间和资源约束
"""
from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set


@dataclass
class Location:
    """物理位置"""
    id: str
    name: str
    type: str  # 'lobby', 'clinic', 'lab', 'imaging', 'pharmacy', 'triage'
    connected_to: List[str] = field(default_factory=list)  # 相邻房间ID
    capacity: int = 1  # 同时容纳人数
    current_occupants: Set[str] = field(default_factory=set)  # 当前在此位置的Agent ID
    available_actions: List[str] = field(default_factory=list)  # 可执行动作
    devices: List[str] = field(default_factory=list)  # 可用设备列表


@dataclass
class QueueEntry:
    """队列条目 - 支持优先级"""
    patient_id: str
    priority: int = 5  # 1-10, 1最高优先级（急诊），10最低
    enqueue_time: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """比较优先级，用于排序"""
        if self.priority != other.priority:
            return self.priority < other.priority  # 数字小的优先
        return self.enqueue_time < other.enqueue_time  # 相同优先级按时间


@dataclass
class Equipment:
    """医疗设备 - 增强版，支持优先级队列、状态管理、预约系统"""
    id: str
    name: str
    location_id: str
    exam_type: str  # 'xray', 'ct', 'mri', 'blood_test', 'ecg', 'ultrasound'
    duration_minutes: int  # 检查所需时间
    is_occupied: bool = False
    occupied_until: Optional[datetime] = None
    current_patient: Optional[str] = None  # 当前正在使用的患者ID
    queue: List[QueueEntry] = field(default_factory=list)  # 优先级队列
    status: str = "available"  # available, occupied, maintenance, offline
    maintenance_until: Optional[datetime] = None  # 维护结束时间
    daily_usage_count: int = 0  # 当天使用次数
    max_daily_usage: int = 50  # 每天最大使用次数
    reservation_slots: Dict[str, str] = field(default_factory=dict)  # 时间槽预约 {"HH:MM": patient_id}
    
    def can_use(self, current_time: datetime) -> bool:
        """检查设备是否可用"""
        # 检查设备状态
        if self.status == "offline":
            return False
        
        if self.status == "maintenance":
            if self.maintenance_until and current_time >= self.maintenance_until:
                self.status = "available"
            else:
                return False
        
        # 检查每日使用限制
        if self.daily_usage_count >= self.max_daily_usage:
            return False
        
        # 检查占用状态
        if not self.is_occupied:
            return True
        
        if self.occupied_until and current_time >= self.occupied_until:
            return True
        
        return False
    
    def start_exam(self, patient_id: str, current_time: datetime, priority: int = 5):
        """开始检查"""
        self.is_occupied = True
        self.status = "occupied"
        self.current_patient = patient_id
        self.occupied_until = current_time + timedelta(minutes=self.duration_minutes)
        self.daily_usage_count += 1
        
        # 从队列中移除
        self.queue = [entry for entry in self.queue if entry.patient_id != patient_id]
    
    def finish_exam(self, current_time: datetime) -> Optional[str]:
        """结束检查（如果时间到了），返回完成检查的患者ID"""
        if self.is_occupied and self.occupied_until and current_time >= self.occupied_until:
            finished_patient = self.current_patient
            self.is_occupied = False
            self.status = "available"
            self.current_patient = None
            self.occupied_until = None
            return finished_patient
        return None
    
    def add_to_queue(self, patient_id: str, priority: int = 5, current_time: datetime = None):
        """加入优先级队列"""
        # 检查是否已在队列
        for entry in self.queue:
            if entry.patient_id == patient_id:
                return  # 已经在队列中
        
        entry = QueueEntry(
            patient_id=patient_id,
            priority=priority,
            enqueue_time=current_time or datetime.now()
        )
        self.queue.append(entry)
        # 按优先级排序
        self.queue.sort()
    
    def get_next_patient(self) -> Optional[str]:
        """获取下一个应该检查的患者（最高优先级）"""
        if self.queue:
            return self.queue[0].patient_id
        return None
    
    def get_wait_time(self, current_time: datetime, patient_id: str = None) -> int:
        """获取预计等待时间（分钟）"""
        if self.status not in ["available", "occupied"]:
            return 999  # 设备不可用
        
        if self.can_use(current_time):
            return 0
        
        wait_minutes = 0
        
        # 当前检查剩余时间
        if self.occupied_until:
            remaining = (self.occupied_until - current_time).total_seconds() / 60
            wait_minutes = max(0, int(remaining))
        
        # 计算队列中该患者前面的等待时间
        if patient_id:
            patient_position = None
            for i, entry in enumerate(self.queue):
                if entry.patient_id == patient_id:
                    patient_position = i
                    break
            
            if patient_position is not None:
                # 只计算前面的人
                wait_minutes += patient_position * self.duration_minutes
            else:
                # 不在队列中，计算所有人
                wait_minutes += len(self.queue) * self.duration_minutes
        else:
            # 没有指定患者，计算队列总时间
            wait_minutes += len(self.queue) * self.duration_minutes
        
        return wait_minutes
    
    def reserve_slot(self, time_slot: str, patient_id: str) -> bool:
        """预约时间槽（格式：HH:MM）"""
        if time_slot in self.reservation_slots:
            return False  # 已被预约
        self.reservation_slots[time_slot] = patient_id
        return True
    
    def cancel_reservation(self, patient_id: str):
        """取消预约"""
        slots_to_remove = [slot for slot, pid in self.reservation_slots.items() if pid == patient_id]
        for slot in slots_to_remove:
            del self.reservation_slots[slot]
    
    def reset_daily_usage(self):
        """重置每日使用计数（每天开始时调用）"""
        self.daily_usage_count = 0
    
    def has_patient_in_queue(self, patient_id: str) -> bool:
        """检查患者是否在队列中"""
        return any(entry.patient_id == patient_id for entry in self.queue)
    
    def __contains__(self, patient_id: str) -> bool:
        """支持 'patient_id in equipment.queue' 语法（实际检查queue中的患者）"""
        return self.has_patient_in_queue(patient_id)


@dataclass
class Symptom:
    """症状数据类 - Level 3 增强"""
    name: str
    severity: float = 5.0  # 0-10，浮点数更精确
    trend: str = "stable"  # improving, stable, worsening
    onset_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    progression_rate: float = 0.1  # 每小时变化率
    treated: bool = False  # 是否已治疗
    treatment_effectiveness: float = 0.0  # 治疗有效性 0-1
    
    def progress(self, hours: float):
        """症状随时间演变"""
        if self.treated:
            # 治疗后症状改善
            change = -self.progression_rate * self.treatment_effectiveness * hours
        else:
            # 未治疗时可能恶化
            if self.severity > 7:
                # 严重症状更容易恶化，确保恶化
                change = self.progression_rate * hours * 1.5
            elif self.severity > 4:
                # 中度症状：轻微恶化或保持稳定
                change = random.uniform(0.0, 0.2) * hours
            else:
                # 轻度症状可能自然波动（包括轻微改善）
                change = random.uniform(-0.1, 0.3) * hours
        
        old_severity = self.severity
        self.severity = max(0.0, min(10.0, self.severity + change))
        
        # 更新趋势
        if self.severity > old_severity + 0.5:
            self.trend = "worsening"
        elif self.severity < old_severity - 0.5:
            self.trend = "improving"
        else:
            self.trend = "stable"
        
        self.last_update = datetime.now()
    
    def apply_treatment(self, effectiveness: float = 0.8):
        """应用治疗"""
        self.treated = True
        self.treatment_effectiveness = effectiveness


@dataclass
class VitalSign:
    """生命体征数据类 - Level 3 增强"""
    name: str
    value: float
    unit: str
    normal_range: tuple[float, float]  # (min, max)
    last_measured: datetime = field(default_factory=datetime.now)
    history: List[tuple[datetime, float]] = field(default_factory=list)  # 历史记录
    
    def is_normal(self) -> bool:
        """检查是否在正常范围"""
        return self.normal_range[0] <= self.value <= self.normal_range[1]
    
    def get_status(self) -> str:
        """获取状态描述"""
        if self.value < self.normal_range[0]:
            deviation = abs(self.value - self.normal_range[0]) / self.normal_range[0] * 100
            if deviation > 20:
                return "严重偏低"
            elif deviation > 10:
                return "偏低"
            else:
                return "略低"
        elif self.value > self.normal_range[1]:
            deviation = abs(self.value - self.normal_range[1]) / self.normal_range[1] * 100
            if deviation > 20:
                return "严重偏高"
            elif deviation > 10:
                return "偏高"
            else:
                return "略高"
        return "正常"
    
    def update(self, new_value: float, current_time: datetime):
        """更新生命体征"""
        self.history.append((self.last_measured, self.value))
        # 保留最近24小时的记录
        cutoff = current_time - timedelta(hours=24)
        self.history = [(t, v) for t, v in self.history if t >= cutoff]
        
        self.value = new_value
        self.last_measured = current_time
    
    def simulate_change(self, hours: float, symptoms: Dict[str, Symptom]):
        """根据症状模拟生命体征变化"""
        # 简单模拟：症状越严重，生命体征越可能异常
        total_severity = sum(s.severity for s in symptoms.values())
        
        if total_severity > 20:  # 多个重症状
            # 向异常方向漂移
            if random.random() > 0.5:
                change = random.uniform(0.5, 2.0) * hours
            else:
                change = random.uniform(-2.0, -0.5) * hours
        else:
            # 轻微波动
            change = random.uniform(-0.5, 0.5) * hours
        
        new_value = self.value + change
        self.update(new_value, datetime.now())


@dataclass
class PhysicalState:
    """物理状态 - Level 3 增强版：动态生理模拟
    
    支持患者和医护人员的物理状态建模：
    - 患者：完整的生理状态（症状、生命体征、体力等）
    - 医护人员：工作状态（体力、工作负荷、连续工作时间）
    """
    patient_id: str
    vital_signs: Dict[str, VitalSign] = field(default_factory=dict)  # 生命体征（患者）
    symptoms: Dict[str, Symptom] = field(default_factory=dict)  # 症状（患者）
    last_update: Optional[datetime] = None  # 最后更新时间
    energy_level: float = 10.0  # 体力水平 0-10
    pain_level: float = 0.0  # 疼痛水平 0-10（患者）
    consciousness_level: str = "alert"  # alert, drowsy, unconscious
    diagnosis: Optional[str] = None  # 诊断（患者）
    medications: List[Dict] = field(default_factory=list)  # 药物列表（患者）
    treatments: List[Dict] = field(default_factory=list)  # 治疗记录
    
    # 【新增】医护人员专属属性
    agent_type: str = "patient"  # patient, doctor, nurse, lab_technician
    work_load: float = 0.0  # 工作负荷 0-10（医护人员）
    consecutive_work_minutes: int = 0  # 连续工作时长（分钟）
    patients_served_today: int = 0  # 今日服务患者数（医护人员）
    last_rest_time: Optional[datetime] = None  # 上次休息时间
    
    def __post_init__(self):
        """初始化默认生命体征"""
        if self.last_update is None:
            self.last_update = datetime.now()
            
        if not self.vital_signs:
            self.vital_signs = {
                "heart_rate": VitalSign("心率", 75.0, "次/分", (60, 100)),
                "blood_pressure_systolic": VitalSign("收缩压", 120.0, "mmHg", (90, 140)),
                "blood_pressure_diastolic": VitalSign("舒张压", 80.0, "mmHg", (60, 90)),
                "temperature": VitalSign("体温", 36.5, "℃", (36.0, 37.5)),
                "respiratory_rate": VitalSign("呼吸频率", 16.0, "次/分", (12, 20)),
                "oxygen_saturation": VitalSign("血氧饱和度", 98.0, "%", (95, 100)),
            }
    
    def add_symptom(self, name: str, severity: float = 5.0, progression_rate: float = 0.1):
        """添加症状"""
        self.symptoms[name] = Symptom(
            name=name,
            severity=severity,
            progression_rate=progression_rate
        )
    
    def update_symptom(self, name: str, severity: float):
        """更新症状严重程度
        
        Args:
            name: 症状名称
            severity: 新的严重程度 (0-10)
        """
        if name in self.symptoms:
            self.symptoms[name].severity = max(0.0, min(10.0, severity))
        else:
            self.add_symptom(name, severity)
    
    def update_vital_sign(self, name: str, value: float):
        """更新生命体征数值
        
        Args:
            name: 生命体征名称（如 'temperature', 'heart_rate'）
            value: 新数值
        """
        if name in self.vital_signs:
            self.vital_signs[name].update(value, datetime.now())
        else:
            # 如果不存在，创建新的生命体征（使用默认范围）
            default_ranges = {
                "heart_rate": (60, 100),
                "blood_pressure_systolic": (90, 140),
                "blood_pressure_diastolic": (60, 90),
                "temperature": (36.0, 37.5),
                "respiratory_rate": (12, 20),
                "oxygen_saturation": (95, 100),
            }
            
            default_units = {
                "heart_rate": "次/分",
                "blood_pressure_systolic": "mmHg",
                "blood_pressure_diastolic": "mmHg",
                "temperature": "℃",
                "respiratory_rate": "次/分",
                "oxygen_saturation": "%",
            }
            
            unit = default_units.get(name, "")
            normal_range = default_ranges.get(name, (0, 100))
            
            self.vital_signs[name] = VitalSign(name, value, unit, normal_range)
    
    def update_physiology(self, current_time: datetime):
        """更新生理状态 - 核心动态模拟方法"""
        elapsed_hours = (current_time - self.last_update).total_seconds() / 3600
        
        if elapsed_hours < 0.1:  # 至少10分钟更新一次
            return
        
        # 1. 症状演变
        for symptom in self.symptoms.values():
            symptom.progress(elapsed_hours)
        
        # 2. 生命体征变化
        for vital_sign in self.vital_signs.values():
            vital_sign.simulate_change(elapsed_hours, self.symptoms)
        
        # 3. 体力消耗
        # 症状越严重，体力消耗越快
        total_severity = sum(s.severity for s in self.symptoms.values())
        energy_loss = elapsed_hours * (1 + total_severity / 50)
        self.energy_level = max(0.0, self.energy_level - energy_loss)
        
        # 4. 疼痛水平计算
        pain_symptoms = ["疼痛", "头痛", "腹痛", "胸痛", "关节痛"]
        self.pain_level = sum(
            self.symptoms[s].severity 
            for s in pain_symptoms 
            if s in self.symptoms
        ) / len(pain_symptoms) if pain_symptoms else 0.0
        
        # 5. 意识水平评估
        self.assess_consciousness()
        
        # 6. 检查危急状态
        self.check_critical_condition()
        
        self.last_update = current_time
    
    def assess_consciousness(self):
        """评估意识水平"""
        # 基于生命体征和症状评估
        vital_abnormalities = sum(
            1 for vs in self.vital_signs.values() 
            if not vs.is_normal()
        )
        
        severe_symptoms = sum(
            1 for s in self.symptoms.values() 
            if s.severity > 8
        )
        
        if vital_abnormalities >= 3 or severe_symptoms >= 2:
            self.consciousness_level = "drowsy"
        elif vital_abnormalities >= 4 or severe_symptoms >= 3:
            self.consciousness_level = "unconscious"
        else:
            self.consciousness_level = "alert"
    
    def check_critical_condition(self) -> bool:
        """检查是否处于危急状态"""
        # 检查生命体征是否危急
        critical_vitals = []
        
        hr = self.vital_signs.get("heart_rate")
        if hr and (hr.value < 40 or hr.value > 150):
            critical_vitals.append("心率异常")
        
        bp_sys = self.vital_signs.get("blood_pressure_systolic")
        if bp_sys and (bp_sys.value < 80 or bp_sys.value > 180):
            critical_vitals.append("血压异常")
        
        temp = self.vital_signs.get("temperature")
        if temp and (temp.value < 35.0 or temp.value > 40.0):
            critical_vitals.append("体温异常")
        
        o2 = self.vital_signs.get("oxygen_saturation")
        if o2 and o2.value < 90:
            critical_vitals.append("血氧过低")
        
        return len(critical_vitals) > 0
    
    def apply_medication(self, medication: str, effectiveness: float = 0.8):
        """应用药物治疗"""
        self.medications.append({
            "name": medication,
            "time": datetime.now(),
            "effectiveness": effectiveness
        })
        
        # 对相关症状应用治疗
        for symptom in self.symptoms.values():
            if not symptom.treated:
                symptom.apply_treatment(effectiveness)
    
    def record_treatment(self, treatment_type: str, details: str):
        """记录治疗"""
        self.treatments.append({
            "type": treatment_type,
            "details": details,
            "time": datetime.now()
        })
    
    def get_status_summary(self) -> str:
        """获取状态摘要"""
        lines = []
        lines.append(f"【患者状态摘要】")
        lines.append(f"意识: {self.consciousness_level}")
        lines.append(f"体力: {self.energy_level:.1f}/10")
        lines.append(f"疼痛: {self.pain_level:.1f}/10")
        
        if self.vital_signs:
            lines.append("\n【生命体征】")
            for vs in self.vital_signs.values():
                status = vs.get_status()
                lines.append(f"  {vs.name}: {vs.value:.1f} {vs.unit} ({status})")
        
        if self.symptoms:
            lines.append("\n【症状】")
            for symptom in self.symptoms.values():
                trend_icon = {"improving": "↓", "stable": "→", "worsening": "↑"}.get(symptom.trend, "→")
                status = "轻度" if symptom.severity <= 3 else ("中度" if symptom.severity <= 6 else "重度")
                treated_mark = " [已治疗]" if symptom.treated else ""
                lines.append(f"  {symptom.name}: {symptom.severity:.1f}/10 ({status}) {trend_icon}{treated_mark}")
        
        if self.check_critical_condition():
            lines.append("\n⚠️ 警告：患者处于危急状态！")
        
        return "\n".join(lines)
    
    def get_vital_signs_dict(self) -> Dict[str, float]:
        """获取生命体征字典（用于兼容旧接口）"""
        return {name: vs.value for name, vs in self.vital_signs.items()}
    
    def get_symptom_severity_dict(self) -> Dict[str, float]:
        """获取症状严重程度字典（用于兼容旧接口）"""
        return {name: s.severity for name, s in self.symptoms.items()}
    
    # ===== 医护人员工作负荷管理和休息恢复 =====
    
    def apply_rest(self, duration_minutes: int, quality: float = 0.7):
        """应用休息恢复
        
        Args:
            duration_minutes: 休息时长（分钟）
            quality: 休息质量（0-1）
        """
        # 休息恢复体力
        recovery = (duration_minutes / 60) * quality * 2.0
        self.energy_level = min(10.0, self.energy_level + recovery)
        
        # 医护人员休息时降低工作负荷
        if self.agent_type in ["doctor", "nurse", "lab_technician"]:
            self.work_load = max(0.0, self.work_load - duration_minutes * 0.1)
            self.last_rest_time = datetime.now()
            self.consecutive_work_minutes = 0
        
        # 良好的休息可能轻微缓解症状（仅患者）
        if quality > 0.6 and self.agent_type == "patient":
            for symptom in self.symptoms.values():
                if symptom.severity < 8:  # 不太严重的症状可能缓解
                    relief = quality * 0.1 * (duration_minutes / 30)
                    symptom.severity = max(0.0, symptom.severity - relief)
        
        self.record_treatment("rest", f"休息{duration_minutes}分钟，质量{quality:.1f}")
    
    # ===== 新增：医护人员工作负荷管理 =====
    
    def add_work_load(self, task_type: str, duration_minutes: int, complexity: float = 0.5):
        """增加医护人员工作负荷
        
        Args:
            task_type: 任务类型（'consultation', 'diagnosis', 'triage', 'lab_test'）
            duration_minutes: 任务持续时间
            complexity: 任务复杂度（0-1）
        """
        if self.agent_type not in ["doctor", "nurse", "lab_technician"]:
            return  # 只对医护人员生效
        
        # 计算工作负荷增量
        base_load = duration_minutes * 0.05  # 基础负荷
        complexity_load = complexity * 0.3  # 复杂度加成
        fatigue_multiplier = 1.0 + (self.consecutive_work_minutes / 180)  # 连续工作疲劳系数
        
        total_load = (base_load + complexity_load) * fatigue_multiplier
        self.work_load = min(10.0, self.work_load + total_load)
        
        # 工作消耗体力
        energy_cost = duration_minutes * 0.02 * (1 + complexity)
        self.energy_level = max(0.0, self.energy_level - energy_cost)
        
        # 累计连续工作时间
        self.consecutive_work_minutes += duration_minutes
        
        # 记录工作
        self.record_treatment(f"work_{task_type}", 
                            f"{task_type}任务{duration_minutes}分钟，复杂度{complexity:.1f}")
    
    def serve_patient(self):
        """记录服务一位患者（医护人员）"""
        if self.agent_type in ["doctor", "nurse", "lab_technician"]:
            self.patients_served_today += 1
    
    def get_work_efficiency(self) -> float:
        """获取工作效率（0-1）
        
        Returns:
            工作效率，受体力和工作负荷影响
        """
        if self.agent_type not in ["doctor", "nurse", "lab_technician"]:
            return 1.0
        
        # 体力影响
        energy_factor = self.energy_level / 10.0
        
        # 工作负荷影响（负荷过高降低效率）
        load_factor = 1.0 if self.work_load < 5.0 else (1.0 - (self.work_load - 5.0) * 0.1)
        
        # 连续工作时间影响
        fatigue_factor = 1.0 if self.consecutive_work_minutes < 120 else \
                        (1.0 - (self.consecutive_work_minutes - 120) * 0.001)
        
        efficiency = energy_factor * load_factor * fatigue_factor
        return max(0.1, min(1.0, efficiency))
    
    def get_staff_status_summary(self) -> str:
        """获取医护人员状态摘要
        
        Returns:
            状态摘要字符串
        """
        if self.agent_type not in ["doctor", "nurse", "lab_technician"]:
            return ""
        
        lines = []
        lines.append(f"【{self.agent_type.upper()}状态】")
        lines.append(f"体力: {self.energy_level:.1f}/10")
        lines.append(f"工作负荷: {self.work_load:.1f}/10")
        lines.append(f"连续工作: {self.consecutive_work_minutes}分钟")
        lines.append(f"今日服务: {self.patients_served_today}人")
        lines.append(f"工作效率: {self.get_work_efficiency()*100:.0f}%")
        
        # 状态建议
        if self.energy_level < 3.0:
            lines.append("⚠️  体力严重不足，建议休息")
        elif self.work_load > 8.0:
            lines.append("⚠️  工作负荷过高，注意调节")
        elif self.consecutive_work_minutes > 180:
            lines.append("⚠️  连续工作过久，需要休息")
        else:
            lines.append("✓ 状态良好")
        
        return "\n".join(lines)


class HospitalWorld:
    """医院世界环境 - 物理空间模拟"""
    
    def __init__(self, start_time: datetime = None):
        """初始化医院世界"""
        # ===== 线程安全：添加可重入锁保护共享状态 =====
        self._lock = threading.RLock()
        
        self.current_time = start_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        
        # 定义医院房间（简化版 - 字典结构）
        self.locations = {
            "lobby": {
                "name": "门诊大厅",
                "devices": ["挂号机", "自助缴费机", "导诊台"],
                "capacity": 50
            },
            "triage": {
                "name": "分诊台",
                "devices": ["分诊系统", "叫号屏", "血压计", "体温计"],
                "capacity": 5
            },
            "waiting_area": {
                "name": "候诊区",
                "devices": ["叫号屏", "座椅", "饮水机"],
                "capacity": 30
            },
            "internal_medicine": {
                "name": "通用诊室",
                "devices": ["诊疗床", "电脑", "听诊器", "血压计", "体温计"],
                "capacity": 10
            },
            "lab": {
                "name": "检验科",
                "devices": ["血液分析仪", "生化分析仪", "尿液分析仪", "采血椅"],
                "capacity": 10
            },
            "pharmacy": {
                "name": "药房",
                "devices": ["药品柜", "发药窗口", "药品管理系统"],
                "capacity": 10
            },
            "cashier": {
                "name": "收费处",
                "devices": ["收费系统", "POS机", "发票打印机"],
                "capacity": 10
            },
            "emergency": {
                "name": "急诊室",
                "devices": ["抢救床", "心电监护仪", "除颤仪", "呼吸机", "急救车"],
                "capacity": 15
            }
        }
        
        # 定义允许的移动路径（有向图）
        # 按照真实就医流程设计：大厅→分诊→候诊→诊室→检验/收费→药房→大厅
        # allowed_moves 将在 _build_hospital() 中根据 Location.connected_to 自动构建
        
        self.equipment: Dict[str, Equipment] = {}
        self.agents: Dict[str, str] = {}  # agent_id -> location_id
        self.physical_states: Dict[str, PhysicalState] = {}
        self.event_log: List[Dict] = []  # 事件日志
        
        # ===== 性能优化：缓存和限制 =====
        # 位置名称缓存（避免重复字典查找）
        self._location_name_cache: Dict[str, str] = {}
        
        # 日志大小限制（防止内存泄漏）
        self._max_log_entries = 10000  # 每类日志最多保留1万条
        self._log_cleanup_threshold = 12000  # 超过此阈值时触发清理
        
        # 工作时间
        self.working_hours = {
            'start': 8,
            'end': 18,
            'lunch_start': 12,
            'lunch_end': 13,
        }
        
        # ===== 医生资源池管理 =====
        # 医生队列：{dept: {doctor_id: {'status': 'available'/'busy', 'current_patient': patient_id, 'queue': [patient_ids]}}}
        self.doctor_pool: Dict[str, Dict[str, Dict]] = {}
        # 患者-医生映射
        self.patient_doctor_map: Dict[str, str] = {}  # patient_id -> doctor_id
        
        # 初始化医院环境
        self._build_hospital()
    
    def _build_hospital(self):
        """构建医院物理结构 - 神经内科专科配置
        
        本系统仅模拟神经内科一个科室，包含以下设施：
        
        【检查设备分类】
        1. 影像科设备:
           - 头颅CT (2台): 急性卒中、颅内出血、脑外伤的首选检查
           - 脑MRI (2台): 脑梗死、脱髓鞘病、肿瘤、血管畸形的精确诊断
        
        2. 神经电生理室设备:
           - 脑电图EEG (2台): 癫痫诊断、意识障碍评估、脑炎监测
           - 肌电图EMG (2台): 周围神经病、肌病、神经肌肉接头疾病诊断
           - TCD超声 (2台): 脑血管狭窄、栓塞评估、血流动力学监测
        
        3. 检验科设备（神经内科常用）:
           - 血常规 (3台): 感染、贫血、血液病筛查
           - 基础生化 (3台): 肝肾功能、血糖、血脂、尿酸等代谢指标
           - 电解质 (2台): 钠、钾、氯、钙、镁等电解质紊乱检测
           - 凝血功能 (2台): 卒中患者凝血状态、抗凝治疗监测
           - 炎症/感染指标 (2台): CRP、PCT、ESR等感染标志物
           - 心脑血管标志物 (2台): 心肌酶、肌钙蛋白、BNP、同型半胱氨酸等卒中风险评估
           - 自免抗体 (2台): 自免性脑炎、多发性硬化、重症肌无力等自免性疾病检测
        
        【资源竞争机制】
        - 每台设备有独立的队列系统，支持优先级排队
        - 设备使用时长根据实际检查时间设定（8-55分钟不等）
        - 每台设备有每日使用次数上限，防止过度使用
        - 支持设备维护、故障等状态模拟
        - 自动推进检查流程，检查完成后自动分配给下一位患者
        """
        # 创建位置 - 仅保留神经内科相关位置
        locations = [
            Location(
                "lobby", 
                "门诊大厅", 
                "lobby", 
                connected_to=["triage", "neuro", "pharmacy", "lab", "imaging"],
                capacity=50,
                available_actions=["register", "wait", "move", "look"],
                devices=["挂号机", "自助缴费机", "导诊台"]
            ),
            
            Location(
                "triage", 
                "分诊台", 
                "triage",
                connected_to=["lobby", "waiting_area"],
                capacity=3,
                available_actions=["triage", "consult", "move", "look"],
                devices=["分诊系统", "叫号屏", "血压计", "体温计"]
            ),
            
            Location(
                "waiting_area",
                "候诊区",
                "waiting",
                connected_to=["triage", "lobby", "neuro"],
                capacity=30,
                available_actions=["wait", "move", "look"],
                devices=["叫号屏", "座椅", "饮水机"]
            ),
            
            Location(
                "cashier",
                "缴费处",
                "billing",
                connected_to=["lobby", "waiting_area"],
                capacity=10,
                available_actions=["pay", "wait", "move", "look"],
                devices=["收费系统", "POS机", "发票打印机", "自助缴费机"]
            ),
            
            Location(
                "neuro", 
                "神经内科诊室", 
                "clinic",
                connected_to=["lobby", "waiting_area", "lab", "imaging", "neurophysiology"],
                capacity=10,
                available_actions=["consult", "examine", "prescribe", "order_test", "move", "look"],
                devices=["HIS系统", "听诊器", "血压计", "神经检查工具", "反射锤"]
            ),
            
            Location(
                "lab", 
                "检验科", 
                "lab",
                connected_to=["lobby", "neuro"],
                capacity=10,
                available_actions=["blood_test", "wait", "move", "look"],
                devices=["LIS系统", "血液分析仪", "生化分析仪", "离心机", "采血台"]
            ),
            
            Location(
                "imaging", 
                "影像科", 
                "imaging",
                connected_to=["lobby", "neuro"],
                capacity=5,
                available_actions=["xray", "ct", "mri", "ultrasound", "wait", "move", "look"],
                devices=["RIS系统", "X光机", "CT机", "MRI机", "B超机"]
            ),
            
            Location(
                "neurophysiology", 
                "神经电生理室", 
                "neurophysiology",
                connected_to=["neuro"],
                capacity=3,
                available_actions=["eeg", "emg", "tcd", "wait", "move", "look"],
                devices=["神经电生理预约系统", "脑电图仪", "肌电图仪", "TCD超声仪", "检查床"]
            ),
            
            Location(
                "pharmacy", 
                "药房", 
                "pharmacy",
                connected_to=["lobby"],
                capacity=10,
                available_actions=["get_medicine", "wait", "move", "look"],
                devices=["药品管理系统", "自动配药机", "药品柜", "发药窗口"]
            ),
        ]
        
        for loc in locations:
            self.locations[loc.id] = loc
        
        # 根据Location的connected_to自动构建allowed_moves（双向图）
        self.allowed_moves = {}
        for loc in locations:
            if loc.id not in self.allowed_moves:
                self.allowed_moves[loc.id] = []
            
            for connected_id in loc.connected_to:
                # 添加单向连接
                if connected_id not in self.allowed_moves[loc.id]:
                    self.allowed_moves[loc.id].append(connected_id)
                
                # 添加反向连接（双向图）
                if connected_id not in self.allowed_moves:
                    self.allowed_moves[connected_id] = []
                if loc.id not in self.allowed_moves[connected_id]:
                    self.allowed_moves[connected_id].append(loc.id)
        
        # 创建设备 - 神经内科专科配置
        equipment_list = [
            # ========== 影像科设备 ==========
            # 头颅CT（用于急性卒中、颅内出血、脑外伤等，15-20分钟/例）
            Equipment("ct_head_1", "头颅CT机1号", "imaging", "ct_head", 20, max_daily_usage=60),
            Equipment("ct_head_2", "头颅CT机2号", "imaging", "ct_head", 20, max_daily_usage=60),
            
            # 脑MRI（用于脑梗死、脱髓鞘、肿瘤、血管畸形等，30-45分钟/例）
            Equipment("mri_brain_1", "脑MRI机1号", "imaging", "mri_brain", 40, max_daily_usage=35),
            Equipment("mri_brain_2", "脑MRI机2号", "imaging", "mri_brain", 40, max_daily_usage=35),
            
            # ========== 神经电生理室设备 ==========
            # 脑电图EEG（用于癫痫、意识障碍、脑炎等，30-40分钟/例）
            Equipment("eeg_1", "脑电图仪1号", "neurophysiology", "eeg", 35, max_daily_usage=25),
            Equipment("eeg_2", "脑电图仪2号", "neurophysiology", "eeg", 35, max_daily_usage=25),
            
            # 肌电图EMG（用于周围神经病、肌病、神经肌肉接头疾病等，30-50分钟/例）
            Equipment("emg_1", "肌电图仪1号", "neurophysiology", "emg", 40, max_daily_usage=20),
            Equipment("emg_2", "肌电图仪2号", "neurophysiology", "emg", 40, max_daily_usage=20),
            
            # TCD超声（用于脑血管狭窄、栓塞评估，20-30分钟/例）
            Equipment("tcd_1", "TCD超声仪1号", "neurophysiology", "tcd", 25, max_daily_usage=30),
            Equipment("tcd_2", "TCD超声仪2号", "neurophysiology", "tcd", 25, max_daily_usage=30),
            
            # ========== 检验科设备（神经内科常用检验项目）==========
            # 血常规CBC（白细胞、红细胞、血小板、血红蛋白等，10-15分钟/例）
            Equipment("cbc_analyzer_1", "血常规分析仪1号", "lab", "cbc", 12, max_daily_usage=250),
            Equipment("cbc_analyzer_2", "血常规分析仪2号", "lab", "cbc", 12, max_daily_usage=250),
            Equipment("cbc_analyzer_3", "血常规分析仪3号", "lab", "cbc", 12, max_daily_usage=250),
            
            # 基础生化（肝肾功能、血糖、血脂、尿酸等，20-30分钟/例）
            Equipment("biochem_basic_1", "基础生化分析仪1号", "lab", "biochem_basic", 25, max_daily_usage=200),
            Equipment("biochem_basic_2", "基础生化分析仪2号", "lab", "biochem_basic", 25, max_daily_usage=200),
            Equipment("biochem_basic_3", "基础生化分析仪3号", "lab", "biochem_basic", 25, max_daily_usage=200),
            
            # 电解质分析（钠、钾、氯、钙、镁等，神经系统功能相关，8-12分钟/例）
            Equipment("electrolyte_1", "电解质分析仪1号", "lab", "electrolyte", 10, max_daily_usage=250),
            Equipment("electrolyte_2", "电解质分析仪2号", "lab", "electrolyte", 10, max_daily_usage=250),
            
            # 凝血功能（PT、APTT、INR、D-二聚体等，卒中患者必查，15-20分钟/例）
            Equipment("coagulation_1", "凝血分析仪1号", "lab", "coagulation", 18, max_daily_usage=180),
            Equipment("coagulation_2", "凝血分析仪2号", "lab", "coagulation", 18, max_daily_usage=180),
            
            # 炎症/感染指标（CRP、PCT、ESR等，脑炎、脑膜炎必查，15-25分钟/例）
            Equipment("inflammation_1", "炎症标志物分析仪1号", "lab", "inflammation", 20, max_daily_usage=180),
            Equipment("inflammation_2", "炎症标志物分析仪2号", "lab", "inflammation", 20, max_daily_usage=180),
            
            # 心肌与血管风险指标（心肌酶、肌钙蛋白、BNP、同型半胱氨酸、脂蛋白a等，卒中风险评估，25-35分钟/例）
            Equipment("cardiac_stroke_1", "心脑血管标志物分析仪1号", "lab", "cardiac_stroke_markers", 30, max_daily_usage=150),
            Equipment("cardiac_stroke_2", "心脑血管标志物分析仪2号", "lab", "cardiac_stroke_markers", 30, max_daily_usage=150),
            
            # 自身免疫抗体（用于自免性脑炎、多发性硬化、重症肌无力等，45-60分钟/例）
            Equipment("autoimmune_1", "自免抗体检测仪1号", "lab", "autoimmune_antibody", 55, max_daily_usage=60),
            Equipment("autoimmune_2", "自免抗体检测仪2号", "lab", "autoimmune_antibody", 55, max_daily_usage=60),
        ]
        
        for eq in equipment_list:
            self.equipment[eq.id] = eq
        
        # 重建位置名称缓存（因为locations被覆盖了）
        self._rebuild_location_cache()
        
        # 输出设备初始化统计
        self._log_equipment_initialization()
    
    def _log_equipment_initialization(self):
        """记录设备初始化统计信息（调试用）"""
        import logging
        logger = logging.getLogger('hospital_agent.world')
        
        # 按类型统计设备
        equipment_by_type = {}
        for eq in self.equipment.values():
            if eq.exam_type not in equipment_by_type:
                equipment_by_type[eq.exam_type] = []
            equipment_by_type[eq.exam_type].append(eq)
        
        logger.info("=" * 70)
        logger.info("🏥 医院物理环境初始化完成 - 神经内科专科配置")
        logger.info("=" * 70)
        
        # 按位置分组展示
        logger.info("\n【影像科设备】位置: imaging")
        for exam_type in ["ct_head", "mri_brain"]:
            if exam_type in equipment_by_type:
                eq_list = equipment_by_type[exam_type]
                logger.info(f"  ├─ {eq_list[0].exam_type.upper()}: {len(eq_list)}台设备")
                for eq in eq_list:
                    logger.info(f"  │  └─ {eq.name} ({eq.duration_minutes}分钟/例, 最大{eq.max_daily_usage}例/日)")
        
        logger.info("\n【神经电生理室设备】位置: neurophysiology")
        for exam_type in ["eeg", "emg", "tcd"]:
            if exam_type in equipment_by_type:
                eq_list = equipment_by_type[exam_type]
                exam_names = {"eeg": "脑电图", "emg": "肌电图", "tcd": "TCD超声"}
                logger.info(f"  ├─ {exam_names.get(exam_type, exam_type)}: {len(eq_list)}台设备")
                for eq in eq_list:
                    logger.info(f"  │  └─ {eq.name} ({eq.duration_minutes}分钟/例, 最大{eq.max_daily_usage}例/日)")
        
        logger.info("\n【检验科设备】位置: lab")
        lab_types = ["cbc", "biochem_basic", "electrolyte", "coagulation", 
                     "inflammation", "cardiac_stroke_markers", "autoimmune_antibody"]
        lab_names = {
            "cbc": "血常规",
            "biochem_basic": "基础生化",
            "electrolyte": "电解质",
            "coagulation": "凝血功能",
            "inflammation": "炎症/感染指标",
            "cardiac_stroke_markers": "心脑血管标志物",
            "autoimmune_antibody": "自免抗体"
        }
        
        for exam_type in lab_types:
            if exam_type in equipment_by_type:
                eq_list = equipment_by_type[exam_type]
                logger.info(f"  ├─ {lab_names.get(exam_type, exam_type)}: {len(eq_list)}台设备")
                for eq in eq_list:
                    logger.info(f"  │  └─ {eq.name} ({eq.duration_minutes}分钟/例, 最大{eq.max_daily_usage}例/日)")
        
        # 统计总数
        total_equipment = len(self.equipment)
        total_daily_capacity = sum(eq.max_daily_usage for eq in self.equipment.values())
        
        logger.info(f"\n📊 设备统计:")
        logger.info(f"  ├─ 总设备数: {total_equipment}台")
        logger.info(f"  ├─ 每日总容量: {total_daily_capacity}例")
        logger.info(f"  └─ 平均处理时间: {sum(eq.duration_minutes for eq in self.equipment.values()) / total_equipment:.1f}分钟/例")
        
        logger.info("\n💡 资源竞争机制:")
        logger.info("  ├─ 优先级队列系统 (1-10级，1最高优先级)")
        logger.info("  ├─ 设备独立队列，自动流转到下一位患者")
        logger.info("  ├─ 每日使用次数限制，防止过度使用")
        logger.info("  └─ 支持设备维护、故障等状态模拟")
        
        logger.info("=" * 70 + "\n")
    
    def is_working_hours(self) -> bool:
        """检查是否在工作时间"""
        hour = self.current_time.hour
        
        # 午休时间
        if self.working_hours['lunch_start'] <= hour < self.working_hours['lunch_end']:
            return False
        
        # 工作时间
        return self.working_hours['start'] <= hour < self.working_hours['end']
    
    def advance_time(self, minutes: int = 1):
        """推进时间并更新所有状态"""
        with self._lock:
            old_time = self.current_time
            self.current_time += timedelta(minutes=minutes)
            
            # 检查是否跨天
            if old_time.date() != self.current_time.date():
                self._reset_daily_counters()
        
        # 更新设备状态并自动推进队列
        for equipment in self.equipment.values():
            # 检查并更新维护状态
            if equipment.status == "maintenance":
                if equipment.maintenance_until and self.current_time >= equipment.maintenance_until:
                    equipment.status = "available"
                    equipment.maintenance_until = None
                    self._log_event("maintenance_complete", {
                        "equipment": equipment.name,
                        "time": self.current_time.strftime("%H:%M")
                    })
            
            finished_patient = equipment.finish_exam(self.current_time)
            
            if finished_patient:
                # 记录检查完成
                self._log_event("exam_complete", {
                    "patient_id": finished_patient,
                    "equipment": equipment.name,
                    "time": self.current_time.strftime("%H:%M")
                })
                
                # 自动开始下一个检查（如果有排队）
                next_patient = equipment.get_next_patient()
                if next_patient and equipment.can_use(self.current_time):
                    # 检查患者是否还在该位置
                    if self.agents.get(next_patient) == equipment.location_id:
                        equipment.start_exam(next_patient, self.current_time)
                        self._log_event("exam_auto_start", {
                            "patient_id": next_patient,
                            "equipment": equipment.name,
                            "time": self.current_time.strftime("%H:%M")
                        })
        
        # 更新患者生理状态
        for state in self.physical_states.values():
            state.update_physiology(self.current_time)
        
        # 记录事件
        self._log_event("time_advance", {
            "from": old_time.strftime("%H:%M"),
            "to": self.current_time.strftime("%H:%M"),
            "minutes": minutes
        })
    
    def _reset_daily_counters(self):
        """重置每日计数器"""
        for equipment in self.equipment.values():
            equipment.reset_daily_usage()
        self._log_event("daily_reset", {"date": self.current_time.strftime("%Y-%m-%d")})
    
    def _find_path(self, start: str, end: str) -> List[str]:
        """使用BFS查找两个位置之间的最短路径
        
        Args:
            start: 起始位置ID
            end: 目标位置ID
            
        Returns:
            路径列表（不包含起点，包含终点），如果无法到达则返回空列表
        """
        if start == end:
            return []
        
        if start not in self.allowed_moves or end not in self.locations:
            return []
        
        # BFS队列：(当前位置, 路径)
        from collections import deque
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            # 获取当前位置可到达的位置
            neighbors = self.allowed_moves.get(current, [])
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                new_path = path + [neighbor]
                
                # 找到目标
                if neighbor == end:
                    return new_path
                
                # 继续搜索
                queue.append((neighbor, new_path))
        
        # 无法到达
        return []
    
    def move_agent(self, agent_id: str, target_location: str) -> tuple[bool, str]:
        """移动智能体到目标位置（支持自动路径查找）
        
        Args:
            agent_id: 智能体ID
            target_location: 目标房间ID
            
        Returns:
            (是否成功, 消息)
        """
        with self._lock:
            # ===== 步骤1：验证前置条件 =====
            
            # 检查智能体是否存在
            if agent_id not in self.agents:
                return False, "智能体不存在"
            
            # 检查目标房间是否存在
            if target_location not in self.locations:
                return False, "目标房间不存在"
        
        # 获取当前位置
        current_loc = self.agents[agent_id]
        
        # 如果已经在目标位置
        if current_loc == target_location:
            return False, f"已经在{self.get_location_name(target_location)}"
        
        # ===== 步骤2：路径查找 =====
        
        # 检查是否允许直接移动
        if current_loc not in self.allowed_moves:
            return False, f"当前位置{self.get_location_name(current_loc)}未配置移动路径"
        
        # 尝试直接移动
        if target_location in self.allowed_moves[current_loc]:
            path = [target_location]
        else:
            # 使用BFS查找路径
            path = self._find_path(current_loc, target_location)
            if not path:
                allowed_names = [self.get_location_name(loc) for loc in self.allowed_moves[current_loc]]
                return False, f"无法从{self.get_location_name(current_loc)}到达{self.get_location_name(target_location)}。直接可达: {', '.join(allowed_names)}"
        
        # ===== 步骤3：执行移动（沿路径）=====
        
        total_moves = len(path)
        from_name = self.get_location_name(current_loc)
        to_name = self.get_location_name(target_location)
        
        # 沿路径移动
        for step_idx, next_loc in enumerate(path, 1):
            # 验证每一步移动的合法性
            if next_loc not in self.allowed_moves.get(self.agents[agent_id], []):
                # 理论上不应该发生（路径已验证）
                return False, f"路径执行失败：无法从{self.get_location_name(self.agents[agent_id])}到{self.get_location_name(next_loc)}"
            
            # 执行单步移动
            prev_loc = self.agents[agent_id]
            self.agents[agent_id] = next_loc
            
            # 推进时间（每步30秒 = 0.5分钟）
            self.advance_time(minutes=0.5)
            
            # 消耗体力（每步0.2）
            if agent_id in self.physical_states:
                state = self.physical_states[agent_id]
                state.energy_level = max(0.0, state.energy_level - 0.2)
            
            # 记录到移动历史
            if not hasattr(self, 'movement_history'):
                self.movement_history = []
            
            self.movement_history.append({
                "time": self.current_time.strftime("%H:%M:%S"),
                "agent": agent_id,
                "from": prev_loc,
                "to": next_loc,
            })
        
        # 性能优化：超过阈值时清理旧记录
        if hasattr(self, 'movement_history') and len(self.movement_history) > self._log_cleanup_threshold:
            self.movement_history = self.movement_history[-self._max_log_entries:]
        
        # ===== 步骤4：返回结果 =====
        
        # 构造成功消息
        if total_moves == 1:
            message = f"已从{from_name}移动到{to_name}"
        else:
            # 多跳移动，显示路径
            path_names = [self.get_location_name(loc) for loc in path]
            message = f"已从{from_name}经{total_moves}步移动到{to_name}"
        
        # 记录日志
        self._log_event("agent_move", {
            "agent_id": agent_id,
            "from": current_loc,
            "to": target_location,
            "steps": total_moves,
            "time": self.current_time.strftime("%H:%M:%S")
        })
        
        return True, message
    
    def use_device(self, agent_id: str, device_name: str) -> tuple[bool, str]:
        """让智能体使用房间内的设备
        
        Args:
            agent_id: 智能体ID
            device_name: 设备名称
            
        Returns:
            (是否成功, 消息)
        """
        with self._lock:
            # ===== 步骤1：验证 =====
            
            # 检查智能体是否存在
            if agent_id not in self.agents:
                return False, "智能体不存在"
        
        # 获取当前位置
        current_loc = self.agents.get(agent_id)
        if not current_loc:
            return False, "智能体位置未知"
        
        # 检查位置是否有该设备
        location_devices = self.get_location_devices(current_loc)
        if device_name not in location_devices:
            available_devices = ", ".join(location_devices) if location_devices else "无"
            return False, f"当前位置({self.get_location_name(current_loc)})没有{device_name}。可用设备: {available_devices}"
        
        # ===== 步骤2：执行 =====
        
        # 定义设备操作时间映射（秒）
        device_time_map = {
            "挂号机": 30,          # 签到机：30秒
            "自助缴费机": 60,      # 缴费机：60秒
            "导诊台": 20,          # 导诊台：20秒
            "分诊系统": 45,        # 分诊系统：45秒
            "叫号屏": 5,           # 叫号屏：5秒
            "血压计": 120,         # 血压计：2分钟
            "体温计": 30,          # 体温计：30秒
            "诊疗床": 600,         # 诊疗床：10分钟
            "电脑": 120,           # 电脑：2分钟
            "听诊器": 180,         # 听诊器：3分钟
            "血液分析仪": 300,     # 检验设备：5分钟
            "生化分析仪": 300,     # 生化分析仪：5分钟
            "尿液分析仪": 180,     # 尿液分析仪：3分钟
            "采血椅": 300,         # 采血椅：5分钟
            "药品柜": 60,          # 药品柜：1分钟
            "发药窗口": 120,       # 发药窗口：2分钟
            "药品管理系统": 30,    # 药品管理系统：30秒
            "收费系统": 60,        # 收费系统：1分钟
            "POS机": 30,           # POS机：30秒
            "发票打印机": 20,      # 发票打印机：20秒
            "抢救床": 1800,        # 抢救床：30分钟
            "心电监护仪": 300,     # 心电监护仪：5分钟
            "除颤仪": 180,         # 除颤仪：3分钟
            "呼吸机": 1800,        # 呼吸机：30分钟
            "急救车": 300,         # 急救车：5分钟
        }
        
        time_cost_seconds = device_time_map.get(device_name, 30)  # 默认30秒
        time_cost_minutes = time_cost_seconds / 60  # 转换为分钟
        
        # 推进时间（以分钟为单位）
        self.advance_time(minutes=time_cost_minutes)
        
        # 记录使用日志（带日志限制）
        if not hasattr(self, 'device_usage_log'):
            self.device_usage_log = []
        
        self.device_usage_log.append({
            "time": self.current_time.strftime("%H:%M:%S"),
            "agent": agent_id,
            "location": current_loc,
            "device": device_name,
            "duration_seconds": time_cost_seconds
        })
        
        # 性能优化：超过阈值时清理旧记录
        if len(self.device_usage_log) > self._log_cleanup_threshold:
            self.device_usage_log = self.device_usage_log[-self._max_log_entries:]
        
        # ===== 步骤3：返回结果 =====
        
        # 构造消息
        if time_cost_seconds >= 60:
            time_display = f"{int(time_cost_minutes)}分钟"
        else:
            time_display = f"{time_cost_seconds}秒"
        
        message = f"已使用{device_name}（耗时{time_display}）"
        
        # 记录日志
        self._log_event("use_device", {
            "agent_id": agent_id,
            "location": current_loc,
            "device": device_name,
            "duration_seconds": time_cost_seconds,
            "time": self.current_time.strftime("%H:%M:%S")
        })
        
        return True, message
    
    def wait(self, agent_id: str, duration_minutes: int) -> tuple[bool, str]:
        """让智能体原地等待
        
        Args:
            agent_id: 智能体ID
            duration_minutes: 等待时长（分钟）
            
        Returns:
            (是否成功, 消息)
        """
        # ===== 步骤1：验证 =====
        
        # 检查智能体是否存在
        if agent_id not in self.agents:
            return False, "智能体不存在"
        
        # 检查时长是否合理
        if duration_minutes <= 0:
            return False, "等待时长必须大于0"
        
        # ===== 步骤2：执行 =====
        
        # 获取当前位置
        current_loc = self.agents.get(agent_id)
        location_name = self.get_location_name(current_loc)
        
        # 推进时间
        self.advance_time(minutes=duration_minutes)
        
        # 特殊处理：候诊区等待恢复体力
        recovery_info = ""
        if current_loc == "waiting_area" and agent_id in self.physical_states:
            ps = self.physical_states[agent_id]
            if ps.energy_level < 10:
                # 候诊区等待每分钟恢复0.1体力（最多恢复到10）
                old_energy = ps.energy_level
                recovery = min(0.1 * duration_minutes, 10 - ps.energy_level)
                ps.energy_level = min(10.0, ps.energy_level + recovery)
                recovery_info = f"，恢复体力 {recovery:.1f}（{old_energy:.1f}→{ps.energy_level:.1f}）"
        
        # ===== 步骤3：返回结果 =====
        
        # 构造消息
        message = f"在{location_name}等待了{duration_minutes}分钟{recovery_info}"
        
        # 记录日志
        self._log_event("wait", {
            "agent_id": agent_id,
            "location": current_loc,
            "duration_minutes": duration_minutes,
            "time": self.current_time.strftime("%H:%M:%S")
        })
        
        return True, message
    
    def record_conversation(self, from_agent: str, to_agent: str, message: str) -> bool:
        """记录智能体之间的对话
        
        Args:
            from_agent: 发送方智能体ID
            to_agent: 接收方智能体ID
            message: 对话内容
            
        Returns:
            是否成功记录
            
        注意:
            这个方法主要用于记录，不影响现有对话逻辑。
            在调用 patient_agent.respond_to_doctor() 等方法后，额外调用此方法即可。
        """
        # ===== 步骤1：验证 =====
        
        # 检查双方是否存在
        if from_agent not in self.agents:
            return False
        
        if to_agent not in self.agents:
            return False
        
        # 检查是否在同一房间
        loc_a = self.agents.get(from_agent)
        loc_b = self.agents.get(to_agent)
        
        if loc_a != loc_b:
            # 不在同一房间，记录警告但不阻止（允许远程通信）
            self._log_event("conversation_warning", {
                "from": from_agent,
                "to": to_agent,
                "reason": f"{from_agent}在{self.get_location_name(loc_a)}，{to_agent}在{self.get_location_name(loc_b)}",
                "time": self.current_time.strftime("%H:%M:%S")
            })
        
        # ===== 步骤2：记录 =====
        
        # 初始化对话日志（如果不存在）
        if not hasattr(self, 'conversation_log'):
            self.conversation_log = []
        
        # 添加记录
        self.conversation_log.append({
            "time": self.current_time.strftime("%H:%M:%S"),
            "location": loc_a,
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "same_room": loc_a == loc_b
        })
        
        # 性能优化：超过阈值时清理旧记录
        if len(self.conversation_log) > self._log_cleanup_threshold:
            self.conversation_log = self.conversation_log[-self._max_log_entries:]
        
        # 推进时间（根据消息长度）
        time_cost_seconds = max(10, len(message) // 10)  # 最少10秒
        self.advance_time(minutes=time_cost_seconds / 60)
        
        # 记录到事件日志
        self._log_event("conversation", {
            "from": from_agent,
            "to": to_agent,
            "location": loc_a,
            "message_length": len(message),
            "duration_seconds": time_cost_seconds,
            "time": self.current_time.strftime("%H:%M:%S")
        })
        
        return True
    
    def perform_exam(self, patient_id: str, exam_type: str, priority: int = 5) -> tuple[bool, str]:
        """执行检查
        
        Args:
            patient_id: 患者ID
            exam_type: 检查类型
            priority: 优先级 (1-10, 1最高)
        """
        patient_loc = self.agents.get(patient_id)
        if not patient_loc:
            return False, "患者位置未知"
        
        # 查找该类型的所有设备（在当前位置）
        all_equipment = [
            eq for eq in self.equipment.values()
            if eq.exam_type == exam_type and eq.location_id == patient_loc
        ]
        
        if not all_equipment:
            return False, f"当前位置没有 {exam_type} 设备，请移动到相应科室"
        
        # 查找空闲设备
        available_equipment = [eq for eq in all_equipment if eq.can_use(self.current_time)]
        
        if available_equipment:
            # 有空闲设备，直接使用（按优先级选择最空闲的）
            equipment = min(available_equipment, key=lambda eq: eq.daily_usage_count)
            equipment.start_exam(patient_id, self.current_time, priority)
            
            # 显示资源竞争状态
            total_equipment = len(all_equipment)
            busy_equipment = len([eq for eq in all_equipment if eq.is_occupied])
            
            self._log_event("exam_start", {
                "patient_id": patient_id,
                "equipment": equipment.name,
                "exam_type": exam_type,
                "priority": priority,
                "start_time": self.current_time.strftime("%H:%M"),
                "estimated_end": equipment.occupied_until.strftime("%H:%M") if equipment.occupied_until else "unknown",
                "resource_status": f"{busy_equipment}/{total_equipment}设备使用中"
            })
            
            return True, f"开始 {equipment.name} 检查，预计 {equipment.duration_minutes} 分钟（预计完成时间: {equipment.occupied_until.strftime('%H:%M')}）[资源: {busy_equipment+1}/{total_equipment}设备使用中]"
        else:
            # 所有设备都在使用中，加入排队
            equipment = all_equipment[0]  # 选择第一个设备的队列
            equipment.add_to_queue(patient_id, priority, self.current_time)
            wait_time = equipment.get_wait_time(self.current_time, patient_id)
            queue_position = next((i+1 for i, entry in enumerate(equipment.queue) if entry.patient_id == patient_id), 0)
            
            # 显示所有设备队列情况
            total_queue = sum(len(eq.queue) for eq in all_equipment)
            
            self._log_event("exam_queue", {
                "patient_id": patient_id,
                "equipment": equipment.name,
                "exam_type": exam_type,
                "queue_position": queue_position,
                "queue_length": len(equipment.queue),
                "total_queue": total_queue,
                "resource_contention": "高" if total_queue > len(all_equipment) else "中"
            })
            
            return False, f"⚠️ 资源竞争: 所有{exam_type}设备繁忙({len(all_equipment)}台全部使用中)，已加入{equipment.name}队列（位置: {queue_position}/{len(equipment.queue)}，总排队: {total_queue}人，预计等待 {wait_time} 分钟）"
    
    def get_observation(self, agent_id: str) -> Dict:
        """获取Agent当前观察"""
        location_id = self.agents.get(agent_id)
        if not location_id:
            return {
                "error": "Agent位置未知",
                "time": self.current_time.strftime("%H:%M"),
                "working_hours": self.is_working_hours()
            }
        
        location = self.locations[location_id]
        
        # 获取相邻位置的详细信息
        nearby_info = []
        for loc_id in location.connected_to:
            nearby_loc = self.locations[loc_id]
            occupancy = f"{len(nearby_loc.current_occupants)}/{nearby_loc.capacity}"
            nearby_info.append(f"{nearby_loc.name} ({occupancy})")
        
        observation = {
            "time": self.current_time.strftime("%H:%M"),
            "day_of_week": self.current_time.strftime("%A"),
            "working_hours": self.is_working_hours(),
            "location": location.name,
            "location_id": location_id,
            "available_actions": location.available_actions,
            "nearby_locations": nearby_info,
            "occupants_count": len(location.current_occupants),
            "capacity": location.capacity,
        }
        
        # 添加设备信息
        location_equipment = [eq for eq in self.equipment.values() if eq.location_id == location_id]
        if location_equipment:
            equipment_status = []
            for eq in location_equipment:
                status = "空闲" if eq.can_use(self.current_time) else f"使用中（还需{eq.get_wait_time(self.current_time)}分钟）"
                queue_info = f"排队{len(eq.queue)}人" if eq.queue else ""
                equipment_status.append(f"{eq.name}: {status} {queue_info}".strip())
            observation["equipment"] = equipment_status
        
        # 如果是患者，添加生理状态
        if agent_id in self.physical_states:
            state = self.physical_states[agent_id]
            observation["symptoms"] = state.get_symptom_severity_dict()
            observation["vital_signs"] = {k: v.value for k, v in state.vital_signs.items()}
            observation["energy_level"] = state.energy_level
        
        return observation
    
    def add_agent(self, agent_id: str, agent_type: str = "patient", initial_location: str = "lobby") -> bool:
        """添加Agent到世界
        
        Args:
            agent_id: Agent唯一标识（如 "patient_001", "doctor_001"）
            agent_type: Agent类型 ('patient', 'doctor', 'nurse', 'lab_technician')
            initial_location: 初始位置ID（默认为大厅）
        
        Returns:
            是否成功添加
        """
        with self._lock:
            # 检查是否已存在
            if agent_id in self.agents:
                return False
        
        # 检查初始位置是否存在
        if initial_location not in self.locations:
            self._log_event("add_agent_failed", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "reason": f"初始位置 {initial_location} 不存在"
            })
            return False
        
        # 直接设置初始位置（首次进入不需要移动验证）
        self.agents[agent_id] = initial_location
        
        # 根据Agent类型初始化不同的生理状态
        if agent_type == "patient":
            # 患者：完整的生理状态（症状、生命体征、体力等）
            state = PhysicalState(
                patient_id=agent_id, 
                agent_type="patient",
                last_update=self.current_time
            )
            self.physical_states[agent_id] = state
            # 生理状态会在 __post_init__ 中自动初始化默认生命体征
            
        elif agent_type in ["doctor", "nurse", "lab_technician"]:
            # 医护人员：简化的工作状态（体力、工作负荷）
            state = PhysicalState(
                patient_id=agent_id,
                agent_type=agent_type,
                last_update=self.current_time
            )
            # 医护人员初始状态良好，无症状
            state.energy_level = 10.0  # 满体力
            state.consciousness_level = "alert"
            state.pain_level = 0.0
            state.work_load = 0.0  # 初始无工作负荷
            state.consecutive_work_minutes = 0
            state.patients_served_today = 0
            state.last_rest_time = self.current_time
            # 清除默认症状（医护人员健康）
            state.symptoms.clear()
            # 清除生命体征（医护人员不需要监测）
            state.vital_signs.clear()
            self.physical_states[agent_id] = state
        
        # 记录添加成功日志
        self._log_event("add_agent", {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "initial_location": initial_location,
            "time": self.current_time.strftime("%H:%M")
        })
        
        return True
    
    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """获取智能体当前位置
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            房间ID字符串，如果智能体不存在则返回None
        """
        return self.agents.get(agent_id)
    
    def get_agents_in_location(self, location_id: str) -> List[str]:
        """获取指定房间内的所有智能体
        
        Args:
            location_id: 房间ID
            
        Returns:
            智能体ID列表
            
        用途：
            检查医患是否在同一房间（用于对话验证）
        """
        return [aid for aid, loc in self.agents.items() if loc == location_id]
    
    def _rebuild_location_cache(self):
        """重建位置名称缓存（内部方法）"""
        self._location_name_cache.clear()
        for loc_id, loc_data in self.locations.items():
            # 兼容两种数据结构：Location对象或字典
            if isinstance(loc_data, Location):
                self._location_name_cache[loc_id] = loc_data.name
            elif isinstance(loc_data, dict):
                self._location_name_cache[loc_id] = loc_data.get('name', '未知位置')
            else:
                self._location_name_cache[loc_id] = '未知位置'
    
    def get_location_name(self, location_id: str) -> str:
        """获取房间的中文名称（使用缓存优化）
        
        Args:
            location_id: 房间ID
            
        Returns:
            房间的中文名称，如果房间不存在则返回'未知位置'
            
        用途：
            日志输出和终端显示
        """
        # 使用缓存快速返回
        if location_id in self._location_name_cache:
            return self._location_name_cache[location_id]
        
        # 缓存未命中，查找并更新缓存
        loc_data = self.locations.get(location_id)
        if loc_data is None:
            name = '未知位置'
        elif isinstance(loc_data, Location):
            name = loc_data.name
        elif isinstance(loc_data, dict):
            name = loc_data.get('name', '未知位置')
        else:
            name = '未知位置'
        
        self._location_name_cache[location_id] = name
        return name
    
    def get_location_devices(self, location_id: str) -> List[str]:
        """获取房间的设备列表
        
        Args:
            location_id: 房间ID
            
        Returns:
            设备名称列表，如果房间不存在或没有设备则返回空列表
        """
        loc_data = self.locations.get(location_id)
        if loc_data is None:
            return []
        elif isinstance(loc_data, dict):
            return loc_data.get('devices', [])
        else:
            # Location 对象有 devices 属性
            return getattr(loc_data, 'devices', [])
    
    def can_move(self, agent_id: str, target_location: str) -> tuple[bool, str]:
        """检查是否允许移动（不执行实际移动）
        
        Args:
            agent_id: 智能体ID
            target_location: 目标房间ID
            
        Returns:
            (是否允许, 消息)
            
        用途：
            UI提示、预先验证
        """
        # 检查智能体是否存在
        if agent_id not in self.agents:
            return False, "智能体不存在"
        
        # 检查目标房间是否存在
        if target_location not in self.locations:
            return False, "目标房间不存在"
        
        # 获取当前位置
        current_loc = self.agents[agent_id]
        
        # 如果已经在目标位置
        if current_loc == target_location:
            return False, f"已经在{self.get_location_name(target_location)}"
        
        # 检查是否允许移动（查询路径表）
        if current_loc not in self.allowed_moves:
            return False, f"当前位置{self.get_location_name(current_loc)}未配置移动路径"
        
        if target_location not in self.allowed_moves[current_loc]:
            allowed_names = [self.get_location_name(loc) for loc in self.allowed_moves[current_loc]]
            return False, f"无法从{self.get_location_name(current_loc)}直接到达{self.get_location_name(target_location)}。可前往: {', '.join(allowed_names)}"
        
        # 允许移动
        return True, f"可以从{self.get_location_name(current_loc)}移动到{self.get_location_name(target_location)}"
    
    def get_movement_history(self, agent_id: str = None) -> List[Dict]:
        """获取移动历史记录
        
        Args:
            agent_id: 智能体ID，如果为None则返回所有智能体的移动历史
            
        Returns:
            移动历史记录列表
        """
        if not hasattr(self, 'movement_history'):
            return []
        
        if agent_id is None:
            return self.movement_history
        
        return [entry for entry in self.movement_history if entry['agent'] == agent_id]
    
    def get_device_usage_log(self, agent_id: str = None) -> List[Dict]:
        """获取设备使用日志
        
        Args:
            agent_id: 智能体ID，如果为None则返回所有智能体的设备使用日志
            
        Returns:
            设备使用日志列表
        """
        if not hasattr(self, 'device_usage_log'):
            return []
        
        if agent_id is None:
            return self.device_usage_log
        
        return [entry for entry in self.device_usage_log if entry['agent'] == agent_id]
    
    def get_conversation_log(self, agent_id: str = None) -> List[Dict]:
        """获取对话记录
        
        Args:
            agent_id: 智能体ID，如果为None则返回所有对话记录
            
        Returns:
            对话记录列表
        """
        if not hasattr(self, 'conversation_log'):
            return []
        
        if agent_id is None:
            return self.conversation_log
        
        # 返回该智能体作为发送方或接收方的所有对话
        return [
            entry for entry in self.conversation_log 
            if entry['from'] == agent_id or entry['to'] == agent_id
        ]
    
    def generate_timeline_report(self, agent_id: str) -> List[Dict]:
        """生成智能体的完整时间线报告
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            按时间排序的所有事件列表，每个事件包含：
            - time: 时间戳
            - type: 事件类型 ('move', 'device', 'conversation')
            - details: 事件详情
        """
        timeline = []
        
        # 收集移动记录
        for entry in self.get_movement_history(agent_id):
            timeline.append({
                'time': entry['time'],
                'type': 'move',
                'details': f"从{self.get_location_name(entry['from'])}移动到{self.get_location_name(entry['to'])}"
            })
        
        # 收集设备使用记录
        for entry in self.get_device_usage_log(agent_id):
            timeline.append({
                'time': entry['time'],
                'type': 'device',
                'details': f"在{self.get_location_name(entry['location'])}使用{entry['device']}（{entry['duration_seconds']}秒）"
            })
        
        # 收集对话记录
        for entry in self.get_conversation_log(agent_id):
            if entry['from'] == agent_id:
                timeline.append({
                    'time': entry['time'],
                    'type': 'conversation',
                    'details': f"对{entry['to']}说: {entry['message'][:50]}{'...' if len(entry['message']) > 50 else ''}"
                })
            else:
                timeline.append({
                    'time': entry['time'],
                    'type': 'conversation',
                    'details': f"收到{entry['from']}的消息: {entry['message'][:50]}{'...' if len(entry['message']) > 50 else ''}"
                })
        
        # 按时间排序
        timeline.sort(key=lambda x: x['time'])
        
        return timeline
    
    def _log_event(self, event_type: str, details: Dict):
        """记录事件（带日志限制）"""
        self.event_log.append({
            "timestamp": self.current_time.isoformat(),
            "type": event_type,
            "details": details
        })
        
        # 性能优化：超过阈值时清理旧记录
        if len(self.event_log) > self._log_cleanup_threshold:
            self.event_log = self.event_log[-self._max_log_entries:]
    
    def get_event_log(self, limit: int = 10) -> List[Dict]:
        """获取最近的事件日志"""
        return self.event_log[-limit:]
    
    def get_world_summary(self) -> str:
        """获取世界状态摘要"""
        lines = [
            f"{'='*60}",
            f"医院环境状态 - {self.current_time.strftime('%Y-%m-%d %H:%M')}",
            f"{'='*60}",
            f"工作状态: {'营业中' if self.is_working_hours() else '休息中'}",
            f"在院人数: {len(self.agents)}",
            "",
            "各区域人数:",
        ]
        
        for loc_id, loc in self.locations.items():
            if loc.current_occupants:
                lines.append(f"  - {loc.name}: {len(loc.current_occupants)}/{loc.capacity}")
        
        lines.append("")
        lines.append("设备使用情况:")
        for eq_id, eq in self.equipment.items():
            status = "使用中" if eq.is_occupied else "空闲"
            queue_info = f" (排队{len(eq.queue)}人)" if eq.queue else ""
            lines.append(f"  - {eq.name}: {status}{queue_info}")
        
        return "\n".join(lines)

    # ========== 神经内科专用设备查询方法 ==========
    
    def get_neuro_equipment_summary(self) -> Dict[str, Dict]:
        """获取神经内科所有设备的摘要信息
        
        Returns:
            按设备类型分组的状态字典，包含设备数量、可用数、队列情况等
        """
        summary = {}
        
        # 定义神经内科设备类型及其友好名称
        neuro_equipment_types = {
            "ct_head": "头颅CT",
            "mri_brain": "脑MRI",
            "eeg": "脑电图",
            "emg": "肌电图",
            "tcd": "TCD超声",
            "cbc": "血常规",
            "biochem_basic": "基础生化",
            "electrolyte": "电解质",
            "coagulation": "凝血功能",
            "inflammation": "炎症/感染指标",
            "cardiac_stroke_markers": "心脑血管标志物",
            "autoimmune_antibody": "自免抗体"
        }
        
        for exam_type, display_name in neuro_equipment_types.items():
            equipment_list = [eq for eq in self.equipment.values() if eq.exam_type == exam_type]
            
            if not equipment_list:
                continue
            
            total_count = len(equipment_list)
            available_count = sum(1 for eq in equipment_list if eq.can_use(self.current_time))
            total_queue = sum(len(eq.queue) for eq in equipment_list)
            avg_wait_time = sum(eq.get_wait_time(self.current_time) for eq in equipment_list) / total_count if total_count > 0 else 0
            
            summary[exam_type] = {
                "display_name": display_name,
                "total_equipment": total_count,
                "available_equipment": available_count,
                "busy_equipment": total_count - available_count,
                "total_queue_length": total_queue,
                "avg_wait_time_minutes": int(avg_wait_time),
                "status": "空闲" if available_count > 0 else f"繁忙(排队{total_queue}人)",
                "equipment_list": [eq.name for eq in equipment_list]
            }
        
        return summary
    
    def get_equipment_recommendations(self, exam_types: List[str]) -> Dict[str, str]:
        """根据检查类型列表推荐最佳设备和时间
        
        Args:
            exam_types: 检查类型列表
            
        Returns:
            推荐字典，包含每个检查类型的最佳时间和预计等待
        """
        recommendations = {}
        
        for exam_type in exam_types:
            equipment_list = [eq for eq in self.equipment.values() if eq.exam_type == exam_type]
            
            if not equipment_list:
                recommendations[exam_type] = "❌ 无此类型设备"
                continue
            
            # 找到等待时间最短的设备
            best_eq = min(equipment_list, key=lambda eq: eq.get_wait_time(self.current_time))
            wait_time = best_eq.get_wait_time(self.current_time)
            
            if wait_time == 0:
                recommendations[exam_type] = f"✅ 立即可用 - {best_eq.name}"
            elif wait_time < 30:
                recommendations[exam_type] = f"⏱️ 短时等待({wait_time}分钟) - {best_eq.name}"
            else:
                recommendations[exam_type] = f"⏳ 较长等待({wait_time}分钟) - {best_eq.name}"
        
        return recommendations

    # ========== Level 2 强化: 资源管理 ==========
    
    def get_equipment_status(self, exam_type: str = None, location_id: str = None) -> List[Dict]:
        equipment_list = list(self.equipment.values())
        if exam_type:
            equipment_list = [eq for eq in equipment_list if eq.exam_type == exam_type]
        if location_id:
            equipment_list = [eq for eq in equipment_list if eq.location_id == location_id]
        status_list = []
        for eq in equipment_list:
            status_list.append({
                'id': eq.id, 'name': eq.name, 'location': eq.location_id,
                'exam_type': eq.exam_type, 'status': eq.status,
                'is_occupied': eq.is_occupied, 'current_patient': eq.current_patient,
                'queue_length': len(eq.queue), 'queue': [entry.patient_id for entry in eq.queue],
                'daily_usage': eq.daily_usage_count, 'max_daily_usage': eq.max_daily_usage,
                'wait_time': eq.get_wait_time(self.current_time) if eq.is_occupied else 0,
            })
        return status_list


    # ========== 医生资源池管理 ==========
    
    def register_doctor(self, doctor_id: str, dept: str):
        """注册医生到资源池
        
        Args:
            doctor_id: 医生ID
            dept: 科室
        """
        with self._lock:
            if dept not in self.doctor_pool:
                self.doctor_pool[dept] = {}
            
            self.doctor_pool[dept][doctor_id] = {
                'status': 'available',  # available/busy
                'current_patient': None,
                'queue': [],  # 等待该医生的患者队列
                'daily_patients': 0,  # 今日已接诊患者数
                'max_daily_patients': 50,  # 每日最大接诊数
            }
    
    def assign_doctor(self, patient_id: str, dept: str, priority: int = 5) -> tuple[Optional[str], int]:
        """为患者分配医生（支持排队和优先级）
        
        Args:
            patient_id: 患者ID
            dept: 科室
            priority: 优先级 (1-10, 1最高)
            
        Returns:
            (医生ID, 预计等待分钟数)
        """
        with self._lock:
            if dept not in self.doctor_pool or not self.doctor_pool[dept]:
                return None, 0  # 无可用医生
            
            # 查找最佳医生（空闲或队列最短）
            best_doctor = None
            min_wait_time = float('inf')
            
            for doctor_id, doctor_info in self.doctor_pool[dept].items():
                # 检查医生是否达到每日接诊上限
                if doctor_info['daily_patients'] >= doctor_info['max_daily_patients']:
                    continue
                
                # 计算等待时间
                wait_time = 0
                if doctor_info['status'] == 'busy':
                    # 假设每个患者平均需要15分钟
                    wait_time = 15
                
                # 加上队列等待时间
                wait_time += len(doctor_info['queue']) * 15
                
                if wait_time < min_wait_time:
                    min_wait_time = wait_time
                    best_doctor = doctor_id
            
            if best_doctor is None:
                return None, 0  # 所有医生都满负荷
            
            # 记录患者-医生映射
            self.patient_doctor_map[patient_id] = best_doctor
            
            # 如果医生空闲，直接分配
            if self.doctor_pool[dept][best_doctor]['status'] == 'available':
                self.doctor_pool[dept][best_doctor]['status'] = 'busy'
                self.doctor_pool[dept][best_doctor]['current_patient'] = patient_id
                self.doctor_pool[dept][best_doctor]['daily_patients'] += 1
                # 添加日志
                import logging
                logger = logging.getLogger('hospital_agent.world')
                logger.info(f"✅ [物理世界] 医生分配: {best_doctor} → 患者 {patient_id}（立即可用）")
                return best_doctor, 0
            
            # 医生忙碌，加入队列（按优先级排序）
            queue_entry = QueueEntry(patient_id=patient_id, priority=priority, enqueue_time=self.current_time)
            self.doctor_pool[dept][best_doctor]['queue'].append(queue_entry)
            self.doctor_pool[dept][best_doctor]['queue'].sort()
            
            # 添加日志
            import logging
            logger = logging.getLogger('hospital_agent.world')
            queue_len = len(self.doctor_pool[dept][best_doctor]['queue'])
            logger.info(f"⏳ [物理世界] 医生忙碌: {best_doctor} 队列+1（当前队列{queue_len}人，预计等待{int(min_wait_time)}分钟）")
            
            return best_doctor, int(min_wait_time)
    
    def release_doctor(self, patient_id: str) -> bool:
        """释放医生资源（患者就诊结束）
        
        Args:
            patient_id: 患者ID
            
        Returns:
            是否成功释放
        """
        with self._lock:
            # 查找患者对应的医生
            if patient_id not in self.patient_doctor_map:
                return False
            
            doctor_id = self.patient_doctor_map[patient_id]
            del self.patient_doctor_map[patient_id]
            
            # 查找医生所在科室
            for dept, doctors in self.doctor_pool.items():
                if doctor_id in doctors:
                    doctor_info = doctors[doctor_id]
                    
                    # 清除当前患者
                    if doctor_info['current_patient'] == patient_id:
                        doctor_info['current_patient'] = None
                    
                    # 从队列中移除（如果在队列中）
                    doctor_info['queue'] = [entry for entry in doctor_info['queue'] 
                                           if entry.patient_id != patient_id]
                    
                    # 检查是否有等待的患者
                    if doctor_info['queue']:
                        # 分配给下一个患者
                        next_entry = doctor_info['queue'].pop(0)
                        doctor_info['status'] = 'busy'
                        doctor_info['current_patient'] = next_entry.patient_id
                        doctor_info['daily_patients'] += 1
                        self.patient_doctor_map[next_entry.patient_id] = doctor_id
                        
                        # 添加日志
                        import logging
                        logger = logging.getLogger('hospital_agent.world')
                        remaining = len(doctor_info['queue'])
                        logger.info(f"🔄 [物理世界] 医生流转: {doctor_id} 完成 {patient_id}，接诊下一位 {next_entry.patient_id}（队列剩余{remaining}人）")
                    else:
                        # 无等待患者，医生变为空闲
                        doctor_info['status'] = 'available'
                        
                        # 添加日志
                        import logging
                        logger = logging.getLogger('hospital_agent.world')
                        logger.info(f"✅ [物理世界] 医生空闲: {doctor_id} 完成 {patient_id}，无队列等待")
                    
                    return True
            
            return False
    
    def get_doctor_status(self, dept: str = None) -> List[Dict]:
        """获取医生状态
        
        Args:
            dept: 科室（可选，不指定则返回所有科室）
            
        Returns:
            医生状态列表
        """
        status_list = []
        
        depts = [dept] if dept else self.doctor_pool.keys()
        
        for d in depts:
            if d not in self.doctor_pool:
                continue
            
            for doctor_id, info in self.doctor_pool[d].items():
                status_list.append({
                    'doctor_id': doctor_id,
                    'dept': d,
                    'status': info['status'],
                    'current_patient': info['current_patient'],
                    'queue_length': len(info['queue']),
                    'queue': [entry.patient_id for entry in info['queue']],
                    'daily_patients': info['daily_patients'],
                    'max_daily_patients': info['max_daily_patients'],
                })
        
        return status_list
    
    def request_equipment(self, patient_id: str, exam_type: str, priority: int = 5) -> tuple[Optional[str], int]:
        """请求检查设备（支持排队和优先级）
        
        Args:
            patient_id: 患者ID
            exam_type: 检查类型
            priority: 优先级 (1-10, 1最高)
            
        Returns:
            (设备ID, 预计等待分钟数)
        """
        with self._lock:
            # 查找该类型的所有设备
            available_equipment = [eq for eq in self.equipment.values() 
                                  if eq.exam_type == exam_type and eq.status != "offline"]
            
            if not available_equipment:
                return None, 0  # 无该类型设备
            
            # 查找最佳设备（空闲或队列最短）
            best_equipment = None
            min_wait_time = float('inf')
            
            for eq in available_equipment:
                wait_time = eq.get_wait_time(self.current_time, patient_id)
                if wait_time < min_wait_time:
                    min_wait_time = wait_time
                    best_equipment = eq
            
            if best_equipment is None:
                return None, 0
            
            # 检查设备是否已被该患者占用
            if best_equipment.current_patient == patient_id:
                # 设备已分配给该患者，无需重复分配
                import logging
                logger = logging.getLogger('hospital_agent.world')
                logger.info(f"♻️  [物理世界] 设备复用: {best_equipment.name} 已被患者 {patient_id} 占用（可继续使用）")
                return best_equipment.id, 0
            
            # 如果设备空闲，直接分配
            if best_equipment.can_use(self.current_time):
                best_equipment.start_exam(patient_id, self.current_time, priority)
                
                # 添加日志
                import logging
                logger = logging.getLogger('hospital_agent.world')
                logger.info(f"✅ [物理世界] 设备分配: {best_equipment.name} → 患者 {patient_id}（立即可用）")
                
                return best_equipment.id, 0
            
            # 设备忙碌，加入队列
            best_equipment.add_to_queue(patient_id, priority, self.current_time)
            
            # 添加日志
            import logging
            logger = logging.getLogger('hospital_agent.world')
            queue_len = len(best_equipment.queue)
            logger.info(f"⏳ [物理世界] 设备忙碌: {best_equipment.name} 队列+1（当前队列{queue_len}人，预计等待{int(min_wait_time)}分钟）")
            
            return best_equipment.id, int(min_wait_time)
    
    def release_equipment(self, equipment_id: str) -> bool:
        """释放设备（检查完成）
        
        Args:
            equipment_id: 设备ID
            
        Returns:
            是否成功释放
        """
        with self._lock:
            if equipment_id not in self.equipment:
                return False
            
            eq = self.equipment[equipment_id]
            finished_patient = eq.finish_exam(self.current_time)
            
            if not finished_patient:
                return False
            
            # 检查是否有等待的患者
            next_patient = eq.get_next_patient()
            if next_patient:
                # 自动分配给下一个患者
                eq.start_exam(next_patient, self.current_time)
            
            return True
