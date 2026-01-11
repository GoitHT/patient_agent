"""
医院物理环境模拟系统 - 基于 ScienceWorld 思想
实现真实的物理空间、时间和资源约束
"""
from __future__ import annotations

import random
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
    
    def is_available(self) -> bool:
        """检查位置是否还有空间"""
        return len(self.current_occupants) < self.capacity
    
    def add_occupant(self, agent_id: str) -> bool:
        """添加占用者"""
        if self.is_available():
            self.current_occupants.add(agent_id)
            return True
        return False
    
    def remove_occupant(self, agent_id: str):
        """移除占用者"""
        self.current_occupants.discard(agent_id)


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
    
    def set_maintenance(self, current_time: datetime, duration_minutes: int = 60):
        """设置设备维护状态"""
        self.status = "maintenance"
        self.maintenance_until = current_time + timedelta(minutes=duration_minutes)
    
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
                # 严重症状更容易恶化
                change = self.progression_rate * hours * 1.5
            else:
                # 轻度症状可能自然波动
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
    """物理状态 - Level 3 增强版：动态生理模拟"""
    patient_id: str
    vital_signs: Dict[str, VitalSign] = field(default_factory=dict)  # 生命体征
    symptoms: Dict[str, Symptom] = field(default_factory=dict)  # 症状
    last_update: Optional[datetime] = None  # 最后更新时间
    energy_level: float = 10.0  # 体力水平 0-10
    pain_level: float = 0.0  # 疼痛水平 0-10
    consciousness_level: str = "alert"  # alert, drowsy, unconscious
    diagnosis: Optional[str] = None  # 诊断
    medications: List[Dict] = field(default_factory=list)  # 药物列表
    treatments: List[Dict] = field(default_factory=list)  # 治疗记录
    
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
class HospitalWorld:
    """医院世界环境 - 物理空间模拟"""
    
    def __init__(self, start_time: datetime = None):
        """初始化医院世界"""
        self.current_time = start_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        self.locations: Dict[str, Location] = {}
        self.equipment: Dict[str, Equipment] = {}
        self.agents: Dict[str, str] = {}  # agent_id -> location_id
        self.physical_states: Dict[str, PhysicalState] = {}
        self.event_log: List[Dict] = []  # 事件日志
        
        # 工作时间
        self.working_hours = {
            'start': 8,
            'end': 18,
            'lunch_start': 12,
            'lunch_end': 13,
        }
        
        # 初始化医院环境
        self._build_hospital()
    
    def _build_hospital(self):
        """构建医院物理结构"""
        # 创建位置
        locations = [
            Location(
                "lobby", 
                "门诊大厅", 
                "lobby", 
                connected_to=["triage", "internal_medicine", "surgery", "gastro", "neuro", "pharmacy"],
                capacity=50,
                available_actions=["register", "wait", "move", "look"]
            ),
            
            Location(
                "triage", 
                "分诊台", 
                "triage",
                connected_to=["lobby"],
                capacity=3,
                available_actions=["triage", "consult", "move", "look"]
            ),
            
            Location(
                "internal_medicine", 
                "内科诊室", 
                "clinic",
                connected_to=["lobby", "lab", "imaging"],
                capacity=10,  # 增加容量以支持多个患者同时在诊室
                available_actions=["consult", "examine", "prescribe", "order_test", "move", "look"]
            ),
            
            Location(
                "surgery", 
                "外科诊室", 
                "clinic",
                connected_to=["lobby", "lab", "imaging"],
                capacity=10,
                available_actions=["consult", "examine", "prescribe", "order_test", "move", "look"]
            ),
            
            Location(
                "gastro", 
                "消化内科诊室", 
                "clinic",
                connected_to=["lobby", "lab", "imaging", "endoscopy"],
                capacity=10,
                available_actions=["consult", "examine", "prescribe", "order_test", "move", "look"]
            ),
            
            Location(
                "neuro", 
                "神经内科诊室", 
                "clinic",
                connected_to=["lobby", "lab", "imaging", "neurophysiology"],
                capacity=2,
                available_actions=["consult", "examine", "prescribe", "order_test", "move", "look"]
            ),
            
            Location(
                "lab", 
                "检验科", 
                "lab",
                connected_to=["internal_medicine", "surgery", "gastro", "neuro"],
                capacity=10,
                available_actions=["blood_test", "wait", "move", "look"]
            ),
            
            Location(
                "imaging", 
                "影像科", 
                "imaging",
                connected_to=["internal_medicine", "surgery", "gastro", "neuro"],
                capacity=5,
                available_actions=["xray", "ct", "mri", "ultrasound", "wait", "move", "look"]
            ),
            
            Location(
                "endoscopy", 
                "内镜中心", 
                "endoscopy",
                connected_to=["gastro"],
                capacity=3,
                available_actions=["endoscopy", "colonoscopy", "wait", "move", "look"]
            ),
            
            Location(
                "neurophysiology", 
                "神经电生理室", 
                "neurophysiology",
                connected_to=["neuro"],
                capacity=3,
                available_actions=["eeg", "emg", "wait", "move", "look"]
            ),
            
            Location(
                "pharmacy", 
                "药房", 
                "pharmacy",
                connected_to=["lobby"],
                capacity=10,
                available_actions=["get_medicine", "wait", "move", "look"]
            ),
        ]
        
        for loc in locations:
            self.locations[loc.id] = loc
        
        # 创建设备
        equipment_list = [
            # 影像科设备
            Equipment("xray_1", "X光机1号", "imaging", "xray", 15),
            Equipment("ct_1", "CT机1号", "imaging", "ct", 30),
            Equipment("mri_1", "MRI机1号", "imaging", "mri", 45),
            Equipment("ultrasound_1", "B超机1号", "imaging", "ultrasound", 20),
            
            # 检验科设备
            Equipment("blood_analyzer_1", "血液分析仪1号", "lab", "blood_test", 20),
            Equipment("biochem_analyzer_1", "生化分析仪1号", "lab", "biochemistry", 25),
            
            # 内镜设备
            Equipment("endoscope_1", "胃镜1号", "endoscopy", "endoscopy", 30),
            Equipment("colonoscope_1", "肠镜1号", "endoscopy", "colonoscopy", 45),
            
            # 神经电生理设备
            Equipment("eeg_1", "脑电图机1号", "neurophysiology", "eeg", 40),
            Equipment("emg_1", "肌电图机1号", "neurophysiology", "emg", 30),
            
            # 诊室设备
            Equipment("ecg_1", "心电图机1号", "internal_medicine", "ecg", 10),
            Equipment("ecg_2", "心电图机2号", "surgery", "ecg", 10),
        ]
        
        for eq in equipment_list:
            self.equipment[eq.id] = eq
    
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
    
    def move_agent(self, agent_id: str, target_location_id: str) -> tuple[bool, str]:
        """移动Agent到目标位置"""
        current_loc_id = self.agents.get(agent_id)
        target_loc = self.locations.get(target_location_id)
        
        if not target_loc:
            return False, f"位置 {target_location_id} 不存在"
        
        # 检查是否相邻（首次进入除外）
        if current_loc_id:
            current_loc = self.locations[current_loc_id]
            if target_location_id not in current_loc.connected_to and current_loc_id != target_location_id:
                nearby = ", ".join([self.locations[loc_id].name for loc_id in current_loc.connected_to])
                return False, f"无法从 {current_loc.name} 直接到达 {target_loc.name}。相邻位置: {nearby}"
        
        # 检查容量
        if not target_loc.is_available():
            return False, f"{target_loc.name} 已满员（{len(target_loc.current_occupants)}/{target_loc.capacity}），请稍后"
        
        # 检查工作时间（某些位置）
        if target_loc.type in ['lab', 'imaging', 'endoscopy', 'neurophysiology']:
            if not self.is_working_hours():
                return False, f"{target_loc.name} 未开放（工作时间: {self.working_hours['start']}:00-{self.working_hours['end']}:00）"
        
        # 执行移动
        if current_loc_id:
            self.locations[current_loc_id].remove_occupant(agent_id)
        
        target_loc.add_occupant(agent_id)
        self.agents[agent_id] = target_location_id
        
        # 移动需要时间
        travel_time = 3 if current_loc_id else 0  # 首次进入不消耗时间
        if travel_time > 0:
            self.advance_time(travel_time)
        
        self._log_event("agent_move", {
            "agent_id": agent_id,
            "from": current_loc_id,
            "to": target_location_id,
            "time": self.current_time.strftime("%H:%M")
        })
        
        return True, f"已移动到 {target_loc.name}（用时{travel_time}分钟）" if travel_time > 0 else f"到达 {target_loc.name}"
    
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
            
            self._log_event("exam_start", {
                "patient_id": patient_id,
                "equipment": equipment.name,
                "exam_type": exam_type,
                "priority": priority,
                "start_time": self.current_time.strftime("%H:%M"),
                "estimated_end": equipment.occupied_until.strftime("%H:%M") if equipment.occupied_until else "unknown"
            })
            
            return True, f"开始 {equipment.name} 检查，预计 {equipment.duration_minutes} 分钟（预计完成时间: {equipment.occupied_until.strftime('%H:%M')}）"
        else:
            # 所有设备都在使用中，加入排队
            equipment = all_equipment[0]  # 选择第一个设备的队列
            equipment.add_to_queue(patient_id, priority, self.current_time)
            wait_time = equipment.get_wait_time(self.current_time, patient_id)
            queue_position = next((i+1 for i, entry in enumerate(equipment.queue) if entry.patient_id == patient_id), 0)
            
            self._log_event("exam_queue", {
                "patient_id": patient_id,
                "equipment": equipment.name,
                "exam_type": exam_type,
                "queue_position": queue_position,
                "queue_length": len(equipment.queue)
            })
            
            return False, f"{equipment.name} 繁忙，已加入排队（排队位置: {queue_position}/{len(equipment.queue)}，预计等待 {wait_time} 分钟）"
        equipment.start_exam(patient_id, self.current_time)
        
        self._log_event("exam_start", {
            "patient_id": patient_id,
            "equipment": equipment.name,
            "exam_type": exam_type,
            "duration": equipment.duration_minutes,
            "start_time": self.current_time.strftime("%H:%M")
        })
        
        # 推进时间
        self.advance_time(equipment.duration_minutes)
        
        return True, f"完成 {equipment.name} 检查（用时 {equipment.duration_minutes} 分钟）"
    
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
        """添加Agent到世界"""
        if agent_id in self.agents:
            return False
        
        success, message = self.move_agent(agent_id, initial_location)
        
        # 如果是患者，初始化生理状态
        if success and agent_type == "patient":
            state = PhysicalState(patient_id=agent_id, last_update=self.current_time)
            self.physical_states[agent_id] = state
            # 生理状态会在 __post_init__ 中自动初始化默认生命体征
        
        return success
    
    def _log_event(self, event_type: str, details: Dict):
        """记录事件"""
        self.event_log.append({
            "timestamp": self.current_time.isoformat(),
            "type": event_type,
            "details": details
        })
    
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

    # ========== Level 2 ǿ: Դ ==========
    
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

    def find_best_equipment(self, exam_type: str, priority: int = 5) -> Optional[Dict]:
        """查找最佳可用设备
        
        Args:
            exam_type: 检查类型
            priority: 患者优先级
            
        Returns:
            最佳设备信息字典,包含等待时间和预计开始时间
        """
        equipment_list = [eq for eq in self.equipment.values() if eq.exam_type == exam_type]
        
        if not equipment_list:
            return None
        
        # 根据等待时间和状态排序
        best_equipment = None
        min_wait = float('inf')
        
        for eq in equipment_list:
            if eq.status == "offline":
                continue
                
            wait_time = eq.get_wait_time(self.current_time)
            if wait_time < min_wait:
                min_wait = wait_time
                best_equipment = eq
        
        if best_equipment:
            estimated_start = self.current_time
            if best_equipment.occupied_until:
                estimated_start = max(self.current_time, best_equipment.occupied_until)
            
            return {
                "id": best_equipment.id,
                "name": best_equipment.name,
                "location_id": best_equipment.location_id,
                "location_name": self.locations[best_equipment.location_id].name,
                "wait_time": min_wait,
                "estimated_start": estimated_start.strftime("%H:%M"),
                "queue_length": len(best_equipment.queue)
            }
        
        return None
    
    def reserve_equipment(self, patient_id: str, exam_type: str, time_slot: str) -> tuple[bool, str]:
        """预约设备
        
        Args:
            patient_id: 患者ID
            exam_type: 检查类型
            time_slot: 时间槽 (格式: "HH:MM")
            
        Returns:
            (成功标志, 消息)
        """
        # 查找该类型的设备
        equipment_list = [eq for eq in self.equipment.values() if eq.exam_type == exam_type]
        
        if not equipment_list:
            return False, f"没有找到 {exam_type} 类型的设备"
        
        # 尝试在任一设备上预约
        for eq in equipment_list:
            if eq.reserve_slot(time_slot, patient_id):
                self._log_event("equipment_reserved", {
                    "patient_id": patient_id,
                    "equipment": eq.name,
                    "time_slot": time_slot
                })
                return True, f"成功预约 {eq.name} 在 {time_slot}"
        
        return False, f"时间槽 {time_slot} 已被预约"
    
    def cancel_equipment_reservation(self, patient_id: str) -> bool:
        """取消设备预约"""
        canceled_count = 0
        for eq in self.equipment.values():
            before = len(eq.reservation_slots)
            eq.cancel_reservation(patient_id)
            after = len(eq.reservation_slots)
            canceled_count += (before - after)
        
        if canceled_count > 0:
            self._log_event("reservation_canceled", {"patient_id": patient_id, "count": canceled_count})
        
        return canceled_count > 0
    
    def get_resource_competition_report(self) -> Dict:
        """生成资源竞争报告"""
        report = {
            "timestamp": self.current_time.strftime("%Y-%m-%d %H:%M"),
            "total_equipment": len(self.equipment),
            "busy_equipment": sum(1 for eq in self.equipment.values() if eq.is_occupied),
            "total_queue_length": sum(len(eq.queue) for eq in self.equipment.values()),
            "hotspots": [],
            "bottlenecks": []
        }
        
        # 找出热点设备（排队超过3人）
        for eq in self.equipment.values():
            if len(eq.queue) >= 3:
                report["hotspots"].append({
                    "equipment": eq.name,
                    "type": eq.exam_type,
                    "queue": len(eq.queue),
                    "wait_time": eq.get_wait_time(self.current_time)
                })
        
        # 找出瓶颈（使用率超过80%）
        for eq in self.equipment.values():
            usage_ratio = eq.daily_usage_count / eq.max_daily_usage
            if usage_ratio >= 0.8:
                report["bottlenecks"].append({
                    "equipment": eq.name,
                    "usage_ratio": f"{usage_ratio*100:.1f}%",
                    "daily_usage": f"{eq.daily_usage_count}/{eq.max_daily_usage}"
                })
        
        return report

