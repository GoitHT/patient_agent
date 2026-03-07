"""
医院协调器 - 管理多医生多患者的并发场景
Hospital Coordinator - Managing multi-doctor multi-patient concurrent scenarios

功能：
1. 医生资源管理（注册、状态跟踪、负载均衡）
2. 患者队列管理（挂号、等候、优先级）
3. 自动分配调度（医生-患者匹配）
4. 会诊调度（跨科室协作）
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Queue, Empty
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

# 优先级老化常量：每等待 PRIORITY_AGING_INTERVAL 秒，有效优先级提升 1 点，上限 PRIORITY_MAX
PRIORITY_AGING_INTERVAL: int = 300  # 5 分钟
PRIORITY_MAX: int = 10

from utils import get_logger, now_iso

logger = get_logger("hospital_agent.coordinator")


class ResourceStatus(Enum):
    """资源状态枚举"""
    AVAILABLE = "available"      # 空闲
    BUSY = "busy"                # 忙碌（正在接诊）
    CONSULTING = "consulting"    # 参与会诊
    OFFLINE = "offline"          # 离线/休息


class PatientStatus(Enum):
    """患者状态枚举"""
    REGISTERED = "registered"          # 已挂号
    WAITING = "waiting"                # 等候就诊
    CONSULTING = "consulting"          # 就诊中
    WAITING_LAB = "waiting_lab"        # 等待检验
    WAITING_IMAGING = "waiting_imaging"  # 等待影像
    RETURNING = "returning"            # 检查后返回
    GETTING_PRESCRIPTION = "getting_prescription"  # 取药
    DISCHARGED = "discharged"          # 离院
    EMERGENCY = "emergency"            # 急诊


@dataclass
class DoctorResource:
    """医生资源"""
    doctor_id: str
    name: str
    dept: str                           # 科室
    status: ResourceStatus = ResourceStatus.AVAILABLE
    current_patient: Optional[str] = None  # 当前正在接诊的患者
    consultation_requests: List[str] = field(default_factory=list)  # 会诊请求队列
    total_patients_today: int = 0       # 今日接诊患者数
    average_consultation_time: float = 15.0  # 平均就诊时间（分钟）
    
    def is_available(self) -> bool:
        """是否可接诊"""
        return self.status == ResourceStatus.AVAILABLE
    
    def start_consultation(self, patient_id: str):
        """开始接诊"""
        self.status = ResourceStatus.BUSY
        self.current_patient = patient_id
        self.total_patients_today += 1
        logger.debug(f"医生 {self.name} 开始接诊患者 {patient_id}")
    
    def end_consultation(self):
        """结束接诊"""
        patient_id = self.current_patient
        self.current_patient = None
        self.status = ResourceStatus.AVAILABLE
        logger.debug(f"医生 {self.name} 结束接诊患者 {patient_id}")
    
    def join_consultation(self, patient_id: str):
        """参与会诊"""
        old_status = self.status
        self.status = ResourceStatus.CONSULTING
        self.consultation_requests.append(patient_id)
        logger.info(f"医生 {self.name} 参与会诊患者 {patient_id} (原状态: {old_status.value})")
    
    def end_consultation_participation(self, patient_id: str):
        """结束会诊参与"""
        if patient_id in self.consultation_requests:
            self.consultation_requests.remove(patient_id)
        
        # 如果没有其他会诊任务，恢复为空闲
        if not self.consultation_requests:
            self.status = ResourceStatus.AVAILABLE
            logger.info(f"医生 {self.name} 结束会诊，恢复空闲")


@dataclass
class PatientSession:
    """患者会话"""
    patient_id: str
    patient_data: Dict[str, Any]
    dept: str                           # 就诊科室
    status: PatientStatus = PatientStatus.REGISTERED
    assigned_doctor: Optional[str] = None  # 分配的医生
    priority: int = 5                   # 优先级（1-10，数字越大越优先）
    arrival_time: str = field(default_factory=now_iso)  # 到达时间
    consultation_start_time: Optional[str] = None  # 就诊开始时间
    consultation_end_time: Optional[str] = None    # 就诊结束时间
    consultation_doctors: Set[str] = field(default_factory=set)  # 参与会诊的医生
    lab_results_ready: bool = False     # 检验结果是否就绪
    imaging_results_ready: bool = False # 影像结果是否就绪
    queue_entry_time: float = 0.0       # 进入等候队列的时间戳（用于老化计算）
    effective_priority: int = 0         # 老化后的有效优先级（动态更新）

    def __lt__(self, other):
        """优先级队列排序（使用老化后的有效优先级，防止饥饿）"""
        if self.effective_priority != other.effective_priority:
            return self.effective_priority > other.effective_priority  # 有效优先级高的先
        return self.arrival_time < other.arrival_time  # 同优先级按到达时间


class HospitalCoordinator:
    """医院协调器 - 中央调度系统"""
    
    def __init__(self, medical_record_service):
        """
        初始化协调器
        
        Args:
            medical_record_service: 医疗记录服务
        """
        self.medical_record_service = medical_record_service
        
        # 资源池
        self.doctors: Dict[str, DoctorResource] = {}  # doctor_id -> DoctorResource
        self.patients: Dict[str, PatientSession] = {}  # patient_id -> PatientSession
        
        # 等候队列（按科室，使用优先级队列）
        self.waiting_queues: Dict[str, PriorityQueue] = defaultdict(PriorityQueue)
        
        # 检验/影像队列
        self.lab_queue: Queue = Queue()
        self.imaging_queue: Queue = Queue()
        
        # 用于去重等待消息
        self._last_waiting_log: Dict[str, float] = {}  # dept -> timestamp (避免重复输出等待消息)
        self._waiting_log_interval = 15.0  # 10秒内不重复输出同一科室的等待消息
        
        # 会诊请求队列
        self.consultation_requests: Queue = Queue()
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "total_patients": 0,
            "total_consultations": 0,
            "total_multi_consultations": 0,
            "average_waiting_time": 0,
        }
        
        # 登记信息（内部处理，简化日志）
        # 不显示启动提示，避免冗余
    
    # ========== 医生管理 ==========
    
    def register_doctor(self, doctor_id: str, name: str, dept: str):
        """
        注册医生资源
        
        Args:
            doctor_id: 医生ID
            name: 医生姓名
            dept: 所属科室
        """
        with self._lock:
            self.doctors[doctor_id] = DoctorResource(
                doctor_id=doctor_id,
                name=name,
                dept=dept
            )
            logger.debug(f"✅ 医生已注册: {name} ({dept}科, ID: {doctor_id})")
    
    def get_doctor(self, doctor_id: str) -> Optional[DoctorResource]:
        """获取医生信息"""
        return self.doctors.get(doctor_id)
    
    def get_available_doctors(self, dept: Optional[str] = None) -> List[DoctorResource]:
        """
        获取空闲医生列表
        
        Args:
            dept: 科室筛选（None表示所有科室）
        
        Returns:
            空闲医生列表
        """
        with self._lock:
            doctors = [
                d for d in self.doctors.values()
                if d.is_available() and (dept is None or d.dept == dept)
            ]
            return doctors
    
    def set_doctor_offline(self, doctor_id: str):
        """设置医生离线"""
        with self._lock:
            doctor = self.doctors.get(doctor_id)
            if doctor:
                doctor.status = ResourceStatus.OFFLINE
                logger.info(f"医生 {doctor.name} 已离线")
    
    # ========== 患者管理 ==========
    
    def register_patient(self, patient_id: str, patient_data: Dict[str, Any], dept: str, priority: int = 5) -> str:
        """
        患者挂号
        
        Args:
            patient_id: 患者ID
            patient_data: 患者数据
            dept: 挂号科室
            priority: 优先级（1-10）
        
        Returns:
            会话ID
        """
        with self._lock:
            # 创建患者会话
            session = PatientSession(
                patient_id=patient_id,
                patient_data=patient_data,
                dept=dept,
                priority=priority,
                status=PatientStatus.REGISTERED
            )
            self.patients[patient_id] = session
            
            # 检查是否已有病例记录
            existing_record = self.medical_record_service.get_record(patient_id)
            
            # 获取患者标识（用于日志输出）
            case_id = patient_data.get("case_id")
            patient_display = f"P{case_id}" if case_id is not None else patient_id
            
            if existing_record:
                logger.info(f"[{patient_display}] ✅ 患者挂号: {patient_display} -> {dept}科 (优先级: {priority}, 病例已存在: {existing_record.record_id})")
            else:
                # 创建病例（注意：此时dept是挂号科室，真实科室需等护士分诊后确定）
                patient_profile = {
                    "name": patient_data.get("name", "患者"),
                    "age": patient_data.get("age", 0),
                    "gender": patient_data.get("gender", "未知"),
                    "dataset_id": patient_data.get("dataset_id"),
                    "case_id": patient_data.get("case_id"),
                    "run_id": patient_data.get("run_id"),
                    # 注意：不在此处设置dept，等护士分诊后再更新
                }
                record = self.medical_record_service.create_record(patient_id, patient_profile)
                logger.info(f"[{patient_display}] ✅ 患者挂号: {patient_display} -> {dept}科 (优先级: {priority}, 病例: {record.record_id})")
            
            self.stats["total_patients"] += 1
            
            return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[PatientSession]:
        """获取患者会话"""
        return self.patients.get(patient_id)
    
    def update_patient_status(self, patient_id: str, status: PatientStatus):
        """更新患者状态"""
        with self._lock:
            session = self.patients.get(patient_id)
            if session:
                old_status = session.status
                session.status = status
                logger.debug(f"患者 {patient_id} 状态: {old_status.value} -> {status.value}")
    
    # ========== 队列管理 ==========
    
    def enqueue_patient(self, patient_id: str):
        """
        患者加入等候队列
        
        Args:
            patient_id: 患者ID
        """
        with self._lock:
            session = self.patients.get(patient_id)
            if not session:
                logger.error(f"患者 {patient_id} 不存在")
                return
            
            dept = session.dept
            session.status = PatientStatus.WAITING
            session.queue_entry_time = time.time()
            session.effective_priority = session.priority  # 初始有效优先级等于原始优先级

            # 加入优先级队列
            self.waiting_queues[dept].put(session)
            
            queue_size = self.waiting_queues[dept].qsize()
            # 显示资源状态
            available_doctors = len([d for d in self.doctors.values() if d.dept == dept and d.is_available()])
            if queue_size > available_doctors:
                logger.warning(f"⚠️  {dept}科资源紧张: {queue_size}名患者竞争{available_doctors}名空闲医生")
            elif queue_size > 0:
                logger.info(f"📋 {dept}科: {queue_size}名患者候诊 | {available_doctors}名医生空闲")
        
        # 尝试自动分配医生（如果有空闲医生，立即分配）
        self._try_assign_doctor(dept)
    
    def get_queue_size(self, dept: str) -> int:
        """获取科室队列长度"""
        return self.waiting_queues[dept].qsize()
    
    # ========== 医生-患者匹配调度 ==========
    
    def _apply_aging_to_queue(self, dept: str) -> None:
        """对等候队列应用优先级老化：每等待 PRIORITY_AGING_INTERVAL 秒提升 1 点有效优先级。

        由于 PriorityQueue 内部堆不会自动重排，需在每次分配前重建队列。
        此方法必须在持有 self._lock 的情况下调用。
        """
        if dept not in self.waiting_queues:
            return
        queue = self.waiting_queues[dept]
        if queue.empty():
            return

        current_time = time.time()
        items: list = []
        while not queue.empty():
            try:
                session = queue.get_nowait()
                wait_seconds = current_time - session.queue_entry_time
                aging_bonus = int(wait_seconds / PRIORITY_AGING_INTERVAL)
                new_effective = min(PRIORITY_MAX, session.priority + aging_bonus)
                if new_effective != session.effective_priority:
                    case_id = session.patient_data.get("case_id")
                    patient_display = f"P{case_id}" if case_id is not None else session.patient_id
                    logger.info(
                        f"[{patient_display}] ⏫ 优先级老化: {session.effective_priority} → {new_effective}"
                        f"（已等待 {wait_seconds:.0f}s）"
                    )
                    session.effective_priority = new_effective
                items.append(session)
            except Empty:
                break

        # 将重新计算后的患者放回队列（heapq 会按 effective_priority 重新排序）
        for item in items:
            queue.put(item)

    def _try_assign_doctor(self, dept: str) -> bool:
        """
        尝试为等候患者分配医生（自动调度）
        优化：循环分配直到队列为空或无可用医生

        Args:
            dept: 科室

        Returns:
            bool: 是否成功分配至少一个
        """
        assigned_count = 0

        while True:
            with self._lock:
                # 分配前更新所有等候患者的有效优先级（老化）
                self._apply_aging_to_queue(dept)
                # 查找空闲医生
                available_doctors = [
                    d for d in self.doctors.values()
                    if d.dept == dept and d.is_available()
                ]
                
                if not available_doctors:
                    waiting_count = self.waiting_queues[dept].qsize() if dept in self.waiting_queues else 0
                    if waiting_count > 0 and assigned_count == 0:
                        # 去重：只在距离上次输出超过指定间隔时才输出
                        import time
                        current_time = time.time()
                        last_log_time = self._last_waiting_log.get(dept, 0)
                        if current_time - last_log_time >= self._waiting_log_interval:
                            logger.info(f"⏳ {dept}科暂无空闲医生，{waiting_count}名患者等候中")
                            self._last_waiting_log[dept] = current_time
                    break
                
                # 从队列取患者
                if dept not in self.waiting_queues:
                    break
                    
                queue = self.waiting_queues[dept]
                if queue.empty():
                    break
                
                # 使用 try-except 处理并发竞争
                try:
                    session = queue.get_nowait()
                except Empty:
                    logger.info(f"⚡ {dept}科队列已空（并发竞争，其他线程已取走）")
                    break
                
                patient_id = session.patient_id
                
                # 选择负载最轻的医生
                doctor = min(available_doctors, key=lambda d: d.total_patients_today)
                
                # 建立分配关系
                session.assigned_doctor = doctor.doctor_id
                session.status = PatientStatus.CONSULTING
                session.consultation_start_time = now_iso()
                
                doctor.start_consultation(patient_id)
                
                # 显示医生分配信息 - 简化输出
                # 获取患者标识（优先使用case_id）
                case_id = session.patient_data.get("case_id")
                patient_display = f"P{case_id}" if case_id is not None else patient_id
                
                logger.info(f"[{patient_display}] ✅ 分配: 患者 {patient_display} → {doctor.name}")
                
                assigned_count += 1
        
        return assigned_count > 0
    
    def assign_doctor_manually(self, patient_id: str, doctor_id: str) -> bool:
        """
        手动分配医生
        
        Args:
            patient_id: 患者ID
            doctor_id: 医生ID
        
        Returns:
            是否分配成功
        """
        with self._lock:
            session = self.patients.get(patient_id)
            doctor = self.doctors.get(doctor_id)
            
            if not session or not doctor:
                logger.error(f"分配失败: 患者或医生不存在")
                return False
            
            if not doctor.is_available():
                logger.warning(f"医生 {doctor.name} 当前不可用")
                return False
            
            session.assigned_doctor = doctor_id
            session.status = PatientStatus.CONSULTING
            session.consultation_start_time = now_iso()
            
            doctor.start_consultation(patient_id)
            
            # 获取患者标识（用于日志输出）
            case_id = session.patient_data.get("case_id")
            patient_display = f"P{case_id}" if case_id is not None else patient_id
            logger.info(f"[{patient_display}] ✅ 手动分配: 患者 {patient_display} -> 医生 {doctor.name}")
            
            return True
    
    def release_doctor(self, doctor_id: str):
        """
        释放医生（就诊结束）
        
        Args:
            doctor_id: 医生ID
        """
        with self._lock:
            doctor = self.doctors.get(doctor_id)
            if not doctor:
                return
            
            # 记录就诊结束时间
            if doctor.current_patient:
                session = self.patients.get(doctor.current_patient)
                if session:
                    session.consultation_end_time = now_iso()
            
            dept = doctor.dept
            doctor.end_consultation()
            
            self.stats["total_consultations"] += 1
        
        # 尝试分配下一位患者
        self._try_assign_doctor(dept)
    
    # ========== 会诊调度 ==========
    
    def request_consultation(self, patient_id: str, requesting_doctor_id: str, 
                           target_dept: str, reason: str = "") -> Optional[str]:
        """
        请求会诊
        
        Args:
            patient_id: 患者ID
            requesting_doctor_id: 发起会诊的医生ID
            target_dept: 目标科室
            reason: 会诊原因
        
        Returns:
            会诊医生ID（如果成功分配）
        """
        with self._lock:
            session = self.patients.get(patient_id)
            requesting_doctor = self.doctors.get(requesting_doctor_id)
            
            if not session or not requesting_doctor:
                logger.error(f"会诊请求失败: 患者或医生不存在")
                return None
            
            # 查找目标科室的空闲医生
            available_doctors = [
                d for d in self.doctors.values()
                if d.dept == target_dept and d.is_available()
            ]
            
            if available_doctors:
                # 立即分配空闲医生
                consulting_doctor = available_doctors[0]
                consulting_doctor.join_consultation(patient_id)
                session.consultation_doctors.add(consulting_doctor.doctor_id)
                
                self.stats["total_multi_consultations"] += 1
                
                logger.info(f"✅ 会诊已建立: {requesting_doctor.name}({requesting_doctor.dept}) "
                          f"-> {consulting_doctor.name}({target_dept}) | 患者: {patient_id}")
                logger.info(f"   会诊原因: {reason}")
                
                return consulting_doctor.doctor_id
            else:
                # 加入会诊等待队列
                consultation_request = {
                    "patient_id": patient_id,
                    "requesting_doctor_id": requesting_doctor_id,
                    "target_dept": target_dept,
                    "reason": reason,
                    "request_time": now_iso()
                }
                self.consultation_requests.put(consultation_request)
                
                logger.info(f"📋 会诊请求已加入队列: {requesting_doctor.name} -> {target_dept}科 (患者: {patient_id})")
                logger.info(f"   原因: {reason}")
                
                return None
    
    def end_consultation_session(self, patient_id: str, consulting_doctor_id: str):
        """
        结束会诊
        
        Args:
            patient_id: 患者ID
            consulting_doctor_id: 会诊医生ID
        """
        with self._lock:
            doctor = self.doctors.get(consulting_doctor_id)
            session = self.patients.get(patient_id)
            
            if doctor:
                doctor.end_consultation_participation(patient_id)
            
            if session and consulting_doctor_id in session.consultation_doctors:
                session.consultation_doctors.remove(consulting_doctor_id)
            
            logger.info(f"✅ 会诊结束: 医生 {doctor.name if doctor else consulting_doctor_id} | 患者: {patient_id}")
        
        # 检查是否有待处理的会诊请求
        self._process_pending_consultation_requests(doctor.dept if doctor else None)
    
    def _process_pending_consultation_requests(self, dept: Optional[str]):
        """处理待处理的会诊请求"""
        if dept is None:
            return
        
        # 检查是否有等待该科室的会诊请求
        pending_requests = []
        while not self.consultation_requests.empty():
            try:
                request = self.consultation_requests.get_nowait()
                if request["target_dept"] == dept:
                    # 尝试分配
                    consulting_doctor_id = self.request_consultation(
                        request["patient_id"],
                        request["requesting_doctor_id"],
                        request["target_dept"],
                        request["reason"]
                    )
                    if consulting_doctor_id is None:
                        # 仍然没有空闲医生，放回队列
                        pending_requests.append(request)
                else:
                    pending_requests.append(request)
            except Empty:
                break
        
        # 放回未处理的请求
        for req in pending_requests:
            self.consultation_requests.put(req)
    
    # ========== 检验/影像管理 ==========
    
    def send_to_lab(self, patient_id: str):
        """患者去检验科"""
        with self._lock:
            session = self.patients.get(patient_id)
            if session:
                session.status = PatientStatus.WAITING_LAB
                self.lab_queue.put(patient_id)
                logger.info(f"🧪 患者 {patient_id} 前往检验科")
    
    def send_to_imaging(self, patient_id: str):
        """患者去影像科"""
        with self._lock:
            session = self.patients.get(patient_id)
            if session:
                session.status = PatientStatus.WAITING_IMAGING
                self.imaging_queue.put(patient_id)
                logger.info(f"📷 患者 {patient_id} 前往影像科")
    
    def complete_lab_test(self, patient_id: str):
        """完成检验"""
        with self._lock:
            session = self.patients.get(patient_id)
            if session:
                session.lab_results_ready = True
                session.status = PatientStatus.RETURNING
                logger.info(f"✅ 患者 {patient_id} 检验完成，返回就诊科室")
                
                # 重新加入等候队列（复诊）
                self.enqueue_patient(patient_id)
    
    def complete_imaging(self, patient_id: str):
        """完成影像检查"""
        with self._lock:
            session = self.patients.get(patient_id)
            if session:
                session.imaging_results_ready = True
                session.status = PatientStatus.RETURNING
                logger.info(f"✅ 患者 {patient_id} 影像检查完成，返回就诊科室")
                
                # 重新加入等候队列（复诊）
                self.enqueue_patient(patient_id)
    
    # ========== 离院管理 ==========
    
    def discharge_patient(self, patient_id: str):
        """患者离院"""
        with self._lock:
            session = self.patients.get(patient_id)
            if session:
                session.status = PatientStatus.DISCHARGED
                logger.info(f"👋 患者 {patient_id} 已离院")
    
    # ========== 状态查询 ==========
    
    def get_doctor_status(self, doctor_id: str) -> Dict[str, Any]:
        """获取医生状态"""
        doctor = self.doctors.get(doctor_id)
        if not doctor:
            return {}
        
        return {
            "doctor_id": doctor_id,
            "name": doctor.name,
            "dept": doctor.dept,
            "status": doctor.status.value,
            "current_patient": doctor.current_patient,
            "consultation_requests": len(doctor.consultation_requests),
            "total_patients_today": doctor.total_patients_today,
        }
    
    def get_dept_status(self, dept: str) -> Dict[str, Any]:
        """获取科室状态"""
        doctors = [d for d in self.doctors.values() if d.dept == dept]
        waiting = self.waiting_queues[dept].qsize() if dept in self.waiting_queues else 0
        
        return {
            "dept": dept,
            "total_doctors": len(doctors),
            "available_doctors": sum(1 for d in doctors if d.is_available()),
            "busy_doctors": sum(1 for d in doctors if d.status == ResourceStatus.BUSY),
            "consulting_doctors": sum(1 for d in doctors if d.status == ResourceStatus.CONSULTING),
            "waiting_patients": waiting,
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        total_doctors = len(self.doctors)
        active_patients = sum(1 for s in self.patients.values() 
                            if s.status not in [PatientStatus.DISCHARGED])
        
        return {
            "total_doctors": total_doctors,
            "available_doctors": sum(1 for d in self.doctors.values() if d.is_available()),
            "total_patients_registered": self.stats["total_patients"],
            "active_patients": active_patients,
            "total_consultations_completed": self.stats["total_consultations"],
            "multi_consultations": self.stats["total_multi_consultations"],
            "pending_consultation_requests": self.consultation_requests.qsize(),
        }
    
    def get_all_dept_status(self) -> List[Dict[str, Any]]:
        """获取所有科室状态"""
        depts = set(d.dept for d in self.doctors.values())
        return [self.get_dept_status(dept) for dept in sorted(depts)]
