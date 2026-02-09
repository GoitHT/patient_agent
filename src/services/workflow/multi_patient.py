"""多患者工作流 - 处理多患者并发诊断流程"""

import random
import threading
import time
from typing import List, Dict, Any

from utils import get_logger
from loaders import load_diagnosis_arena_case, _get_dataset_size
from processing import LangGraphMultiPatientProcessor
from display import format_patient_log, get_patient_color
from logging_utils import should_log
from config import Config


logger = get_logger("hospital_agent.workflow")


class MultiPatientWorkflow:
    """多患者并发诊断工作流"""
    
    def __init__(
        self,
        config: Config,
        coordinator: Any,
        retriever: Any,
        llm: Any,
        services: Any,
        medical_record_service: Any
    ):
        self.config = config
        self.coordinator = coordinator
        self.retriever = retriever
        self.llm = llm
        self.services = services
        self.medical_record_service = medical_record_service
        self.processor = None
        self.monitoring_active = threading.Event()
    
    def register_doctors(self, num_doctors: int = 3) -> None:
        """注册医生到协调器
        
        Args:
            num_doctors: 医生数量
        """
        logger.info(f"🏥 注册医生: {num_doctors}名")
        for i in range(num_doctors):
            doc_id = f"DOC{i+1:03d}"
            doc_name = f"神经内科医生{i+1}"
            self.coordinator.register_doctor(doc_id, doc_name, "neurology")
    
    def initialize_processor(self, num_patients: int) -> None:
        """初始化多患者处理器
        
        Args:
            num_patients: 患者数量
        """
        logger.info("⚙️  初始化处理器")
        self.processor = LangGraphMultiPatientProcessor(
            coordinator=self.coordinator,
            retriever=self.retriever,
            llm=self.llm,
            services=self.services,
            medical_record_service=self.medical_record_service,
            max_questions=self.config.agent.max_questions,
            max_workers=num_patients,
        )
    
    def select_patient_cases(self, num_patients: int) -> List[int]:
        """从数据集随机选择患者病例
        
        Args:
            num_patients: 需要的患者数量
        
        Returns:
            病例ID列表
        """
        try:
            max_case_id = _get_dataset_size(None)
        except Exception as e:
            logger.warning(f"⚠️  无法获取数据集: {e}")
            max_case_id = 100
        
        logger.info(f"🎲 选择 {num_patients} 个病例 (from {max_case_id})")
        available_case_ids = list(range(max_case_id))
        random.shuffle(available_case_ids)
        return available_case_ids[:num_patients]
    
    def calculate_priority_by_symptoms(self, chief_complaint: str) -> int:
        """根据主诉中的症状严重程度判断优先级
        
        Args:
            chief_complaint: 主诉
        
        Returns:
            优先级（1-10，数字越大越紧急）
        """
        urgent_keywords = ["胸痛", "胸闷", "呼吸困难", "气促", "昏迷", "意识不清",
                          "大出血", "出血不止", "休克", "抽搐", "癫痫发作",
                          "窒息", "严重外伤", "骨折", "剧烈头痛"]
        severe_keywords = ["剧烈疼痛", "持续发热", "高热", "呕血", "黑便", "便血",
                          "咯血", "晕厥", "持续呕吐", "腹痛加重", "无法忍受",
                          "突发", "急性"]
        moderate_keywords = ["疼痛", "不适", "发热", "咳嗽", "头晕", "乏力",
                            "腹泻", "恶心", "反酸", "烧心"]
        
        complaint_lower = chief_complaint.lower()
        
        if any(keyword in complaint_lower for keyword in urgent_keywords):
            return random.randint(9, 10)
        elif any(keyword in complaint_lower for keyword in severe_keywords):
            return random.randint(7, 8)
        elif any(keyword in complaint_lower for keyword in moderate_keywords):
            return random.randint(5, 6)
        else:
            return random.randint(3, 4)
    
    def submit_patient(self, i: int, case_id: int, total_patients: int) -> str:
        """提交一个患者到处理队列
        
        Args:
            i: 患者索引
            case_id: 病例ID
            total_patients: 总患者数
        
        Returns:
            任务ID
        """
        # 加载病例获取主诉和 Patient-SN
        try:
            case_bundle = load_diagnosis_arena_case(case_id)
            known_case = case_bundle["known_case"]
            case_info = known_case.get("Case Information", "")
            dataset_index = known_case.get('id', 'unknown')
            original_case_id = known_case.get('Patient-SN', 'N/A')
            
            # 使用 Patient-SN 作为患者ID（如果可用）
            if original_case_id and original_case_id != 'N/A':
                patient_id = f"patient_{original_case_id}"
            else:
                patient_id = f"patient_{case_id:03d}"
            
            # 提取主诉
            if "主诉：" in case_info:
                start_idx = case_info.find("主诉：") + 3
                remaining = case_info[start_idx:]
                end_markers = ["现病史：", "既往史：", "个人史：", "家族史：", "体格检查：", "\n\n"]
                end_idx = len(remaining)
                for marker in end_markers:
                    pos = remaining.find(marker)
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
                chief_complaint = remaining[:end_idx].strip()
            else:
                chief_complaint = case_info[:100].strip()
            
            priority = self.calculate_priority_by_symptoms(chief_complaint)
        except Exception as e:
            logger.warning(f"⚠️  无法加载病例 {case_id} 的主诉，使用随机优先级: {e}")
            priority = random.randint(5, 7)
            chief_complaint = "未知"
            dataset_index = case_id
            original_case_id = "N/A"
        
        # 显示患者到达信息（简洁版）
        color = get_patient_color(i)
        priority_icon = "🚨" if priority >= 9 else "⚠️" if priority >= 7 else "📋"
        
        if total_patients == 1:
            logger.info(f"{color}▶ P{dataset_index} 就诊 ({priority_icon}P{priority})\033[0m")
        else:
            logger.info(f"{color}▶ P{dataset_index} [{i+1}/{total_patients}] ({priority_icon}P{priority})\033[0m")
        
        # 提交患者
        task_id = self.processor.submit_patient(
            patient_id=patient_id,
            case_id=case_id,
            dept="neurology",
            priority=priority
        )
        
        # 不显示线程启动提示，避免冗余输出
        pass
        
        return task_id
    
    def schedule_patients(self, case_ids: List[int], interval: float) -> List[str]:
        """按时间间隔调度患者
        
        Args:
            case_ids: 病例ID列表
            interval: 患者间隔时间（秒）
        
        Returns:
            任务ID列表
        """
        task_ids = []
        timers = []
        total_patients = len(case_ids)
        
        for i, case_id in enumerate(case_ids):
            delay = i * interval
            timer = threading.Timer(
                delay,
                lambda idx=i, cid=case_id: task_ids.append(
                    self.submit_patient(idx, cid, total_patients)
                )
            )
            timer.start()
            timers.append(timer)
        
        # 等待所有定时器完成
        for timer in timers:
            timer.join()
        
        return task_ids
    
    def start_monitoring(self) -> threading.Thread:
        """启动状态监控线程
        
        Returns:
            监控线程对象
        """
        self.monitoring_active.set()
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def _monitor_loop(self) -> None:
        """监控循环（内部方法）"""
        iteration = 0
        while self.monitoring_active.is_set():
            time.sleep(60)
            iteration += 1
            
            # 检查是否有活跃患者，但要考虑患者可能还在路上
            active_count = self.processor.get_active_count()
            if active_count == 0:
                # 等待一段时间，防止患者还未到达就退出
                time.sleep(10)
                active_count = self.processor.get_active_count()
                if active_count == 0:
                    break
            
            if not should_log(2, "main", "monitor") and iteration % 4 != 0:
                continue
            
            self._display_system_status(active_count)
    
    def _display_system_status(self, active_count: int) -> None:
        """显示系统状态（内部方法）"""
        sys_stats = self.coordinator.get_system_stats()
        logger.info(f"🟢 活动:{active_count} | 医生:{sys_stats['available_doctors']}/{sys_stats['total_doctors']} | 完成:{sys_stats['total_consultations_completed']}")
    
    def stop_monitoring(self, monitor_thread: threading.Thread) -> None:
        """停止监控
        
        Args:
            monitor_thread: 监控线程对象
        """
        self.monitoring_active.clear()
        monitor_thread.join(timeout=2)
    
    def wait_for_completion(self, num_patients: int, timeout: int = None) -> List[Dict[str, Any]]:
        """等待所有患者完成
        
        Args:
            num_patients: 患者数量
            timeout: 超时时间（秒），None表示使用默认计算
        
        Returns:
            结果列表
        """
        if timeout is None:
            timeout = max(600, num_patients * 600)
        
        return self.processor.wait_all(timeout=timeout)
    
    def shutdown(self) -> None:
        """关闭处理器"""
        if self.processor:
            self.processor.shutdown()
