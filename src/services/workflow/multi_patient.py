"""多患者工作流 - 处理多患者并发诊断流程"""

import random
import threading
import time
from datetime import datetime
from typing import List, Dict, Any

from utils import get_logger
from loaders import load_diagnosis_arena_case, _get_dataset_size
from processing import LangGraphMultiPatientProcessor
from display import format_patient_log, get_patient_color
from config import Config
from logging_utils import log_throughput, log_treatment_duration_summary
from logging_utils import log_effective_rounds_summary, log_diagnosis_accuracy_summary
from logging_utils import log_avg_rounds_summary, flush_rag_metric_summaries


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
        self.workflow_run_id = datetime.now().strftime("workflow_%Y%m%d_%H%M%S")
        self._throughput_start_ts = 0.0
        self._throughput_start_iso = ""
        self._total_requests = 0
    
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
        """根据主诉判断就诊优先级，优先使用LLM语义理解，失败时降级为默认优先级为5
        
        Args:
            chief_complaint: 主诉
        
        Returns:
            优先级（1-10，数字越大越紧急）
        """
        # 优先使用LLM语义判断
        if self.llm:
            try:
                system_prompt = (
                    "你是急诊分诊护士，根据患者主诉判断就诊紧急程度。\n"
                    "评分标准：\n"
                    "  9-10分：危及生命，需立即处理（昏迷、休克、大出血、严重呼吸困难等）\n"
                    "  7-8分：病情严重，需尽快处理（高热、剧烈疼痛、晕厥、呕血等）\n"
                    "  5-6分：病情中等，需较快处理（发热、头晕、持续疼痛、恶心呕吐等）\n"
                    "  3-4分：病情较轻，可正常排队（轻微不适、慢性症状复查等）\n"
                    "只返回JSON，格式：{\"priority\": <整数>, \"reason\": \"<简短理由>\"}"
                )
                obj, _, _ = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=f"患者主诉：{chief_complaint}",
                    fallback=lambda: {"priority": 5, "reason": "fallback"},
                    temperature=0.1,
                )
                priority = int(obj.get("priority", 5))
                priority = max(1, min(10, priority))  # 限制在1-10范围内
                logger.debug(f"  🤖 LLM优先级评估: {priority}分 - {obj.get('reason', '')}")
                return priority
            except Exception as e:
                logger.debug(f"  ⚠️  LLM优先级评估失败，返回默认值5: {e}")

        return 5  # LLM不可用或失败时返回中等优先级
    
    def submit_patient(self, i: int, case_id: int, total_patients: int) -> str:
        """提交一个患者到处理队列
        
        Args:
            i: 患者索引
            case_id: 病例ID
            total_patients: 总患者数
        
        Returns:
            任务ID
        """
        # 加载病例获取主诉
        try:
            case_bundle = load_diagnosis_arena_case(case_id)
            known_case = case_bundle["known_case"]
            dataset_index = known_case.get('id', case_id)
            patient_id = f"patient_{case_id:03d}"
            
            # 仅使用新字段主诉
            chief_complaint = str(known_case.get("主诉", "")).strip()
            if not chief_complaint:
                raise ValueError("病例缺少新字段'主诉'")
            
            priority = self.calculate_priority_by_symptoms(chief_complaint)
        except Exception as e:
            logger.warning(f"⚠️  无法加载病例 {case_id} 的主诉，使用随机优先级: {e}")
            priority = random.randint(5, 7)
            chief_complaint = "未知"
            dataset_index = case_id
            patient_id = f"patient_{case_id:03d}"
        
        # 显示患者到达信息
        color = get_patient_color(i)
        priority_icon = "🚨" if priority >= 9 else "⚠️" if priority >= 7 else "📋"
        priority_label = f"{priority_icon} 优先级 P{priority}"
        complaint_preview = chief_complaint[:]

        if total_patients == 1:
            logger.info(
                f"{color}▶ 患者到达 | 病例编号: P{dataset_index} | ID: {patient_id} | "
                f"{priority_label}\033[0m"
            )
        else:
            logger.info(
                f"{color}▶ 患者到达 [{i+1}/{total_patients}] | 病例编号: P{dataset_index} | ID: {patient_id} | "
                f"{priority_label}\033[0m"
            )
        
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
    
    def schedule_patients(self, case_ids: List[int], interval: float) -> None:
        """按时间间隔调度患者
        
        Args:
            case_ids: 病例ID列表
            interval: 患者间隔时间（秒）
        """
        total_patients = len(case_ids)
        self._total_requests = total_patients
        self._throughput_start_ts = time.time()
        self._throughput_start_iso = datetime.now().isoformat()
        timers = []
        
        for i, case_id in enumerate(case_ids):
            delay = i * interval
            timer = threading.Timer(
                delay,
                lambda idx=i, cid=case_id: self.submit_patient(idx, cid, total_patients)
            )
            timer.start()
            timers.append(timer)
        
        for timer in timers:
            timer.join()
    
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
        """监控循环（内部方法）- 每完成一例患者时显示系统状态"""
        def _sleep_interruptible(seconds: int) -> bool:
            end_time = time.time() + seconds
            while self.monitoring_active.is_set() and time.time() < end_time:
                time.sleep(0.2)
            return self.monitoring_active.is_set()

        prev_completed = 0
        while self.monitoring_active.is_set():
            if not _sleep_interruptible(15):
                break

            # 检查是否有活跃患者，但要考虑患者可能还在路上
            active_count = self.processor.get_active_count()
            if active_count == 0:
                # 等待一段时间，防止患者还未到达就退出
                if not _sleep_interruptible(10):
                    break
                active_count = self.processor.get_active_count()
                if active_count == 0:
                    break

            # 检测完成数量变化，有新患者完成时显示状态
            sys_stats = self.coordinator.get_system_stats()
            current_completed = sys_stats['total_consultations_completed']
            if current_completed > prev_completed:
                prev_completed = current_completed
                self._display_system_status(active_count)
    
    def _display_system_status(self, active_count: int) -> None:
        """显示系统状态（内部方法）"""
        sys_stats = self.coordinator.get_system_stats()
        available = sys_stats['available_doctors']
        total = sys_stats['total_doctors']
        busy = total - available
        completed = sys_stats['total_consultations_completed']
        logger.info(
            f"📊 系统状态 | "
            f"🏃 就诊中: {active_count}人 | "
            f"👨‍⚕️ 医生: {available}空闲/{busy}忙碌/{total}总计 | "
            f"✅ 已完成: {completed}例"
        )
    
    def stop_monitoring(self, monitor_thread: threading.Thread) -> None:
        """停止监控
        
        Args:
            monitor_thread: 监控线程对象
        """
        self.monitoring_active.clear()
        monitor_thread.join(timeout=30)
    
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

        results = self.processor.wait_all(timeout=timeout)

        # 系统性能与流程效率指标：并发吞吐量
        if self._throughput_start_ts > 0:
            test_end_ts = time.time()
            duration_seconds = max(0.001, test_end_ts - self._throughput_start_ts)
            completed_requests = len(results)
            throughput = completed_requests / duration_seconds

            # 计算峰值吞吐：按每秒完成数分箱取最大值
            peak_throughput = throughput
            completion_times = [
                item.get("completion_timestamp")
                for item in results
                if isinstance(item.get("completion_timestamp"), (int, float))
            ]
            if completion_times:
                sec_bins: dict[int, int] = {}
                for ts in completion_times:
                    sec = int(max(0.0, float(ts) - self._throughput_start_ts))
                    sec_bins[sec] = sec_bins.get(sec, 0) + 1
                if sec_bins:
                    peak_throughput = float(max(sec_bins.values()))

            log_throughput(
                test_start=self._throughput_start_iso,
                test_end=datetime.now().isoformat(),
                total_requests=int(self._total_requests or num_patients),
                completed_requests=int(completed_requests),
                test_duration_seconds=float(duration_seconds),
                throughput_req_per_sec=float(throughput),
                peak_throughput_req_per_sec=float(peak_throughput),
                run_id=self.workflow_run_id,
            )

        # 系统性能与流程效率指标：平均诊疗时长（按患者）
        durations = []
        for item in results:
            value = item.get("simulated_duration_minutes")
            if isinstance(value, (int, float)):
                durations.append(float(value))
        if durations:
            avg_duration = sum(durations) / len(durations)
            log_treatment_duration_summary(
                avg_duration_minutes=float(avg_duration),
                patient_count=len(durations),
                run_id=self.workflow_run_id,
            )

        # 问诊与诊断效果：平均有效问诊轮次（运行级）
        effective_rounds_list = [
            float(item.get("effective_rounds", 0.0))
            for item in results
            if isinstance(item.get("effective_rounds", 0), (int, float))
        ]
        if effective_rounds_list:
            avg_effective_rounds = sum(effective_rounds_list) / len(effective_rounds_list)
            log_effective_rounds_summary(
                patient_count=len(effective_rounds_list),
                avg_effective_rounds=float(avg_effective_rounds),
                run_id=self.workflow_run_id,
            )

        # 问诊与诊断效果：AvgRounds（总问诊轮次均值）
        log_avg_rounds_summary(run_id=self.workflow_run_id)

        # 问诊与诊断效果：诊断准确率（运行级）
        total_cases = len([r for r in results if r.get("status") == "completed"])
        if total_cases > 0:
            correct_cases = sum(1 for r in results if bool(r.get("diagnosis_correct", False)))
            accuracy = correct_cases / total_cases
            log_diagnosis_accuracy_summary(
                total_cases=total_cases,
                correct_cases=correct_cases,
                accuracy=float(accuracy),
                run_id=self.workflow_run_id,
            )

        # RAG指标：运行级汇总（Recall@k / Groundedness / Latency）
        flush_rag_metric_summaries(run_id=self.workflow_run_id)

        return results
    
    def shutdown(self) -> None:
        """关闭处理器"""
        if self.processor:
            self.processor.shutdown()
