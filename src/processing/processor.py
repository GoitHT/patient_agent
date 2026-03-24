"""
LangGraph 多患者处理器 - 与 LangGraph 流程深度集成
LangGraph Multi-Patient Processor - Deep integration with LangGraph workflows

功能：
1. 为每个患者执行完整的 LangGraph 诊断流程
2. 支持多医生并发接诊
3. 医生资源通过 HospitalCoordinator 统一调度
4. 物理环境模拟与 LangGraph 节点集成
"""

import concurrent.futures
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from agents import PatientAgent, DoctorAgent, NurseAgent, LabAgent
from environment import HospitalWorld
from graphs.router import build_common_graph, build_dept_subgraphs, build_services
from coordination import HospitalCoordinator, PatientStatus
from loaders import load_diagnosis_arena_case, _build_case_info_text
from logging_utils import create_patient_detail_logger, close_patient_detail_logger, get_patient_detail_logger
from logging_utils import log_treatment_duration
from logging_utils import log_consultation_quality, log_effective_rounds, log_diagnosis_accuracy, log_avg_rounds
from rag import AdaptiveRAGRetriever
from services.llm_client import LLMClient
from services.medical_record import MedicalRecordService
from services.medical_record_integration import MedicalRecordIntegration
from state.schema import BaseState
from utils import get_logger, make_run_id

logger = get_logger("hospital_agent.langgraph_multi_patient")


def _normalize_diagnosis_text(text: str) -> str:
    t = (text or "").lower().strip()
    for ch in [" ", "\t", "\n", "，", ",", "。", ".", "；", ";", "、", "(", ")", "（", "）", ":", "：", "-", "_"]:
        t = t.replace(ch, "")
    return t


def _is_diagnosis_match(predicted: str, ground_truth: str) -> bool:
    p_norm = _normalize_diagnosis_text(predicted)
    g_norm = _normalize_diagnosis_text(ground_truth)
    if not p_norm or not g_norm:
        return False
    if p_norm in g_norm or g_norm in p_norm:
        return True
    raw_parts = [x.strip() for x in str(ground_truth).replace("；", ",").replace("、", ",").replace("/", ",").split(",")]
    parts = [_normalize_diagnosis_text(x) for x in raw_parts if x.strip()]
    return any(part and (part in p_norm or p_norm in part) for part in parts)


def _compute_doctor_information_coverage(
    *,
    questions: list[str],
    history: dict[str, Any],
    chief_complaint: str,
) -> float:
    """Doctor-side information coverage.

    Coverage targets:
    1) key symptoms
    2) medical history
    3) risk factors
    """
    q_blob = " ".join(str(q) for q in questions if q).lower()
    h_keys = " ".join(str(k).lower() for k in history.keys()) if isinstance(history, dict) else ""
    h_values = " ".join(str(v).lower() for v in history.values()) if isinstance(history, dict) else ""
    cc = str(chief_complaint or "").lower()

    symptom_keywords = ["症状", "哪里不舒服", "部位", "性质", "伴随", "麻木", "无力", "头痛", "恶心"]
    history_keywords = ["既往", "病史", "过敏", "手术", "家族史", "用药", "慢性病", "高血压", "糖尿病"]
    risk_keywords = ["吸烟", "抽烟", "饮酒", "喝酒", "职业", "接触", "暴露", "危险因素", "肥胖", "高脂"]

    has_key_symptoms = bool(cc) or any(k in q_blob for k in symptom_keywords) or ("associated_symptoms" in h_keys)
    has_history = any(k in q_blob for k in history_keywords) or any(k in h_keys for k in ["history", "past", "allergy", "family", "medication"]) or any(k in h_values for k in ["既往", "过敏", "家族", "手术"])
    has_risk = any(k in q_blob for k in risk_keywords) or any(k in h_values for k in ["吸烟", "饮酒", "职业", "暴露", "高血压", "糖尿病"])

    covered = int(has_key_symptoms) + int(has_history) + int(has_risk)
    return covered / 3.0


def _compute_patient_information_completeness(
    *,
    answers: list[str],
    history: dict[str, Any],
) -> float:
    """Patient-side information completeness.

    Completeness targets:
    1) symptom details
    2) duration
    3) severity
    """
    a_blob = " ".join(str(a) for a in answers if a).lower()
    h = history if isinstance(history, dict) else {}

    has_symptom_detail = (
        any(k in a_blob for k in ["部位", "性质", "伴随", "放射", "具体", "哪个", "什么样", "头", "胸", "腹", "肢体"]) 
        or bool(h.get("associated_symptoms"))
        or bool(h.get("symptom_detail"))
    )
    has_duration = bool(h.get("duration")) or any(k in a_blob for k in ["天", "周", "月", "年", "多久", "起病", "开始"])
    has_severity = bool(h.get("severity")) or any(k in a_blob for k in ["轻", "中", "重", "严重", "剧烈", "几分", "无法", "明显", "加重"])

    covered = int(has_symptom_detail) + int(has_duration) + int(has_severity)
    return covered / 3.0


# ANSI颜色代码 - 用于区分不同患者的输出
class Colors:
    """终端颜色代码"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 前景色
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    
    # 背景色（用于高亮患者ID）
    BG_CYAN = '\033[46m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_RED = '\033[41m'
    
    @staticmethod
    def get_patient_color(patient_index: int) -> tuple:
        """根据患者索引获取颜色（循环使用）"""
        colors = [
            (Colors.CYAN, Colors.BG_CYAN),
            (Colors.GREEN, Colors.BG_GREEN),
            (Colors.YELLOW, Colors.BG_YELLOW),
            (Colors.MAGENTA, Colors.BG_MAGENTA),
            (Colors.BLUE, Colors.BG_BLUE),
        ]
        return colors[patient_index % len(colors)]


class LangGraphPatientExecutor:
    """为单个患者执行完整的 LangGraph 流程"""
    
    def __init__(
        self,
        patient_id: str,
        case_id: int,
        dept: str,
        priority: int,
        coordinator: HospitalCoordinator,
        retriever: AdaptiveRAGRetriever,
        llm: LLMClient,
        services: Any,
        medical_record_service: MedicalRecordService,
        max_questions: int = 3,  # 最底层默认值，通常从config传入
        shared_world: HospitalWorld = None,  # 新增：共享物理环境
        shared_nurse_agent: NurseAgent = None,  # 新增：共享护士
        shared_lab_agent: LabAgent = None,  # 新增：共享检验科
        doctor_agents: Dict[str, DoctorAgent] = None,  # 新增：医生agents字典
    ):
        self.patient_id = patient_id
        self.case_id = case_id
        self.dept = dept
        self.priority = priority
        self.coordinator = coordinator
        self.retriever = retriever
        self.llm = llm
        self.services = services
        self.medical_record_service = medical_record_service
        self.max_questions = max_questions
        self.logger = get_logger(f"patient.{patient_id}")
        
        # 使用共享资源
        self.world = shared_world
        self.nurse_agent = shared_nurse_agent
        self.lab_agent = shared_lab_agent
        self.doctor_agents = doctor_agents or {}
        
        # 创建患者详细日志记录器
        self.detail_logger = None  # 延迟到execute时创建（需要case_id）
    
    def _generate_appointment_info(self) -> dict:
        """
        根据物理世界时间和患者特征动态生成预约信息
        
        Returns:
            包含 channel, timeslot 的字典
        """
        import random
        from datetime import datetime
        
        # 根据物理世界时间判断时段（如果可用），否则随机分配
        if self.world:
            current_hour = self.world.current_time.hour
        else:
            current_hour = datetime.now().hour
        
        if 6 <= current_hour < 12:
            timeslot = "上午"
        elif 12 <= current_hour < 18:
            timeslot = "下午"
        else:
            timeslot = "晚上"  # 18:00-次日06:00 都算晚上
        
        # 根据优先级和时段选择就诊渠道
        if self.priority >= 9:
            # 高优先级：更可能是现场挂号
            channel = random.choices(
                ["线下", "APP", "电话"],
                weights=[0.6, 0.2, 0.2]
            )[0]
        elif self.priority >= 7:
            # 中高优先级：混合渠道
            channel = random.choices(
                ["APP", "线下", "微信小程序", "电话"],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]
        else:
            # 普通优先级：主要通过线上预约
            channel = random.choices(
                ["APP", "微信小程序", "电话", "线下"],
                weights=[0.5, 0.3, 0.1, 0.1]
            )[0]
        
        return {
            "channel": channel,
            "timeslot": timeslot
        }
    
    def _extract_patient_info_from_case(self, case_data: dict) -> dict:
        """
        从新结构化字段中提取患者基本信息（姓名、年龄、性别）
        
        Args:
            case_data: 原始病例数据（from known_case）
        
        Returns:
            包含 name, age, gender 的字典
        """
        # ---- 仅从新结构化字段获取 ----
        name: Any = case_data.get("姓名") or case_data.get("name") or case_data.get("patient_name")
        # 年龄字段可能是字符串（如 "45岁" 或 "45"）
        _age_raw = case_data.get("年龄") or case_data.get("age")
        if _age_raw:
            _age_str = str(_age_raw).replace("岁", "").strip()
            try:
                age: Any = int(_age_str)
            except ValueError:
                age = None
        else:
            age = None
        gender: Any = case_data.get("性别") or case_data.get("gender") or case_data.get("sex")
        
        # 缺失值填充
        if not name:
            name = f"患者{self.patient_id}"
        if not age or age == 0:
            age = 0
        if not gender:
            gender = "未知"
        
        return {
            "name": name,
            "age": age,
            "gender": gender
        }
    
    def _wait_for_doctor_assignment(self, timeout: int = 600) -> Optional[str]:
        """
        等待 coordinator 分配医生（优化：主动重试）
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            分配的医生ID，超时返回 None
        """
        import time
        start_time = time.time()
        check_interval = 0.5  # 检查间隔（秒）
        retry_interval = 5  # 重试间隔（秒）
        last_warning_time = start_time
        last_retry_time = start_time
        
        while time.time() - start_time < timeout:
            session = self.coordinator.get_patient(self.patient_id)
            if session and session.assigned_doctor:
                elapsed = time.time() - start_time
                self.logger.info(f"✅ 医生分配成功（等待 {elapsed:.1f}秒）")
                return session.assigned_doctor
            
            # 每5秒主动重试一次分配
            current_time = time.time()
            if current_time - last_retry_time >= retry_interval:
                if session:
                    self.coordinator._try_assign_doctor(session.dept)
                last_retry_time = current_time
            
            # 每30秒输出一次等待提示
            current_time = time.time()
            if current_time - last_warning_time > 30:
                elapsed = current_time - start_time
                self.logger.info(f"⏳ 仍在等待医生分配... (已等待 {elapsed:.0f}秒)")
                last_warning_time = current_time
            
            time.sleep(check_interval)
        
        # 超时，输出详细的资源状态
        session = self.coordinator.get_patient(self.patient_id)
        if session:
            dept = session.dept
            queue_size = self.coordinator.get_queue_size(dept)
            available_doctors = len(self.coordinator.get_available_doctors(dept))
            total_doctors = len([d for d in self.coordinator.doctors.values() if d.dept == dept])
            
            self.logger.error(f"❌ 等待医生分配超时 ({timeout}秒)")
            self.logger.error(f"   科室: {dept}")
            self.logger.error(f"   队列长度: {queue_size}")
            self.logger.error(f"   可用/总医生: {available_doctors}/{total_doctors}")
        else:
            self.logger.error(f"❌ 等待医生分配超时 ({timeout}秒)")
        
        return None
    
    def execute(self) -> Dict[str, Any]:
        """执行完整的患者诊断流程"""
        try:
            # 创建患者详细日志记录器
            self.detail_logger = create_patient_detail_logger(self.patient_id, self.case_id)
            
            # 为患者分配颜色（基于case_id的哈希）
            patient_hash = hash(str(self.case_id)) % 5
            fg_color, bg_color = Colors.get_patient_color(patient_hash)
            
            # 终端显示开始信息
            patient_tag = f"{bg_color} P{self.case_id} {Colors.RESET}"
            priority_icon = "🚨" if self.priority >= 9 else "⚠️" if self.priority >= 7 else "📋"
            self.logger.info(
                f"{fg_color}▶ {patient_tag} 就诊开始 | "
                f"患者ID: {self.patient_id} | "
                f"科室: {self.dept} | "
                f"{priority_icon} 优先级: P{self.priority}{Colors.RESET}"
            )
            
            # 记录开始时间
            import time
            start_time = time.time()
            
            # 详细日志中记录完整信息
            self.detail_logger.section("开始诊断流程")
            self.detail_logger.info(f"案例ID: {self.case_id}")
            self.detail_logger.info(f"患者ID: {self.patient_id}")
            self.detail_logger.info(f"科室: {self.dept}")
            self.detail_logger.info(f"记录生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.detail_logger.info(f"模拟起始时间: 08:00（医院开始营业）")
            self.detail_logger.info(f"系统配置: 最大问诊轮数={self.max_questions}")
            self.detail_logger.info("")
            
            # 1. 加载病例数据
            self.detail_logger.subsection("加载病例数据")
            case_bundle = load_diagnosis_arena_case(self.case_id)
            known_case = case_bundle["known_case"]
            ground_truth = case_bundle["ground_truth"]
            medical_data = case_bundle["medical_data"]      # 患者不可见：体格检查 + 辅助检查
            _ref_full_case = case_bundle["full_case"]       # Excel原始完整数据（仅用于日志参考）

            # 新版结构化字段日志输出（兼容旧版）
            basic_info_parts = []
            for label in ["姓名", "性别", "年龄", "民族", "职业", "病史陈述者"]:
                value = str(known_case.get(label, "")).strip()
                if value:
                    basic_info_parts.append(f"{label}: {value}")
            if basic_info_parts:
                self.detail_logger.info(f"👤 基本信息: {' | '.join(basic_info_parts)}")

            # 现病史结构化摘要
            present_illness_items = []
            for label in ["现病史_详细描述", "现病史_起病情况", "现病史_病程", "现病史_病情发展"]:
                value = str(known_case.get(label, "")).strip()
                if value:
                    present_illness_items.append(f"{label.replace('现病史_', '')}: {value}")
            if present_illness_items:
                self.detail_logger.info("🩺 现病史摘要:")
                for item in present_illness_items:
                    self.detail_logger.info(f"    {item}")

            # 既往史/个人史/家族史结构化摘要（仅展示非空字段）
            history_items = []
            for label in [
                "既往史_疾病史", "既往史_手术史", "既往史_过敏史",
                "个人史_饮酒史", "个人史_抽烟史", "个人史_月经史",
                "婚育史", "家族史_疾病"
            ]:
                value = str(known_case.get(label, "")).strip()
                if value:
                    history_items.append(f"{label}: {value}")
            if history_items:
                self.detail_logger.info("📚 病史要点:")
                for item in history_items[:6]:
                    self.detail_logger.info(f"    {item}")
                if len(history_items) > 6:
                    self.detail_logger.info(f"    ... 还有 {len(history_items) - 6} 项")
            
            # 提取原始主诉
            case_info = _build_case_info_text(known_case)
            original_chief_complaint = str(known_case.get("主诉", "")).strip()
            if not original_chief_complaint:
                raise ValueError("病例缺少新字段'主诉'，无法继续问诊流程")
            
            # 详细日志中记录完整病例信息
            # 处理原始主诉的显示
            formatted_complaint = original_chief_complaint.replace('\\n', '\n    ')  # 将转义的换行符转为实际换行并缩进
            if len(formatted_complaint) > 300:
                formatted_complaint = formatted_complaint[:300] + "..."
            self.detail_logger.info(f"📋 原始主诉:\n    {formatted_complaint}")
            
            # 参考诊断（仅新字段）
            ref_diagnosis = ground_truth.get('初步诊断', '')
            if ref_diagnosis:
                self.detail_logger.info(f"\n🎯 参考诊断: {ref_diagnosis}")
            
            # 参考治疗方案（来自 Excel 原始数据，仅用于初始日志输出。实际治疗内容将由 LLM 生成后写入数据库）
            if _ref_full_case.get('治疗原则'):
                treatment_plan = _ref_full_case.get('治疗原则', '')
                # 处理转义的换行符
                treatment_plan = treatment_plan.replace('\\n', '\n    ')
                # 智能截断
                if len(treatment_plan) > 250:
                    # 尝试在句号处截断
                    truncate_pos = treatment_plan.rfind('。', 0, 250)
                    if truncate_pos == -1:
                        truncate_pos = 250
                    treatment_plan = treatment_plan[:truncate_pos+1] + "..."
                self.detail_logger.info(f"\n💡 参考治疗方案:\n    {treatment_plan}")

            # 参考辅助检查（来自 medical_data，患者不可见）
            auxiliary_exam = str(medical_data.get('辅助检查', '')).strip()
            if auxiliary_exam:
                display_aux = auxiliary_exam.replace('\\n', '\n    ')
                if len(display_aux) > 250:
                    display_aux = display_aux[:250] + "..."
                self.detail_logger.info(f"\n🔬 参考辅助检查:\n    {display_aux}")

            # 参考医患问答（坥自 Excel 原始数据，不属于 ground_truth 评估字段）
            qa_ref_q = str(_ref_full_case.get('医患问答参考_问', '')).strip()
            qa_ref_a = str(_ref_full_case.get('医患问答参考_答', '')).strip()
            if qa_ref_q or qa_ref_a:
                self.detail_logger.info("\n💬 医患问答参考:")
                if qa_ref_q:
                    show_q = qa_ref_q if len(qa_ref_q) <= 120 else qa_ref_q[:120] + "..."
                    self.detail_logger.info(f"    问: {show_q}")
                if qa_ref_a:
                    show_a = qa_ref_a if len(qa_ref_a) <= 160 else qa_ref_a[:160] + "..."
                    self.detail_logger.info(f"    答: {show_a}")
            
            
            # 2. 使用共享物理环境
            world = self.world  # 使用传入的共享 world
            
            # 患者已在 submit_patient 时添加到 world
            # 3. 初始化 State
            run_id = make_run_id(self.dept)
            
            # 动态生成预约信息
            appointment_info = self._generate_appointment_info()
            
            state = BaseState(
                run_id=run_id,
                dept=self.dept,
                patient_profile={"case_text": case_info},
                appointment=appointment_info,  # 使用动态生成的预约信息
                original_chief_complaint=original_chief_complaint,
                chief_complaint="",
                case_data=known_case,
                ground_truth=ground_truth,
                medical_data=medical_data,
                patient_id=self.patient_id,
                current_location="lobby",
                agent_config={
                    "max_questions": self.max_questions,
                    "use_agents": True,
                },
            )
            
            # 集成物理环境和病例库
            state.world_context = world
            medical_record_integration = MedicalRecordIntegration(self.medical_record_service, world)
            state.medical_record_integration = medical_record_integration
            
            # 注入患者详细日志记录器到 state
            state.patient_detail_logger = self.detail_logger
            
            # 注入 coordinator 和 doctor_agents（供 C4 节点使用）
            state.coordinator = self.coordinator
            state.doctor_agents = self.doctor_agents
            
            # 准备患者基本信息（从病例文本中智能提取）
            extracted_info = self._extract_patient_info_from_case(state.case_data)
            patient_profile = {
                "name": extracted_info["name"],
                "age": extracted_info["age"],
                "gender": extracted_info["gender"],
                "case_id": self.case_id,
                "dataset_id": state.case_data.get("dataset_id"),
                "run_id": run_id,
                "case_data": state.case_data,  # 传递结构化病例字段给数据库层
            }
            
            # 更新state.patient_profile以包含提取的患者信息
            state.patient_profile.update({
                "name": extracted_info["name"],
                "age": extracted_info["age"],
                "gender": extracted_info["gender"],
            })
            
            # 详细日志记录病例和患者信息（合并为一行，减少重复）
            self.detail_logger.info(f"\n👤 患者信息: {extracted_info['name']}, {extracted_info['age']}岁, {extracted_info['gender']}")
            self.detail_logger.info(f"📅 预约信息: {appointment_info['channel']}预约 | 就诊时段: {appointment_info['timeslot']}")
            self.detail_logger.info("")  # 空行分隔
            
            # 4. 准备 Agents
            # 重置护士状态（避免患者之间状态污染）
            self.nurse_agent.reset()
            self.logger.debug(f"  🔄 护士Agent已重置")
            
            # 创建患者专属 Agent
            patient_agent = PatientAgent(
                known_case=state.case_data,
                llm=self.llm,
                chief_complaint=original_chief_complaint
            )
            
            # ===== 5. 执行护士分诊 =====
            nurse_agent = self.nurse_agent
            dept_cn_names = {"neurology": "神经内科"}

            self.detail_logger.section("护士分诊")
            self.detail_logger.staff_info("护士", "nurse_001", "护士")
            world.move_agent(self.patient_id, "triage")

            patient_description = patient_agent.describe_to_nurse()
            triaged_dept = nurse_agent.triage(patient_description=patient_description)

            # 更新科室、run_id 及病历会话
            state.dept = triaged_dept
            run_id = make_run_id(triaged_dept)
            state.run_id = run_id
            patient_profile["run_id"] = run_id

            record_id = medical_record_integration.on_patient_entry(self.patient_id, patient_profile)
            state.history["present_illness"] = patient_description
            state.chief_complaint = ""  # 留空，等待医生总结

            # 获取分诊摘要及理由
            triage_summary = nurse_agent.get_triage_summary()
            state.agent_interactions["nurse_triage"] = triage_summary
            triage_reason = ""
            if triage_summary.get("history"):
                triage_reason = triage_summary["history"][-1].get("reason", "")

            if state.medical_record_integration:
                state.medical_record_integration.on_triage(
                    state, nurse_id="nurse_001", nurse_name="护士"
                )

            # 终端简要显示
            dept_display = dept_cn_names.get(triaged_dept, triaged_dept)
            self.logger.info(f"{fg_color}├ {patient_tag} 分诊→{dept_display}{Colors.RESET}")

            # 详细日志
            self.detail_logger.info(f"📋 患者主诉:\n    {patient_description}\n")
            self.detail_logger.info(
                f"✅ 就诊会话已创建，病例ID: {record_id} | 科室: {triaged_dept}({dept_display})"
            )
            if triage_reason:
                self.detail_logger.info(f"    分诊理由: {triage_reason}")

            # ===== 6. 通过 Coordinator 注册患者 =====
            
            # 准备患者数据（复用已提取的信息）
            patient_data = {
                "name": patient_profile["name"],
                "age": patient_profile["age"],
                "gender": patient_profile["gender"],
                "case_id": self.case_id,
                "dataset_id": state.case_data.get("dataset_id"),
                "run_id": state.run_id,
            }
            
            # 注册患者到 coordinator（不立即分配医生）
            self.coordinator.register_patient(
                patient_id=self.patient_id,
                patient_data=patient_data,
                dept=triaged_dept,
                priority=self.priority
            )
            
            # 加入等候队列（医生分配将在 C4 节点中执行）
            self.coordinator.enqueue_patient(self.patient_id)
            
            # 记录候诊信息到详细日志
            queue_size = self.coordinator.get_queue_size(triaged_dept)
            available_doctors = len(self.coordinator.get_available_doctors(triaged_dept))
            self.detail_logger.info("")
            self.detail_logger.info(f"✅ 患者已加入候诊队列")
            self.detail_logger.info(f"    队列位置: 第{queue_size}位")
            self.detail_logger.info(f"    可用医生: {available_doctors}名")
            self.detail_logger.info(f"    ⏳ 等待叫号...（医生将在候诊室分配）")
            self.detail_logger.info("")
            
        
            # doctor_agent 将在 C4 节点中根据分配的医生ID获取
            
            # 7. 构建 LangGraph
            self.detail_logger.subsection("构建执行图")
            self.detail_logger.info(f"    执行引擎: LangGraph")
            self.detail_logger.info(f"    流程图: {state.dept}_specialty_graph")
            self.detail_logger.info(f"    配置参数: max_questions={self.max_questions}, use_agents=True")
            
            # 注入 patient_agent 到 state
            state.patient_agent = patient_agent
            
            # 构建图时不传入特定 doctor_agent（在 C4 动态分配）
            dept_subgraphs = build_dept_subgraphs(
                retriever=self.retriever,
                llm=self.llm,
                doctor_agent=None,  # 将在 C4 节点中动态设置
                patient_agent=patient_agent,
                max_questions=self.max_questions
            )
            
            graph = build_common_graph(
                dept_subgraphs,
                retriever=self.retriever,
                services=self.services,
                llm=self.llm,
                llm_reports=False,
                use_agents=True,
                patient_agent=patient_agent,
                doctor_agent=None,  # 将在 C4 节点中动态设置
                nurse_agent=self.nurse_agent,
                lab_agent=self.lab_agent,
                max_questions=self.max_questions,
                world=self.world,
            )
            
            # 8. 执行 LangGraph 流程
            self.logger.info(f"{fg_color}🏥 {patient_tag} {fg_color}| 门诊流程开始{Colors.RESET}")
            
            # 记录流程开始时的模拟世界时钟，作为最终兜底用于计算就诊时长
            _world_start_time = self.world.current_time if self.world else None
            
            self.detail_logger.section("执行门诊流程")
            self.detail_logger.info("🔄 开始执行 LangGraph 工作流...")
            self.detail_logger.info("")
            
            node_count = 0
            node_names = []  # 记录节点名称
            node_time_map: dict = {}  # 记录每个节点完成时的模拟时钟 HH:MM
            out = None
            final_state = state  # 保存最终状态，初始为输入状态
            last_diagnosis_state = None  # 记录最近一次产生诊断的状态
            last_simulated_minutes = None  # 追踪最新的模拟就诊时长
            
            for chunk in graph.stream(state):
                node_count += 1
                if isinstance(chunk, dict) and len(chunk) > 0:
                    node_name = list(chunk.keys())[0]
                    node_names.append(node_name)
                    out = chunk[node_name]
                    # 记录该节点完成时的模拟时钟（节点内部已推进完毕）
                    # 若节点设置了 node_log_time（在 advance_time 后、LLM 调用前写入），
                    # 优先使用该意图时间戳，避免并发患者线程在 LLM I/O 期间推进共享时钟
                    # 导致本节点耗时虚高
                    if self.world:
                        _intended = getattr(out, 'node_log_time', '') if isinstance(out, BaseState) else \
                                    (out.get('node_log_time', '') if isinstance(out, dict) else '')
                        if _intended:
                            node_time_map[node_name] = _intended
                        else:
                            try:
                                node_time_map[node_name] = self.world.patient_current_time(self.patient_id).strftime('%H:%M')
                            except Exception:
                                node_time_map[node_name] = self.world.current_time.strftime('%H:%M')
                        # 消费后清空，防止被下一节点继承
                        if _intended and isinstance(out, BaseState):
                            out.node_log_time = ""
                    
                    # 更新最终状态（接受BaseState或字典类型）
                    if isinstance(out, BaseState):
                        final_state = out
                        
                        # 跟踪最近有诊断的状态
                        if isinstance(out.diagnosis, dict) and out.diagnosis.get("name"):
                            last_diagnosis_state = out
                        # 追踪模拟就诊时长（防止后续节点更新覆盖丢失）
                        if hasattr(out, 'appointment') and isinstance(out.appointment, dict):
                            val = out.appointment.get('simulated_duration_minutes')
                            if val is not None:
                                last_simulated_minutes = val
                    elif isinstance(out, dict):
                        # 【修复】LangGraph可能返回字典而非Pydantic对象
                        # 尝试将字典转换为BaseState
                        try:
                            final_state = BaseState.model_validate(out)
                            
                            # 跟踪最近有诊断的状态
                            if isinstance(final_state.diagnosis, dict) and final_state.diagnosis.get("name"):
                                last_diagnosis_state = final_state
                            # 追踪模拟就诊时长
                            if isinstance(final_state.appointment, dict):
                                val = final_state.appointment.get('simulated_duration_minutes')
                                if val is not None:
                                    last_simulated_minutes = val
                        except Exception as e:
                            if node_name in ["C12", "C13", "C14", "C15", "C16"]:
                                self.detail_logger.warning(f"⚠️  [{node_name}] 从字典转换为BaseState失败: {e}")
                        # 即使转换失败，也尝试从字典直接提取
                        if isinstance(out, dict) and isinstance(out.get('appointment'), dict):
                            val = out['appointment'].get('simulated_duration_minutes')
                            if val is not None:
                                last_simulated_minutes = val
                    
                    # 详细日志记录每个节点的执行
                    self.detail_logger.info(f"{'─'*80}")
                    self.detail_logger.info(f"节点 #{node_count}: {node_name}")
                    self.detail_logger.info(f"{'─'*80}")
                    
                    # 记录节点输出的关键信息
                    if hasattr(out, '__dict__'):
                        # 记录位置变化
                        if hasattr(out, 'current_location'):
                            self.detail_logger.info(f"  📍 当前位置: {out.current_location}")
                        
                        # 记录诊断信息
                        if hasattr(out, 'diagnosis') and out.diagnosis:
                            if isinstance(out.diagnosis, dict):
                                diag_name = out.diagnosis.get('name', '未知')
                                self.detail_logger.info(f"  🔬 诊断: {diag_name}")
                                if out.diagnosis.get('confidence'):
                                    self.detail_logger.info(f"      置信度: {out.diagnosis['confidence']}")
                        
                        # 记录开具的检查
                        if hasattr(out, 'ordered_tests') and out.ordered_tests:
                            self.detail_logger.info(f"  📋 开具检查: {len(out.ordered_tests)}项")
                            for i, test in enumerate(out.ordered_tests[:3], 1):
                                test_name = test.get('name', test.get('test_name', '未知'))
                                self.detail_logger.info(f"      {i}. {test_name}")
                            if len(out.ordered_tests) > 3:
                                self.detail_logger.info(f"      ... 还有 {len(out.ordered_tests) - 3} 项")
                        
                        # 记录检查结果
                        if hasattr(out, 'test_results') and out.test_results:
                            self.detail_logger.info(f"  🧪 检查结果: {len(out.test_results)}项完成")
                        
                        # 记录处方
                        if hasattr(out, 'treatment_plan') and out.treatment_plan:
                            if isinstance(out.treatment_plan, dict):
                                if out.treatment_plan.get('medications'):
                                    meds = out.treatment_plan['medications']
                                    med_count = len(meds) if isinstance(meds, list) else 1
                                    self.detail_logger.info(f"  💊 处方药物: {med_count}种")
                    
                    self.detail_logger.info("")
            
            # 计算总耗时
            import time
            program_execution_time = time.time() - start_time if 'start_time' in locals() else 0
            
            # 获取患者就诊时间
            # 优先使用 world 的患者个人计时（多患者并发下不会被其他患者动作污染）
            simulated_minutes = None
            if self.world is not None and self.patient_id:
                try:
                    simulated_minutes = self.world.get_patient_elapsed_minutes(self.patient_id)
                except Exception:
                    simulated_minutes = None

            # 次优先：使用循环中追踪到的最新值（防止被后续节点覆盖丢失）
            if simulated_minutes is None or simulated_minutes <= 0:
                simulated_minutes = last_simulated_minutes
            if simulated_minutes is None and final_state and hasattr(final_state, 'appointment'):
                simulated_minutes = final_state.appointment.get('simulated_duration_minutes')
            # 兜底：直接通过物理世界时钟和预约开始时间计算
            if simulated_minutes is None and self.world is not None:
                visit_start_str = None
                if final_state and hasattr(final_state, 'appointment'):
                    visit_start_str = final_state.appointment.get('visit_start_time')
                if visit_start_str is None and hasattr(state, 'appointment'):
                    visit_start_str = state.appointment.get('visit_start_time')
                if visit_start_str:
                    try:
                        import datetime as _dt
                        visit_start = _dt.datetime.fromisoformat(visit_start_str)
                        simulated_minutes = (self.world.current_time - visit_start).total_seconds() / 60
                    except Exception:
                        pass
            # 最终兜底：用流程开始前记录的世界时钟起点计算
            if simulated_minutes is None and _world_start_time is not None and self.world is not None:
                simulated_minutes = (self.world.current_time - _world_start_time).total_seconds() / 60
            
            # 用于终端简要显示
            total_time_seconds = simulated_minutes * 60 if simulated_minutes else program_execution_time
            
            self.detail_logger.section("诊断完成")
            self.detail_logger.info("")
            self.detail_logger.info("📋 执行概要:")
            self.detail_logger.info(f"  • 总节点数: {node_count}个")
            # 统一显示格式：就诊时间 | 系统时间
            if simulated_minutes is not None:
                self.detail_logger.info(f"  • 总耗时: {simulated_minutes:.0f}分钟（患者就诊时间） | {program_execution_time:.1f}秒（系统运行时间）")
            else:
                self.detail_logger.info(f"  • 总耗时: {program_execution_time:.1f}秒（系统运行时间）")
            if node_count > 0:
                if simulated_minutes is not None:
                    self.detail_logger.info(f"  • 平均每节点: {simulated_minutes/node_count:.1f}分钟 | {program_execution_time/node_count:.1f}秒")
                else:
                    self.detail_logger.info(f"  • 平均每节点: {program_execution_time/node_count:.1f}秒")
            self.detail_logger.info("")
            self.detail_logger.info("📍 完整节点路径:")
            self.detail_logger.info(f"  {' → '.join(node_names)}")
            self.detail_logger.info("")
            
            # 9. 提取结果
            # 使用最终状态而不是最后一个节点输出
            # 安全提取诊断结果（检查final_state是否存在，以及diagnosis是否为有效字典）
            
            final_diagnosis = "未明确"
            state_for_diagnosis = final_state
            
            # 优先使用last_diagnosis_state（最近一次更新诊断的状态）
            # 因为在LangGraph的stream过程中，final_state的diagnosis可能被后续节点重置
            if last_diagnosis_state is not None and isinstance(last_diagnosis_state.diagnosis, dict) and last_diagnosis_state.diagnosis.get("name"):
                # 优先使用last_diagnosis_state
                final_diagnosis = last_diagnosis_state.diagnosis.get("name", "未明确")
                self.detail_logger.info(f"✅ 从last_diagnosis_state提取诊断: {final_diagnosis}")
            elif (
                state_for_diagnosis
                and isinstance(state_for_diagnosis, BaseState)
                and isinstance(state_for_diagnosis.diagnosis, dict)
                and state_for_diagnosis.diagnosis.get("name")
            ):
                # 再检查final_state
                final_diagnosis = state_for_diagnosis.diagnosis.get("name", "未明确")
                self.detail_logger.info(f"✅ 从final_state提取诊断: {final_diagnosis}")
            else:
                self.detail_logger.warning("⚠️  未找到有效诊断状态，诊断将标记为未明确")
            
            result = {
                "status": "completed",
                "patient_id": self.patient_id,
                "case_id": self.case_id,
                "dept": triaged_dept,
                "diagnosis": final_diagnosis,
                "node_count": node_count,
                "node_names": node_names,  # 添加节点名称列表
                "record_id": record_id,
                "detail_log_file": self.detail_logger.get_log_file_path() if hasattr(self, 'detail_logger') and self.detail_logger else "",  # 添加详细日志路径
                "simulated_duration_minutes": simulated_minutes,
                "wall_time_seconds": program_execution_time,
                "visit_start_time": (final_state.appointment.get("visit_start_time") if final_state and hasattr(final_state, "appointment") and isinstance(final_state.appointment, dict) else ""),
                "visit_end_time": (final_state.appointment.get("visit_end_time") if final_state and hasattr(final_state, "appointment") and isinstance(final_state.appointment, dict) else ""),
                "run_id": (final_state.run_id if final_state and hasattr(final_state, "run_id") else ""),
                "completion_timestamp": time.time(),
            }

            # 系统性能与流程效率指标：记录患者诊疗时长
            log_treatment_duration(
                visit_start_time=str(result.get("visit_start_time", "")),
                visit_end_time=str(result.get("visit_end_time", "")),
                visit_duration_minutes=(float(simulated_minutes) if simulated_minutes is not None else None),
                simulated_duration_minutes=(float(simulated_minutes) if simulated_minutes is not None else None),
                wall_time_seconds=float(program_execution_time),
                run_id=str(result.get("run_id", "")),
                patient_id=str(self.patient_id),
                case_id=str(self.case_id),
            )

            # 问诊与诊断效果指标：问诊质量（Consultation Quality）
            qa_quality = {}
            interview_quality = {}
            if final_state and hasattr(final_state, "agent_interactions") and isinstance(final_state.agent_interactions, dict):
                qa_quality = final_state.agent_interactions.get("qa_quality_scores", {}) or {}
                interview_quality = final_state.agent_interactions.get("interview_quality", {}) or {}

            detailed_scores = qa_quality.get("detailed_scores", []) if isinstance(qa_quality, dict) else []

            def _avg_metric(path: tuple[str, ...], default: float = 0.0) -> float:
                if not detailed_scores:
                    return default
                vals = []
                for item in detailed_scores:
                    cur = item
                    ok = True
                    for key in path:
                        if isinstance(cur, dict) and key in cur:
                            cur = cur[key]
                        else:
                            ok = False
                            break
                    if ok and isinstance(cur, (int, float)):
                        vals.append(float(cur))
                if not vals:
                    return default
                return sum(vals) / len(vals)

            doctor_specificity = _avg_metric(("doctor_metrics", "specificity"), float(qa_quality.get("avg_doctor_quality", 0.0) or 0.0))
            doctor_purposefulness = _avg_metric(("doctor_metrics", "targetedness"), float(qa_quality.get("avg_doctor_quality", 0.0) or 0.0))
            doctor_professionalism = _avg_metric(("doctor_metrics", "professionalism"), float(qa_quality.get("avg_doctor_quality", 0.0) or 0.0))

            patient_relevance = _avg_metric(("patient_metrics", "relevance"), float(qa_quality.get("avg_patient_ability", 0.0) or 0.0))
            patient_faithfulness = _avg_metric(("patient_metrics", "faithfulness"), float(qa_quality.get("avg_patient_ability", 0.0) or 0.0))
            patient_consistency_robustness = _avg_metric(("patient_metrics", "robustness"), float(qa_quality.get("avg_patient_ability", 0.0) or 0.0))

            qa_pairs = []
            if final_state and hasattr(final_state, "agent_interactions") and isinstance(final_state.agent_interactions, dict):
                raw_pairs = final_state.agent_interactions.get("doctor_patient_qa", [])
                if isinstance(raw_pairs, list):
                    qa_pairs = [p for p in raw_pairs if isinstance(p, dict)]

            questions = [str(p.get("question", "")) for p in qa_pairs if p.get("question")]
            answers = [str(p.get("answer", "")) for p in qa_pairs if p.get("answer")]
            state_history = final_state.history if final_state and hasattr(final_state, "history") and isinstance(final_state.history, dict) else {}
            state_cc = final_state.chief_complaint if final_state and hasattr(final_state, "chief_complaint") else ""

            # 医生信息覆盖度：关键症状、病史、危险因素
            doctor_information_coverage = _compute_doctor_information_coverage(
                questions=questions,
                history=state_history,
                chief_complaint=str(state_cc),
            )

            # 患者信息完整性：症状细节、持续时间、程度
            patient_information_completeness = _compute_patient_information_completeness(
                answers=answers,
                history=state_history,
            )

            # 兜底：避免极端空数据导致全零
            if doctor_information_coverage <= 0:
                doctor_information_coverage = float(interview_quality.get("completeness_score", 0.0) or 0.0) / 100.0
            if patient_information_completeness <= 0:
                patient_information_completeness = float(interview_quality.get("depth_score", 0.0) or 0.0) / 100.0

            doctor_quality = (doctor_specificity + doctor_purposefulness + doctor_professionalism + doctor_information_coverage) / 4.0
            patient_quality = (patient_relevance + patient_faithfulness + patient_information_completeness + patient_consistency_robustness) / 4.0
            consultation_quality_score = (doctor_quality + patient_quality) / 2.0

            log_consultation_quality(
                doctor_specificity=doctor_specificity,
                doctor_purposefulness=doctor_purposefulness,
                doctor_professionalism=doctor_professionalism,
                doctor_information_coverage=doctor_information_coverage,
                patient_relevance=patient_relevance,
                patient_faithfulness=patient_faithfulness,
                patient_information_completeness=patient_information_completeness,
                patient_consistency_robustness=patient_consistency_robustness,
                consultation_quality_score=consultation_quality_score,
                run_id=str(result.get("run_id", "")),
                patient_id=str(self.patient_id),
                case_id=str(self.case_id),
            )

            # 问诊与诊断效果指标：平均有效问诊轮次（按高质量阈值判定单轮是否有效）
            total_rounds = int(final_state.node_qa_counts.get("global_total", 0)) if final_state and hasattr(final_state, "node_qa_counts") else 0
            effective_rounds = 0
            if detailed_scores:
                effective_rounds = sum(1 for item in detailed_scores if isinstance(item, dict) and float(item.get("overall_score", 0.0) or 0.0) >= 0.7)
            if total_rounds == 0 and isinstance(qa_quality.get("total_rounds"), int):
                total_rounds = int(qa_quality.get("total_rounds", 0))
            avg_effective_rounds = float(effective_rounds)

            log_effective_rounds(
                total_rounds=total_rounds,
                effective_rounds=effective_rounds,
                avg_effective_rounds=avg_effective_rounds,
                run_id=str(result.get("run_id", "")),
                patient_id=str(self.patient_id),
                case_id=str(self.case_id),
            )

            # 问诊与诊断效果指标：AvgRounds（每病例总问诊轮次）
            if total_rounds >= 0:
                log_avg_rounds(
                    rounds=total_rounds,
                    run_id=str(result.get("run_id", "")),
                    patient_id=str(self.patient_id),
                    case_id=str(self.case_id),
                )

            # 问诊与诊断效果指标：诊断准确率（单病例）
            gt_text = ""
            if final_state and hasattr(final_state, "ground_truth") and isinstance(final_state.ground_truth, dict):
                gt_text = str(final_state.ground_truth.get("初步诊断") or final_state.ground_truth.get("diagnosis") or "")
            is_correct = _is_diagnosis_match(final_diagnosis, gt_text)
            log_diagnosis_accuracy(
                predicted_diagnosis=final_diagnosis,
                ground_truth_diagnosis=gt_text,
                is_correct=is_correct,
                run_id=str(result.get("run_id", "")),
                patient_id=str(self.patient_id),
                case_id=str(self.case_id),
            )

            result["consultation_quality_score"] = consultation_quality_score
            result["effective_rounds"] = effective_rounds
            result["total_rounds"] = total_rounds
            result["diagnosis_correct"] = is_correct
            
            self.logger.info(f"{fg_color}└ {patient_tag} 诊断→{final_diagnosis} ({total_time_seconds/60:.0f}min){Colors.RESET}")
            
            # 详细日志记录完整诊断结果
            self.detail_logger.info("🎯 诊断结果:")
            self.detail_logger.info(f"  • AI诊断: {final_diagnosis}")
            self.detail_logger.info("")
            
            # 问诊质量评估
            if hasattr(final_state, 'collected_info'):
                info_items = len([k for k, v in final_state.collected_info.items() if v])
                self.detail_logger.info("📊 问诊质量评估:")
                self.detail_logger.info(f"  • 收集信息项: {info_items}项")
                if hasattr(final_state, 'test_results'):
                    self.detail_logger.info(f"  • 完成检查: {len(final_state.test_results)}项")
                self.detail_logger.info("")
            
            # 关键决策点
            self.detail_logger.info("📌 关键决策点:")
            if hasattr(final_state, 'ordered_tests') and final_state.ordered_tests:
                self.detail_logger.info(f"  • 开单检查: {len(final_state.ordered_tests)}项")
                for test in final_state.ordered_tests[:5]:  # 最多显示5项
                    self.detail_logger.info(f"    - {test.get('name', '未知')} ({test.get('type', '未知')})")
            if hasattr(final_state, 'escalations') and final_state.escalations:
                self.detail_logger.info(f"  • 升级建议: {len(final_state.escalations)}项")
                for esc in final_state.escalations[:3]:
                    self.detail_logger.info(f"    - {esc}")
            self.detail_logger.info("")
            
            self.detail_logger.info("📋 病例记录:")
            self.detail_logger.info(f"  • 记录ID: {record_id}")
            self.detail_logger.info(f"  • 详细日志: {self.detail_logger.get_log_file_path()}")
            self.detail_logger.info("")
            
            # 添加诊疗流程总结
            self.detail_logger.section("诊疗流程总结")
            self.detail_logger.info("")
            self.detail_logger.info("📋 就诊流程回顾:")
            self.detail_logger.info(f"  1️⃣  患者到达 → 护士分诊 → {triaged_dept}")
            self.detail_logger.info(f"  2️⃣  问诊收集信息 → {node_count}个节点")
            if hasattr(out, 'ordered_tests') and out.ordered_tests:
                self.detail_logger.info(f"  3️⃣  开单检查 → {len(out.ordered_tests)}项检查")
            if hasattr(out, 'test_results') and out.test_results:
                self.detail_logger.info(f"  4️⃣  检查结果 → {len(out.test_results)}份报告")
            self.detail_logger.info(f"  5️⃣  诊断结论 → {final_diagnosis}")
            self.detail_logger.info("")
            
            # 质量指标
            self.detail_logger.info("📊 质量指标:")
            if simulated_minutes is not None:
                self.detail_logger.info(f"  • 流程效率: {simulated_minutes:.1f}分钟 / {node_count}节点")
            else:
                self.detail_logger.info(f"  • 流程效率: {program_execution_time:.1f}秒 / {node_count}节点")
            if hasattr(out, 'ordered_tests'):
                test_coverage = "充分" if len(out.ordered_tests) >= 3 else "一般" if len(out.ordered_tests) >= 1 else "不足"
                self.detail_logger.info(f"  • 检查覆盖: {test_coverage} ({len(out.ordered_tests)}项)")
            
            # 资源使用统计
            session = self.coordinator.get_patient(self.patient_id)
            if session and session.assigned_doctor:
                doctor = self.coordinator.get_doctor(session.assigned_doctor)
                if doctor:
                    self.detail_logger.info(f"  • 接诊医生: {doctor.name} (今日第{doctor.total_patients_today}位患者)")
            self.detail_logger.info("")
            
            # 改进建议
            self.detail_logger.info("💡 流程改进建议:")
            if hasattr(out, 'ordered_tests') and len(out.ordered_tests) == 0:
                self.detail_logger.info("  ⚠️  未开具任何检查，可能影响诊断准确性")
            if node_count > 20:
                self.detail_logger.info("  ℹ️  流程节点较多，考虑优化诊疗路径")
            # 使用模拟时间判断（如果有），否则使用程序执行时间
            if simulated_minutes is not None and simulated_minutes > 60:
                self.detail_logger.info(f"  ℹ️  就诊时间较长（{simulated_minutes:.0f}分钟），考虑优化检查流程")
            elif simulated_minutes is None and program_execution_time > 300:
                self.detail_logger.info("  ℹ️  程序执行时间较长，考虑优化响应速度")
            if hasattr(out, 'ordered_tests') and len(out.ordered_tests) >= 3:
                self.detail_logger.info("  ✅ 诊疗流程规范，质量良好")
            self.detail_logger.info("")
            
            # 最终状态总结
            self.detail_logger.section("就诊完成总结")
            self.detail_logger.info(f"✅ 患者 {self.patient_id} 就诊流程完成")
            # 统一显示格式
            if simulated_minutes is not None:
                self.detail_logger.info(f"📊 总耗时: {simulated_minutes:.0f}分钟（患者就诊时间） | {program_execution_time:.1f}秒（系统运行时间）")
            else:
                self.detail_logger.info(f"📊 总耗时: {program_execution_time:.1f}秒（系统运行时间）")
            self.detail_logger.info(f"📋 诊断: {final_diagnosis}")
            if hasattr(out, 'ordered_tests'):
                self.detail_logger.info(f"🔬 检查项数: {len(out.ordered_tests)}项")
            if hasattr(out, 'test_results'):
                self.detail_logger.info(f"📊 完成检查: {len(out.test_results)}项")
            self.detail_logger.info("")

            # ─── 就诊时间线 ──────────────────────────────────────────────────────
            self.detail_logger.section("就诊时间线")
            _node_zh = {
                "C1": "开始就诊", "C2": "挂号登记", "C3": "候诊签到",
                "C4": "呼叫就诊", "C5": "问诊准备", "C6": "专科接诊",
                "C7": "路径决策", "C8": "开具检查", "C9": "缴费预约",
                "C10": "执行检查", "C11": "复诊回诊", "C12": "综合诊断",
                "C13": "制定方案", "C14": "开具文书", "C15": "健康教育",
                "C16": "完成就诊",
                # 专科子图节点（通过C6调用，一般不直接出现）
                "S1": "专科问诊", "S2": "体格检查", "S3": "初步判断",
            }
            _node_desc = {
                "C1": "患者入院，初始化就诊流程",
                "C2": "完成预约挂号，生成就诊序号",
                "C3": "签到并进入候诊队列",
                "C4": "被呼叫进入诊室",
                "C5": "医生准备，阅读病历",
                "C6": "专科问诊+体检+初步判断",
                "C7": "判断是否需要辅助检查",
                "C8": "向患者解释检查项目",
                "C9": "缴费并预约检查时段",
                "C10": "前往各科室完成检查",
                "C11": "携带报告返回诊室",
                "C12": "结合检查结果综合诊断",
                "C13": "制定诊疗处置方案",
                "C14": "开具处方/检查单/住院证",
                "C15": "健康宣教及随访安排",
                "C16": "办理离院，结束就诊",
            }
            _movements = (
                (final_state.movement_history or [])
                if (final_state and hasattr(final_state, 'movement_history'))
                else []
            )
            # 就诊基准时间（分钟数）
            _base_min: Optional[int] = None
            if final_state and hasattr(final_state, 'appointment') and isinstance(final_state.appointment, dict):
                _vst_str = final_state.appointment.get('visit_start_time')
                if _vst_str:
                    try:
                        _vst_dt = datetime.fromisoformat(_vst_str)
                        _base_min = _vst_dt.hour * 60 + _vst_dt.minute
                    except Exception:
                        pass

            def _hhmm_to_min(s: str) -> Optional[int]:
                try:
                    h, m = map(int, str(s).split(':'))
                    return h * 60 + m
                except Exception:
                    return None

            def _normalize_event_minutes(events: list[dict]) -> list[Optional[int]]:
                """
                将事件 HH:MM 时间归一化为单调不减的绝对分钟序列。

                目标：
                - 允许跨天（23:xx -> 00:xx）
                - 对同次就诊中的异常回跳（并发/旧时间戳污染）进行钳制，
                  避免出现 1369min 这类取模异常耗时。
                """
                abs_list: list[Optional[int]] = []
                prev_abs: Optional[int] = None
                day_offset = 0

                for ev in events:
                    cur = _hhmm_to_min(ev.get('time', ''))
                    if cur is None:
                        abs_list.append(None)
                        continue

                    cand = cur + day_offset * 1440
                    if prev_abs is not None and cand < prev_abs:
                        prev_clock = prev_abs % 1440
                        # 典型跨天场景：前一时刻接近午夜，当前时刻在凌晨
                        if prev_clock >= 1320 and cur <= 180:
                            day_offset += 1
                            cand = cur + day_offset * 1440
                        else:
                            # 非跨天的回跳，视为时间戳异常，钳制为不回退
                            cand = prev_abs

                    abs_list.append(cand)
                    prev_abs = cand

                return abs_list

            def _abs_min_to_hhmm(abs_min: Optional[int]) -> str:
                if abs_min is None:
                    return '--:--'
                clock = abs_min % 1440
                h = clock // 60
                m = clock % 60
                return f"{h:02d}:{m:02d}"

            # 构建 node -> movement list 映射
            _move_by_node: dict = {}
            for _mv in _movements:
                _mn = _mv.get('node', '')
                _move_by_node.setdefault(_mn, []).append(_mv)

            # 构建统一事件列表：movement nodes 使用移动记录时间，in-place nodes 使用 node_time_map
            _cur_loc = "入院"
            _timeline_events = []
            # 读取专科子图各节点完成时钟（由子图节点写入 appointment）
            _appt_times = (
                final_state.appointment
                if (final_state and hasattr(final_state, 'appointment') and isinstance(final_state.appointment, dict))
                else {}
            )
            _subgraph_times = {
                'S1': _appt_times.get('_s1_end_time'),
                'S2': _appt_times.get('_s2_end_time'),
                'S3': _appt_times.get('_s3_end_time'),
            }
            # S1/S2/S3 节点描述
            _subgraph_meta = {
                'S1': ('专科问诊', '多轮医患对话，问诊+质量评估'),
                'S2': ('体格检查', '体征采集，生命体征+专科查体'),
                'S3': ('初步判断', '综合分析，决定是否需要辅助检查'),
            }
            for _nd in node_names:
                _lbl  = _node_zh.get(_nd, _nd)
                _desc = _node_desc.get(_nd, '')
                _mvs  = _move_by_node.get(_nd, [])
                # C6 是专科子图入口，展开为 S1/S2/S3 子事件
                if _nd == 'C6':
                    _sub_expanded = False
                    for _snd in ('S1', 'S2', 'S3'):
                        _st = _subgraph_times.get(_snd)
                        if _st:
                            _slbl, _sdesc = _subgraph_meta[_snd]
                            _timeline_events.append({
                                'time': _st,
                                'node': _snd, 'label': _slbl, 'desc': _sdesc,
                                'kind': 'stay', 'loc': _cur_loc,
                            })
                            _sub_expanded = True
                    if not _sub_expanded:
                        # 子图时钟未记录，回退到整个 C6 显示
                        _t_node = node_time_map.get(_nd, '--:--')
                        _timeline_events.append({
                            'time': _t_node,
                            'node': _nd, 'label': _lbl, 'desc': _desc,
                            'kind': 'stay', 'loc': _cur_loc,
                        })
                    continue
                if _mvs:
                    for _mv in _mvs:
                        _fr_n = _mv.get('from', _cur_loc)
                        _to_n = _mv.get('to', '?')
                        _cur_loc = _to_n
                        _timeline_events.append({
                            'time': _mv.get('time', '--:--'),
                            'node': _nd, 'label': _lbl, 'desc': _desc,
                            'kind': 'move', 'fr': _fr_n, 'to': _to_n,
                        })
                else:
                    _t_node = node_time_map.get(_nd, '--:--')
                    _timeline_events.append({
                        'time': _t_node,
                        'node': _nd, 'label': _lbl, 'desc': _desc,
                        'kind': 'stay', 'loc': _cur_loc,
                    })

            if _timeline_events:
                _hdr = f"  {'时间':^5}  {'累计':^7}  {'耗时':^5}  {'节点':<14}  {'位置 / 说明'}"
                _sep = f"  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*14}  {'─'*30}"
                self.detail_logger.info(_hdr)
                self.detail_logger.info(_sep)
                _abs_minutes = _normalize_event_minutes(_timeline_events)
                # 累计时间以时间线首个有效事件为基准，保证 C1 从 +0min 开始。
                # 说明：此前公式会把 HH:MM 的分钟值错误叠加，出现 +544min 之类异常。
                _base_abs: Optional[int] = next((x for x in _abs_minutes if x is not None), None)
                _last_abs: Optional[int] = None
                for _i, _ev in enumerate(_timeline_events):
                    _t_raw = _ev['time']
                    _nd  = _ev['node']
                    _tc_abs = _abs_minutes[_i] if _i < len(_abs_minutes) else None
                    if _tc_abs is not None:
                        _last_abs = _tc_abs
                    _t = _abs_min_to_hhmm(_tc_abs) if _tc_abs is not None else _t_raw
                    # 累计（相对就诊基准）
                    _elap_s = ''
                    if _base_abs is not None and _tc_abs is not None:
                        _e = max(0, _tc_abs - _base_abs)
                        _elap_s = f'+{_e}min'
                    # 耗时 = 本节点完成时刻 − 上一事件完成时刻（本节点自己占用的模拟时间）
                    _dur_s = ''
                    if _i > 0 and _tc_abs is not None:
                        for _j in range(_i - 1, -1, -1):
                            _tp_abs = _abs_minutes[_j] if _j < len(_abs_minutes) else None
                            if _tp_abs is not None:
                                _d = max(0, _tc_abs - _tp_abs)
                                _dur_s = f'{_d}min'
                                break
                    # 位置/说明文本
                    if _ev['kind'] == 'move':
                        _loc_s = f"{_ev['fr']} → {_ev['to']}"
                    else:
                        _loc_s = f"@ {_ev['loc']}  {_ev['desc']}"
                    _nd_lbl = f"[{_nd}]{_ev['label']}"
                    self.detail_logger.info(
                        f"  {_t:^5}  {_elap_s:^7}  {_dur_s:^5}  {_nd_lbl:<14}  {_loc_s}"
                    )
                self.detail_logger.info(f"  {'─'*65}")
                # 时间线总时长使用同口径（首末事件差值），保证与节点耗时列一致
                timeline_total: Optional[int] = None
                if _base_abs is not None and _last_abs is not None:
                    timeline_total = max(0, _last_abs - _base_abs)

                if timeline_total is not None:
                    self.detail_logger.info(f"  总就诊时长: {timeline_total} 分钟")
                    if simulated_minutes is not None:
                        _delta = int(round(simulated_minutes - timeline_total))
                        if _delta != 0:
                            self.detail_logger.info(
                                f"  口径差异说明: 患者总时长={simulated_minutes:.0f} 分钟，"
                                f"时间线统计={timeline_total} 分钟，差值={_delta} 分钟"
                            )
                elif simulated_minutes is not None:
                    self.detail_logger.info(f"  总就诊时长: {simulated_minutes:.0f} 分钟")
            else:
                self.detail_logger.info("  （无执行记录）")
            self.detail_logger.info("")

            return result
            
        except Exception as e:
            # 使用红色显示错误
            patient_tag = f"{Colors.BG_RED} P{self.case_id} {Colors.RESET}"
            self.logger.error(f"{Colors.RED}✗ {patient_tag} 失败: {str(e)[:50]}{Colors.RESET}")
            
            # 如果已分配医生，需要释放（改进：使用 finally 确保清理）
            return self._cleanup_and_return_error(str(e))
        finally:
            # 确保资源清理（即使在异常情况下）
            try:
                # 关闭患者详细日志记录器
                if hasattr(self, 'detail_logger') and self.detail_logger:
                    from logging_utils import close_patient_detail_logger
                    close_patient_detail_logger(self.patient_id)
                
                session = self.coordinator.get_patient(self.patient_id)
                if session and session.assigned_doctor:
                    doctor_id = session.assigned_doctor
                    # 检查医生是否仍在接诊该患者
                    doctor = self.coordinator.get_doctor(doctor_id)
                    if doctor and doctor.current_patient == self.patient_id:
                        self.coordinator.release_doctor(doctor_id)
                        # 资源清理日志移到详细日志中
                        if hasattr(self, 'detail_logger') and self.detail_logger:
                            self.detail_logger.info(f"清理资源：已释放医生 {doctor_id}")
            except Exception as cleanup_error:
                self.logger.error(f"⚠️ 资源清理失败: {cleanup_error}")
    
    def _cleanup_and_return_error(self, error_msg: str) -> Dict[str, Any]:
        """清理资源并返回错误结果"""
        # 安全地获取日志文件路径（detail_logger可能未创建）
        log_file = ""
        if hasattr(self, 'detail_logger') and self.detail_logger:
            try:
                log_file = self.detail_logger.get_log_file_path()
            except Exception:
                pass
        
        return {
            "status": "failed",
            "patient_id": self.patient_id,
            "case_id": self.case_id,
            "error": error_msg,
            "detail_log_file": log_file,  # 即使失败也返回日志路径
        }


class LangGraphMultiPatientProcessor:
    """LangGraph 多患者并发处理器"""
    
    def __init__(
        self,
        coordinator: HospitalCoordinator,
        retriever: AdaptiveRAGRetriever,
        llm: LLMClient,
        services: Any,
        medical_record_service: MedicalRecordService,
        max_questions: int = 3,
        max_workers: int = 10,
    ):
        """
        初始化处理器
        
        Args:
            coordinator: 医院协调器
            retriever: RAG 检索器
            llm: LLM 客户端
            services: 服务组件
            medical_record_service: 病例库服务
            max_questions: 最大问题数
            max_workers: 最大并发数
        """
        self.coordinator = coordinator
        self.retriever = retriever
        self.llm = llm
        self.services = services
        self.medical_record_service = medical_record_service
        self.max_questions = max_questions
        self.max_workers = max_workers
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        
        # ===== 创建共享的物理环境（核心改动）=====
        logger.info("🏥 初始化物理环境")
        self.shared_world = HospitalWorld(start_time=None)
        
        # 添加共享的医护人员到 world
        self.shared_world.add_agent("nurse_001", agent_type="nurse", initial_location="triage")
        self.shared_world.add_agent("lab_tech_001", agent_type="lab_technician", initial_location="lab")
        
        # 根据 coordinator 中注册的医生添加到 world
        logger.info(f"   → 注册医生: {len(self.coordinator.doctors)}名")
        for doctor_id, doctor in self.coordinator.doctors.items():
            dept_location = self._get_dept_location(doctor.dept)
            self.shared_world.add_agent(doctor_id, agent_type="doctor", initial_location=dept_location)
        
        # 初始化共享设备
        self._setup_shared_equipment()
        
        # 创建共享的 Nurse 和 Lab Agent（所有患者共用）
        self.shared_nurse_agent = NurseAgent(llm=self.llm)
        self.shared_lab_agent = LabAgent(llm=self.llm)
        
        # 为每个医生创建 DoctorAgent 实例（映射到 coordinator 的医生）
        self.doctor_agents: Dict[str, DoctorAgent] = {}
        for doctor_id, doctor in self.coordinator.doctors.items():
            self.doctor_agents[doctor_id] = DoctorAgent(
                dept=doctor.dept,
                retriever=self.retriever,
                llm=self.llm,
                max_questions=self.max_questions
            )
            
            # 【资源管理】注册医生到物理世界的资源池
            if self.shared_world:
                self.shared_world.register_doctor(doctor_id, doctor.dept)
        
        logger.info(f"✅ 处理器启动 (并发: {max_workers} | 医生: {len(self.coordinator.doctors)}名)")
        logger.info("")
    
    def _get_dept_location(self, dept: str) -> str:
        """获取科室对应的物理位置
        
        Args:
            dept: 科室代码
        
        Returns:
            位置ID
        """
        dept_location_map = {
            "neurology": "neuro",  # 神经医学使用神经内科诊室
        }
        return dept_location_map.get(dept, "neuro")
    
    def _setup_shared_equipment(self):
        """设置共享设备（可选，暂时简化实现）"""
        # 这里可以添加共享设备的初始化逻辑
        # 例如：限制检验设备数量、配置队列等
        pass  # 不显示初始化提示
    
    def submit_patient(
        self,
        patient_id: str,
        case_id: int,
        dept: str,
        priority: int = 5,
    ) -> str:
        """
        提交患者任务
        
        Args:
            patient_id: 患者ID
            case_id: 病例ID
            dept: 就诊科室
            priority: 优先级
        
        Returns:
            任务ID
        """
        # 先将患者添加到共享 world
        success = self.shared_world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
        if not success:
            logger.warning(f"⚠️  患者 {patient_id} 已在 world 中，跳过添加")
        
        # 创建执行器，传入共享 world 和共享 agents
        executor = LangGraphPatientExecutor(
            patient_id=patient_id,
            case_id=case_id,
            dept=dept,
            priority=priority,
            coordinator=self.coordinator,
            retriever=self.retriever,
            llm=self.llm,
            services=self.services,
            medical_record_service=self.medical_record_service,

            max_questions=self.max_questions,
            shared_world=self.shared_world,  # 传入共享 world
            shared_nurse_agent=self.shared_nurse_agent,  # 传入共享 nurse
            shared_lab_agent=self.shared_lab_agent,  # 传入共享 lab agent
            doctor_agents=self.doctor_agents,  # 传入医生 agents 字典
        )
        
        # 提交任务
        with self._lock:
            future = self.executor.submit(executor.execute)
            self.active_tasks[patient_id] = future
        
        # 不显示提交提示，避免冗余输出
        
        return patient_id
    
    
    def wait_all(self, timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """等待所有任务完成"""
        results = []
        
        with self._lock:
            futures = list(self.active_tasks.items())
        
        for patient_id, future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                logger.warning(f"任务超时: {patient_id}")
                results.append({"status": "timeout", "patient_id": patient_id})
            except Exception as e:
                logger.error(f"任务执行失败 ({patient_id}): {e}")
                results.append({"status": "error", "patient_id": patient_id, "error": str(e)})
        
        logger.info(f"✅ 所有任务完成: {len(results)} 个")
        
        return results
    
    def get_active_count(self) -> int:
        """获取活跃任务数"""
        with self._lock:
            return len([f for f in self.active_tasks.values() if not f.done()])
    
    def shutdown(self, wait: bool = True):
        """关闭处理器"""
        logger.info("关闭 LangGraph 多患者处理器...")
        self.executor.shutdown(wait=wait)
        logger.info("✅ 处理器已关闭")
