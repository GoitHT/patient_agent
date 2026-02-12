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
from loaders import load_diagnosis_arena_case
from logging_utils import create_patient_detail_logger, close_patient_detail_logger, get_patient_detail_logger
from rag import AdaptiveRAGRetriever
from services.llm_client import LLMClient
from services.medical_record import MedicalRecordService
from services.medical_record_integration import MedicalRecordIntegration
from state.schema import BaseState
from utils import get_logger, make_run_id

logger = get_logger("hospital_agent.langgraph_multi_patient")


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
    
    def _extract_patient_info_from_case(self, case_info: str, case_data: dict) -> dict:
        """
        从病例文本中提取患者基本信息（姓名、年龄、性别）
        
        Args:
            case_info: 病例文本信息
            case_data: 原始病例数据
        
        Returns:
            包含 name, age, gender 的字典
        """
        import re
        
        # 优先从 case_data 字段中获取
        name = case_data.get("name") or case_data.get("patient_name")
        age = case_data.get("age")
        gender = case_data.get("gender") or case_data.get("sex")
        
        # 如果 case_data 中没有，尝试从文本中解析
        if not name or not age or not gender:
            # 模式1: "患者，女性，45岁" 或 "患者，男，60岁"
            pattern1 = r'患者[，,]\s*([男女])[性]?[，,]\s*(\d+)岁'
            match1 = re.search(pattern1, case_info)
            if match1:
                if not gender:
                    gender = match1.group(1)
                if not age:
                    age = int(match1.group(2))
            
            # 模式2: "姓名：张三" "年龄：50" "性别：男"
            if not name:
                name_match = re.search(r'姓名[：:]\s*([^，,\s]+)', case_info)
                if name_match:
                    name = name_match.group(1)
            
            if not age:
                age_match = re.search(r'年龄[：:]\s*(\d+)', case_info)
                if age_match:
                    age = int(age_match.group(1))
            
            if not gender:
                gender_match = re.search(r'性别[：:]\s*([男女])', case_info)
                if gender_match:
                    gender = gender_match.group(1)
            
            # 模式3: "45岁女性" 或 "60岁男性患者"
            if not age or not gender:
                pattern3 = r'(\d+)岁([男女])性'
                match3 = re.search(pattern3, case_info)
                if match3:
                    if not age:
                        age = int(match3.group(1))
                    if not gender:
                        gender = match3.group(2)
        
        # 如果仍然没有提取到，使用合理的默认值
        if not name:
            name = f"患者{self.patient_id}"
        if not age or age == 0:
            # 尝试提取任何年龄数字（作为最后手段）
            age_match = re.search(r'(\d{1,3})岁', case_info)
            if age_match:
                extracted_age = int(age_match.group(1))
                # 合理性检查：年龄应该在 0-120 之间
                if 0 < extracted_age <= 120:
                    age = extracted_age
                else:
                    age = 0  # 不合理的年龄，保持为0
            else:
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
            
            # 终端显示简洁的开始信息
            patient_tag = f"{bg_color} P{self.case_id} {Colors.RESET}"
            
            self.logger.info(f"{fg_color}▶ {patient_tag} 就诊开始{Colors.RESET}")
            
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
            
            # 提取原始主诉
            case_info = known_case.get("Case Information", "")
            if "主诉：" in case_info:
                start_idx = case_info.find("主诉：") + 3
                remaining = case_info[start_idx:]
                end_markers = ["现病史：", "既往史：", "个人史：", "家族史：", "体格检查：", "\n\n"]
                end_idx = len(remaining)
                for marker in end_markers:
                    pos = remaining.find(marker)
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
                original_chief_complaint = remaining[:end_idx].strip()
            else:
                original_chief_complaint = case_info[:200].strip()
            
            # 详细日志中记录完整病例信息
            # 处理原始主诉的显示
            formatted_complaint = original_chief_complaint.replace('\\n', '\n    ')  # 将转义的换行符转为实际换行并缩进
            if len(formatted_complaint) > 300:
                formatted_complaint = formatted_complaint[:300] + "..."
            self.detail_logger.info(f"📋 原始主诉:\n    {formatted_complaint}")
            
            # 参考诊断
            if ground_truth.get('diagnosis'):
                self.detail_logger.info(f"\n🎯 参考诊断: {ground_truth['diagnosis']}")
            
            # 参考治疗方案 - 改进格式化
            if ground_truth.get('treatment_plan'):
                treatment_plan = ground_truth['treatment_plan']
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
            
            # 建议检查
            if ground_truth.get('recommended_tests'):
                self.detail_logger.info(f"\n🔬 建议检查: {', '.join(ground_truth['recommended_tests'])}")
            self.detail_logger.info("")
            
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
            extracted_info = self._extract_patient_info_from_case(case_info, state.case_data)
            patient_profile = {
                "name": extracted_info["name"],
                "age": extracted_info["age"],
                "gender": extracted_info["gender"],
                "case_id": self.case_id,
                "dataset_id": state.case_data.get("dataset_id"),
                "run_id": run_id,
            }
            
            # 更新state.patient_profile以包含提取的患者信息
            state.patient_profile.update({
                "name": extracted_info["name"],
                "age": extracted_info["age"],
                "gender": extracted_info["gender"],
            })
            
            # 获取已创建的病例（在 coordinator.register_patient 时已创建）
            existing_record = self.medical_record_service.get_record(self.patient_id)
            if existing_record:
                record_id = existing_record.record_id
                self.detail_logger.info(f"✅ 使用已创建的病例: {record_id}")
            else:
                # 容错：如果病例不存在（不应发生），则创建
                record_id = medical_record_integration.on_patient_entry(self.patient_id, patient_profile)
                self.detail_logger.warning(f"⚠️  病例不存在，已创建新病例: {record_id}")
            
            # 详细日志记录病例和患者信息（合并为一行，减少重复）
            self.detail_logger.info(f"\n👤 患者信息: {extracted_info['name']}, {extracted_info['age']}岁, {extracted_info['gender']} | 病例ID: {record_id}")
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
            
            self.detail_logger.section("护士分诊")
            # 记录分诊护士信息
            nurse_id = "nurse_001"
            nurse_name = "分诊护士"
            self.detail_logger.staff_info("分诊护士", nurse_id, nurse_name)
            world.move_agent(self.patient_id, "triage")
            
            patient_description = patient_agent.describe_to_nurse()
            
            
            # 调用分诊（使用LLM判断）
            triaged_dept = nurse_agent.triage(
                patient_description=patient_description
            )
            
            # 更新科室和 run_id
            state.dept = triaged_dept
            run_id = make_run_id(triaged_dept)
            state.run_id = run_id
            # 患者的详细描述保存到present_illness，chief_complaint留给医生总结
            state.history["present_illness"] = patient_description
            state.chief_complaint = ""  # 留空，等待医生总结
            
            triage_summary = nurse_agent.get_triage_summary()
            state.agent_interactions["nurse_triage"] = triage_summary
            
            # 从分诊历史中获取分诊理由
            triage_reason = ""
            if triage_summary.get("history"):
                latest_triage = triage_summary["history"][-1]
                triage_reason = latest_triage.get("reason", "")
                # 详细日志：记录分诊理由
                self.detail_logger.info(f"LLM分诊分析: {triage_reason}")
            
            if state.medical_record_integration:
                state.medical_record_integration.on_triage(
                    state, 
                    nurse_id="nurse_001",
                    nurse_name="分诊护士"
                )
            
            # 终端显示分诊结果
            dept_cn_names = {
                "neurology": "神经内科",
                "cardiology": "心内科",
                "gastroenterology": "消化内科",
                "respiratory": "呼吸内科",
                "endocrinology": "内分泌科"
            }
            dept_display = dept_cn_names.get(triaged_dept, triaged_dept)
            self.logger.info(f"{fg_color}├ {patient_tag} 分诊→{dept_display}{Colors.RESET}")
            
            # 详细日志记录分诊信息
            self.detail_logger.info("")
            self.detail_logger.info("📋 患者主诉:")
            self.detail_logger.info(f"    {patient_description}")
            self.detail_logger.info("")
            self.detail_logger.info("✅ 分诊结果:")
            dept_name_map = {
                'neurology': '神经内科',
                'cardiology': '心内科',
                'gastroenterology': '消化内科',
                'respiratory': '呼吸内科',
                'endocrinology': '内分泌科'
            }
            self.detail_logger.info(f"    科室代码: {triaged_dept}")
            self.detail_logger.info(f"    科室名称: {dept_name_map.get(triaged_dept, triaged_dept)}")
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
            
            # 创建患者专属的 PatientAgent
            patient_agent = PatientAgent(
                known_case=state.case_data,
                llm=self.llm,
                chief_complaint=original_chief_complaint
            )
            
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
            
            self.detail_logger.section("执行门诊流程")
            self.detail_logger.info("🔄 开始执行 LangGraph 工作流...")
            self.detail_logger.info("")
            
            node_count = 0
            node_names = []  # 记录节点名称
            out = None
            final_state = state  # 保存最终状态，初始为输入状态
            last_diagnosis_state = None  # 记录最近一次产生诊断的状态
            
            for chunk in graph.stream(state):
                node_count += 1
                if isinstance(chunk, dict) and len(chunk) > 0:
                    node_name = list(chunk.keys())[0]
                    node_names.append(node_name)
                    out = chunk[node_name]
                    
                    # 更新最终状态（接受BaseState或字典类型）
                    if isinstance(out, BaseState):
                        final_state = out
                        
                        # 跟踪最近有诊断的状态
                        if isinstance(out.diagnosis, dict) and out.diagnosis.get("name"):
                            last_diagnosis_state = out
                    elif isinstance(out, dict):
                        # 【修复】LangGraph可能返回字典而非Pydantic对象
                        # 尝试将字典转换为BaseState
                        try:
                            final_state = BaseState.model_validate(out)
                            
                            # 跟踪最近有诊断的状态
                            if isinstance(final_state.diagnosis, dict) and final_state.diagnosis.get("name"):
                                last_diagnosis_state = final_state
                        except Exception as e:
                            if node_name in ["C12", "C13", "C14", "C15", "C16"]:
                                self.detail_logger.warning(f"⚠️  [{node_name}] 从字典转换为BaseState失败: {e}")
                    
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
            
            # 获取患者就诊时间（如果有）
            simulated_minutes = None
            if final_state and hasattr(final_state, 'appointment'):
                simulated_minutes = final_state.appointment.get('simulated_duration_minutes')
            
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
            }
            
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
        self.shared_nurse_agent = NurseAgent(llm=self.llm, max_triage_questions=3)
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
    
    def submit_batch(self, patients: List[Dict[str, Any]]) -> List[str]:
        """
        批量提交患者任务
        
        Args:
            patients: 患者列表，每个元素包含:
                - patient_id: 患者ID
                - case_id: 病例ID
                - dept: 科室（可选，默认为 "internal_medicine"，会被护士分诊覆盖）
                - priority: 优先级（可选，默认为 5）
        
        Returns:
            任务ID列表
        """
        task_ids = []
        
        for patient_info in patients:
            patient_id = patient_info["patient_id"]
            case_id = patient_info["case_id"]
            dept = patient_info.get("dept", "internal_medicine")  # 默认科室，会被护士分诊覆盖
            priority = patient_info.get("priority", 5)
            
            task_id = self.submit_patient(patient_id, case_id, dept, priority)
            task_ids.append(task_id)
            
            # 稍微错开提交时间，避免资源竞争
            time.sleep(0.1)
        
        logger.info(f"✅ 批量提交完成: {len(task_ids)} 个患者")
        
        return task_ids
    
    def wait_for_patient(self, patient_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """等待单个患者任务完成"""
        with self._lock:
            future = self.active_tasks.get(patient_id)
        
        if not future:
            return {"status": "not_found", "patient_id": patient_id}
        
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            return {"status": "timeout", "patient_id": patient_id}
        except Exception as e:
            logger.error(f"任务执行失败 ({patient_id}): {e}")
            return {"status": "error", "patient_id": patient_id, "error": str(e)}
    
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
