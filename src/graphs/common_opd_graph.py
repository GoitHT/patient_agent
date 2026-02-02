from __future__ import annotations

"""
神经内科门诊流程图：

注意：C0（护士分诊+Agent初始化）已移至main.py的初始化阶段执行

- 通用前置：
  C1 开始 -> C2 挂号（预约挂号） -> C3 签到候诊 -> C4 叫号入诊室
- 专科中段：
  N4-N6（在 C6 Specialty Dispatch 调用）
- 通用后置：
  若 need_aux_tests=True：C8 开单并解释准备 -> C9 缴费与预约 -> C10 执行检查取报告 -> C11 回诊
  最终：C12 综合分析明确诊断/制定方案 -> C13 处置 -> C14 文书 -> C15 宣教随访 -> C16 结束
"""

import time
import json
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from graphs.log_helpers import _log_node_start, _log_node_end, _log_detail, _log_physical_state
from rag import ChromaRetriever
from services.appointment import AppointmentService
from services.billing import BillingService
from services.llm_client import LLMClient
from state.schema import BaseState, make_audit_entry
from logging_utils import should_log, get_output_level, OutputFilter, SUPPRESS_UNCHECKED_LOGS
from utils import (
    parse_json_with_retry,
    get_logger,
    load_prompt,
    apply_safety_rules,
    disclaimer_text,
    contains_any_positive,
)

# 初始化logger
logger = get_logger("hospital_agent.graph")

# 应用输出过滤器来抑制未被should_log包装的日志
if SUPPRESS_UNCHECKED_LOGS:
    logger.addFilter(OutputFilter("common_opd_graph"))



@dataclass(frozen=True)
class Services:
    """保留的必要服务：预约和计费系统"""
    appointment: AppointmentService
    billing: BillingService


def _default_channel() -> str:
    return "APP"  # 默认使用APP预约


def _chunks_for_prompt(chunks: list[dict[str, Any]], *, max_chars: int = 1600) -> str:
    lines: list[str] = []
    total = 0
    for c in chunks:
        text = str(c.get("text") or "").replace("\n", " ").strip()
        line = f"[{c.get('doc_id')}#{c.get('chunk_id')}] {text[:260]}"
        lines.append(line)
        total += len(line) + 1
        if total >= max_chars:
            break
    return "\n".join(lines)


class CommonOPDGraph:
    def __init__(
        self,
        *,
        retriever: ChromaRetriever,
        dept_subgraphs: dict[str, Any],
        services: Services,
        llm: LLMClient | None = None,
        llm_reports: bool = False,
        use_agents: bool = True,  # 总是使用三智能体模式
        patient_agent: Any | None = None,
        doctor_agent: Any | None = None,
        nurse_agent: Any | None = None,
        lab_agent: Any | None = None,
        max_questions: int = 3,  # 最底层默认值，通常从config传入
        world: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.dept_subgraphs = dept_subgraphs
        self.services = services
        self.llm = llm
        self.llm_reports = llm_reports
        self.use_agents = use_agents
        self.patient_agent = patient_agent
        self.doctor_agent = doctor_agent
        self.nurse_agent = nurse_agent
        self.lab_agent = lab_agent
        self.max_questions = max_questions
        self.world = world
    
    def _map_test_to_equipment_type(self, test_name: str, test_type: str) -> str:
        """
        映射检查项目名称到物理设备类型（神经内科专科配置）
        
        Args:
            test_name: 检查项目名称（如"头颅CT"、"血常规"）
            test_type: 检查类型（lab/imaging/exam等）
            
        Returns:
            设备类型字符串，对应 hospital_world.py 中的 exam_type
        """
        test_lower = test_name.lower()
        type_lower = test_type.lower()
        
        # ========== 影像检查设备 ==========
        if any(keyword in test_lower for keyword in ["头颅ct", "颅脑ct", "ct头", "head ct", "头部ct"]):
            return "ct_head"
        if any(keyword in test_lower for keyword in ["脑mri", "颅脑mri", "mri脑", "brain mri", "头部mri", "mri头"]):
            return "mri_brain"
        
        # ========== 神经电生理检查设备 ==========
        if any(keyword in test_lower for keyword in ["脑电图", "eeg", "脑电", "脑波"]):
            return "eeg"
        if any(keyword in test_lower for keyword in ["肌电图", "emg", "神经传导", "肌电"]):
            return "emg"
        if any(keyword in test_lower for keyword in ["tcd", "经颅多普勒", "脑血流", "颅内多普勒"]):
            return "tcd"
        
        # ========== 检验科检查设备（按检验项目分类）==========
        # 血常规
        if any(keyword in test_lower for keyword in ["血常规", "cbc", "血细胞", "血液常规", "全血细胞"]):
            return "cbc"
        
        # 基础生化（肝肾功能、血糖、血脂等）
        if any(keyword in test_lower for keyword in [
            "生化", "肝功", "肾功", "血糖", "血脂", "尿酸", "肌酐", "尿素氮", 
            "转氨酶", "胆红素", "白蛋白", "总蛋白", "甘油三酯", "胆固醇",
            "biochem", "liver", "kidney", "glucose", "lipid"
        ]):
            return "biochem_basic"
        
        # 电解质
        if any(keyword in test_lower for keyword in ["电解质", "钠", "钾", "氯", "钙", "镁", "electrolyte", "na+", "k+"]):
            return "electrolyte"
        
        # 凝血功能
        if any(keyword in test_lower for keyword in [
            "凝血", "pt", "aptt", "inr", "d-二聚体", "纤维蛋白", 
            "凝血酶原", "活化部分凝血活酶", "coagulation", "d-dimer"
        ]):
            return "coagulation"
        
        # 炎症/感染指标
        if any(keyword in test_lower for keyword in [
            "crp", "c反应蛋白", "降钙素原", "pct", "血沉", "esr", 
            "炎症", "感染", "inflammation", "infection"
        ]):
            return "inflammation"
        
        # 心肌与血管风险指标（卒中相关）
        if any(keyword in test_lower for keyword in [
            "心肌酶", "肌钙蛋白", "troponin", "bnp", "nt-probnp", 
            "同型半胱氨酸", "脂蛋白", "lp(a)", "homocysteine", 
            "心脑血管", "卒中标志", "cardiac", "stroke marker"
        ]):
            return "cardiac_stroke_markers"
        
        # 自身免疫抗体
        if any(keyword in test_lower for keyword in [
            "自免", "抗体", "自身免疫", "ana", "抗核抗体", "抗神经", 
            "抗磷脂", "autoimmune", "antibody", "抗nmda", "抗mog"
        ]):
            return "autoimmune_antibody"
        
        # ========== 默认映射（根据类型）==========
        if type_lower == "lab":
            # 默认检验项目使用基础生化设备（更通用，适合多种检验）
            # 注：皮肤科、微生物检验等特殊项目也会使用此设备
            logger.info(f"ℹ️  检查项目 '{test_name}' 使用通用检验设备 (biochem_basic)")
            return "biochem_basic"
        elif type_lower == "imaging":
            # 默认影像检查使用CT
            logger.info(f"ℹ️  影像检查 '{test_name}' 使用默认CT设备")
            return "ct_head"
        else:
            # 完全未知的情况，使用基础生化设备作为后备
            logger.warning(f"⚠️  未识别的检查项目 '{test_name}' (类型: {test_type})，默认使用通用检验设备 (biochem_basic)")
            return "biochem_basic"

    def build(self):
        graph = StateGraph(BaseState)

        def c1_start(state: BaseState) -> BaseState:
            """C1: 开始门诊流程 - 验证状态、记录开始时间、显示患者概览"""
            _log_node_start("C1", "开始", state)
            
            # 1. 验证必要的状态字段
            required_fields = {
                "dept": state.dept,
                "run_id": state.run_id,
                "chief_complaint": state.chief_complaint,
            }
            
            missing_fields = [k for k, v in required_fields.items() if not v]
            if missing_fields:
                logger.error(f"❌ 缺少必要字段: {', '.join(missing_fields)}")
                raise ValueError(f"State validation failed: missing {missing_fields}")
            
            # 2. 记录流程开始时间（使用物理世界时间，保证一致性）
            import datetime
            if self.world:
                # 使用物理世界的时间系统
                start_timestamp = self.world.current_time.isoformat()
            else:
                # 如果没有启用物理世界，使用系统时间
                start_timestamp = datetime.datetime.now().isoformat()
            state.appointment["visit_start_time"] = start_timestamp
            
            # 3. 科室显示名称映射（与C4节点保持一致）
            dept_display_names = {
                "neurology": "神经医学科",
            }
            dept_display_name = dept_display_names.get(state.dept, state.dept)
            
            # 4. 详细日志记录患者信息
            _log_detail(f"就诊科室: {dept_display_name}", state, 2, "C1")
            # 显示针对医生的主诉（医学专业描述），而不是患者对护士说的口语化版本
            _log_detail(f"主诉: {state.original_chief_complaint}", state, 2, "C1")
            
            # 5. 显示完整物理世界信息（如果启用）
            _log_physical_state(state, self.world, "C1", level=2)
            
            # 6. 初始化流程追踪
            if "nurse_triage" in state.agent_interactions:
                triage_info = state.agent_interactions["nurse_triage"]
                triaged_dept_code = triage_info.get('triaged_dept', 'N/A')
                # 将分诊科室代码映射为中文显示名称
                triaged_dept_display = dept_display_names.get(triaged_dept_code, triaged_dept_code) if triaged_dept_code != 'N/A' else 'N/A'
                logger.info(f"  💉 分诊结果: {triaged_dept_display}")
                if triage_info.get("reasoning"):
                    logger.info(f"     理由: {triage_info['reasoning'][:60]}...")
            
            # 7. 设置流程状态标记
            state.appointment["status"] = "visit_started"
            state.appointment["current_stage"] = "C1_start"
            
            state.add_audit(
                make_audit_entry(
                    node_name="C1 Start Visit",
                    inputs_summary={
                        "dept": state.dept,
                        "dept_display_name": dept_display_name,
                        "chief_complaint": state.chief_complaint[:40],
                        "triage_completed": "nurse_triage" in state.agent_interactions,
                        "physical_world_enabled": bool(self.world and state.patient_id),
                    },
                    outputs_summary={
                        "run_id": state.run_id,
                        "start_time": start_timestamp,
                        "status": "visit_started",
                        "current_location": state.current_location if self.world else "N/A",
                    },
                    decision="验证状态完整性，记录流程开始，初始化就诊追踪，同步物理世界状态",
                    chunks=[],
                    flags=["VISIT_START"],
                )
            )
            logger.info("  ✅ C1完成")
            return state

        def c2_registration(state: BaseState) -> BaseState:
            if should_log(1, "common_opd_graph", "C2"):
                logger.info("📝 C2: 预约挂号")
            
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            if detail_logger:
                detail_logger.subsection("C2: 预约挂号")
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C2", level=2)
            
            channel = state.appointment.get("channel") or _default_channel()
            timeslot = state.appointment.get("timeslot") or "上午"
            if detail_logger:
                detail_logger.info(f"预约渠道: {channel}")
                detail_logger.info(f"时间段: {timeslot}")
            
            appt = self.services.appointment.create_appointment(
                channel=channel, dept=state.dept, timeslot=timeslot
            )
            state.appointment = appt
            
            # 推进时间（挂号约需3分钟）
            if self.world:
                self.world.advance_time(minutes=3)
                state.sync_physical_state()
            
            if detail_logger:
                detail_logger.info(f"挂号成功 - 预约ID: {appt.get('appointment_id')}")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C2 Registration",
                    inputs_summary={"channel": channel, "timeslot": timeslot},
                    outputs_summary={"appointment_id": appt.get("appointment_id")},
                    decision="完成预约挂号",
                    chunks=[],
                )
            )
            if should_log(1, "common_opd_graph", "C2"):
                logger.info("  ✅ C2完成")
            return state

        def c3_checkin_waiting(state: BaseState) -> BaseState:
            if should_log(1, "common_opd_graph", "C3"):
                logger.info("✍️ C3: 签到候诊")
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            if detail_logger:
                detail_logger.subsection("C3: 签到候诊")
            
            # 物理环境：移动到候诊区
            if self.world and state.patient_id:
                # 移动到候诊区
                success, msg = self.world.move_agent(state.patient_id, "waiting_area")
                if success:
                    _log_detail(f"  🚶 移动: 门诊大厅 → 候诊区", state, 2, "C3")
                    state.current_location = "waiting_area"
                    state.sync_physical_state()
                    self.world.advance_time(minutes=2)
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C3", level=2)
            
            state.appointment = self.services.appointment.checkin(state.appointment)
            
            if should_log(1, "common_opd_graph", "C3"):
                logger.info(f"✅ 签到成功 - 状态: {state.appointment.get('status')}")
            
            # 候诊等待（5-10分钟）
            if self.world and state.patient_id:
                wait_time = 7  # 固定等待7分钟
                success, msg = self.world.wait(state.patient_id, wait_time)
                if success:
                    logger.info(f"  ⏳ 候诊等待: {wait_time}分钟")
                    state.sync_physical_state()
                    logger.info(f"  🕐 当前时间: {self.world.current_time.strftime('%H:%M')}")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C3 Checkin & Waiting",
                    inputs_summary={"appointment_id": state.appointment.get("appointment_id")},
                    outputs_summary={"status": state.appointment.get("status")},
                    decision="完成签到并进入候诊",
                    chunks=[],
                )
            )
            if should_log(1, "common_opd_graph", "C3"):
                logger.info("✅ C3节点完成\n")
            return state

        def c4_call_in(state: BaseState) -> BaseState:
            """C4: 叫号进诊 - 叫号患者并分配医生"""
            _log_node_start("C4", "叫号进诊", state)
            
            state.appointment = self.services.appointment.call_patient(state.appointment)
            
            _log_detail(f"✅ 叫号成功 - 状态: {state.appointment.get('status')}", state, 2, "C4")
            
            # 【物理环境】将患者从候诊区移动到对应科室诊室
            if self.world and state.patient_id:
                # 科室到诊室位置的映射
                dept_location_map = {
                    "neurology": "neuro",
                }
                
                # 科室中文名称映射
                dept_display_names = {
                    "neurology": "神经医学诊室",
                }
                
                # 获取目标诊室位置和显示名称
                target_clinic = dept_location_map.get(state.dept, "neuro")
                dept_display_name = dept_display_names.get(state.dept, "神经医学诊室")
                
                # 在state中存储科室显示名称，供后续节点使用
                state.dept_display_name = dept_display_name
                
                # 移动患者到诊室
                success, msg = self.world.move_agent(state.patient_id, target_clinic)
                if success:
                    # 使用科室的真实名称而不是物理位置的名称
                    _log_detail(f"🚶 已从候诊区移动到{dept_display_name}", state, 2, "C4")
                    
                    # 更新状态中的位置信息
                    state.current_location = target_clinic
                    state.sync_physical_state()
                    
                    # 推进时间（叫号和入诊大约2分钟）
                    self.world.advance_time(minutes=2)
                else:
                    _log_detail(f"⚠️  患者移动失败: {msg}", state, 2, "C4")
                
                # 【资源竞争】分配医生
                if hasattr(state, 'assigned_doctor_id') and state.assigned_doctor_id:
                    doctor_id, wait_time = self.world.assign_doctor(
                        patient_id=state.patient_id,
                        dept=state.dept,
                        priority=3 if "emergency" in state.escalations else 5
                    )
                    
                    if doctor_id:
                        state.assigned_doctor_id = doctor_id
                        if wait_time > 0:
                            _log_detail(f"⏳ 医生忙碌，预计等待{wait_time}分钟", state, 1, "C4")
                            _log_detail(f"   患者已加入队列，等待医生 {doctor_id} 完成当前就诊", state, 2, "C4")
                            # 真实等待
                            success, msg = self.world.wait(state.patient_id, wait_time)
                            if success:
                                state.sync_physical_state()
                                _log_detail(f"✅ 等待完成，开始就诊", state, 2, "C4")
                        else:
                            _log_detail(f"✅ 医生空闲，立即分配: {doctor_id}", state, 1, "C4")
                    else:
                        _log_detail(f"⚠️  暂无可用医生，加入候诊队列", state, 1, "C4")
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C4", level=2)
            
            state.add_audit(
                make_audit_entry(
                    node_name="C4 Call In",
                    inputs_summary={"appointment_id": state.appointment.get("appointment_id")},
                    outputs_summary={"status": state.appointment.get("status")},
                    decision="叫号进入诊室",
                    chunks=[],
                )
            )
            _log_node_end("C4", state)
            return state

        def c5_prepare_intake(state: BaseState) -> BaseState:
            """C5: 问诊准备 - 检索通用SOP并初始化问诊记录（实际问诊在C6专科子图中进行）"""
            _log_node_start("C5", "问诊准备", state)
            
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C5", level=2)
            
            _log_detail("🔍 检索医院通用SOP与免责声明...", state, 2, "C5")
            chunks = self.retriever.retrieve(
                f"门诊 问诊要点 分流 免责声明 {state.chief_complaint}",
                filters={"dept": "hospital", "type": "sop"},
                k=4,
            )
            _log_detail(f"  ✅ 检索到 {len(chunks)} 个知识片段", state, 2, "C5")
            state.add_retrieved_chunks(chunks)

            # 初始化问诊对话记录（实际问诊在C6专科子图中进行）
            _log_detail("\n💬 注：详细问诊将在C6专科子图中进行", state, 2, "C5")
            state.agent_interactions["doctor_patient_qa"] = []
            
            state.add_audit(
                make_audit_entry(
                    node_name="C5 Prepare Intake",
                    inputs_summary={"chief_complaint": state.chief_complaint[:40]},
                    outputs_summary={"sop_chunks": len(chunks)},
                    decision="检索医院通用SOP/免责声明，初始化问诊记录（实际问诊在C6专科子图执行）",
                    chunks=chunks,
                    flags=["AGENT_MODE"],
                )
            )
            _log_node_end("C5", state)
            return state

        def c6_specialty_dispatch(state: BaseState) -> BaseState:
            if should_log(1, "common_opd_graph", "C6"):
                logger.info("\n" + "="*60)
                logger.info(f"🏭 C6: 专科流程调度 ({state.dept})")
                logger.info("="*60)
            
            sub = self.dept_subgraphs.get(state.dept)
            if sub is None:
                raise ValueError(f"Unknown dept: {state.dept}")
            
            if should_log(1, "common_opd_graph", "C6"):
                logger.info(f"🔀 调用 {state.dept} 子图...")
            out = sub.invoke(state)
            state = BaseState.model_validate(out)
            
            if should_log(1, "common_opd_graph", "C6"):
                logger.info(f"✅ 专科流程完成 - 需要辅助检查: {state.need_aux_tests}")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C6 Specialty Dispatch",
                    inputs_summary={"dept": state.dept},
                    outputs_summary={"need_aux_tests": state.need_aux_tests},
                    decision="执行专科子图并回填专科结构化结果",
                    chunks=[],
                )
            )
            if should_log(1, "common_opd_graph", "C6"):
                logger.info("✅ C6节点完成\n")
            return state

        def c7_decide_path(state: BaseState) -> BaseState:
            """C7: 路径决策 - 根据need_aux_tests标志决定是否进入辅助检查流程
            注：此节点目前仅做简单判断，未来可扩展为更复杂的决策逻辑（如急诊分流、转诊判断等）
            """
            _log_node_start("C7", "路径决策", state)
            
            _log_detail(f"❓ 需要辅助检查: {state.need_aux_tests}", state, 1, "C7")
            if state.need_aux_tests:
                _log_detail(f"📝 待开单项目数: {len(state.ordered_tests)}", state, 2, "C7")
                for test in state.ordered_tests:
                    _log_detail(f"  - {test.get('name', 'N/A')} ({test.get('type', 'N/A')})", state, 2, "C7")
            else:
                _log_detail("✅ 无需辅助检查，直接进入诊断", state, 1, "C7")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C7 Decide Path",
                    inputs_summary={"need_aux_tests": state.need_aux_tests},
                    outputs_summary={"ordered_tests_count": len(state.ordered_tests)},
                    decision="根据need_aux_tests标志选择后续路径（with_tests或no_tests）",
                    chunks=[],
                )
            )
            _log_node_end("C7", state)
            return state

        def c8_order_explain_tests(state: BaseState) -> BaseState:
            """
            C8: 开单与检查准备说明
            职责：
            1. 检索医院缴费/预约流程SOP
            2. 检索专科检查准备知识（禁忌、注意事项、准备步骤）
            3. 生成完整的检查准备说明（不包含具体预约信息）
            """
            _log_node_start("C8", "开单与准备说明", state)
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C8", level=2)
            
            # 检索医院通用流程SOP
            _log_detail("🔍 检索医院通用流程...", state, 1, "C8")
            hospital_chunks = self.retriever.retrieve(
                "缴费 预约 报告领取 回诊 流程",
                filters={"dept": "hospital", "type": "sop"},
                k=4,
            )
            state.add_retrieved_chunks(hospital_chunks)
            _log_detail(f"  ✅ 检索到 {len(hospital_chunks)} 个通用流程SOP", state, 1, "C8")

            dept_chunks: list[dict[str, Any]] = []
            prep_items: list[dict[str, Any]] = []
            
            # 为每个检查项目检索准备知识
            _log_detail(f"\n📋 检索 {len(state.ordered_tests)} 个检查项目的准备知识...", state, 1, "C8")
            for t in state.ordered_tests:
                test_name = t.get('name', '')
                test_type = t.get('type', 'unknown')
                
                _log_detail(f"  🔍 {test_name} ({test_type})", state, 1, "C8")
                
                # 检索专科检查准备知识
                q = f"{state.dept} {test_name} 准备 禁忌 注意事项 禁食"
                cs = self.retriever.retrieve(q, filters={"dept": state.dept}, k=4)
                dept_chunks.extend(cs)
                state.add_retrieved_chunks(cs)
                _log_detail(f"     ✅ 检索到 {len(cs)} 个准备知识片段", state, 1, "C8")

                # 生成准备说明（不包含预约调度信息）
                prep_item = {
                    "test_name": test_name,
                    "test_type": test_type,
                    "need_schedule": bool(t.get("need_schedule", False)),
                    "need_prep": bool(t.get("need_prep", False)),
                    "body_part": t.get("body_part", []),
                    "prep_notes": [
                        "按下方宣教于SOP完成检查准备",
                        "如有基础病史、药物过敏、长期用药请提前告知区域",
                        "检查当天请携带身份证和缴费凭证",
                    ],
                    "contraindications": ["存在特殊禁忌症时请咨询医生进行评估"],
                    "reference_chunks": len(cs),  # 记录引用的知识片段数
                }
                
                prep_items.append(prep_item)

            state.test_prep = prep_items
            _log_detail(f"\n✅ 开单与准备说明生成完成，共 {len(prep_items)} 项检查", state, 1, "C8")

            all_chunks = hospital_chunks + dept_chunks
            state.add_audit(
                make_audit_entry(
                    node_name="C8 Order & Explain Tests",
                    inputs_summary={"ordered_tests": [t.get("name") for t in state.ordered_tests]},
                    outputs_summary={
                        "test_prep_count": len(prep_items),
                        "knowledge_chunks": len(all_chunks),
                        "need_schedule_count": sum(1 for p in prep_items if p.get("need_schedule")),
                    },
                    decision="开单并检索准备知识（通用SOP+专科准备说明），不包含预约调度",
                    chunks=all_chunks,
                )
            )
            _log_node_end("C8", state)
            return state

        def c9_billing_scheduling(state: BaseState) -> BaseState:
            """
            C9: 缴费与预约调度
            职责：
            1. 生成订单并完成缴费
            2. 调度检查项目预约时间
            3. 生成检查准备清单（checklist）
            """
            logger.info("\n" + "="*60)
            logger.info("💳 C9: 缴费与预约")
            logger.info("="*60)
            
            # 物理环境：移动到收费处
            if self.world and state.patient_id:
                # 移动到收费处
                success, msg = self.world.move_agent(state.patient_id, "cashier")
                if success:
                    _log_detail(f"  🚶 移动: 诊室 → 收费处", state, 2, "C9")
                    state.current_location = "cashier"
                    state.sync_physical_state()
                    self.world.advance_time(minutes=2)
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C9", level=2)
            
            # 1. 生成订单并缴费
            order_id = f"ORD-{state.run_id}-{len(state.ordered_tests)}"
            logger.info(f"📝 订单ID: {order_id}")
            
            payment = self.services.billing.pay(order_id=order_id)
            logger.info(f"✅ 缴费完成 - 金额: {payment.get('amount', 0)}元")
            state.appointment["billing"] = payment
            
            # 缴费等待（3-5分钟）
            if self.world and state.patient_id:
                wait_time = 4  # 固定等待4分钟
                success, msg = self.world.wait(state.patient_id, wait_time)
                if success:
                    logger.info(f"  ⏳ 缴费等待: {wait_time}分钟")
                    state.sync_physical_state()
                    logger.info(f"  🕐 当前时间: {self.world.current_time.strftime('%H:%M')}")
                logger.info("")

            # 2. 预约调度与准备清单生成
            logger.info("\n📅 调度检查预约...")
            
            # 验证test_prep和ordered_tests长度一致
            if len(state.test_prep) != len(state.ordered_tests):
                logger.error(f"⚠️  数据不一致: test_prep({len(state.test_prep)}) != ordered_tests({len(state.ordered_tests)})")
                raise ValueError("test_prep和ordered_tests长度不匹配")
            
            scheduled_count = 0
            for prep, t in zip(state.test_prep, state.ordered_tests, strict=False):
                test_name = t.get("name")
                test_type = t.get("type")
                
                # 处理需要预约的检查
                if t.get("need_schedule"):
                    logger.info(f"  🕒 预约: {test_name}")
                    
                    if test_type == "endoscopy":
                        # 内镜检查：生成预约信息（24小时后）
                        from datetime import timedelta
                        scheduled_time = self.world.current_time + timedelta(hours=24)
                        prep["schedule"] = {
                            "procedure": test_name,
                            "scheduled": True,
                            "schedule_id": f"END-{int(time.time() * 1000) % 100000}",
                            "scheduled_at": scheduled_time.strftime("%Y-%m-%d %H:%M"),
                            "location": "内镜中心",
                        }
                        # 根据检查类型生成准备清单
                        if "结肠" in test_name or "肠镜" in test_name:
                            prep["prep_checklist"] = [
                                {"item": "检查前3天低渣饮食", "required": True},
                                {"item": "检查前1天清流质饮食", "required": True},
                                {"item": "按医嘱服用肠道清洁剂", "required": True},
                                {"item": "抗凝/抗血小板药物需提前评估", "required": True},
                            ]
                        else:
                            prep["prep_checklist"] = [
                                {"item": "检查前6-8小时禁食禁饮", "required": True},
                                {"item": "如需镇静需家属陪同", "required": True},
                            ]
                        logger.info(f"     ✅ 预约时间: {scheduled_time.strftime('%H:%M')} (明天)")
                    elif test_type == "imaging":
                        # 影像检查：通用预约（2小时后）
                        from datetime import timedelta
                        scheduled_time = self.world.current_time + timedelta(hours=2)
                        prep["schedule"] = {
                            "scheduled": True,
                            "procedure": test_name,
                            "scheduled_at": scheduled_time.strftime("%Y-%m-%d %H:%M"),
                            "location": "影像科",
                        }
                        logger.info(f"     ✅ 预约时间: {scheduled_time.strftime('%H:%M')} (约2小时)")
                    elif test_type == "neurophysiology":
                        # 神经电生理检查（24小时后）
                        from datetime import timedelta
                        scheduled_time = self.world.current_time + timedelta(hours=24)
                        prep["schedule"] = {
                            "scheduled": True,
                            "procedure": test_name,
                            "scheduled_at": scheduled_time.strftime("%Y-%m-%d %H:%M"),
                            "location": "神经电生理室",
                        }
                        logger.info(f"     ✅ 预约时间: {scheduled_time.strftime('%H:%M')} (明天)")
                    else:
                        # 其他检查（2小时后）
                        from datetime import timedelta
                        scheduled_time = self.world.current_time + timedelta(hours=2)
                        prep["schedule"] = {
                            "scheduled": True,
                            "procedure": test_name,
                            "scheduled_at": scheduled_time.strftime("%Y-%m-%d %H:%M"),
                        }
                        logger.info(f"     ✅ 预约完成: {scheduled_time.strftime('%H:%M')} (约2小时)")
                    
                    scheduled_count += 1
                else:
                    # 不需要预约（如普通检验）
                    prep["schedule"] = {
                        "scheduled": False,
                        "immediate": True,
                        "location": "检验科" if test_type == "lab" else "相关科室",
                    }
                
                # 生成准备清单（如果需要且还没有）
                if t.get("need_prep") and "prep_checklist" not in prep:
                    prep["prep_checklist"] = [
                        {"item": "按医生建议完成检查准备", "required": True},
                        {"item": "检查前阅读注意事项", "required": True},
                    ]
            
            logger.info(f"\n✅ 预约调度完成：{scheduled_count}/{len(state.ordered_tests)} 项需要预约")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C9 Billing & Scheduling",
                    inputs_summary={
                        "order_id": order_id,
                        "tests_to_schedule": sum(1 for t in state.ordered_tests if t.get("need_schedule")),
                    },
                    outputs_summary={
                        "paid": payment.get("paid"),
                        "amount": payment.get("amount"),
                        "scheduled_count": scheduled_count,
                        "total_tests": len(state.ordered_tests),
                    },
                    decision="完成缴费与检查项目预约调度，生成准备清单",
                    chunks=[],
                )
            )
            logger.info("✅ C9节点完成\n")
            return state

        def c10a_fetch_test_results(state: BaseState) -> BaseState:
            """C10a: 获取检查结果数据（从数据集或LLM生成）"""
            logger.info("\n" + "="*60)
            logger.info("🧪 C10a: 获取检查结果")
            logger.info("="*60)
            
            # 物理环境：模拟检查过程（移动到检验科/影像科等）
            if self.world and state.patient_id:
                # 移动到检验科（简化处理，实际可能有多个地点）
                success, msg = self.world.move_agent(state.patient_id, "lab")
                if success:
                    _log_detail(f"  🚶 移动: 收费处 → 检验科", state, 2, "C10a")
                    state.current_location = "lab"
                    state.sync_physical_state()
                    self.world.advance_time(minutes=2)
                
                # 【资源竞争】为每个检查项目请求设备
                _log_detail(f"\n🏥 开始{len(state.ordered_tests)}项检查的设备分配...", state, 2, "C10a")
                total_wait_time = 0
                
                # 按设备类型去重：同一类型设备只分配一次
                allocated_exam_types = set()  # 已分配的设备类型
                
                for test in state.ordered_tests:
                    test_name = test.get("test_name", test.get("name", ""))
                    test_type = test.get("test_type", test.get("type", "lab"))
                    
                    # 映射检查类型到设备类型（神经内科专科配置）
                    exam_type = self._map_test_to_equipment_type(test_name, test_type)
                    
                    # 如果该设备类型已分配，跳过（避免重复分配）
                    if exam_type in allocated_exam_types:
                        _log_detail(f"  ♻️  [{test_name}] 使用已分配的{exam_type}设备（批量检测）", state, 2, "C10a")
                        continue
                    
                    # 请求设备
                    equipment_id, wait_time = self.world.request_equipment(
                        patient_id=state.patient_id,
                        exam_type=exam_type,
                        priority=3 if "emergency" in state.escalations else 5
                    )
                    
                    if equipment_id:
                        allocated_exam_types.add(exam_type)  # 记录已分配的设备类型
                        if wait_time > 0:
                            _log_detail(f"  ⏳ [{test_name}] 设备{equipment_id}忙碌，排队等待{wait_time}分钟", state, 2, "C10a")
                            total_wait_time += wait_time
                        else:
                            _log_detail(f"  ✅ [{test_name}] 已分配设备{equipment_id}", state, 2, "C10a")
                    else:
                        _log_detail(f"  ⚠️  [{test_name}] 暂无可用{exam_type}设备", state, 2, "C10a")
                
                # 真实等待（包括排队和检查时间）
                if total_wait_time > 0:
                    _log_detail(f"  ⏳ 总等待时间: {total_wait_time}分钟", state, 2, "C10a")
                    success, msg = self.world.wait(state.patient_id, total_wait_time)
                    if success:
                        state.sync_physical_state()
                
                # 检查完成后释放设备
                for test in state.ordered_tests:
                    # 找到对应的设备并释放
                    for eq_id, eq in self.world.equipment.items():
                        if eq.current_patient == state.patient_id:
                            self.world.release_equipment(eq_id)
                
                _log_detail(f"  ✅ 所有检查完成", state, 2, "C10a")
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C10a", level=2)
            
            # 🔬 使用检验科Agent智能生成检查结果
            logger.info("\n🔬 检验科Agent执行检查并生成结果...")
            
            # 获取数据集中的真实检查结果作为参考（如果有）
            real_diagnostic_tests = state.ground_truth.get("Diagnostic Tests", "").strip()
            
            # 准备检验科Agent需要的上下文信息
            lab_context = {
                "ordered_tests": state.ordered_tests,  # 医生开具的检查项目
                "chief_complaint": state.chief_complaint,  # 患者主诉
                "case_info": state.patient_profile.get("case_text", ""),  # 完整病例信息
                "real_tests_reference": real_diagnostic_tests if real_diagnostic_tests else None,  # 真实结果作为参考
                "dept": state.dept,  # 就诊科室
                "patient_id": state.patient_id,
            }
            
            # 显示检验科接收的检查项目
            logger.info(f"📋 接收医生开具的检查单: {len(state.ordered_tests)} 项")
            for idx, test in enumerate(state.ordered_tests, 1):
                logger.info(f"  [{idx}] {test.get('name')} ({test.get('type')})")
            
            if real_diagnostic_tests:
                logger.info(f"\n📚 参考数据集真实结果: {len(real_diagnostic_tests)} 字符")
                logger.info(f"  预览: {real_diagnostic_tests[:100]}...")
            
            # 调用检验科Agent生成检查结果
            results: list[dict[str, Any]] = []
            used_fallback = False
            
            if self.lab_agent:
                logger.info("\n🤖 检验科Agent分析并生成检查结果...")
                try:
                    # 使用lab_agent生成结果
                    lab_results = self.lab_agent.generate_test_results(lab_context)
                    
                    if lab_results and isinstance(lab_results, list):
                        results = lab_results
                        logger.info(f"  ✅ 检验科Agent成功生成 {len(results)} 项检查结果")
                        
                        # 统计异常结果
                        abnormal_count = sum(1 for r in results if r.get("abnormal"))
                        logger.info(f"  ⚠️  异常结果: {abnormal_count}/{len(results)}")
                        
                        # 显示每项结果概要
                        for idx, r in enumerate(results, 1):
                            test_name = r.get("test_name", "未知")
                            abnormal = r.get("abnormal", False)
                            status = "⚠️ 异常" if abnormal else "✓ 正常"
                            logger.info(f"  [{idx}] {test_name}: {status}")
                        
                        # 标记数据来源
                        for r in results:
                            r["source"] = "lab_agent"
                            if real_diagnostic_tests:
                                r["reference_data"] = "dataset"
                    else:
                        logger.warning("  ⚠️  检验科Agent返回结果格式错误")
                        used_fallback = True
                        
                except Exception as e:
                    logger.error(f"  ❌ 检验科Agent生成失败: {e}")
                    used_fallback = True
            else:
                logger.warning("  ⚠️  检验科Agent未初始化")
                used_fallback = True
            
            # 如果lab_agent失败，使用备用方案（简化版生成）
            if used_fallback or not results:
                logger.warning("\n⚠️  使用备用方案生成检查结果...")
                
                # 为每项检查生成基本结果结构
                results = []
                for t in state.ordered_tests:
                    result = {
                        "test_name": t.get("name"),
                        "test": t.get("name"),
                        "type": t.get("type"),
                        "body_part": t.get("body_part", ["未知"]),
                        "summary": "检查已完成，详见报告",
                        "abnormal": False,  # 默认正常
                        "detail": f"{t.get('name')}检查已完成，结果正常范围内。",
                        "source": "fallback_simple",
                        "reference_data": "dataset" if real_diagnostic_tests else None,
                    }
                    results.append(result)
                
                logger.info(f"  ✅ 备用方案生成 {len(results)} 项基础结果")
            
            # 保存原始检查结果（未增强）
            state.test_results = results
            state.appointment["reports_ready"] = bool(results)
            
            # 【病例库】记录检验结果
            if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
                state.medical_record_integration.on_lab_test_completed(state, lab_tech_id="lab_tech_001")
                logger.info("  📋 检验结果已记录到病例库")
            
            # 安全获取data_source（防止索引错误）
            data_source = results[0].get("source") if results else "none"
            has_reference = bool(real_diagnostic_tests)
            
            logger.info(f"\n✅ 检查结果生成完成")
            logger.info(f"  数据来源: {data_source}")
            logger.info(f"  参考数据: {'有（数据集）' if has_reference else '无'}")
            logger.info(f"  结果数量: {len(results)} 项")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C10a Fetch Test Results",
                    inputs_summary={
                        "ordered_tests_count": len(state.ordered_tests),
                        "has_reference_data": has_reference,
                        "patient_complaint": state.chief_complaint[:40],
                        "dept": state.dept,
                    },
                    outputs_summary={
                        "results_count": len(results), 
                        "abnormal_count": sum(1 for r in results if r.get("abnormal")),
                        "data_source": data_source,
                        "lab_agent_used": data_source == "lab_agent",
                    },
                    decision="检验科Agent根据医生开具的检查项目、患者主诉和病例信息智能生成检查结果",
                    chunks=[],
                    flags=["LAB_AGENT"] if data_source == "lab_agent" else (["FALLBACK"] if used_fallback else ["GENERATED"]),
                )
            )
            
            logger.info("✅ C10a节点完成\n")
            return state

        def c10b_enhance_reports(state: BaseState) -> BaseState:
            """C10b: 增强检查报告（生成叙述和解读）"""
            logger.info("\n" + "="*60)
            logger.info("📝 C10b: 增强检查报告")
            logger.info("="*60)
            
            results = state.test_results
            
            # 检查是否需要增强报告
            if not results:
                logger.info("⚠️  无检查结果，跳过报告增强")
                state.add_audit(
                    make_audit_entry(
                        node_name="C10b Enhance Reports",
                        inputs_summary={"results_count": 0},
                        outputs_summary={"enhanced": False},
                        decision="无检查结果，跳过增强",
                        chunks=[],
                        flags=["SKIPPED"]
                    )
                )
                logger.info("✅ C10b节点完成\n")
                return state
            
            system_prompt = load_prompt("common_system.txt")
            enhanced_count = 0
            failed_count = 0
            
            # 为每个结果生成个性化叙述
            for idx, result in enumerate(results):
                test_name = result.get("test_name") or result.get("test", "未知检查")
                body_part = result.get("body_part", ["相关部位"])
                abnormal = result.get("abnormal", False)
                summary = result.get("summary", "")
                detail = result.get("detail", "")
                
                # 构建增强提示词
                user_prompt = (
                    f"请为以下检查结果生成1-2句专业、清晰的医学报告叙述。\n\n"
                    f"【检查信息】\n"
                    f"- 检查名称：{test_name}\n"
                    f"- 检查部位：{', '.join(body_part) if isinstance(body_part, list) else body_part}\n"
                    f"- 是否异常：{'是' if abnormal else '否'}\n"
                    f"- 结果摘要：{summary}\n"
                )
                
                if detail:
                    user_prompt += f"- 详细结果：{detail[:500]}\n"
                
                user_prompt += (
                    "\n【要求】\n"
                    "1. 叙述要包含检查部位和关键发现\n"
                    "2. 明确指出异常或正常\n"
                    "3. 使用专业医学术语但保持可读性\n"
                    "4. 简洁明了，1-2句话\n\n"
                    "请仅输出报告叙述文本。"
                )
                
                try:
                    narrative = self.llm.generate_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.2,
                        max_tokens=150
                    )
                    result["narrative"] = narrative.strip()
                    result["llm_enhanced"] = True
                    enhanced_count += 1
                    logger.info(f"  ✓ [{idx+1}/{len(results)}] {test_name}")
                except Exception as e:
                    logger.warning(f"  ✗ [{idx+1}/{len(results)}] {test_name}: {e}")
                    result["narrative"] = f"{test_name}：{summary}"
                    result["llm_enhanced"] = False
                    failed_count += 1
            
            logger.info(f"\n✅ 报告叙述增强完成: {enhanced_count}成功, {failed_count}失败")
            
            # 更新状态中的检查结果
            state.test_results = results
            
            state.add_audit(
                make_audit_entry(
                    node_name="C10b Enhance Reports",
                    inputs_summary={"results_count": len(results)},
                    outputs_summary={
                        "enhanced_count": enhanced_count,
                        "failed_count": failed_count,
                        "success_rate": f"{enhanced_count}/{len(results)}"
                    },
                    decision=f"完成报告增强：{enhanced_count}项成功",
                    chunks=[],
                    flags=["LLM_USED"] if enhanced_count > 0 else ["LLM_FAILED"],
                )
            )
            logger.info("✅ C10b节点完成\n")
            return state

        def c11_return_visit(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("🔙 C11: 报告回诊")
            logger.info("="*60)
            
            # 物理环境：从检验科返回诊室
            if self.world and state.patient_id:
                current_time_before = self.world.current_time.strftime('%H:%M')
                logger.info(f"\n🏥 物理环境状态:")
                logger.info(f"  🕐 时间: {current_time_before}")
                
                # 返回诊室（使用之前保存的目标诊室位置）
                target_clinic = state.current_location  # 从state获取之前的诊室位置
                # 如果没有记录，根据科室重新映射
                if target_clinic == "lab":
                    dept_location_map = {
                        "neurology": "neuro",
                    }
                    target_clinic = dept_location_map.get(state.dept, "neuro")
                
                success, msg = self.world.move_agent(state.patient_id, target_clinic)
                if success:
                    dept_display_name = state.dept_display_name if hasattr(state, 'dept_display_name') else "诊室"
                    _log_detail(f"  🚶 移动: 检验科 → {dept_display_name}", state, 2, "C11")
                    state.current_location = target_clinic
                    state.sync_physical_state()
                    self.world.advance_time(minutes=2)
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C11", level=2)
            
            state.appointment["return_visit"] = {"status": "returned", "reports_ready": True}
            logger.info("✅ 患者携报告返回诊室")
            
            # 初始化变量（防止作用域错误）
            need_followup = False
            followup_reason = []
            
            # 医生基于检查结果进行智能补充问诊
            if self.doctor_agent and self.patient_agent and state.test_results:
                # 统计异常结果
                abnormal_results = [r for r in state.test_results if r.get("abnormal")]
                logger.info(f"\n📊 检查结果统计: {len(state.test_results)}项，异常{len(abnormal_results)}项")
                
                # 智能判断：是否需要补充问诊
                followup_reason = []
                max_followup_questions = 0
                
                # 判断条件1：有异常检查结果
                if abnormal_results:
                    followup_reason.append(f"{len(abnormal_results)}项异常结果")
                    max_followup_questions = min(len(abnormal_results) + 1, self.max_questions)
                
                # 判断条件2：检查结果提示需要进一步问诊的关键词
                key_findings = [
                    r.get("test_name") for r in state.test_results
                    if any(kw in str(r.get("summary", "")).lower() 
                          for kw in ["建议", "复查", "进一步", "随访", "注意", "监测", "评估"])
                ]
                if key_findings:
                    followup_reason.append(f"{len(key_findings)}项提示需进一步评估")
                    max_followup_questions = max(max_followup_questions, 2)
                
                # 判断条件3：初步诊断不确定
                uncertainty = state.specialty_summary.get("uncertainty", "low") if state.specialty_summary else "low"
                if uncertainty in ["high", "medium"]:
                    followup_reason.append(f"诊断不确定性{uncertainty}")
                    max_followup_questions = max(max_followup_questions, 2)
                
                # 判断条件4：检查结果与主诉不符或出现意外发现
                unexpected_findings = [r for r in state.test_results if r.get("unexpected", False)]
                if unexpected_findings:
                    followup_reason.append(f"{len(unexpected_findings)}项意外发现")
                    max_followup_questions = max(max_followup_questions, 3)
                
                need_followup = bool(followup_reason)  # 有任何原因即需要问诊
                
                # 最终决策
                if need_followup:
                    logger.info(f"\n💬 需要补充问诊（原因: {', '.join(followup_reason)}）")
                    logger.info(f"  📋 计划问诊轮数: 最多{max_followup_questions}轮")
                    
                    # 显示完整检查报告（让医生判断，不预先标注正常/异常）
                    if state.test_results:
                        logger.info("\n" + "="*60)
                        logger.info("📋 检验科检查报告")
                        logger.info("="*60)
                        for idx, result in enumerate(state.test_results, 1):
                            test_name = result.get('test_name', '未知检查')
                            test_type = result.get('type', 'lab')
                            result_text = result.get('result', 'N/A')
                            
                            logger.info(f"\n【报告 {idx}/{len(state.test_results)}】{test_name} ({test_type})")
                            logger.info("-" * 60)
                            # 显示完整的检查结果内容
                            for line in result_text.split('\n'):
                                if line.strip():
                                    logger.info(f"  {line}")
                            logger.info("-" * 60)
                        logger.info("")
                else:
                    logger.info("\n✅ 检查结果正常且明确，无需补充问诊")
                
                qa_list = state.agent_interactions.get("doctor_patient_qa", [])
                
                # 使用全局共享计数器
                global_qa_count = state.node_qa_counts.get("global_total", 0)
                remaining_global_questions = max(0, self.max_questions - global_qa_count)
                logger.info(f"  全局已问 {global_qa_count} 个问题，剩余配额 {remaining_global_questions} 个")
                
                # 根据剩余配额调整C11的问诊轮数
                max_followup_questions = min(max_followup_questions, remaining_global_questions)
                
                questions_asked_in_this_stage = 0
                
                # 构建检查结果摘要供医生参考
                test_summary = []
                for r in state.test_results:
                    test_summary.append({
                        "test": r.get("test_name"),
                        "abnormal": r.get("abnormal", False),
                        "summary": r.get("summary", ""),
                        "value": r.get("value"),
                        "unexpected": r.get("unexpected", False)
                    })
                
                # 只有在需要时才进行问诊
                if need_followup and max_followup_questions > 0:
                    logger.info("\n💬 开始检查后补充问诊（一问一答）...")
                    
                    # 逐个生成基于检查结果的问题
                    for i in range(max_followup_questions):
                        logger.info(f"\n  📝 检查后第 {i + 1} 轮问诊:")
                        
                        # 医生基于检查结果生成问题
                        question = self.doctor_agent.generate_question_based_on_tests(
                            test_results=test_summary,
                            chief_complaint=state.chief_complaint,
                            collected_info=self.doctor_agent.collected_info
                        )
                        
                        if not question:
                            logger.info("    ℹ️  医生判断信息已充足，提前结束问诊")
                            break
                        
                        logger.info(f"    🧑‍⚕️  医生问: {question}")
                        
                        # 患者回答
                        answer = self.patient_agent.respond_to_doctor(question)
                        logger.info(f"    👤 患者答: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                        
                        # 医生处理回答
                        self.doctor_agent.process_patient_answer(question, answer)
                        
                        # 【重要】同步更新医生的对话历史记录
                        self.doctor_agent.collected_info.setdefault("conversation_history", [])
                        self.doctor_agent.collected_info["conversation_history"].append({
                            "question": question,
                            "answer": answer
                        })
                        
                        # 记录对话到state
                        qa_list.append({
                            "question": question, 
                            "answer": answer, 
                            "stage": "post_test_followup",
                            "triggered_by": "test_results"
                        })
                        questions_asked_in_this_stage += 1
                        # 更新全局计数器
                        state.node_qa_counts["global_total"] = global_qa_count + questions_asked_in_this_stage
                    
                    if questions_asked_in_this_stage > 0:
                        final_global_count = state.node_qa_counts.get("global_total", 0)
                        logger.info(f"\n  ✅ 检查后补充问诊完成，新增 {questions_asked_in_this_stage} 轮，全局总计 {final_global_count} 轮")
                
                else:
                    logger.info("\n  ℹ️  检查结果完整，无需补充问诊")
                
                # 更新医生和患者交互信息
                state.agent_interactions["doctor_patient_qa"] = qa_list
                # 注意：doctor_summary和patient_summary包含智能体的内部状态（collected_info等）
                # 不应该重复记录qa_pairs，因为已经在doctor_patient_qa中了
                state.agent_interactions["doctor_summary"] = {
                    "questions_count": len(self.doctor_agent.questions_asked),
                    "collected_info": self.doctor_agent.collected_info
                }
                state.agent_interactions["patient_summary"] = {
                    "total_turns": len(self.doctor_agent.questions_asked),  # 使用医生问题数作为对话轮数
                    "case_info": self.patient_agent.case_info
                }
            
            state.add_audit(
                make_audit_entry(
                    node_name="C11 Return Visit",
                    inputs_summary={
                        "reports_ready": bool(state.appointment.get("reports_ready")),
                        "abnormal_count": sum(1 for r in state.test_results if r.get("abnormal")),
                        "need_followup": need_followup if state.test_results else False
                    },
                    outputs_summary={
                        "status": "returned",
                        "post_test_qa": len([qa for qa in state.agent_interactions.get("doctor_patient_qa", []) 
                                            if qa.get("stage") == "post_test_followup"]),
                        "followup_reason": followup_reason if state.test_results and need_followup else []
                    },
                    decision="模拟携带报告回诊" + (f" + 智能补充问诊({', '.join(followup_reason)})" if state.test_results and need_followup else " + 无需补充问诊"),
                    chunks=[],
                    flags=["AGENT_MODE", "INTELLIGENT_FOLLOWUP"] if state.test_results and need_followup else ["AGENT_MODE"]
                )
            )
            logger.info("✅ C11节点完成\n")
            return state

        def c12_final_synthesis(state: BaseState) -> BaseState:
            _log_node_start("C12", "综合分析与诊断", state)
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C12", level=2)
            
            # 显示输入信息
            _log_detail("\n📋 输入信息:", state, 1, "C12")
            _log_detail(f"  • 主诉: {state.chief_complaint[:50]}...", state, 1, "C12")
            _log_detail(f"  • 科室: {state.dept}", state, 1, "C12")
            if state.test_results:
                _log_detail(f"  • 检查结果: {len(state.test_results)}项", state, 1, "C12")
                for i, result in enumerate(state.test_results[:3], 1):
                    status = "⚠️  异常" if result.get("abnormal") else "✅ 正常"
                    _log_detail(f"    [{i}] {result.get('test_name', '未知')}: {status}", state, 1, "C12")
            else:
                _log_detail(f"  • 检查结果: 无", state, 1, "C12")
            
            _log_detail("\n🔍 检索诊断相关知识...", state, 1, "C12")
            chunks_forms = self.retriever.retrieve(
                "门诊病历 诊断证明 病假条 宣教单 模板",
                filters={"dept": "forms"},
                k=4,
            )
            chunks_hospital = self.retriever.retrieve(
                "诊后处置 随访 SOP",
                filters={"dept": "hospital", "type": "sop"},
                k=4,
            )
            chunks_dept_plan = self.retriever.retrieve(
                f"{state.dept} plan 随访 模板",
                filters={"dept": state.dept, "type": "plan"},
                k=4,
            )
            all_chunks = chunks_forms + chunks_hospital + chunks_dept_plan
            _log_detail(f"  ✅ 检索到 {len(all_chunks)} 个知识片段", state, 1, "C12")
            state.add_retrieved_chunks(all_chunks)

            # 定义fallback函数（统一管理默认值）
            def get_fallback_response():
                return {
                    "diagnosis": {
                        "name": "待明确诊断",
                        "evidence": [],
                        "reasoning": "诊断生成失败，需人工判断",
                        "uncertainty": "high",
                        "rule_out": ["需排除严重器质性病变"],
                        "disclaimer": disclaimer_text(),
                    },
                    "treatment_plan": {
                        "symptomatic": ["对症治疗"],
                        "etiology": ["根据检查结果进一步治疗"],
                        "tests": [t.get("name") for t in state.ordered_tests] if state.need_aux_tests else [],
                        "referral": [],
                        "admission": [],
                        "followup": ["按随访计划复诊"],
                        "disclaimer": disclaimer_text(),
                    },
                    "followup_plan": {
                        "when": "1-2周内复诊",
                        "monitoring": ["症状变化"],
                        "emergency": ["出现红旗症状立即急诊"],
                        "long_term_goals": ["明确诊断", "症状控制"],
                        "disclaimer": disclaimer_text(),
                    },
                    "escalations": [],
                }

            used_fallback = False
            if self.llm is not None:
                _log_detail("\n🤖 使用LLM生成诊断与方案...", state, 1, "C12")
                system_prompt = load_prompt("common_system.txt")
                
                # 构建证据结构
                evidence_summary = {
                    "问诊信息": {
                        "主诉": state.chief_complaint,
                        "病史": state.history,
                        "专科问诊": state.specialty_summary
                    }
                }
                
                # 引用医生的初步诊断
                if state.agent_interactions.get("doctor_diagnosis"):
                    evidence_summary["医生初步诊断"] = state.agent_interactions["doctor_diagnosis"]
                    _log_detail("  ✓ 引用医生初步诊断", state, 1, "C12")
                
                if state.test_results:
                    evidence_summary["检查结果"] = []
                    _log_detail(f"  ✓ 整合 {len(state.test_results)} 项检查结果", state, 1, "C12")
                    for r in state.test_results:
                        evidence_summary["检查结果"].append({
                            "项目": r.get("test"),
                            "部位": r.get("body_part", ["未知"]),
                            "结果": r.get("summary"),
                            "异常": "是" if r.get("abnormal") else "否",
                            "叙述": r.get("narrative", "")
                        })
                
                # 安全加载专科方案模板（神经内科）
                dept_plan_prompt = ""
                try:
                    dept_plan_prompt = load_prompt("neuro_plan.txt")
                except Exception as e:
                    logger.warning(f"⚠️  无法加载神经内科专科模板: {e}")
                    dept_plan_prompt = "请根据神经内科科室特点制定方案。"
                
                user_prompt = (
                    load_prompt("common_finalize.txt")
                    + "\n\n【专科方案模板】\n"
                    + dept_plan_prompt
                    + "\n\n【证据链要求】\n"
                    + "诊断必须明确引用以下证据来源：\n"
                    + "1. **问诊证据**：症状描述、持续时间、伴随症状等\n"
                    + "2. **检查证据**：具体检查项目名称、检查部位、异常结果\n"
                    + "3. **排除依据**：哪些检查结果正常，排除了哪些疾病\n\n"
                    + "在diagnosis字段中必须包含：\n"
                    + "- name: 明确的诊断名称（如存在多个假设，用'/'分隔或选主要假设）\n"
                    + "- evidence: 列出支持诊断的具体证据（格式：'问诊：XXX'、'检查：XXX部位XXX项目显示XXX'）\n"
                    + "- reasoning: 诊断推理过程（为何这些证据支持该诊断）\n"
                    + "- uncertainty: 诊断确定程度（high/medium/low）\n"
                    + "- rule_out: 已排除的诊断及排除依据\n\n"
                    + "【输入结构化信息】\n"
                    + json.dumps(evidence_summary, ensure_ascii=False, indent=2)
                    + "\n\n【引用片段（可追溯）】\n"
                    + _chunks_for_prompt(all_chunks)
                    + "\n\n请仅输出 JSON，必须包含以下字段：\n"
                    + "- diagnosis: {\n"
                    + "    name, evidence: [列表], reasoning,\n"
                    + "    uncertainty, rule_out: [列表]\n"
                    + "  }\n"
                    + "- treatment_plan: {symptomatic, etiology, tests, referral, admission, followup}\n"
                    + "- followup_plan: {when, monitoring, emergency, long_term_goals}\n"
                    + "- escalations: [列表，可选]"
                )
                
                # 调用LLM生成诊断
                obj, used_fallback, _raw = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=get_fallback_response,
                    temperature=0.2,
                    max_tokens=2500,
                )
                
                # 保存结果（使用fallback作为安全默认值）
                fallback_data = get_fallback_response()
                state.diagnosis = dict(obj.get("diagnosis") or fallback_data["diagnosis"])
                state.treatment_plan = dict(obj.get("treatment_plan") or fallback_data["treatment_plan"])
                state.followup_plan = dict(obj.get("followup_plan") or fallback_data["followup_plan"])
                if isinstance(obj.get("escalations"), list):
                    state.escalations = [str(x) for x in obj.get("escalations") if str(x)]
                
                _log_detail(f"  ✅ 最终诊断: {state.diagnosis.get('name', 'N/A')}", state, 1, "C12")
                
                # 显示诊断详情
                _log_detail("\n🎯 诊断结果:", state, 1, "C12")
                _log_detail(f"  • 诊断名称: {state.diagnosis.get('name', '未明确')}", state, 1, "C12")
                _log_detail(f"  • 确定程度: {state.diagnosis.get('uncertainty', 'unknown')}", state, 1, "C12")
                
                # 显示证据链
                evidence_list = state.diagnosis.get("evidence", [])
                if evidence_list:
                    _log_detail(f"  • 证据支持: {len(evidence_list)}项", state, 1, "C12")
                    for i, ev in enumerate(evidence_list[:3], 1):
                        _log_detail(f"    [{i}] {ev if isinstance(ev, str) else str(ev)[:50]}", state, 1, "C12")
                else:
                    _log_detail("  ⚠️  缺少证据引用", state, 1, "C12")
                
                # 显示鉴别诊断
                rule_out = state.diagnosis.get('rule_out', [])
                if rule_out:
                    logger.info(f"  • 鉴别诊断: {len(rule_out)}项")
                    for i, ro in enumerate(rule_out[:2], 1):
                        logger.info(f"    [{i}] {ro}")
                else:
                    logger.info("  • 鉴别诊断: 无")
                
                # 显示治疗方案
                logger.info("\n💊 治疗方案:")
                symptomatic = state.treatment_plan.get('symptomatic', [])
                if symptomatic:
                    logger.info(f"  • 对症治疗: {len(symptomatic)}项")
                    for i, s in enumerate(symptomatic[:2], 1):
                        logger.info(f"    [{i}] {s}")
                
                etiology = state.treatment_plan.get('etiology', [])
                if etiology:
                    logger.info(f"  • 病因治疗: {len(etiology)}项")
                
                # 显示随访计划
                logger.info("\n📅 随访计划:")
                logger.info(f"  • 复诊时间: {state.followup_plan.get('when', '未设置')}")
                monitoring = state.followup_plan.get('monitoring', [])
                if monitoring:
                    logger.info(f"  • 监测项目: {', '.join(monitoring[:3])}")
                
                # 【病例库】记录诊断
                if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
                    state.medical_record_integration.on_diagnosis(state, doctor_id="doctor_001")
                    logger.info("  📋 诊断信息已记录到病例库")
                
                # 【病例库】记录处方（如果有药物）
                if state.treatment_plan.get("medications"):
                    if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
                        state.medical_record_integration.on_prescription(state, doctor_id="doctor_001")
                        logger.info("  📋 处方已记录到病例库")
                
                # 显示证据引用
                evidence_list = state.diagnosis.get("evidence", [])
                logger.info(f"  ✓ 证据引用: {len(evidence_list)}条" if evidence_list else "  ⚠️  缺少证据引用")
                
                if state.escalations:
                    # 终端只显示简要信息
                    logger.info(f"  ⚠️  升级建议: {len(state.escalations)}项 (详见患者日志)")
                    # 详细内容记录到患者日志
                    _log_detail(f"\n⚠️  升级建议 ({len(state.escalations)}项):", state, 1, "C13")
                    for i, esc in enumerate(state.escalations, 1):
                        _log_detail(f"    [{i}] {esc}", state, 1, "C13")

            else:
                # 无LLM时使用fallback
                fallback_data = get_fallback_response()
                state.diagnosis = fallback_data["diagnosis"]
                state.treatment_plan = fallback_data["treatment_plan"]
                state.followup_plan = fallback_data["followup_plan"]
                used_fallback = True

            # 确保所有字段都有disclaimer
            state.diagnosis.setdefault("disclaimer", disclaimer_text())
            state.treatment_plan.setdefault("disclaimer", disclaimer_text())
            state.followup_plan.setdefault("disclaimer", disclaimer_text())

            apply_safety_rules(state)
            _log_detail("  ✅ 安全规则应用完成", state, 1, "C12")

            state.add_audit(
                make_audit_entry(
                    node_name="C12 Final Synthesis",
                    inputs_summary={
                        "dept": state.dept,
                        "need_aux_tests": state.need_aux_tests,
                        "results_count": len(state.test_results),
                    },
                    outputs_summary={
                        "diagnosis": state.diagnosis.get("name"),
                        "escalations": state.escalations,
                    },
                    decision="综合分析形成诊断与方案（含表单/随访/专科模板检索）",
                    chunks=all_chunks,
                    flags=["LLM_PARSE_FALLBACK"]
                    if used_fallback
                    else (["LLM_USED"] if self.llm else []),
                )
            )
            _log_node_end("C12", state)
            return state

        def c13_disposition(state: BaseState) -> BaseState:
            _log_node_start("C13", "处置决策", state)
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C13", level=2)
            
            disposition: list[str] = []
            if "急诊" in state.escalations:
                disposition.append("建议立即急诊评估")
                _log_detail("⚠️  建议立即急诊评估", state, 1, "C13")
            if "住院" in state.escalations:
                disposition.append("建议住院进一步检查治疗")
                _log_detail("⚠️  建议住院治疗", state, 1, "C13")
            if not disposition:
                disposition.append("门诊对症处理/取药/观察")
                _log_detail("✅ 门诊对症处理", state, 1, "C13")
            
            state.treatment_plan["disposition"] = disposition
            state.add_audit(
                make_audit_entry(
                    node_name="C13 Disposition",
                    inputs_summary={"escalations": state.escalations},
                    outputs_summary={"disposition": disposition},
                    decision="根据方案与升级触发处置",
                    chunks=[],
                )
            )
            _log_node_end("C13", state)
            return state

        def c14_documents(state: BaseState) -> BaseState:
            """C14: 使用LLM生成门诊医疗文书"""
            _log_node_start("C14", "生成文书", state)
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C14", level=2)
            
            # 显示输入信息
            _log_detail("\n📋 输入信息:", state, 1, "C14")
            _log_detail(f"  • 诊断: {state.diagnosis.get('name', '未明确')}", state, 1, "C14")
            _log_detail(f"  • 科室: {state.dept}", state, 1, "C14")
            _log_detail(f"  • 治疗方案: 已制定", state, 1, "C14")
            
            docs = []
            doc_types = ["门诊病历", "诊断证明", "病假条", "宣教单"]
            
            logger.info("\n🤖 使用LLM生成专业医疗文书...")
            
            # 准备文书生成所需的上下文
            context = {
                "dept": state.dept,
                "chief_complaint": state.chief_complaint,
                "history": state.history,
                "diagnosis": state.diagnosis,
                "treatment_plan": state.treatment_plan,
                "test_results": [{
                    "test": r.get("test_name"),
                    "result": r.get("summary")
                } for r in state.test_results] if state.test_results else [],
                "followup_plan": state.followup_plan,
            }
            
            system_prompt = load_prompt("common_system.txt")
            
            # 逐个生成每种文书
            for idx, doc_type in enumerate(doc_types, 1):
                logger.info(f"  [{idx}/{len(doc_types)}] 📝 正在生成{doc_type}...")
                
                user_prompt = (
                    f"请生成一份专业的{doc_type}。\n\n"
                    + "【患者信息】\n"
                    + json.dumps(context, ensure_ascii=False, indent=2)
                    + "\n\n【文书要求】\n"
                )
                
                if doc_type == "门诊病历":
                    user_prompt += (
                        "1. 包含：主诉、现病史、体格检查、辅助检查、诊断、治疗计划\n"
                        "2. 格式规范，使用医学术语\n"
                        "3. 内容完整准确\n"
                    )
                elif doc_type == "诊断证明":
                    user_prompt += (
                        "1. 简洁明了，突出诊断\n"
                        "2. 包含就诊日期、诊断名称\n"
                        "3. 医学术语准确\n"
                    )
                elif doc_type == "病假条":
                    user_prompt += (
                        "1. 根据诊断建议合理休息天数\n"
                        "2. 格式正式\n"
                        "3. 包含就诊日期和诊断\n"
                    )
                elif doc_type == "宣教单":
                    user_prompt += (
                        "1. 通俗易懂，便于患者理解\n"
                        "2. 包含疾病知识、注意事项、复诊提醒\n"
                        "3. 强调红旗症状\n"
                    )
                
                user_prompt += "\n请直接输出文书内容，不要添加标题或其他说明。"
                
                try:
                    content = self.llm.generate_text(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.2,
                        max_tokens=800
                    )
                    
                    docs.append({
                        "doc_type": doc_type,
                        "content": content.strip(),
                        "generated_by": "llm"
                    })
                    # 显示文书预览
                    preview = content[:60].replace('\n', ' ')
                    _log_detail(f"      ✅ 完成 ({len(content)}字): {preview}...", state, 1, "C14")
                except Exception as e:
                    logger.warning(f"      ❌ 生成失败: {e}，使用简化版本")
                    docs.append({
                        "doc_type": doc_type,
                        "content": f"{doc_type}生成失败",
                        "generated_by": "fallback",
                        "error": str(e)
                    })
            
            state.discharge_docs = docs
            
            # 显示文书汇总
            _log_detail("\n🎯 文书生成结果:", state, 1, "C14")
            for i, doc in enumerate(docs, 1):
                doc_type = doc.get('doc_type', '未知')
                content_length = len(doc.get('content', ''))
                generated_by = doc.get('generated_by', 'unknown')
                _log_detail(f"  [{i}] {doc_type}: {content_length}字 (生成方式: {generated_by})", state, 1, "C14")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C14 Documents",
                    inputs_summary={"need_docs": True},
                    outputs_summary={
                        "docs": [d.get("doc_type") for d in docs],
                        "generation_method": "LLM" if self.llm else "Template"
                    },
                    decision="使用LLM生成专业门诊文书（病历、证明、病假条、宣教单）",
                    chunks=[],
                    flags=["LLM_USED"] if self.llm else ["TEMPLATE_FALLBACK"],
                )
            )
            _log_node_end("C14", state)
            return state

        def c15_education_followup(state: BaseState) -> BaseState:
            _log_node_start("C15", "宣教与随访", state)
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C15", level=2)
            
            # 显示输入信息
            _log_detail("\n📋 输入信息:", state, 1, "C15")
            _log_detail(f"  • 诊断: {state.diagnosis.get('name', '未明确')}", state, 1, "C15")
            _log_detail(f"  • 科室: {state.dept}", state, 1, "C15")
            _log_detail(f"  • 治疗方案: 已制定", state, 1, "C15")
            
            _log_detail("\n🔍 检索宣教知识...", state, 1, "C15")
            chunks_common = self.retriever.retrieve(
                "门诊 宣教 随访 红旗 应急处理",
                filters={"dept": "hospital", "type": "education"},
                k=4,
            )
            chunks_dept = self.retriever.retrieve(
                f"{state.dept} 宣教 随访 注意事项",
                filters={"dept": state.dept, "type": "education"},
                k=4,
            )
            all_chunks = chunks_common + chunks_dept
            _log_detail(f"  ✅ 检索到 {len(all_chunks)} 个宣教片段", state, 1, "C15")
            _log_detail(f"    - 通用宣教: {len(chunks_common)}个", state, 1, "C15")
            _log_detail(f"    - 专科宣教: {len(chunks_dept)}个", state, 1, "C15")
            state.add_retrieved_chunks(all_chunks)

            # 神经内科默认宣教内容
            education = [
                "监测：头痛/眩晕频率与诱因记录",
                "如有癫痫样发作风险，避免危险作业并按医嘱用药",
                "出现意识障碍/肢体无力/言语不清等立即急诊",
            ]

            used_fallback = False
            if self.llm is not None:
                logger.info("\n🤖 使用LLM生成宣教内容...")
                system_prompt = load_prompt("common_system.txt")
                user_prompt = (
                    load_prompt("common_education.txt")
                    + "\n\n【输入结构化信息】\n"
                    + json.dumps(
                        {
                            "dept": state.dept,
                            "diagnosis": state.diagnosis,
                            "treatment_plan": state.treatment_plan,
                            "followup_plan": state.followup_plan,
                            "escalations": state.escalations,
                            "education_fallback": education,
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n【参考宣教片段（可追溯）】\n"
                    + _chunks_for_prompt(all_chunks)
                    + "\n\n请仅输出 JSON，可包含 education(list) 与 followup_plan(dict)。"
                )
                try:
                    obj, used_fallback, _raw = self.llm.generate_json(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        fallback=lambda: {
                            "education": education,
                            "followup_plan": {
                                "when": state.followup_plan.get("when", "1-2周内复诊"),
                                "monitoring": state.followup_plan.get("monitoring", ["症状变化"]),
                                "emergency": state.followup_plan.get("emergency", ["出现红旗症状立即急诊"])[:3],  # 限制最多3项
                                "long_term_goals": state.followup_plan.get("long_term_goals", ["明确诊断", "症状控制"]),
                            },
                            "disclaimer": disclaimer_text(),
                        },
                        temperature=0.2,
                        max_tokens=1500,  # 增加token限制，确保JSON完整
                    )
                    parsed = obj
                    if used_fallback:
                        logger.warning("  ⚠️  LLM生成失败，使用默认宣教内容")
                        # 显示原始响应以便调试（warning级别，便于排查问题）
                        if _raw:
                            logger.warning(f"  原始响应长度: {len(_raw)} 字符")
                            logger.warning(f"  原始响应前300字符: {str(_raw)[:300]}...")
                            logger.warning(f"  原始响应后100字符: ...{str(_raw)[-100:]}")
                    else:
                        logger.info("  ✅ LLM生成成功")
                        logger.info(f"  • 生成教育项目: {len(parsed.get('education', []))}条")
                except Exception as e:
                    logger.error(f"  ❌ LLM调用异常: {e}")
                    # 使用fallback
                    parsed = {
                        "education": education,
                        "followup_plan": state.followup_plan,
                        "disclaimer": disclaimer_text(),
                    }
                    used_fallback = True
            else:
                logger.warning("\n⚠️  未配置LLM，使用默认宣教内容")
                llm_text = json.dumps(
                    {"education": education, "disclaimer": disclaimer_text()}, ensure_ascii=False
                )
                parsed, used_fallback = parse_json_with_retry(
                    llm_text,
                    fallback=lambda: {"education": education, "disclaimer": disclaimer_text()},
                )

            state.followup_plan.setdefault("education", [])
            state.followup_plan["education"] = list(parsed.get("education", education))
            if isinstance(parsed.get("followup_plan"), dict):
                # 选择性更新，避免覆盖异常数据
                new_followup = dict(parsed.get("followup_plan"))
                # 验证并清理emergency列表
                if "emergency" in new_followup:
                    emergency_list = new_followup["emergency"]
                    if isinstance(emergency_list, list):
                        # 过滤非字符串项，限制最多5项
                        new_followup["emergency"] = [str(e) for e in emergency_list if e][:5]
                state.followup_plan.update(new_followup)
            state.followup_plan["disclaimer"] = str(parsed.get("disclaimer", disclaimer_text()))

            # 显示宣教内容
            logger.info("\n🎯 宣教内容详情:")
            education_items = state.followup_plan.get('education', [])
            if education_items:
                logger.info(f"  共 {len(education_items)} 项宣教内容:")
                for i, item in enumerate(education_items, 1):
                    # 截取显示，避免太长
                    display_text = item[:80] + "..." if len(item) > 80 else item
                    logger.info(f"    [{i}] {display_text}")
            else:
                logger.warning("  ⚠️  未生成宣教内容")
            
            # 显示随访计划更新
            logger.info("\n📅 随访计划详情:")
            logger.info(f"  • 复诊时间: {state.followup_plan.get('when', '未设置')}")
            
            monitoring = state.followup_plan.get('monitoring', [])
            if monitoring:
                logger.info(f"  • 监测项目: {len(monitoring)}项")
                for i, mon in enumerate(monitoring[:3], 1):
                    logger.info(f"    [{i}] {mon}")
                if len(monitoring) > 3:
                    logger.info(f"    ... 及其他{len(monitoring)-3}项")
            else:
                logger.info("  • 监测项目: 无")
            
            emergency = state.followup_plan.get('emergency', [])
            if emergency:
                # 验证数据类型，过滤无效项
                valid_emergency = [str(e) for e in emergency if e and isinstance(e, (str, dict))]
                if len(valid_emergency) != len(emergency):
                    _log_detail(f"  ⚠️  检测到 {len(emergency)-len(valid_emergency)} 个无效紧急情况项，已过滤", state, 1, "C15")
                
                if valid_emergency:
                    # 异常数据警告（超过10项可能有问题）
                    if len(valid_emergency) > 10:
                        _log_detail(f"  ⚠️  紧急情况项数异常多({len(valid_emergency)}项)，可能存在数据问题", state, 1, "C15")
                    
                    # 终端只显示简要信息
                    logger.info(f"  ⚠️  紧急情况: {len(valid_emergency)}项 (详见患者日志)")
                    
                    # 详细内容记录到患者日志
                    _log_detail(f"  ⚠️  紧急情况({len(valid_emergency)}项):", state, 1, "C15")
                    for i, emg in enumerate(valid_emergency, 1):
                        _log_detail(f"    [{i}] {emg}", state, 1, "C15")
                else:
                    logger.info("  • 紧急情况: 无有效项")
            else:
                logger.info("  • 紧急情况: 无")
            
            # 显示免责声明
            disclaimer = state.followup_plan.get('disclaimer', '')
            if disclaimer:
                logger.info(f"  • 免责声明: {disclaimer[:50]}...")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C15 Education & Follow-up",
                    inputs_summary={"dept": state.dept},
                    outputs_summary={"education_items": len(state.followup_plan.get("education", []))},
                    decision="生成宣教与随访计划（含通用与专科检索）",
                    chunks=all_chunks,
                    flags=["LLM_PARSE_FALLBACK"]
                    if used_fallback
                    else (["LLM_USED"] if self.llm else []),
                )
            )
            _log_node_end("C15", state)
            return state

        def c16_end(state: BaseState) -> BaseState:
            _log_node_start("C16", "结束流程", state)
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            # 显示物理环境状态
            _log_physical_state(state, self.world, "C16", level=2)
            
            # 记录流程结束时间和统计信息
            import datetime
            end_timestamp = datetime.datetime.now().isoformat()
            state.appointment["visit_end_time"] = end_timestamp
            state.appointment["status"] = "visit_completed"
            
            # 计算流程耗时
            start_time_str = state.appointment.get("visit_start_time")
            if start_time_str:
                try:
                    start_time = datetime.datetime.fromisoformat(start_time_str)
                    end_time = datetime.datetime.fromisoformat(end_timestamp)
                    duration = end_time - start_time
                    duration_minutes = duration.total_seconds() / 60
                    state.appointment["visit_duration_minutes"] = duration_minutes
                    _log_detail(f"\n⏱️  流程耗时: {duration_minutes:.1f} 分钟", state, 1, "C16")
                except Exception:
                    pass
            
            # 显示流程统计摘要
            _log_detail("\n📊 流程统计摘要:", state, 1, "C16")
            _log_detail(f"  🏥 科室: {state.dept}", state, 1, "C16")
            _log_detail(f"  🗣️  主诉: {state.chief_complaint}", state, 1, "C16")
            _log_detail(f"  💬 问诊轮数: {len(state.agent_interactions.get('doctor_patient_qa', []))}", state, 1, "C16")
            _log_detail(f"  🧪 开单项目: {len(state.ordered_tests)}", state, 1, "C16")
            _log_detail(f"  📋 检查结果: {len(state.test_results)}", state, 1, "C16")
            _log_detail(f"  🩺 最终诊断: {state.diagnosis.get('name', 'N/A')}", state, 1, "C16")
            if state.escalations:
                _log_detail(f"  ⚠️  升级建议: {', '.join(state.escalations)}", state, 1, "C16")
            
            # 【资源释放】释放医生资源
            if self.world and state.patient_id:
                released = self.world.release_doctor(state.patient_id)
                if released:
                    _log_detail(f"  ✅ 已释放医生资源", state, 2, "C16")
            
            # 【病例库】患者出院，记录出院信息
            if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
                state.medical_record_integration.on_discharge(state, doctor_id="doctor_001")
                logger.info("  📋 出院信息已记录到病例库")
                
                # 显示病例摘要
                summary = state.medical_record_integration.get_patient_history(state.patient_id)
                if summary:
                    logger.info(f"\n📋 病例摘要:")
                    logger.info(f"  病例号: {summary['record_id']}")
                    logger.info(f"  总记录数: {summary['total_entries']} 条")
                    logger.info(f"  诊断次数: {summary['diagnoses_count']}")
                    logger.info(f"  检验次数: {summary['lab_tests_count']}")
                    logger.info(f"  处方次数: {summary['prescriptions_count']}")
            
            # 评估诊断准确性
            if state.ground_truth:
                logger.info("\n📊 评估诊断准确性...")
                doctor_diagnosis = state.diagnosis.get("name", "")
                correct_diagnosis = state.ground_truth.get("Final Diagnosis", "")
                
                logger.info(f"  👨‍⚕️  医生诊断: {doctor_diagnosis}")
                
                # 使用LLM进行语义相似度评估
                accuracy = 0.0
                accuracy_method = "LLM语义评估"
                
                if self.llm:
                    try:
                        logger.info("  🤖 使用LLM评估诊断准确性...")
                        system_prompt = "你是一位医学专家，擅长评估医学诊断的准确性。"
                        user_prompt = (
                            f"请评估以下两个诊断的相似度（0-100分）：\n\n"
                            f"医生诊断：{doctor_diagnosis}\n"
                            f"标准答案：{correct_diagnosis}\n\n"
                            f"评分标准：\n"
                            f"- 100分：完全一致或同义词\n"
                            f"- 80-99分：核心诊断正确，表述略有差异\n"
                            f"- 60-79分：大方向正确，但有遗漏或冗余\n"
                            f"- 40-59分：部分正确，但有明显错误\n"
                            f"- 0-39分：完全错误或无关\n\n"
                            f"请仅输出一个0-100之间的整数分数，不要有其他文字。"
                        )
                        
                        score_text = self.llm.generate_text(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=0.1,
                            max_tokens=10
                        ).strip()
                        
                        # 提取数字
                        import re
                        match = re.search(r'\d+', score_text)
                        if match:
                            semantic_score = int(match.group())
                            accuracy = min(100, max(0, semantic_score)) / 100.0
                            logger.info(f"  🎯 诊断准确率: {accuracy*100:.0f}分")
                        else:
                            logger.warning(f"  ⚠️  无法解析LLM评分: {score_text}")
                            accuracy_method = "解析失败"
                    except Exception as e:
                        logger.warning(f"  ⚠️  LLM评估失败: {e}")
                        accuracy_method = "评估失败"
                else:
                    logger.warning("  ⚠️  未配置LLM，跳过评估")
                    accuracy_method = "无LLM"
                
                evaluation = {
                    "doctor_diagnosis": doctor_diagnosis,
                    "correct_diagnosis": correct_diagnosis,
                    "accuracy": accuracy,
                    "accuracy_method": accuracy_method,
                    "questions_asked": len(state.agent_interactions.get("doctor_patient_qa", [])),
                    "tests_ordered": len(state.ordered_tests),
                }
                
                state.agent_interactions["evaluation"] = evaluation
                
                # 显示评估结果（仅记录到日志文件）
                accuracy_pct = accuracy * 100
                if accuracy_pct >= 80:
                    logger.debug(f"  ✅ 诊断准确性评级: 优秀 ({accuracy_pct:.0f}分)")
                elif accuracy_pct >= 60:
                    logger.debug(f"  ⚠️  诊断准确性评级: 良好 ({accuracy_pct:.0f}分)")
                elif accuracy_pct > 0:
                    logger.debug(f"  ⚠️  诊断准确性评级: 需改进 ({accuracy_pct:.0f}分)")
                else:
                    logger.debug(f"  ❌ 未能完成评估")
                
                logger.debug(f"  💬 问诊轮数: {evaluation['questions_asked']}")
                logger.debug(f"  🧪 开单数量: {evaluation['tests_ordered']}")


            
            state.add_audit(
                make_audit_entry(
                    node_name="C16 End Visit",
                    inputs_summary={
                        "run_id": state.run_id,
                        "start_time": state.appointment.get("visit_start_time"),
                    },
                    outputs_summary={
                        "done": True,
                        "end_time": end_timestamp,
                        "duration_minutes": state.appointment.get("visit_duration_minutes"),
                        "has_evaluation": bool(state.agent_interactions.get("evaluation")),
                        "final_diagnosis": state.diagnosis.get("name"),
                    },
                    decision="记录流程结束时间，生成统计摘要，评估诊断准确性",
                    chunks=[],
                    flags=["VISIT_END", "EVALUATION"] if state.ground_truth else ["VISIT_END"],
                )
            )
            _log_detail("\n🎉 门诊流程全部完成!", state, 1, "C16")
            return state

        # 添加所有节点（C0已移至初始化阶段）
        graph.add_node("C1", c1_start)
        graph.add_node("C2", c2_registration)
        graph.add_node("C3", c3_checkin_waiting)
        graph.add_node("C4", c4_call_in)
        graph.add_node("C5", c5_prepare_intake)  # 更名：准确反映其准备问诊的功能
        graph.add_node("C6", c6_specialty_dispatch)
        graph.add_node("C7", c7_decide_path)
        graph.add_node("C8", c8_order_explain_tests)
        graph.add_node("C9", c9_billing_scheduling)
        graph.add_node("C10a", c10a_fetch_test_results)
        graph.add_node("C10b", c10b_enhance_reports)
        graph.add_node("C11", c11_return_visit)
        graph.add_node("C12", c12_final_synthesis)
        graph.add_node("C13", c13_disposition)
        graph.add_node("C14", c14_documents)
        graph.add_node("C15", c15_education_followup)
        graph.add_node("C16", c16_end)

        # 设置入口点和连接边（C0已移至初始化阶段，直接从C1开始）
        graph.set_entry_point("C1")
        graph.add_edge("C1", "C2")
        graph.add_edge("C2", "C3")
        graph.add_edge("C3", "C4")
        graph.add_edge("C4", "C5")
        graph.add_edge("C5", "C6")
        graph.add_edge("C6", "C7")

        def _path(state: BaseState) -> str:
            return "with_tests" if state.need_aux_tests else "no_tests"

        graph.add_conditional_edges(
            "C7",
            _path,
            {
                "with_tests": "C8",
                "no_tests": "C12",
            },
        )

        graph.add_edge("C8", "C9")
        graph.add_edge("C9", "C10a")
        graph.add_edge("C10a", "C10b")
        graph.add_edge("C10b", "C11")
        graph.add_edge("C11", "C12")
        graph.add_edge("C12", "C13")
        graph.add_edge("C13", "C14")
        graph.add_edge("C14", "C15")
        graph.add_edge("C15", "C16")
        graph.add_edge("C16", END)

        return graph.compile()
