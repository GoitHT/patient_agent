from __future__ import annotations

"""
门诊流程图（与 gastro/neuro 两张流程图一致）：

注意：C0（护士分诊+Agent初始化）已移至main.py的初始化阶段执行

- 通用前置（两科一致，合并实现一次）：
  C1 开始 -> C2 挂号（预约挂号） -> C3 签到候诊 -> C4 叫号入诊室
- 专科中段（唯一差异点，通过可插拔 LangGraph 子图注入）：
  gastro: G4-G6 / neuro: N4-N6（均在 C6 Specialty Dispatch 调用）
- 通用后置（两科一致，合并实现一次）：
  若 need_aux_tests=True：C8 开单并解释准备 -> C9 缴费与预约 -> C10 执行检查取报告 -> C11 回诊
  最终：C12 综合分析明确诊断/制定方案 -> C13 处置 -> C14 文书 -> C15 宣教随访 -> C16 结束
"""

import random
import json
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from rag import ChromaRetriever
from services.appointment import AppointmentService
from services.billing import BillingService
from services.llm_client import LLMClient
from state.schema import BaseState, make_audit_entry
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


@dataclass(frozen=True)
class Services:
    """保留的必要服务：预约和计费系统"""
    appointment: AppointmentService
    billing: BillingService


def _default_channel(rng: random.Random) -> str:
    return rng.choice(["APP", "公众号", "电话", "自助机", "窗口"])


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
        rng: random.Random,
        llm: LLMClient | None = None,
        llm_reports: bool = False,
        use_agents: bool = True,  # 总是使用三智能体模式
        patient_agent: Any | None = None,
        doctor_agent: Any | None = None,
        nurse_agent: Any | None = None,
        max_questions: int = 3,
    ) -> None:
        self.retriever = retriever
        self.dept_subgraphs = dept_subgraphs
        self.services = services
        self.rng = rng
        self.llm = llm
        self.llm_reports = llm_reports
        self.use_agents = use_agents
        self.patient_agent = patient_agent
        self.doctor_agent = doctor_agent
        self.nurse_agent = nurse_agent
        self.max_questions = max_questions

    def build(self):
        graph = StateGraph(BaseState)

        def c1_start(state: BaseState) -> BaseState:
            """C1: 开始门诊流程 - 验证状态、记录开始时间、显示患者概览"""
            logger.info("\n" + "="*60)
            logger.info("🏁 C1: 开始门诊流程")
            logger.info("="*60)
            
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
            
            logger.info(f"✓ 状态验证通过")
            
            # 2. 记录流程开始时间
            import datetime
            start_timestamp = datetime.datetime.now().isoformat()
            state.appointment["visit_start_time"] = start_timestamp
            
            # 3. 显示患者就诊概览
            logger.info("\n📋 患者就诊信息:")
            logger.info(f"  🏥 就诊科室: {state.dept}")
            logger.info(f"  🏷️  流程ID: {state.run_id}")
            logger.info(f"  🗣️  主诉: {state.chief_complaint}")
            logger.info(f"  🕐 开始时间: {start_timestamp}")
            
            # 4. 初始化流程追踪
            if "nurse_triage" in state.agent_interactions:
                triage_info = state.agent_interactions["nurse_triage"]
                logger.info(f"  💉 分诊结果: {triage_info.get('triaged_dept', 'N/A')}")
                if triage_info.get("reasoning"):
                    logger.info(f"     理由: {triage_info['reasoning'][:60]}...")
            
            # 5. 设置流程状态标记
            state.appointment["status"] = "visit_started"
            state.appointment["current_stage"] = "C1_start"
            
            state.add_audit(
                make_audit_entry(
                    node_name="C1 Start Visit",
                    inputs_summary={
                        "dept": state.dept,
                        "chief_complaint": state.chief_complaint[:40],
                        "triage_completed": "nurse_triage" in state.agent_interactions,
                    },
                    outputs_summary={
                        "run_id": state.run_id,
                        "start_time": start_timestamp,
                        "status": "visit_started",
                    },
                    decision="验证状态完整性，记录流程开始，初始化就诊追踪",
                    chunks=[],
                    flags=["VISIT_START"],
                )
            )
            logger.info("✅ C1节点完成 - 就诊流程正式启动\n")
            return state

        def c2_registration(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("📝 C2: 预约挂号")
            logger.info("="*60)
            
            channel = state.appointment.get("channel") or _default_channel(self.rng)
            timeslot = state.appointment.get("timeslot") or "上午"
            logger.info(f"📱 预约渠道: {channel}")
            logger.info(f"⏰ 时间段: {timeslot}")
            
            appt = self.services.appointment.create_appointment(
                channel=channel, dept=state.dept, timeslot=timeslot
            )
            state.appointment = appt
            
            logger.info(f"✅ 挂号成功 - 预约ID: {appt.get('appointment_id')}")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C2 Registration",
                    inputs_summary={"channel": channel, "timeslot": timeslot},
                    outputs_summary={"appointment_id": appt.get("appointment_id")},
                    decision="完成预约挂号",
                    chunks=[],
                )
            )
            logger.info("✅ C2节点完成\n")
            return state

        def c3_checkin_waiting(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("✍️ C3: 签到候诊")
            logger.info("="*60)
            
            state.appointment = self.services.appointment.checkin(state.appointment)
            
            logger.info(f"✅ 签到成功 - 状态: {state.appointment.get('status')}")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C3 Checkin & Waiting",
                    inputs_summary={"appointment_id": state.appointment.get("appointment_id")},
                    outputs_summary={"status": state.appointment.get("status")},
                    decision="完成签到并进入候诊",
                    chunks=[],
                )
            )
            logger.info("✅ C3节点完成\n")
            return state

        def c4_call_in(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("🔔 C4: 叫号进诊")
            logger.info("="*60)
            
            state.appointment = self.services.appointment.call_patient(state.appointment)
            
            logger.info(f"✅ 叫号成功 - 状态: {state.appointment.get('status')}")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C4 Call In",
                    inputs_summary={"appointment_id": state.appointment.get("appointment_id")},
                    outputs_summary={"status": state.appointment.get("status")},
                    decision="叫号进入诊室",
                    chunks=[],
                )
            )
            logger.info("✅ C4节点完成\n")
            return state

        def c5_prepare_intake(state: BaseState) -> BaseState:
            """C5: 问诊准备 - 检索通用SOP并初始化问诊记录（实际问诊在C6专科子图中进行）"""
            logger.info("\n" + "="*60)
            logger.info("🩺 C5: 问诊准备")
            logger.info("="*60)
            
            logger.info("🔍 检索医院通用SOP与免责声明...")
            chunks = self.retriever.retrieve(
                f"门诊 问诊要点 分流 免责声明 {state.chief_complaint}",
                filters={"dept": "hospital", "type": "sop"},
                k=4,
            )
            logger.info(f"  ✅ 检索到 {len(chunks)} 个知识片段")
            state.add_retrieved_chunks(chunks)

            # 初始化问诊对话记录（实际问诊在C6专科子图中进行）
            logger.info("\n💬 注：详细问诊将在C6专科子图中进行")
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
            logger.info("✅ C5节点完成\n")
            return state

        def c6_specialty_dispatch(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info(f"🏭 C6: 专科流程调度 ({state.dept})")
            logger.info("="*60)
            
            sub = self.dept_subgraphs.get(state.dept)
            if sub is None:
                raise ValueError(f"Unknown dept: {state.dept}")
            
            logger.info(f"🔀 调用 {state.dept} 子图...")
            out = sub.invoke(state)
            state = BaseState.model_validate(out)
            
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
            logger.info("✅ C6节点完成\n")
            return state

        def c7_decide_path(state: BaseState) -> BaseState:
            """C7: 路径决策 - 根据need_aux_tests标志决定是否进入辅助检查流程
            注：此节点目前仅做简单判断，未来可扩展为更复杂的决策逻辑（如急诊分流、转诊判断等）
            """
            logger.info("\n" + "="*60)
            logger.info("🔀 C7: 路径决策")
            logger.info("="*60)
            
            logger.info(f"❓ 需要辅助检查: {state.need_aux_tests}")
            if state.need_aux_tests:
                logger.info(f"📝 待开单项目数: {len(state.ordered_tests)}")
                for test in state.ordered_tests:
                    logger.info(f"  - {test.get('name', 'N/A')} ({test.get('type', 'N/A')})")
            else:
                logger.info("✅ 无需辅助检查，直接进入诊断")
            
            state.add_audit(
                make_audit_entry(
                    node_name="C7 Decide Path",
                    inputs_summary={"need_aux_tests": state.need_aux_tests},
                    outputs_summary={"ordered_tests_count": len(state.ordered_tests)},
                    decision="根据need_aux_tests标志选择后续路径（with_tests或no_tests）",
                    chunks=[],
                )
            )
            logger.info("✅ C7节点完成\n")
            return state

        def c8_order_explain_tests(state: BaseState) -> BaseState:
            """
            C8: 开单与检查准备说明
            职责：
            1. 检索医院缴费/预约流程SOP
            2. 检索专科检查准备知识（禁忌、注意事项、准备步骤）
            3. 生成完整的检查准备说明（不包含具体预约信息）
            """
            logger.info("\n" + "="*60)
            logger.info("🧪 C8: 开单与准备说明")
            logger.info("="*60)
            
            # 检索医院通用流程SOP
            logger.info("🔍 检索医院通用流程...")
            hospital_chunks = self.retriever.retrieve(
                "缴费 预约 报告领取 回诊 流程",
                filters={"dept": "hospital", "type": "sop"},
                k=4,
            )
            state.add_retrieved_chunks(hospital_chunks)
            logger.info(f"  ✅ 检索到 {len(hospital_chunks)} 个通用流程SOP")

            dept_chunks: list[dict[str, Any]] = []
            prep_items: list[dict[str, Any]] = []
            
            # 为每个检查项目检索准备知识
            logger.info(f"\n📋 检索 {len(state.ordered_tests)} 个检查项目的准备知识...")
            for t in state.ordered_tests:
                test_name = t.get('name', '')
                test_type = t.get('type', 'unknown')
                
                logger.info(f"  🔍 {test_name} ({test_type})")
                
                # 检索专科检查准备知识
                q = f"{state.dept} {test_name} 准备 禁忌 注意事项 禁食"
                cs = self.retriever.retrieve(q, filters={"dept": state.dept}, k=4)
                dept_chunks.extend(cs)
                state.add_retrieved_chunks(cs)
                logger.info(f"     ✅ 检索到 {len(cs)} 个准备知识片段")

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
            logger.info(f"\n✅ 开单与准备说明生成完成，共 {len(prep_items)} 项检查")

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
            logger.info("✅ C8节点完成\n")
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
            
            # 1. 生成订单并缴费
            order_id = f"ORD-{state.run_id}-{len(state.ordered_tests)}"
            logger.info(f"📝 订单ID: {order_id}")
            
            payment = self.services.billing.pay(order_id=order_id)
            logger.info(f"✅ 缴费完成 - 金额: {payment.get('amount', 0)}元")
            state.appointment["billing"] = payment

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
                        # 内镜检查：生成预约信息
                        prep["schedule"] = {
                            "procedure": test_name,
                            "scheduled": True,
                            "schedule_id": f"END-{self.rng.randint(10000, 99999)}",
                            "scheduled_at": "T+2d",
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
                        logger.info(f"     ✅ 预约时间: 后天")
                    elif test_type == "imaging":
                        # 影像检查：通用预约
                        prep["schedule"] = {
                            "scheduled": True,
                            "procedure": test_name,
                            "scheduled_at": "T+1d",
                            "location": "影像科",
                        }
                        logger.info(f"     ✅ 预约时间: 明天")
                    elif test_type == "neurophysiology":
                        # 神经电生理检查
                        prep["schedule"] = {
                            "scheduled": True,
                            "procedure": test_name,
                            "scheduled_at": "T+2d",
                            "location": "神经电生理室",
                        }
                        logger.info(f"     ✅ 预约时间: 后天")
                    else:
                        # 其他检查
                        prep["schedule"] = {
                            "scheduled": True,
                            "procedure": test_name,
                            "scheduled_at": "T+1d",
                        }
                        logger.info(f"     ✅ 预约完成")
                    
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
            
            # 优先从数据集获取真实检查结果
            real_diagnostic_tests = state.ground_truth.get("Diagnostic Tests", "").strip()
            results: list[dict[str, Any]] = []
            used_fallback = False
            
            if real_diagnostic_tests:
                logger.info("📋 使用数据集中的真实检查结果")
                logger.info(f"  原始数据: {real_diagnostic_tests[:300]}{'...' if len(real_diagnostic_tests) > 300 else ''}")
                
                # 使用LLM将文本结构化为检查结果列表
                system_prompt = load_prompt("common_system.txt")
                
                # 构建已开检查项目列表
                ordered_tests_str = "\n".join([
                    f"- {t.get('name')} ({t.get('type')}, {t.get('body_part', ['未知部位'])})"
                    for t in state.ordered_tests
                ])
                
                user_prompt = (
                    "请将以下真实检查结果文本结构化为JSON格式的检查结果列表。\n\n"
                    + "【已开检查项目】\n"
                    + ordered_tests_str + "\n\n"
                    + "【真实检查结果文本】\n"
                    + f"{real_diagnostic_tests}\n\n"
                    + "【任务要求】\n"
                    + "1. 从文本中提取所有检查结果，每项检查对应一个结果对象\n"
                    + "2. 尽量匹配已开检查项目，但也要包含文本中提到的其他检查\n"
                    + "3. 每项检查结果包含：\n"
                    + "   - test: 检查名称（与已开项目匹配或从文本提取）\n"
                    + "   - test_name: 同test\n"
                    + "   - type: 检查类型（lab/imaging/endoscopy/neurophysiology）\n"
                    + "   - body_part: 检查部位（从已开项目获取或从文本推断）\n"
                    + "   - summary: 结果摘要（简短描述）\n"
                    + "   - abnormal: 是否异常（true/false）\n"
                    + "   - value: 具体数值或描述（如有）\n"
                    + "   - reference: 参考范围（如有）\n"
                    + "   - detail: 详细结果文本（保持原文）\n"
                    + "4. 保持原始结果的准确性，不要修改数值或结论\n"
                    + "5. 判断abnormal时要准确：如果结果明确提示异常/超标/阳性，则为true\n\n"
                    + "【输出格式】\n"
                    + "请输出JSON：{\"test_results\": [{检查结果1}, {检查结果2}, ...]}"
                )
                
                obj, used_fallback, _raw = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=lambda: {
                        "test_results": [{
                            "test": "综合检查",
                            "test_name": "综合检查",
                            "type": "lab",
                            "body_part": ["全身"],
                            "summary": "见详细报告",
                            "abnormal": False,
                            "detail": real_diagnostic_tests[:500] + "...",  # 限制长度避免超token
                            "source": "dataset_fallback"
                        }]
                    },
                    temperature=0.1,  # 低温度保证忠实原文
                    max_tokens=2000,
                )
                
                results = list(obj.get("test_results") or [])
                
                # 标记数据来源
                for r in results:
                    r["source"] = "dataset_real"
                    r["raw_text"] = real_diagnostic_tests
                
                logger.info(f"  ✅ 从真实数据提取 {len(results)} 项检查结果")
                abnormal_count = sum(1 for r in results if r.get("abnormal"))
                logger.info(f"  ⚠️  异常结果: {abnormal_count}/{len(results)}")
            
            else:
                # 如果数据集没有检查结果，使用LLM基于ordered_tests生成合理的检查结果
                logger.info("⚠️  数据集无检查结果，使用LLM生成合理的检查结果")
                
                if self.llm is None:
                    logger.error("❌ 无LLM配置，无法生成检查结果")
                    results = []
                else:
                    # 构建已开检查项目列表
                    ordered_tests_str = "\n".join([
                        f"- {t.get('name')} ({t.get('type')}, {t.get('body_part', ['未知部位'])}): {t.get('reason', '诊断需要')}"
                        for t in state.ordered_tests
                    ])
                    
                    system_prompt = load_prompt("common_system.txt")
                    user_prompt = (
                        "请为以下检查项目生成合理的检查结果。\n\n"
                        + "【患者信息】\n"
                        + f"主诉：{state.chief_complaint}\n"
                        + f"科室：{state.dept}\n"
                        + f"专科诊断：{state.specialty_summary.get('diagnosis', 'N/A')}\n\n"
                        + "【已开检查项目】\n"
                        + ordered_tests_str + "\n\n"
                        + "【任务要求】\n"
                        + "1. 为每项检查生成临床上合理的结果\n"
                        + "2. 结果应与患者主诉和初步诊断相关联\n"
                        + "3. 适当设置异常结果以支持诊断（约20-40%异常率）\n"
                        + "4. 每项检查结果包含：\n"
                        + "   - test_name: 检查名称\n"
                        + "   - type: 检查类型（lab/imaging/endoscopy/neurophysiology）\n"
                        + "   - body_part: 检查部位\n"
                        + "   - summary: 结果摘要（简短描述）\n"
                        + "   - abnormal: 是否异常（true/false）\n"
                        + "   - value: 具体数值或描述（如有）\n"
                        + "   - reference: 参考范围（如有）\n"
                        + "   - detail: 详细结果描述\n"
                        + "5. 保持医学专业性和临床合理性\n\n"
                        + "【输出格式】\n"
                        + "请输出JSON：{\"test_results\": [{检查结果1}, {检查结果2}, ...]}"
                    )
                    
                    obj, used_fallback, _raw = self.llm.generate_json(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        fallback=lambda: {
                            "test_results": [{
                                "test_name": t.get("name"),
                                "type": t.get("type"),
                                "body_part": t.get("body_part", ["未知"]),
                                "summary": "检查结果生成失败",
                                "abnormal": False,
                                "detail": "LLM生成失败，请人工审核",
                                "source": "llm_fallback"
                            } for t in state.ordered_tests]
                        },
                        temperature=0.3,  # 适度随机性以生成合理变化
                        max_tokens=2000,
                    )
                    
                    results = list(obj.get("test_results") or [])
                    
                    # 标记数据来源
                    for r in results:
                        r["source"] = "llm_generated"
                    
                    logger.info(f"\n✅ LLM生成检查结果完成，共 {len(results)} 项")
                    abnormal_count = sum(1 for r in results if r.get("abnormal"))
                    logger.info(f"  ⚠️  异常结果: {abnormal_count}/{len(results)}")
            
            # 保存原始检查结果（未增强）
            state.test_results = results
            state.appointment["reports_ready"] = bool(results)
            
            # 安全获取data_source（防止索引错误）
            data_source = results[0].get("source") if results else "none"
            
            state.add_audit(
                make_audit_entry(
                    node_name="C10a Fetch Test Results",
                    inputs_summary={"ordered_tests_count": len(state.ordered_tests), "has_real_data": bool(real_diagnostic_tests)},
                    outputs_summary={
                        "results_count": len(results), 
                        "abnormal_count": sum(1 for r in results if r.get("abnormal")),
                        "data_source": data_source
                    },
                    decision="获取检查结果" + ("（使用数据集真实结果）" if real_diagnostic_tests else "（LLM生成）"),
                    chunks=[],
                    flags=["REAL_DATA"] if real_diagnostic_tests else (["LLM_PARSE_FALLBACK"] if used_fallback else ["LLM_USED"]),
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
            
            if not self.llm or not self.llm_reports:
                logger.info("ℹ️  未启用LLM报告增强，保持原始结果")
                state.add_audit(
                    make_audit_entry(
                        node_name="C10b Enhance Reports",
                        inputs_summary={"results_count": len(results)},
                        outputs_summary={"enhanced": False},
                        decision="未启用LLM报告增强",
                        chunks=[],
                        flags=["SKIPPED"]
                    )
                )
                logger.info("✅ C10b节点完成\n")
                return state
            
            # 使用LLM为检查结果生成个性化报告叙述
            logger.info(f"🤖 使用LLM为 {len(results)} 项检查结果生成报告叙述...")
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
                    
                    if abnormal_results:
                        logger.info("  ⚠️  异常项目:")
                        for result in abnormal_results:
                            logger.info(f"     - {result.get('test_name')}: {result.get('summary', 'N/A')}")
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
                        
                        # 记录对话
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
                    "total_turns": len(self.patient_agent.conversation_history),
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
            logger.info("\n" + "="*60)
            logger.info("🔬 C12: 综合分析与诊断")
            logger.info("="*60)
            
            logger.info("🔍 检索诊断相关知识...")
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
            logger.info(f"  ✅ 检索到 {len(all_chunks)} 个知识片段")
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
                logger.info("\n🤖 使用LLM生成诊断与方案...")
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
                    logger.info("  ✓ 引用医生初步诊断")
                
                if state.test_results:
                    evidence_summary["检查结果"] = []
                    for r in state.test_results:
                        evidence_summary["检查结果"].append({
                            "项目": r.get("test"),
                            "部位": r.get("body_part", ["未知"]),
                            "结果": r.get("summary"),
                            "异常": "是" if r.get("abnormal") else "否",
                            "叙述": r.get("narrative", "")
                        })
                
                # 安全加载专科方案模板
                dept_plan_prompt = ""
                if state.dept in ["gastro", "neuro"]:
                    try:
                        dept_plan_prompt = load_prompt(
                            "gastro_plan.txt" if state.dept == "gastro" else "neuro_plan.txt"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  无法加载{state.dept}专科模板: {e}")
                        dept_plan_prompt = f"请根据{state.dept}科室特点制定方案。"
                
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
                
                logger.info(f"  ✅ 最终诊断: {state.diagnosis.get('name', 'N/A')}")
                
                # 显示诊断信息
                evidence_list = state.diagnosis.get("evidence", [])
                logger.info(f"  ✓ 证据引用: {len(evidence_list)}条" if evidence_list else "  ⚠️  缺少证据引用")
                
                if state.escalations:
                    logger.warning(f"  ⚠️  升级建议: {', '.join(state.escalations)}")

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
            logger.info("  ✅ 安全规则应用完成")

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
            logger.info("✅ C12节点完成\n")
            return state

        def c13_disposition(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("👨‍⚕️ C13: 处置决策")
            logger.info("="*60)
            
            disposition: list[str] = []
            if "急诊" in state.escalations:
                disposition.append("建议立即急诊评估")
                logger.warning("⚠️  建议立即急诊评估")
            if "住院" in state.escalations:
                disposition.append("建议住院进一步检查治疗")
                logger.warning("⚠️  建议住院治疗")
            if not disposition:
                disposition.append("门诊对症处理/取药/观察")
                logger.info("✅ 门诊对症处理")
            
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
            logger.info("✅ C13节点完成\n")
            return state

        def c14_documents(state: BaseState) -> BaseState:
            """C14: 使用LLM生成门诊医疗文书"""
            logger.info("\n" + "="*60)
            logger.info("📄 C14: 生成文书")
            logger.info("="*60)
            
            docs = []
            doc_types = ["门诊病历", "诊断证明", "病假条", "宣教单"]
            
            if self.llm is None:
                logger.warning("⚠️  未配置LLM，使用基础模板生成文书")
                # 简单的fallback文书
                docs = [
                    {"doc_type": "门诊病历", "content": f"主诉：{state.chief_complaint}\n诊断：{state.diagnosis.get('name')}"},
                    {"doc_type": "诊断证明", "content": f"诊断：{state.diagnosis.get('name')}"},
                    {"doc_type": "病假条", "content": "建议休息3-7天"},
                    {"doc_type": "宣教单", "content": "\n".join(state.followup_plan.get("education", []))},
                ]
            else:
                logger.info("🤖 使用LLM生成专业医疗文书...")
                
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
                for doc_type in doc_types:
                    logger.info(f"  📝 生成{doc_type}...")
                    
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
                        logger.info(f"     ✅ {doc_type}生成完成")
                    except Exception as e:
                        logger.warning(f"     ⚠️  {doc_type}生成失败: {e}，使用简化版本")
                        docs.append({
                            "doc_type": doc_type,
                            "content": f"{doc_type}生成失败",
                            "generated_by": "fallback",
                            "error": str(e)
                        })
            
            state.discharge_docs = docs
            logger.info(f"\n✅ 文书生成完成，共 {len(docs)} 份")
            for doc in docs:
                logger.info(f"     - {doc.get('doc_type')}")
            
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
            logger.info("✅ C14节点完成\n")
            return state

        def c15_education_followup(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("📚 C15: 宣教与随访")
            logger.info("="*60)
            
            logger.info("🔍 检索宣教知识...")
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
            logger.info(f"  ✅ 检索到 {len(all_chunks)} 个宣教片段")
            state.add_retrieved_chunks(all_chunks)

            if state.dept == "gastro":
                education = [
                    "饮食：避免辛辣油腻与酒精，规律进食",
                    "按医嘱用药；如行Hp检测/治疗需按疗程并复查",
                    "出现黑便/呕血/进行性消瘦等立即急诊",
                ]
            else:
                education = [
                    "监测：头痛/眩晕频率与诱因记录",
                    "如有癫痫样发作风险，避免危险作业并按医嘱用药",
                    "出现意识障碍/肢体无力/言语不清等立即急诊",
                ]

            used_fallback = False
            if self.llm is not None:
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
                obj, used_fallback, _raw = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=lambda: {
                        "education": education,
                        "followup_plan": state.followup_plan,
                        "disclaimer": disclaimer_text(),
                    },
                    temperature=0.2,
                    max_tokens=900,
                )
                parsed = obj
            else:
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
                state.followup_plan.update(dict(parsed.get("followup_plan")))
            state.followup_plan["disclaimer"] = str(parsed.get("disclaimer", disclaimer_text()))

            logger.info(f"\n✅ 宣教内容生成完成，共 {len(state.followup_plan.get('education', []))}条")
            
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
            logger.info("✅ C15节点完成\n")
            return state

        def c16_end(state: BaseState) -> BaseState:
            logger.info("\n" + "="*60)
            logger.info("✅ C16: 结束流程")
            logger.info("="*60)
            
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
                    logger.info(f"\n⏱️  流程耗时: {duration_minutes:.1f} 分钟")
                except Exception:
                    pass
            
            # 显示流程统计摘要
            logger.info("\n📊 流程统计摘要:")
            logger.info(f"  🏥 科室: {state.dept}")
            logger.info(f"  🗣️  主诉: {state.chief_complaint}")
            logger.info(f"  💬 问诊轮数: {len(state.agent_interactions.get('doctor_patient_qa', []))}")
            logger.info(f"  🧪 开单项目: {len(state.ordered_tests)}")
            logger.info(f"  📋 检查结果: {len(state.test_results)}")
            logger.info(f"  🩺 最终诊断: {state.diagnosis.get('name', 'N/A')}")
            if state.escalations:
                logger.info(f"  ⚠️  升级建议: {', '.join(state.escalations)}")
            
            # 评估诊断准确性
            if state.ground_truth:
                logger.info("\n📊 评估诊断准确性...")
                doctor_diagnosis = state.diagnosis.get("name", "")
                correct_diagnosis = state.ground_truth.get("Final Diagnosis", "")
                
                logger.info(f"  👨‍⚕️  医生诊断: {doctor_diagnosis}")
                logger.info(f"  🎯 标准答案: {correct_diagnosis}")
                
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
                
                # 显示评估结果
                accuracy_pct = accuracy * 100
                if accuracy_pct >= 80:
                    logger.info(f"  ✅ 诊断准确性评级: 优秀 ({accuracy_pct:.0f}分)")
                elif accuracy_pct >= 60:
                    logger.warning(f"  ⚠️  诊断准确性评级: 良好 ({accuracy_pct:.0f}分)")
                elif accuracy_pct > 0:
                    logger.warning(f"  ⚠️  诊断准确性评级: 需改进 ({accuracy_pct:.0f}分)")
                else:
                    logger.error(f"  ❌ 未能完成评估")
                
                logger.info(f"  💬 问诊轮数: {evaluation['questions_asked']}")
                logger.info(f"  🧪 开单数量: {evaluation['tests_ordered']}")


            
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
            logger.info("\n🎉 门诊流程全部完成!\n")
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
