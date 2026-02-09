"""通用专科子图：支持所有科室的专科问诊、体检、初步判断"""
from __future__ import annotations

import json
import random
from typing import Any

from langgraph.graph import END, StateGraph

from graphs.log_helpers import _log_detail
from rag import ChromaRetriever
from services.llm_client import LLMClient
from state.schema import BaseState, make_audit_entry
from utils import load_prompt, contains_any_positive, get_logger
from environment.staff_tracker import StaffTracker  # 导入医护人员状态追踪器
from logging_utils import should_log, OutputFilter, SUPPRESS_UNCHECKED_LOGS  # 导入输出配置

# 初始化logger
logger = get_logger("hospital_agent.specialty_subgraph")

# 应用输出过滤器来抑制未被should_log包装的日志
if SUPPRESS_UNCHECKED_LOGS:
    logger.addFilter(OutputFilter("specialty_subgraph"))


# Type标准化映射常量（将各种变体映射到标准type）
TEST_TYPE_MAPPING = {
    "血液检查": "lab",
    "血液": "lab",
    "检验": "lab",
    "实验室": "lab",
    "化验": "lab",
    "尿液检查": "lab",
    "大便检查": "lab",
    "免疫学检查": "lab",
    "炎症标志物": "lab",
    "血清学检查": "lab",
    "影像检查": "imaging",
    "影像": "imaging",
    "放射": "imaging",
    "超声": "imaging",
    "内镜检查": "endoscopy",
    "内镜": "endoscopy",
    "镜检": "endoscopy",
    "功能检查": "neurophysiology",
    "电生理": "neurophysiology",
    "神经电生理": "neurophysiology",
}


def _validate_and_normalize_test(test: dict[str, Any], dept: str, dept_config: dict) -> dict[str, Any] | None:
    """
    标准化检查项目（不做白名单校验，完全信任LLM判断）
    
    Args:
        test: 原始检查项目
        dept: 科室代码
        dept_config: 科室配置
        
    Returns:
        标准化后的检查项目
    """
    test_name = str(test.get("name", "")).strip()
    test_type = str(test.get("type", "lab")).lower()
    
    if not test_name:
        logger.warning(f"  ⚠️  检查项目名称为空，跳过")
        return None
    
    # 限制检查名称长度，避免JSON解析问题
    if len(test_name) > 100:
        logger.warning(f"  ⚠️  检查名称过长({len(test_name)}字符)，截断: {test_name[:50]}...")
        test_name = test_name[:100]
    
    # 如果type不是标准值，尝试映射
    if test_type not in ["lab", "imaging", "endoscopy", "neurophysiology"]:
        test_type = TEST_TYPE_MAPPING.get(test_type, "lab")  # 默认为lab
        logger.debug(f"  🔄 检查类型标准化: {test.get('type')} → {test_type}")
    
    # 获取检查部位（如果配置了）
    test_body_parts = dept_config.get("test_body_parts", {})
    body_part = test_body_parts.get(test_name, ["相关部位"])
    
    return {
        "dept": dept,
        "type": test_type,
        "name": test_name,
        "reason": test.get("reason", "进一步明确诊断"),
        "priority": test.get("priority", "routine"),
        "need_prep": bool(test.get("need_prep", test_type in ["endoscopy"])),
        "need_schedule": bool(test.get("need_schedule", test_type in ["endoscopy", "neurophysiology"])),
        "body_part": body_part,
    }


def _chunks_for_prompt(chunks: list[dict[str, Any]], *, max_chars: int = 1400) -> str:
    lines: list[str] = []
    total = 0
    for c in chunks:
        text = str(c.get("text") or "").replace("\n", " ").strip()
        line = f"[{c.get('doc_id')}#{c.get('chunk_id')}] {text[:240]}"
        lines.append(line)
        total += len(line) + 1
        if total >= max_chars:
            break
    return "\n".join(lines)


# 科室配置映射（当前只保留 neurology，其他科室配置已删除以减少冗余）
DEPT_CONFIG = {
    "neurology": {
        "name": "神经医学",
        "interview_keys": ["onset_time", "frequency", "severity", "triggers", "relievers", "red_flags"],
        "alarm_keywords": ["突发", "偏瘫", "肢体无力", "言语不清", "意识障碍", "抽搐"],
        "exam_area": "neurological",
        "common_tests": ["头颅CT", "头颅MRI", "脑电图", "肌电图"],
    },
}


def build_common_specialty_subgraph(
    *, 
    retriever: ChromaRetriever,
    llm: LLMClient | None = None,
    doctor_agent=None, 
    patient_agent=None, 
    max_questions: int = 3  # 最底层默认值，通常从config.yaml传入
):
    """构建通用专科子图，适用于所有科室
    
    Args:
        max_questions: 医生最多问诊次数（从config.agent.max_questions传入）
    """
    graph = StateGraph(BaseState)
    
    # 注意：不在这里判断 use_agents，而是在节点执行时动态判断
    # 因为 doctor_agent 是在 C4 节点中动态分配到 state 的

    def s4_specialty_interview(state: BaseState) -> BaseState:
        """S4: 通用专科问诊节点"""
        dept = state.dept
        dept_config = DEPT_CONFIG.get(dept, DEPT_CONFIG.get("internal_medicine", {}))
        dept_name = dept_config.get("name", "通用科室")
        
        # 终端简洁输出
        if should_log(1, "specialty_subgraph", "S4"):
            logger.info(f"🏫 S4: {dept_name}专科问诊")
        
        # 详细日志记录
        detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
        if detail_logger:
            detail_logger.section(f"{dept_name}专科问诊")
        
        # 动态判断是否启用Agent模式（检查state中是否有Agent）
        use_agents = (hasattr(state, 'doctor_agent') and state.doctor_agent is not None and
                     hasattr(state, 'patient_agent') and state.patient_agent is not None)
        
        # 从 state 中获取医生 Agent（C4 节点中分配的）
        doctor_agent = None
        patient_agent = None
        if use_agents:
            doctor_agent = state.doctor_agent
            patient_agent = state.patient_agent
            if hasattr(state, 'assigned_doctor_name'):
                logger.info(f"  👨‍⚕️ 使用 C4 分配的医生 Agent: {state.assigned_doctor_name}")
            else:
                logger.info(f"  👨‍⚕️ 使用医生 Agent 进行问诊")
        else:
            logger.info(f"  ℹ️  Agent模式未启用，将使用LLM模式")
            if detail_logger:
                detail_logger.info("未检测到Doctor Agent或Patient Agent，使用LLM模式提取信息")
        
        # 如果是Agent模式，确保医生智能体的科室设置正确
        if use_agents and doctor_agent:
            doctor_agent.dept = dept
            logger.info(f"  🏥 医生科室: {dept_name}")
        
        # 检索该科室的专科知识
        # 注意：此时chief_complaint还未设置（医生尚未从患者处获得），使用科室信息检索
        query = f"{dept} {dept_name} 红旗 检查建议 鉴别诊断"
        logger.info(f"🔍 检索{dept_name}知识...")
        chunks = retriever.retrieve(query, filters={"dept": dept}, k=4)
        state.add_retrieved_chunks(chunks)

        cc = state.chief_complaint
        
        # 获取科室配置用于提示词
        alarm_keywords = dept_config.get("alarm_keywords", [])
        interview_keys = dept_config.get("interview_keys", ["symptoms_detail"])

        # 获取节点专属计数器
        node_key = f"s4_{dept}"
        
        # Agent模式：逐步一问一答，然后从doctor_agent收集结构化信息
        if use_agents:
            # 获取最大问诊轮数（优先使用state.agent_config，其次使用函数参数）
            # 确保使用配置文件设置的值，而不是硬编码的默认值
            if state.agent_config and "max_questions" in state.agent_config:
                max_qs = state.agent_config["max_questions"]
            else:
                max_qs = max_questions  # 使用函数参数（来自配置文件）
            
            # 开始问诊
            logger.info(f"  💬 问诊开始")
            
            if detail_logger:
                detail_logger.subsection("医生问诊")
            
            # ===== 物理环境集成：问诊前检查患者状态 =====
            if state.world_context:
                # 检查物理状态影响（静默处理，仅记录紧急情况）
                impact = state.get_physical_impact_on_diagnosis()
                if impact.get("has_impact"):
                    # 调整问诊轮数（内部逻辑，不显示）
                    physical_max_questions = impact.get("max_questions", max_qs)
                    if physical_max_questions < max_qs:
                        max_qs = physical_max_questions
                    
                    # 仅记录紧急情况（意识异常）
                    if impact.get("emergency"):
                        logger.error("🚨 紧急情况：患者意识异常，建议立即转急诊")
                        state.escalations.append("患者意识异常，建议急诊评估")
                        max_qs = 0
            
            # 使用全局共享计数器
            global_qa_count = state.node_qa_counts.get("global_total", 0)
            questions_asked_this_node = state.node_qa_counts.get(node_key, 0)
            
            # 计算本节点剩余问题数：本节点配额 - 本节点已问数
            # 不使用全局计数器限制，因为每个专科节点应该有独立的问诊机会
            remaining_questions = max(0, max_qs - questions_asked_this_node)
            
            if detail_logger:
                detail_logger.info(f"全局已问 {global_qa_count} 个，本节点已问 {questions_asked_this_node} 个，本节点剩余 {remaining_questions} 个")
            
            # 如果剩余问题数为0，记录原因
            if remaining_questions == 0:
                reason = ""
                if max_qs == 0:
                    reason = "max_questions配置为0或因紧急情况跳过问诊"
                elif questions_asked_this_node >= max_qs:
                    reason = f"本节点已完成全部 {max_qs} 轮问诊"
                
                logger.info(f"  ℹ️  跳过问诊：{reason}")
                if detail_logger:
                    detail_logger.info(f"⚠️ 跳过问诊：{reason}")

            
            # 逐个生成问题并获取回答
            qa_list = state.agent_interactions.get("doctor_patient_qa", [])
            
            # 获取患者详细日志记录器（如果存在）
            detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
            
            for i in range(remaining_questions):
                # 终端只显示简洁信息
                if should_log(1, "specialty_subgraph", "S4"):
                    logger.info(f"  💬 问诊第 {questions_asked_this_node + i + 1} 轮")
                
                # 医生基于当前信息生成一个问题
                context_desc = f"{dept_name}专科问诊，关注：{', '.join(interview_keys)}"
                if alarm_keywords:
                    context_desc += f"，警报症状：{', '.join(alarm_keywords)}"
                
                # 第一个问题：医生总是先用开放式问题询问患者哪里不舒服
                # 这符合真实医疗场景：医生首先让患者自己描述主要症状
                if i == 0 and not doctor_agent.questions_asked:
                    question = "您好，请问您哪里不舒服？"
                else:
                    # 后续问题：使用收集到的信息生成针对性问题
                    # 注意：不直接使用state.chief_complaint，而是使用doctor_agent已收集的信息
                    question = doctor_agent.generate_one_question(
                        chief_complaint=doctor_agent.collected_info.get("chief_complaint", ""),
                        context=context_desc,
                        rag_chunks=chunks
                    )
                
                if not question:
                    if should_log(1, "specialty_subgraph", "S4"):
                        logger.info("  ℹ️  医生提前结束问诊")
                    if detail_logger:
                        detail_logger.info("医生判断信息已充足，提前结束问诊")
                    break
                
                # 患者回答（传入物理状态）
                physical_state = state.physical_state_snapshot if state.world_context else None
                answer = patient_agent.respond_to_doctor(question, physical_state=physical_state)
                
                # 详细日志：记录完整的问诊对话
                if detail_logger:
                    detail_logger.qa_round(questions_asked_this_node + i + 1, question, answer)
                
                # 医生处理回答
                doctor_agent.process_patient_answer(question, answer)
                
                # 【重要】同步更新医生的对话历史记录（用于下次生成问题时参考）
                doctor_agent.collected_info.setdefault("conversation_history", [])
                doctor_agent.collected_info["conversation_history"].append({
                    "question": question,
                    "answer": answer
                })
                
                # 记录对话到state
                qa_list.append({
                    "question": question, 
                    "answer": answer, 
                    "stage": f"{dept}_specialty"
                })
                
                # 更新该节点和全局计数器
                state.node_qa_counts[node_key] = questions_asked_this_node + i + 1
                state.node_qa_counts["global_total"] = global_qa_count + i + 1
            
            state.agent_interactions["doctor_patient_qa"] = qa_list
            
            # ===== StaffTracker集成：区生专科问诊工作 =====
            if state.world_context:
                actual_questions = state.node_qa_counts.get(node_key, 0) - questions_asked_this_node
                if actual_questions > 0:
                    # 每轮问诊约2-3分钟
                    consultation_time = actual_questions * 2.5
                    StaffTracker.update_doctor_consultation(
                        world=state.world_context,
                        duration_minutes=int(consultation_time),
                        complexity=0.6  # 专科问诊复杂度中等偏上
                    )
                    logger.info(f"  👨‍⚕️  医生完成{dept_name}专科问诊（{actual_questions}轮，耗时{int(consultation_time)}分钟）")
            
            # ===== 物理环境集成：问诊后更新物理状态 =====
            if state.world_context:
                qa_count = len([qa for qa in qa_list if qa.get('stage') == f"{dept}_specialty"])
                if qa_count > 0:
                    duration = qa_count * 3  # 每轮约3分钟
                    energy_cost = 0.5 * qa_count  # 每轮消耗0.5体力
                    
                    logger.info(f"\n{'─'*60}")
                    logger.info(f"🌍 物理环境模拟 - 问诊过程")
                    logger.info(f"{'─'*60}")
                    start_time = state.world_context.current_time.strftime('%H:%M')
                    
                    result = state.update_physical_world(
                        action="consult",
                        duration_minutes=duration,
                        energy_cost=energy_cost
                    )
                    end_time = state.world_context.current_time.strftime('%H:%M')
                    
                    logger.info(f"💬 问诊轮数: {qa_count}轮")
                    logger.info(f"⏱️  总耗时: {duration}分钟")
                    logger.info(f"🕐 时间: {start_time} → {end_time}")
                    logger.info(f"{'─'*60}")
                    
                    # 如果出现危急警报
                    if result.get("critical_warning"):
                        logger.warning(f"🚨 警告：患者出现危急状态 (意识: {result.get('consciousness')})")
            
            # 从医生收集的信息更新state
            state.history.update(doctor_agent.collected_info.get("history", {}))
            
            final_qa_count = state.node_qa_counts.get(node_key, 0)
            final_global_count = state.node_qa_counts.get("global_total", 0)
            logger.info(f"  ✅ {dept_name}专科问诊完成，本节点 {final_qa_count} 轮，全局总计 {final_global_count} 轮")
            
            # ===== 医生总结专业主诉 =====
            # 总是让医生基于问诊总结专业主诉，覆盖患者向护士说的口语化描述
            summarized_cc = doctor_agent.summarize_chief_complaint()
            if summarized_cc:
                # 保存原始主诉（患者向护士说的）供参考
                if state.chief_complaint and state.chief_complaint != summarized_cc:
                    state.original_chief_complaint = state.chief_complaint
                # 更新为医生总结的专业主诉
                state.chief_complaint = summarized_cc
                logger.info(f"\n  📋 医生总结主诉（专业版）: {summarized_cc}")
                
                # 更新数据库中的chief_complaint字段
                if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
                    record = state.medical_record_integration.mrs.get_record(state.patient_id)
                    if record:
                        state.medical_record_integration.mrs.dao.update_medical_case(record.record_id, {
                            "chief_complaint": summarized_cc
                        })
            
            # ===== 新增：问诊质量评估 =====
            # 只有在实际问了问题时才显示评估
            if len(doctor_agent.questions_asked) > 0:
                logger.info(f"\n{'━'*60}")
                logger.info("📊 问诊质量评估")
                logger.info(f"{'━'*60}")
                
                quality_report = doctor_agent.assess_interview_quality()
                
                # 显示评估结果
                logger.info(f"  📈 综合评分: {quality_report['overall_score']}/100")
                logger.info(f"     • 完整性: {quality_report['completeness_score']:.0f}/100")
                logger.info(f"     • 深度: {quality_report['depth_score']:.0f}/100")
                logger.info(f"     • 效率: {quality_report['efficiency_score']:.0f}/100")
                
                if quality_report['warning']:
                    if quality_report['overall_score'] < 50:
                        logger.warning(f"  {quality_report['warning']}")
                    elif quality_report['overall_score'] < 70:
                        logger.info(f"  {quality_report['warning']}")
                    else:
                        logger.info(f"  {quality_report['warning']}")
                
                # 显示缺失信息
                if quality_report['missing_areas']:
                    logger.info(f"\n  ❌ 缺失关键信息 ({len(quality_report['missing_areas'])}项):")
                    for area in quality_report['missing_areas']:
                        logger.info(f"     • {area}")
                
                # 显示改进建议
                if quality_report['suggestions']:
                    logger.info(f"\n  💡 改进建议:")
                    for suggestion in quality_report['suggestions'][:3]:  # 最多显示3条
                        logger.info(f"     • {suggestion}")
                
                logger.info(f"{'━'*60}\n")
                
                # 保存评估结果到state
                state.agent_interactions["interview_quality"] = quality_report
            
            # Agent模式：直接从医生智能体获取结构化信息，不再用LLM重复提取
            interview = doctor_agent.collected_info.get(f"{dept}_interview", {})
            if not interview:
                # 如果医生没有特定科室信息，使用通用history
                interview = {
                    "collected_from_agent": True,
                    "alarm_symptoms": [],  # Agent会在对话中处理警报症状
                }
                # 只更新非警报症状相关的字段（避免将"不详"字符串赋值给警报症状字段）
                for key in interview_keys:
                    if key not in ["alarm_symptoms", "red_flags"]:
                        interview[key] = doctor_agent.collected_info.get("history", {}).get(key, "不详")
            
            # 从 Agent 收集信息
            if detail_logger:
                detail_logger.info("\n从 Agent收集的专科信息已整合")
        
        # 非Agent模式：使用LLM提取专科信息
        else:
            # 使用LLM提取
            if detail_logger:
                detail_logger.subsection("使用LLM提取专科信息")
            system_prompt = load_prompt("common_system.txt")
            
            # 根据科室选择不同的prompt
            specialty_prompt_file = f"{dept}_specialty.txt"
            try:
                specialty_prompt = load_prompt(specialty_prompt_file)
            except:
                specialty_prompt = f"请提取{dept_name}相关的专科信息。"
            
            # 简化的提示词
            user_prompt = (
                specialty_prompt
                + f"\n\n【任务】从病例中提取{dept_name}专科结构化信息\n"
                + f"【关注点】{', '.join(interview_keys)}\n"
                + f"【警报症状】{', '.join(alarm_keywords)}\n\n"
                + f"【病例】{cc}\n\n"
                + "【参考知识】\n" + _chunks_for_prompt(chunks) + "\n\n"
                + f"【输出】JSON格式，字段名: {dept}_interview，包含上述关注点及alarm_symptoms列表"
            )
            
            obj, used_fallback, _raw = llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {f"{dept}_interview": {key: "不详" for key in interview_keys} | {"alarm_symptoms": []}},
                temperature=0.2,
            )
            interview = dict(obj.get(f"{dept}_interview") or {})
            # 提取完成
            if detail_logger:
                detail_logger.info("专科信息提取完成")

        state.dept_payload.setdefault(dept, {})
        state.dept_payload[dept]["interview"] = interview

        # 统一警报症状检测（从LLM返回的interview中获取）
        # 安全地提取警报症状，检查类型避免将字符串拆分成字符列表
        raw_alarms = interview.get("alarm_symptoms") or interview.get("red_flags") or []
        if isinstance(raw_alarms, list):
            alarm_list = [str(a) for a in raw_alarms if a]  # 过滤空值
        elif isinstance(raw_alarms, str) and raw_alarms not in ["不详", "无", ""]:
            alarm_list = [raw_alarms]  # 单个字符串转为列表
        else:
            alarm_list = []  # 忽略其他无效值
        
        if alarm_list:
            if detail_logger:
                detail_logger.warning(f"⚠️  发现警报症状: {', '.join(str(a) for a in alarm_list)}")
            # 终端输出（需要output level >= 2）
            if should_log(2, "specialty_subgraph", "S4"):
                logger.warning(f"  ⚠️  发现警报症状: {', '.join(str(a) for a in alarm_list)}")

        # 记录节点问答轮数
        node_qa_turns = state.node_qa_counts.get(node_key, 0)
        
        # ===== 保存问诊记录到数据库 =====
        if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
            state.medical_record_integration.on_doctor_consultation(state, doctor_id="doctor_001")
        
        state.add_audit(
            make_audit_entry(
                node_name=f"S4 {dept_name} Specialty Interview",
                inputs_summary={"chief_complaint": state.chief_complaint, "use_agents": use_agents, "dept": dept, "max_questions": max_questions},
                outputs_summary={"alarm_symptoms": alarm_list, "node_qa_turns": node_qa_turns},
                decision=f"完成{dept_name}专科问诊（本节点{node_qa_turns}轮）" + ("（Agent模式）" if use_agents else ("（LLM模式）" if not used_fallback else "（Fallback）")),
                chunks=chunks,
                flags=["AGENT_MODE"] if use_agents else (["LLM_PARSE_FALLBACK"] if used_fallback else ["LLM_USED"]),
            )
        )
        
        # 详细日志输出节点执行摘要
        if detail_logger:
            detail_logger.info("")
            detail_logger.info("📤 S4 专科问诊输出:")
            detail_logger.info(f"  • 问诊轮数: {node_qa_turns}轮")
            if alarm_list:
                detail_logger.info(f"  • 危险症状: {', '.join(alarm_list)}")
            if use_agents and doctor_agent:
                collected = doctor_agent.collected_info
                if collected.get('chief_complaint'):
                    detail_logger.info(f"  • 主诉: {collected['chief_complaint']}")
                if collected.get('history', {}).get('duration'):
                    detail_logger.info(f"  • 病程: {collected['history']['duration']}")
            detail_logger.info(f"✅ S4 专科问诊完成")
            detail_logger.info("")
        
        if should_log(1, "specialty_subgraph", "S4"):
            logger.info(f"  ✅ S4完成\n")
        return state

    def s5_physical_exam(state: BaseState) -> BaseState:
        """S5: 通用体检节点"""
        dept = state.dept
        dept_config = DEPT_CONFIG.get(dept, DEPT_CONFIG.get("internal_medicine", {}))
        dept_name = dept_config.get("name", "通用")
        exam_area = dept_config.get("exam_area", "general")
        alarm_keywords = dept_config.get("alarm_keywords", [])
        
        # 获取详细日志记录器
        detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
        
        if should_log(1, "specialty_subgraph", "S5"):
            logger.info(f"� S5: {dept_name}体格检查")
        
        if detail_logger:
            detail_logger.section(f"{dept_name}体格检查")
        
        # 当前数据源只有case_character，使用LLM生成体检结果
        data_source = "llm_generated"
        real_physical_exam = None  # 数据集中没有体格检查数据
        
        logger.info(f"📋 使用LLM生成体检结果")
        
        # 统一结构化处理流程
        system_prompt = load_prompt("common_system.txt")
        
        # LLM生成：基于主诉和专科信息
        interview_info = state.dept_payload.get(dept, {}).get("interview", {})
        interview_str = json.dumps(interview_info, ensure_ascii=False) if interview_info else "无"
        
        # 获取问诊对话历史以提供更多上下文
        qa_history = ""
        if state.agent_interactions.get("doctor_patient_qa"):
            qa_list = [qa for qa in state.agent_interactions["doctor_patient_qa"] if qa.get('stage') == f"{dept}_specialty"]
            if qa_list:
                qa_history = "\n【问诊对话】\n"
                for i, qa in enumerate(qa_list[:3], 1):  # 最多显示3轮对话
                    qa_history += f"Q{i}: {qa.get('question', '')}\n"
                    qa_history += f"A{i}: {qa.get('answer', '')}\n"
        
        user_prompt = (
                f"根据{dept_name}科室特点和患者主诉，生成符合临床实际的体格检查结果。\n\n"
                + f"【主诉】{state.chief_complaint}\n"
                + f"【专科问诊】{interview_str}\n"
                + qa_history
                + f"\n【要求】\n"
                + f"1. 生命体征（vital_signs）：体温、脉搏、血压、呼吸频率等，给出具体数值\n"
                + f"2. 一般情况（general）：神志、精神状态、营养状况、体型等，描述具体\n"
                + f"3. {exam_area}专科体检：根据主诉和科室特点，生成相关阳性或阴性体征\n"
                + f"4. 阳性体征与主诉相符，阴性体征用于排除相关疾病\n"
                + f"5. 考虑警报症状：{', '.join(alarm_keywords)}\n"
                + f"6. 体检结果应真实可信，符合医学常识\n\n"
                + "【输出】JSON格式：\n"
                + "{\n"
                + "  \"exam\": {\n"
                + "    \"vital_signs\": {\n"
                + "      \"temperature\": \"36.5°C\",\n"
                + "      \"pulse\": \"78次/分\",\n"
                + "      \"blood_pressure\": \"120/80mmHg\",\n"
                + "      \"respiration\": \"18次/分\"\n"
                + "    },\n"
                + "    \"general\": \"神志清楚，精神可，营养中等，体型正常\",\n"
                + f"    \"{exam_area}_exam\": {{具体专科体检项目及结果}},\n"
                + "    \"positive_signs\": [\"阳性体征1\", \"阳性体征2\"],\n"
                + "    \"negative_signs\": [\"阴性体征1\", \"阴性体征2\"]\n"
                + "  }\n"
                + "}\n\n"
                + "⚠️ 注意：所有数值和描述应基于主诉合理推测，不要生成不相关的异常"
        )
        fallback_data = {
            "exam": {
                "vital_signs": {"temperature": "正常", "pulse": "正常", "blood_pressure": "正常"},
                "general": "一般情况可",
                "note": f"{dept_name}体格检查"
            }
        }
        temp = 0.2
        
        # 检查LLM是否可用
        if llm is None:
            logger.error("⚠️  未LLM配置，无法生成体格检查结果")
            exam = fallback_data["exam"]
            exam["source"] = "no_llm"
            used_fallback = True
            
            # 使用fallback时也输出到患者日志
            if detail_logger:
                detail_logger.warning("⚠️  LLM不可用，使用默认体检结果")
                detail_logger.info("\n📋 体格检查结果:")
                detail_logger.info(f"  【一般情况】{exam.get('general', '无')}")
                if exam.get('vital_signs'):
                    detail_logger.info(f"  【生命体征】正常范围")
        else:
            # 执行LLM调用
            obj, used_fallback, _raw = llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: fallback_data,
                temperature=temp,
            )
            exam = dict(obj.get("exam") or {})
            exam["source"] = data_source
            logger.info("  ✅ 体格检查处理完成")
            
            # 输出体检结果到患者日志
            if detail_logger and exam:
                detail_logger.info("\n📋 体格检查结果:")
                
                # 生命体征
                vital_signs = exam.get("vital_signs", {})
                if vital_signs:
                    detail_logger.info("  【生命体征】")
                    for key, value in vital_signs.items():
                        detail_logger.info(f"    • {key}: {value}")
                
                # 一般情况
                general = exam.get("general")
                if general:
                    detail_logger.info(f"  【一般情况】{general}")
                
                # 专科体检
                specialty_exam = exam.get(f"{exam_area}_exam")
                if specialty_exam:
                    detail_logger.info(f"  【{dept_name}专科体检】")
                    if isinstance(specialty_exam, dict):
                        for key, value in specialty_exam.items():
                            detail_logger.info(f"    • {key}: {value}")
                    else:
                        detail_logger.info(f"    {specialty_exam}")
                
                # 阳性体征
                positive_signs = exam.get("positive_signs", [])
                if positive_signs:
                    detail_logger.info("  【阳性体征】")
                    for sign in positive_signs:
                        detail_logger.info(f"    ✓ {sign}")
                
                # 阴性体征
                negative_signs = exam.get("negative_signs", [])
                if negative_signs:
                    detail_logger.info("  【阴性体征】")
                    for sign in negative_signs:  # 显示所有阴性体征
                        detail_logger.info(f"    - {sign}")
        
        state.exam_findings.setdefault(exam_area, {})
        state.exam_findings[exam_area] = exam
        
        # 推进时间（体格检查约5分钟）
        if state.world_context:
            state.world_context.advance_time(minutes=5)
            state.sync_physical_state()

        state.add_audit(
            make_audit_entry(
                node_name=f"S5 {dept_name} Physical Exam",
                inputs_summary={"exam_area": exam_area, "dept": dept, "has_real_data": bool(real_physical_exam)},
                outputs_summary={"exam_completed": True, "data_source": exam.get("source", "unknown")},
                decision=f"完成{dept_name}体格检查记录" + ("（使用数据集真实数据）" if real_physical_exam else "（LLM生成）"),
                chunks=[],
                flags=["REAL_DATA"] if real_physical_exam else (["LLM_PARSE_FALLBACK"] if used_fallback else ["LLM_USED"]),
            )
        )
        
        # 患者日志总结
        if detail_logger:
            detail_logger.info(f"✅ S5 {dept_name}体格检查完成")
            detail_logger.info("")
        
        logger.info("✅ S5节点完成\n")
        return state

    def s6_preliminary_judgment(state: BaseState) -> BaseState:
        """S6: 通用初步判断与开单节点"""
        dept = state.dept
        dept_config = DEPT_CONFIG.get(dept, DEPT_CONFIG.get("internal_medicine", {}))
        dept_name = dept_config.get("name", "通用")
        alarm_keywords = dept_config.get("alarm_keywords", [])
        common_tests = dept_config.get("common_tests", ["血常规"])
                # 获取详细日志记录器
        detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
        logger.info("\n" + "="*60)
        logger.info(f"� S6: {dept_name}初步判断")
        logger.info("="*60)
        
        query = f"{dept} {dept_name} 检查选择 适应症 {state.chief_complaint}"
        logger.info(f"🔍 检索{dept_name}检查指南...")
        chunks = retriever.retrieve(query, filters={"dept": dept}, k=4)
        state.add_retrieved_chunks(chunks)

        cc = state.chief_complaint
        
        # 使用LLM生成检查方案
        logger.info("\n🤖 使用LLM生成检查方案...")
        system_prompt = load_prompt("common_system.txt")
        
        # 尝试加载科室特定prompt
        specialty_prompt_file = f"{dept}_specialty.txt"
        try:
            specialty_prompt = load_prompt(specialty_prompt_file)
        except:
            specialty_prompt = f"请根据{dept_name}症状制定检查方案。"
        
        # 强化提示词：明确type标准，精准开具关键检查
        user_prompt = (
            specialty_prompt
            + "\n\n【任务】根据患者情况，判断是否需要辅助检查并给出初步评估。\n\n"
            + "【核心原则：精准开单，避免过度检查】\n"
            + "1. 优先级评估：\n"
            + "   - 仅开具对明确诊断和治疗方案有实质性帮助的检查\n"
            + "   - 每项检查都应有清晰的临床目的和诊断价值\n"
            + "   - 避免「预防性」或「以防万一」的检查\n\n"
            + "2. 检查数量控制：\n"
            + "   - 首次就诊通常1-3项核心检查即可\n"
            + "   - 症状明确且轻微：可不开检查，给予对症建议\n"
            + "   - 症状复杂或有警报信号：开具2-4项针对性检查\n"
            + "   - 避免「检查套餐」式的大规模筛查\n\n"
            + "3. 临床决策逻辑：\n"
            + f"   - 警报症状（必须重视）：{', '.join(alarm_keywords)}\n"
            + f"   - 常规基础检查参考：{', '.join(common_tests[:2])}（仅在必要时开具）\n"
            + "   - 影像学检查（CT/MRI）：仅在高度怀疑结构性病变时开具\n"
            + "   - 电生理检查（EEG/EMG）：仅在明确神经功能评估需求时开具\n\n"
            + "4. 决策示例：\n"
            + "   - 轻度头痛，无警报症状 → 不开检查，观察随访\n"
            + "   - 头痛伴呕吐、视物模糊 → 血常规+头颅CT（排除颅内病变）\n"
            + "   - 癫痫发作史 → EEG（评估异常放电）\n"
            + "   - 四肢麻木无力 → 肌电图（评估周围神经）\n\n"
            + "【患者信息】\n"
            + json.dumps(
                {
                    "chief_complaint": state.chief_complaint,
                    "history": state.history,
                    "exam_findings": state.exam_findings,
                    f"{dept}_interview": state.dept_payload.get(dept, {}).get("interview", {}),
                },
                ensure_ascii=False,
                indent=2
            )
            + "\n\n【参考知识】\n" + _chunks_for_prompt(chunks)
            + "\n\n【输出要求】必须严格按照以下JSON格式输出（不要遗漏任何逗号或括号）：\n"
            + "{\n"
            + "  \"need_aux_tests\": true/false,\n"
            + "  \"ordered_tests\": [\n"
            + "    {\n"
            + "      \"dept\": \"科室代码\",\n"
            + "      \"type\": \"lab\"/\"imaging\"/\"endoscopy\"/\"neurophysiology\",\n"
            + "      \"name\": \"检查名称\",\n"
            + "      \"reason\": \"开具原因\",\n"
            + "      \"priority\": \"urgent\"/\"routine\",\n"
            + "      \"need_prep\": true/false,\n"
            + "      \"need_schedule\": true/false\n"
            + "    }\n"
            + "  ],\n"
            + "  \"specialty_summary\": {\n"
            + "    \"problem_list\": [\"问题1\", \"问题2\"],\n"
            + "    \"assessment\": \"评估内容\",\n"
            + "    \"plan_direction\": \"计划方向\",\n"
            + "    \"red_flags\": [\"警报信号1\"]\n"
            + "  }\n"
            + "}\n\n"
            + "⚠️ 关键要求：\n"
            + "1. type字段必须是：lab/imaging/endoscopy/neurophysiology（小写英文）\n"
            + "2. need_prep和need_schedule必须是布尔值（true/false，小写）\n"
            + "3. 每个对象内部最后一个字段后面不要加逗号\n"
            + "4. 数组最后一个元素后面不要加逗号\n"
            + "5. 确保所有括号和引号正确配对\n"
            + "6. 检查name应简洁明了，不超过50个字符（如：\"血常规\"、\"头颅MRI\"、\"抗核抗体\"）"
        )
        
        # 检查LLM是否可用
        if llm is None:
            logger.error("⚠️  未LLM配置，无法生成检查方案")
            # 使用保守的fallback
            obj = {
                "need_aux_tests": False,
                "ordered_tests": [],
                "specialty_summary": {
                    "problem_list": [f"{dept_name}症状待评估"],
                    "assessment": "LLM不可用，无法生成检查方案",
                    "plan_direction": "需配置LLM",
                    "red_flags": []
                },
            }
            used_fallback = True
        else:
            # 优化fallback为保守策略
            obj, used_fallback, _raw = llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=lambda: {
                "need_aux_tests": False,  # 改为保守策略：不确定时不开单
                "ordered_tests": [],
                "specialty_summary": {
                    "problem_list": [f"{dept_name}症状待评估"],
                    "assessment": "信息不足，建议进一步问诊",
                    "plan_direction": "完善病史采集",
                    "red_flags": []
                },
            },
            temperature=0.2,
        )
        need_aux_tests = bool(obj.get("need_aux_tests", False))
        ordered = list(obj.get("ordered_tests") or [])
        summary = dict(obj.get("specialty_summary") or {})
        logger.info("  ✅ 检查方案生成完成")
        
        # 推进时间（医生初步判断与开单约5分钟）
        if state.world_context:
            state.world_context.advance_time(minutes=5)
            state.sync_physical_state()

        # 标准化检查项目（不做白名单过滤，完全信任LLM判断）
        normalized: list[dict[str, Any]] = []
        for t in ordered:
            if not isinstance(t, dict):
                continue
            normalized_test = _validate_and_normalize_test(t, dept, dept_config)
            if normalized_test:
                normalized.append(normalized_test)
            else:
                logger.warning(f"  ⚠️  检查项目 '{t.get('name')}' 标准化失败，跳过")
        
        ordered = normalized
        
        # 如果标准化后没有项目，更新状态
        if need_aux_tests and not ordered:
            logger.warning("  ⚠️  原计划开单但标准化后无有效项目，改为不开单")
            need_aux_tests = False
        
        # 更新状态
        state.need_aux_tests = need_aux_tests
        state.ordered_tests = ordered
        state.specialty_summary = summary
        
        decision = "需要辅助检查以明确诊断" if need_aux_tests else "暂无需辅助检查，给出对症方向"
        
        logger.info(f"\n  📋 开单决策: need_aux_tests={state.need_aux_tests}")
        if ordered:
            logger.info(f"  📝 开单项目 ({len(ordered)}项):")
            for test in ordered:
                logger.info(f"     - {test['name']} ({test['type']}) - {test.get('priority', 'routine')}")

        state.dept_payload.setdefault(dept, {})
        state.dept_payload[dept]["preliminary"] = {
            "need_aux_tests": state.need_aux_tests,
            "ordered_tests_count": len(ordered),
        }

        state.add_audit(
            make_audit_entry(
                node_name=f"S6 {dept_name} Preliminary Judgment",
                inputs_summary={"chief_complaint": state.chief_complaint, "dept": dept},
                outputs_summary={
                    "need_aux_tests": state.need_aux_tests,
                    "ordered_tests": [t["name"] for t in ordered],
                },
                decision=decision,
                chunks=chunks,
                flags=["LLM_PARSE_FALLBACK"] if used_fallback else ["LLM_USED"],
            )
        )
        
        # 详细日志输出节点执行摘要
        if detail_logger:
            detail_logger.info("")
            detail_logger.info("📤 S6 初步判断输出:")
            detail_logger.info(f"  • 需要辅助检查: {'是' if need_aux_tests else '否'}")
            if ordered:
                detail_logger.info(f"  • 开具检查: {len(ordered)}项")
                for test in ordered[:3]:  # 最多显示3项
                    detail_logger.info(f"      - {test['name']} ({test['type']})")
                if len(ordered) > 3:
                    detail_logger.info(f"      ... 还有 {len(ordered) - 3} 项")
            if summary:
                if summary.get('problem_list'):
                    detail_logger.info(f"  • 问题列表: {', '.join(summary['problem_list'][:3])}")
                if summary.get('assessment'):
                    detail_logger.info(f"  • 评估: {summary['assessment'][:80]}...")
            detail_logger.info(f"✅ S6 初步判断完成")
            detail_logger.info("")
        
        logger.info("✅ S6节点完成\n")
        return state

    # 构建图结构
    graph.add_node("S4", s4_specialty_interview)
    graph.add_node("S5", s5_physical_exam)
    graph.add_node("S6", s6_preliminary_judgment)

    graph.set_entry_point("S4")
    graph.add_edge("S4", "S5")
    graph.add_edge("S5", "S6")
    graph.add_edge("S6", END)
    
    return graph.compile()
