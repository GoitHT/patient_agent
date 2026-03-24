"""通用专科子图：支持所有科室的专科问诊、体检、初步判断"""
from __future__ import annotations

import json
import random
from typing import Any

from langgraph.graph import END, StateGraph

from graphs.log_helpers import _log_detail
from rag import AdaptiveRAGRetriever, DialogueQualityEvaluator
from rag.query_optimizer import QueryContext, get_query_optimizer
from rag.keyword_generator import RAGKeywordGenerator, NodeContext
from services.llm_client import LLMClient
from state.schema import BaseState, make_audit_entry
from utils import load_prompt, contains_any_positive, get_logger
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
    retriever: AdaptiveRAGRetriever,
    llm: LLMClient | None = None,
    doctor_agent=None, 
    patient_agent=None, 
    max_questions: int = 3,  # 最底层默认值，通常从config.yaml传入
    enable_eval: bool = True,  # 是否对照参考数据评估开单准确率
):
    """构建通用专科子图，适用于所有科室
    
    Args:
        max_questions: 医生最多问诊次数（从config.agent.max_questions传入）
    """
    graph = StateGraph(BaseState)
    
    # 注意：不在这里判断 use_agents，而是在节点执行时动态判断
    # 因为 doctor_agent 是在 C4 节点中动态分配到 state 的

    def s1_specialty_interview(state: BaseState) -> BaseState:
        """S1: 通用专科问诊节点"""
        dept = state.dept
        dept_config = DEPT_CONFIG.get(dept, DEPT_CONFIG.get("internal_medicine", {}))
        dept_name = dept_config.get("name", "通用科室")
        
        # 终端简洁输出
        if should_log(1, "specialty_subgraph", "S1"):
            logger.info(f"🏫 S1: {dept_name}专科问诊")
        
        # 详细日志记录
        detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
        if detail_logger:
            detail_logger.section(f"{dept_name}专科问诊")
        
        # 从 state 中获取医生 Agent（C4 节点中分配的）
        doctor_agent = getattr(state, 'doctor_agent', None)
        patient_agent = getattr(state, 'patient_agent', None)
        if doctor_agent is None or patient_agent is None:
            raise RuntimeError("S1 节点缺少 DoctorAgent 或 PatientAgent，请检查 C4 节点是否正确分配医生")
        
        use_agents = True
        if hasattr(state, 'assigned_doctor_name'):
            logger.info(f"  👨‍⚕️ 使用 C4 分配的医生 Agent: {state.assigned_doctor_name}")
        else:
            logger.info(f"  👨‍⚕️ 使用医生 Agent 进行问诊")
        
        # 如果是Agent模式，确保医生智能体的科室设置正确
        if use_agents and doctor_agent:
            doctor_agent.dept = dept
            logger.info(f"  🏥 医生科室: {dept_name}")
        
        # 检索该科室的专科知识
        # 注意：此时chief_complaint还未设置（医生尚未从患者处获得），使用科室信息检索
        _log_detail(f"🔍 检索{dept_name}专科知识库...", state, 2, "S1")
        
        # 使用关键词生成器构建节点上下文（优先使用医生主诉，回退到原始主诉）
        keyword_generator = RAGKeywordGenerator()
        complaint_seed = state.chief_complaint or state.original_chief_complaint
        node_ctx = NodeContext(
            node_id="S1",
            node_name="专科问诊",
            dept=dept,
            dept_name=dept_name,
            chief_complaint=complaint_seed,
            patient_age=state.patient_profile.get("age") if state.patient_profile else None,
            patient_gender=state.patient_profile.get("gender") if state.patient_profile else None,
        )
        
        # 【增强RAG】1. 检索专科知识库（使用关键词生成器）
        # 【单一数据库检索】只查询医学指南库(MedicalGuide_db) - 检索专科基础知识、Red Flags、鉴别诊断
        query = keyword_generator.generate_keywords(node_ctx, "MedicalGuide_db")
        chunks = retriever.retrieve(query, filters={"db_name": "MedicalGuide_db"}, k=4)
        state.add_retrieved_chunks(chunks)
        from graphs.log_helpers import _log_rag_retrieval
        _log_rag_retrieval(query, chunks, state, filters={"db_name": "MedicalGuide_db"}, node_name="S1", purpose=f"{dept_name}专科知识[医学指南库]")
        
        # 【增强RAG】2. 检索高质量问诊库（使用关键词生成器）
        # 【单一数据库检索】只查询高质量问诊库(HighQualityQA_db) - 检索推荐的问诊问题
        qa_query = keyword_generator.generate_keywords(node_ctx, "HighQualityQA_db")
        qa_chunks = retriever.retrieve(
            qa_query,
            filters={"db_name": "HighQualityQA_db"},
            k=3
        )
        # 无论是否有结果，都记录检索日志
        from graphs.log_helpers import _log_rag_retrieval
        _log_rag_retrieval(qa_query, qa_chunks, state, filters={"db_name": "HighQualityQA_db"}, node_name="S1", purpose="高质量问诊参考[高质量问诊库]")
        if qa_chunks:
            state.add_retrieved_chunks(qa_chunks)
        
        # 【增强RAG】3. 检索相似症状的临床案例（使用关键词生成器）
        # 【单一数据库检索】只查询临床案例库(ClinicalCase_db) - 检索相似症状的患者案例
        if state.patient_id:
            case_query = keyword_generator.generate_keywords(node_ctx, "ClinicalCase_db")
            case_chunks = retriever.retrieve(
                case_query,
                filters={"db_name": "ClinicalCase_db"},
                k=2
            )
            # 无论是否有结果，都记录检索日志
            from graphs.log_helpers import _log_rag_retrieval
            _log_rag_retrieval(case_query, case_chunks, state, filters={"db_name": "ClinicalCase_db"}, node_name="S1", purpose="临床案例参考[临床案例库]")
            if case_chunks:
                state.add_retrieved_chunks(case_chunks)

        cc = state.chief_complaint
        
        # 获取科室配置用于提示词
        alarm_keywords = dept_config.get("alarm_keywords", [])
        interview_keys = dept_config.get("interview_keys", ["symptoms_detail"])

        # 获取节点专属计数器
        node_key = f"s1_{dept}"
        
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
            
            # 【新增】初始化对话质量评估器
            qa_evaluator = None
            qa_scores = []  # 存储每轮对话的质量评分
            high_quality_count = 0  # 高质量对话计数
            
            if hasattr(state, 'retriever') and state.retriever:
                try:
                    # 获取SPLLM根目录
                    import sys
                    from pathlib import Path
                    spllm_root = None
                    for path in sys.path:
                        candidate = Path(path).parent / "SPLLM-RAG1"
                        if candidate.exists():
                            spllm_root = candidate
                            break
                    
                    if not spllm_root:
                        # 尝试从retriever获取
                        if hasattr(retriever, 'spllm_root'):
                            spllm_root = retriever.spllm_root
                    
                    if spllm_root:
                        qa_evaluator = DialogueQualityEvaluator(
                            llm=llm if llm else None,
                            spllm_root=spllm_root,
                            high_quality_threshold=0.7
                        )
                        logger.info("  ✅ 对话质量评估器已启用")
                except Exception as e:
                    logger.warning(f"  ⚠️  对话质量评估器初始化失败: {e}")
            
            for i in range(remaining_questions):
                # 终端只显示简洁信息
                if should_log(1, "specialty_subgraph", "S1"):
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
                    # 【增强】传入检索到的知识片段（包括高质量问诊库）作为参考
                    # 注意：不直接使用state.chief_complaint，而是使用doctor_agent已收集的信息
                    question = doctor_agent.generate_one_question(
                        chief_complaint=doctor_agent.collected_info.get("chief_complaint", ""),
                        context=context_desc,
                        rag_chunks=chunks + qa_chunks + case_chunks  # 合并所有检索结果
                    )
                
                if not question:
                    if should_log(1, "specialty_subgraph", "S1"):
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
                
                # 【新增】对话质量评估与存储
                if qa_evaluator and question and answer:
                    try:
                        # 准备患者信息（用于忠实性评估）
                        patient_info = {
                            "chief_complaint": state.chief_complaint,
                            "history": state.history,
                            "patient_profile": state.patient_profile,
                        }
                        
                        # 准备问诊上下文（用于医生提问评估）
                        context = {
                            "dept": dept,
                            "dept_name": dept_name,
                            "stage": "specialty_interview",
                            "collected_info": doctor_agent.collected_info if doctor_agent else {}
                        }
                        
                        # 评估对话质量
                        dialogue_score = qa_evaluator.evaluate_dialogue(
                            question=question,
                            answer=answer,
                            patient_info=patient_info,
                            context=context
                        )
                        
                        qa_scores.append(dialogue_score)
                        
                        # 如果是高质量对话，存储到向量库
                        if dialogue_score.is_high_quality():
                            success = qa_evaluator.store_high_quality_dialogue(
                                dialogue_score=dialogue_score,
                                patient_id=state.patient_id,
                                metadata={
                                    "dept": dept,
                                    "stage": "specialty_interview",
                                    "round": questions_asked_this_node + i + 1
                                }
                            )
                            if success:
                                high_quality_count += 1
                        
                        # 详细日志（仅在debug级别显示）
                        if should_log(3, "specialty_subgraph", "S1"):
                            logger.debug(
                                f"  📊 Q{i+1} 质量评分: "
                                f"医生={dialogue_score.doctor_metrics.quality:.2f}, "
                                f"患者={dialogue_score.patient_metrics.ability:.2f}, "
                                f"综合={dialogue_score.overall_score:.2f}"
                            )
                    except Exception as e:
                        logger.warning(f"  ⚠️  对话质量评估失败 (Q{i+1}): {e}")
                
                # 更新该节点和全局计数器
                state.node_qa_counts[node_key] = questions_asked_this_node + i + 1
                state.node_qa_counts["global_total"] = global_qa_count + i + 1
            
            state.agent_interactions["doctor_patient_qa"] = qa_list
            
            # ===== 物理环境集成：问诊后更新物理状态 =====
            if state.world_context:
                qa_count = len([qa for qa in qa_list if qa.get('stage') == f"{dept}_specialty"])
                if qa_count > 0:
                    duration = qa_count * 1  # 每轮约3分钟
                    energy_cost = 0.5 * qa_count  # 每轮消耗0.5体力
                    
                    logger.info(f"\n{'─'*60}")
                    logger.info(f"🌍 物理环境模拟 - 问诊过程")
                    logger.info(f"{'─'*60}")
                    start_time = state.world_context.patient_current_time(state.patient_id).strftime('%H:%M')
                    
                    result = state.update_physical_world(
                        action="consult",
                        duration_minutes=duration,
                        energy_cost=energy_cost
                    )
                    end_time = state.world_context.patient_current_time(state.patient_id).strftime('%H:%M')
                    
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
            
            # 【新增】展示对话质量统计
            if qa_scores:
                logger.info(f"\n{'━'*60}")
                logger.info("💎 对话质量统计")
                logger.info(f"{'━'*60}")
                
                # 计算平均分数
                avg_doctor_quality = sum(s.doctor_metrics.quality for s in qa_scores) / len(qa_scores)
                avg_patient_ability = sum(s.patient_metrics.ability for s in qa_scores) / len(qa_scores)
                avg_overall = sum(s.overall_score for s in qa_scores) / len(qa_scores)
                
                logger.info(f"  📈 对话轮数: {len(qa_scores)} 轮")
                logger.info(f"  👨‍⚕️  医生平均质量: {avg_doctor_quality:.2f}/1.0")
                logger.info(f"     • 具体性: {sum(s.doctor_metrics.specificity for s in qa_scores) / len(qa_scores):.2f}")
                logger.info(f"     • 针对性: {sum(s.doctor_metrics.targetedness for s in qa_scores) / len(qa_scores):.2f}")
                logger.info(f"     • 专业性: {sum(s.doctor_metrics.professionalism for s in qa_scores) / len(qa_scores):.2f}")
                logger.info(f"  👤 患者平均能力: {avg_patient_ability:.2f}/1.0")
                logger.info(f"     • 相关性: {sum(s.patient_metrics.relevance for s in qa_scores) / len(qa_scores):.2f}")
                logger.info(f"     • 忠实性: {sum(s.patient_metrics.faithfulness for s in qa_scores) / len(qa_scores):.2f}")
                logger.info(f"     • 鲁棒性: {sum(s.patient_metrics.robustness for s in qa_scores) / len(qa_scores):.2f}")
                logger.info(f"  🎯 综合得分: {avg_overall:.2f}/1.0")
                
                if high_quality_count > 0:
                    logger.info(f"  ✨ 高质量对话: {high_quality_count}/{len(qa_scores)} 轮已存入知识库")
                else:
                    logger.info(f"  ℹ️  本次问诊暂无高质量对话达到存储阈值")
                
                # 保存质量评分到state（用于审计）
                state.agent_interactions["qa_quality_scores"] = {
                    "total_rounds": len(qa_scores),
                    "avg_doctor_quality": avg_doctor_quality,
                    "avg_patient_ability": avg_patient_ability,
                    "avg_overall_score": avg_overall,
                    "high_quality_count": high_quality_count,
                    "detailed_scores": [s.to_dict() for s in qa_scores]
                }
                
                logger.info(f"{'━'*60}\n")
            
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
            if should_log(2, "specialty_subgraph", "S1"):
                logger.warning(f"  ⚠️  发现警报症状: {', '.join(str(a) for a in alarm_list)}")

        # 记录节点问答轮数
        node_qa_turns = state.node_qa_counts.get(node_key, 0)
        
        # ===== 保存问诊记录到数据库 =====
        if hasattr(state, 'medical_record_integration') and state.medical_record_integration:
            state.medical_record_integration.on_doctor_consultation(state, doctor_id="doctor_001")
        
        state.add_audit(
            make_audit_entry(
                node_name=f"S1 {dept_name} Specialty Interview",
                inputs_summary={"chief_complaint": state.chief_complaint, "use_agents": use_agents, "dept": dept, "max_questions": max_questions},
                outputs_summary={"alarm_symptoms": alarm_list, "node_qa_turns": node_qa_turns},
                decision=f"完成{dept_name}专科问诊（本节点{node_qa_turns}轮）（Agent模式）",
                chunks=chunks,
                flags=["AGENT_MODE"],
            )
        )
        
        # 详细日志输出节点执行摘要
        if detail_logger:
            detail_logger.info("")
            detail_logger.info("📤 S1 专科问诊输出:")
            detail_logger.info(f"  • 问诊轮数: {node_qa_turns}轮")
            if alarm_list:
                detail_logger.info(f"  • 危险症状: {', '.join(alarm_list)}")
            if use_agents and doctor_agent:
                collected = doctor_agent.collected_info
                if collected.get('chief_complaint'):
                    detail_logger.info(f"  • 主诉: {collected['chief_complaint']}")
                if collected.get('history', {}).get('duration'):
                    detail_logger.info(f"  • 病程: {collected['history']['duration']}")
            detail_logger.info(f"✅ S1 专科问诊完成")
            detail_logger.info("")
        
        if should_log(1, "specialty_subgraph", "S1"):
            logger.info(f"  ✅ S1完成\n")
        # 记录 S1 完成时的模拟时钟（供就诊时间线使用）
        if state.world_context and isinstance(state.appointment, dict):
            state.appointment["_s1_end_time"] = state.world_context.patient_current_time(state.patient_id).strftime('%H:%M')
        return state

    def s2_physical_exam(state: BaseState) -> BaseState:
        """S2: 通用体检节点"""
        dept = state.dept
        dept_config = DEPT_CONFIG.get(dept, DEPT_CONFIG.get("internal_medicine", {}))
        dept_name = dept_config.get("name", "通用")
        exam_area = dept_config.get("exam_area", "general")
        alarm_keywords = dept_config.get("alarm_keywords", [])
        
        # 获取详细日志记录器
        detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
        
        if should_log(1, "specialty_subgraph", "S2"):
            logger.info(f"� S2: {dept_name}体格检查")
        
        if detail_logger:
            detail_logger.section(f"{dept_name}体格检查")
        
        # 优先使用新结构化字段中的体格检查数据，缺失时再使用LLM生成
        case_data = state.case_data if isinstance(state.case_data, dict) else {}
        structured_exam_fields = {
            "vital_signs": case_data.get("体格检查_生命体征", ""),
            "skin_mucosa": case_data.get("体格检查_皮肤黏膜", ""),
            "superficial_lymph_nodes": case_data.get("体格检查_浅表淋巴结", ""),
            "head_neck": case_data.get("体格检查_头颈部", ""),
            "cardiopulmonary_vascular": case_data.get("体格检查_心肺血管", ""),
            "abdomen": case_data.get("体格检查_腹部", ""),
            "spine_limbs": case_data.get("体格检查_脊柱四肢", ""),
            "nervous_system": case_data.get("体格检查_神经系统", ""),
        }
        has_structured_exam = any(str(v).strip() for v in structured_exam_fields.values())
        data_source = "dataset_structured_fields" if has_structured_exam else "llm_generated"
        real_physical_exam = structured_exam_fields if has_structured_exam else None

        logger.info("📋 使用数据集结构化体检结果" if has_structured_exam else "📋 使用LLM生成体检结果")
        
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
        
        # 推进时间（体格检查约5分钟）- 在LLM调用前推进，避免并发时钟漂移影响时间戳
        if state.world_context:
            state.world_context.advance_time(minutes=5, patient_id=state.patient_id)
            state.sync_physical_state()
            if isinstance(state.appointment, dict):
                state.appointment["_s2_end_time"] = state.world_context.patient_current_time(state.patient_id).strftime('%H:%M')

        used_fallback = False
        if has_structured_exam:
            exam = {
                "vital_signs": structured_exam_fields["vital_signs"],
                "general": "",
                f"{exam_area}_exam": {
                    "皮肤黏膜": structured_exam_fields["skin_mucosa"],
                    "浅表淋巴结": structured_exam_fields["superficial_lymph_nodes"],
                    "头颈部": structured_exam_fields["head_neck"],
                    "心肺血管": structured_exam_fields["cardiopulmonary_vascular"],
                    "腹部": structured_exam_fields["abdomen"],
                    "脊柱四肢": structured_exam_fields["spine_limbs"],
                    "神经系统": structured_exam_fields["nervous_system"],
                },
                "positive_signs": [],
                "negative_signs": [],
                "source": data_source,
            }
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
                for sign in negative_signs:
                    detail_logger.info(f"    - {sign}")
        
        state.exam_findings.setdefault(exam_area, {})
        state.exam_findings[exam_area] = exam

        state.add_audit(
            make_audit_entry(
                node_name=f"S2 {dept_name} Physical Exam",
                inputs_summary={"exam_area": exam_area, "dept": dept, "has_real_data": bool(real_physical_exam)},
                outputs_summary={"exam_completed": True, "data_source": exam.get("source", "unknown")},
                decision=f"完成{dept_name}体格检查记录" + ("（使用数据集真实数据）" if real_physical_exam else "（LLM生成）"),
                chunks=[],
                flags=["REAL_DATA"] if real_physical_exam else (["LLM_PARSE_FALLBACK"] if used_fallback else ["LLM_USED"]),
            )
        )
        
        # 患者日志总结
        if detail_logger:
            detail_logger.info(f"✅ S2 {dept_name}体格检查完成")
            detail_logger.info("")
        
        logger.info("✅ S2节点完成\n")
        return state

    def s3_preliminary_judgment(state: BaseState) -> BaseState:
        """S3: 通用初步判断与开单节点"""
        dept = state.dept
        dept_config = DEPT_CONFIG.get(dept, DEPT_CONFIG.get("internal_medicine", {}))
        dept_name = dept_config.get("name", "通用")
        alarm_keywords = dept_config.get("alarm_keywords", [])
        common_tests = dept_config.get("common_tests", ["血常规"])
                # 获取详细日志记录器
        detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
        logger.info("\n" + "="*60)
        logger.info(f"📊 S3: {dept_name}初步判断")
        logger.info("="*60)
        
        # 【增强RAG】S3: 检索医学指南库 + 临床案例库（使用关键词生成器）
        # S3节点用途：综合患者信息和医学指南和相关案例得出是否需要辅助检查
        # 使用：医学指南库(MedicalGuide_db) - 检索专科诊疗指南、检查指征、适应症
        #      临床案例库(ClinicalCase_db) - 检索相似案例
        
        # 使用关键词生成器
        keyword_generator = RAGKeywordGenerator()
        node_ctx = NodeContext(
            node_id="S3",
            node_name="初步判断",
            dept=dept,
            dept_name=dept_name,
            chief_complaint=state.chief_complaint,
            patient_age=state.patient_profile.get("age") if state.patient_profile else None,
            patient_gender=state.patient_profile.get("gender") if state.patient_profile else None,
            specialty_summary=state.specialty_summary,
        )
        
        # 1. 检索医学指南库（使用关键词生成器）
        query = keyword_generator.generate_keywords(node_ctx, "MedicalGuide_db")
        logger.info(f"🔍 检索{dept_name}检查指南...")
        _log_detail(f"\n🔍 检索{dept_name}诊疗指南与检查指征[医学指南库]...", state, 2, "S3")
        
        # 【单一数据库检索】只查询医学指南库
        chunks_guide = retriever.retrieve(query, filters={"db_name": "MedicalGuide_db"}, k=4)
        state.add_retrieved_chunks(chunks_guide)
        
        # 使用详细的 RAG 日志记录
        from graphs.log_helpers import _log_rag_retrieval
        _log_rag_retrieval(query, chunks_guide, state, 
                         filters={"db_name": "MedicalGuide_db"}, 
                         node_name="S3", 
                         purpose=f"{dept_name}诊疗指南与检查指征[医学指南库]")
        
        # 2. 检索临床案例库（使用关键词生成器）
        case_query = keyword_generator.generate_keywords(node_ctx, "ClinicalCase_db")
        _log_detail(f"\n🔍 检索相似临床案例[临床案例库]...", state, 2, "S3")
        
        # 【单一数据库检索】只查询临床案例库
        chunks_cases = retriever.retrieve(
            case_query,
            filters={"db_name": "ClinicalCase_db"},
            k=3
        )
        state.add_retrieved_chunks(chunks_cases)
        
        _log_rag_retrieval(case_query, chunks_cases, state, 
                         filters={"db_name": "ClinicalCase_db"}, 
                         node_name="S3", 
                         purpose=f"相似临床案例[临床案例库]")
        
        # 合并所有检索结果
        chunks = chunks_guide + chunks_cases
        _log_detail(f"  ✅ 共检索到 {len(chunks)} 个知识片段", state, 2, "S3")

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
        
        # 推进时间（医生初步判断与开单约5分钟）- 在LLM调用前推进，避免并发时钟漂移影响时间戳
        if state.world_context:
            state.world_context.advance_time(minutes=5, patient_id=state.patient_id)
            state.sync_physical_state()
            if isinstance(state.appointment, dict):
                state.appointment["_s3_end_time"] = state.world_context.patient_current_time(state.patient_id).strftime('%H:%M')

        # 优化fallback为保守策略
        obj, used_fallback, _raw = llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=lambda: {
                "need_aux_tests": False,
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

        # ── S3 评估：对照 medical_data["辅助检查"] 评估开单的数量与准确率 ──
        ref_aux_exam = str(state.medical_data.get("辅助检查", "")).strip() if state.medical_data else ""
        s3_eval: dict[str, Any] = {}
        if enable_eval and ref_aux_exam:
            import re as _re
            # 按常见中英文分隔符拆分参考检查项，过滤掉长描述文本（保留检查名称关键词）
            ref_items = [
                item.strip()
                for item in _re.split(r'[；;、，,\n\r]+', ref_aux_exam)
                if 2 <= len(item.strip()) <= 30
            ]

            ordered_names = [t.get("name", "") for t in ordered]
            hit_list: list[str] = []
            miss_list: list[str] = []
            for ref_item in ref_items:
                matched = any(ref_item in name or name in ref_item for name in ordered_names)
                if matched:
                    hit_list.append(ref_item)
                else:
                    miss_list.append(ref_item)

            ref_count = len(ref_items)
            ordered_count = len(ordered)
            hit_count = len(hit_list)
            # 覆盖率（召回率）：参考检查中被正确开具的比例
            coverage_rate = round(hit_count / ref_count, 4) if ref_count > 0 else 0.0
            # 精准率：已开单中与参考匹配的比例（防过度开单）
            precision_rate = round(hit_count / ordered_count, 4) if ordered_count > 0 else 0.0

            s3_eval = {
                "ref_aux_exam_text": ref_aux_exam,
                "ref_items": ref_items,
                "ref_count": ref_count,
                "ordered_count": ordered_count,
                "hit_count": hit_count,
                "hit_items": hit_list,
                "miss_items": miss_list,
                "coverage_rate": coverage_rate,
                "precision_rate": precision_rate,
            }

            logger.info(f"\n📊 S3 开单评估（对照参考辅助检查）:")
            logger.info(f"  参考检查项数: {ref_count}  |  实际开单: {ordered_count}  |  命中: {hit_count}")
            logger.info(f"  覆盖率: {coverage_rate:.0%}  |  精准率: {precision_rate:.0%}")
            if hit_list:
                logger.info(f"  ✅ 命中: {', '.join(hit_list)}")
            if miss_list:
                logger.info(f"  ❌ 遗漏: {', '.join(miss_list)}")
            if detail_logger:
                detail_logger.info("")
                detail_logger.info("📊 S3 开单评估（对照参考辅助检查）:")
                detail_logger.info(f"  • 参考检查项数: {ref_count}  |  实际开单: {ordered_count}  |  命中: {hit_count}")
                detail_logger.info(f"  • 覆盖率(召回率): {coverage_rate:.0%}  |  精准率: {precision_rate:.0%}")
                if hit_list:
                    detail_logger.info(f"  • 命中项: {', '.join(hit_list[:5])}")
                if miss_list:
                    detail_logger.info(f"  • 遗漏项: {', '.join(miss_list[:5])}")

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
        if s3_eval:
            state.dept_payload[dept]["s3_eval"] = s3_eval

        state.add_audit(
            make_audit_entry(
                node_name=f"S3 {dept_name} Preliminary Judgment",
                inputs_summary={"chief_complaint": state.chief_complaint, "dept": dept},
                outputs_summary={
                    "need_aux_tests": state.need_aux_tests,
                    "ordered_tests": [t["name"] for t in ordered],
                    **(
                        {
                            "s3_eval_ordered_count": s3_eval["ordered_count"],
                            "s3_eval_ref_count": s3_eval["ref_count"],
                            "s3_eval_hit_count": s3_eval["hit_count"],
                            "s3_eval_coverage_rate": s3_eval["coverage_rate"],
                            "s3_eval_precision_rate": s3_eval["precision_rate"],
                        }
                        if s3_eval else {}
                    ),
                },
                decision=decision,
                chunks=chunks,
                flags=["LLM_PARSE_FALLBACK"] if used_fallback else ["LLM_USED"],
            )
        )
        
        # 详细日志输出节点执行摘要
        if detail_logger:
            detail_logger.info("")
            detail_logger.info("📤 S3 初步判断输出:")
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
            if s3_eval:
                detail_logger.info(
                    f"  • 开单评估: 覆盖率 {s3_eval['coverage_rate']:.0%}"
                    f" | 精准率 {s3_eval['precision_rate']:.0%}"
                    f" | 命中 {s3_eval['hit_count']}/{s3_eval['ref_count']}"
                )
            detail_logger.info(f"✅ S3 初步判断完成")
            detail_logger.info("")
        
        logger.info("✅ S3节点完成\n")
        return state

    # 构建图结构
    graph.add_node("S1", s1_specialty_interview)
    graph.add_node("S2", s2_physical_exam)
    graph.add_node("S3", s3_preliminary_judgment)

    graph.set_entry_point("S1")
    graph.add_edge("S1", "S2")
    graph.add_edge("S2", "S3")
    graph.add_edge("S3", END)
    
    return graph.compile()
