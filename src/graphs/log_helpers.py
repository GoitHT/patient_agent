"""
日志辅助函数 - 为 graphs 模块提供统一的日志输出工具
Log Helpers - Unified logging utilities for graph modules
"""
from typing import Any
from state.schema import BaseState
from logging_utils import should_log
from utils import get_logger

logger = get_logger("hospital_agent.graph")


def _log_node_start(node_name: str, node_desc: str, state: BaseState):
    """统一的节点开始日志输出
    
    Args:
        node_name: 节点名称（如"C1"）
        node_desc: 节点描述（如"开始"）
        state: 当前状态对象（会自动从state.world获取物理世界对象）
    """
    # 根据配置决定是否输出到终端
    if should_log(1, "common_opd_graph", node_name):
        logger.info(f"{node_name}: {node_desc}")
    
    # 详细日志总是记录
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    if detail_logger:
        detail_logger.info("")
        detail_logger.info(f"{'─'*80}")
        detail_logger.info(f"▶ {node_name}: {node_desc}")
        detail_logger.info(f"{'─'*80}")
        
        # 记录当前位置（转换为中文）
        if hasattr(state, 'current_location') and state.current_location:
            current_loc = state.current_location
            # 从state.world获取world对象
            world = getattr(state, 'world', None)
            # 如果有world对象，转换为中文名称
            if world:
                loc_name = world.get_location_name(current_loc)
                # 如果有dept_display_name属性，优先使用（用于诊室）
                if hasattr(state, 'dept_display_name') and state.dept_display_name:
                    loc_name = state.dept_display_name
            else:
                # 没有world对象时，直接使用位置ID或dept_display_name
                loc_name = getattr(state, 'dept_display_name', current_loc) if hasattr(state, 'dept_display_name') and state.dept_display_name else current_loc
            
            detail_logger.info(f"  📍 当前位置: {loc_name}")
        
        # 记录诊断状态
        if hasattr(state, 'diagnosis') and state.diagnosis:
            if isinstance(state.diagnosis, dict) and state.diagnosis.get('name'):
                detail_logger.info(f"  🔬 诊断状态: {state.diagnosis['name']}")
        
        # 记录检查状态
        if hasattr(state, 'ordered_tests') and state.ordered_tests:
            detail_logger.info(f"  📋 待检查: {len(state.ordered_tests)}项")
            for test in state.ordered_tests:
                test_name = test.get('name', '未知检查')
                test_type = test.get('type', 'unknown')
                detail_logger.info(f"    - {test_name} ({test_type})")
        if hasattr(state, 'test_results') and state.test_results:
            detail_logger.info(f"  🧪 已完成检查: {len(state.test_results)}项")


def _log_node_end(node_name: str, state: BaseState, outputs_summary: dict = None):
    """统一的节点结束日志输出
    
    Args:
        node_name: 节点名称
        state: 状态对象
        outputs_summary: 输出摘要（可选），例如 {"诊断": "偏头痛", "检查": "3项"}
    """
    if should_log(1, "common_opd_graph", node_name):
        logger.info(f"  ✅ {node_name}完成")
    
    # 详细日志记录节点输出
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    if detail_logger:
        if outputs_summary:
            detail_logger.info("")
            detail_logger.info("📤 节点输出:")
            for key, value in outputs_summary.items():
                detail_logger.info(f"  • {key}: {value}")
        detail_logger.info(f"✅ {node_name} 完成")
        detail_logger.info("")


def _log_detail(message: str, state: BaseState, level: int = 2, node_name: str = ""):
    """记录详细信息（只在详细日志中）"""
    # 终端只在高详细级别时输出
    if should_log(level, "common_opd_graph", node_name):
        logger.info(message)
    
    # 详细日志总是记录
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    if detail_logger:
        detail_logger.info(message)


def _log_physical_state(state: BaseState, node_name: str = "", level: int = 2):
    """统一的物理环境状态显示函数
    
    Args:
        state: 当前状态（会自动从state.world获取物理世界对象）
        node_name: 节点名称（用于日志标记）
        level: 日志级别
    """
    world = getattr(state, 'world', None)
    if not world or not state.patient_id:
        return
    
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    
    # 同步物理状态
    state.sync_physical_state()
    
    # 获取当前时间
    current_time = world.current_time.strftime('%H:%M')
    
    # 获取当前位置
    current_loc = state.current_location or world.get_agent_location(state.patient_id)
    loc_name = world.get_location_name(current_loc) if current_loc else "未知位置"
    
    # 如果有dept_display_name属性，优先使用（用于诊室）
    if hasattr(state, 'dept_display_name') and state.dept_display_name:
        loc_name = state.dept_display_name
    
    # 输出物理环境信息
    _log_detail(f"\n🏥 物理环境状态:", state, level, node_name)
    _log_detail(f"  🕐 时间: {current_time}", state, level, node_name)
    _log_detail(f"  📍 位置: {loc_name}", state, level, node_name)


def _log_rag_retrieval(
    query: str,
    chunks: list[dict[str, Any]],
    state: BaseState,
    filters: dict[str, Any] | None = None,
    node_name: str = "",
    level: int = 2,
    purpose: str = "检索"
):
    """详细记录 RAG 检索过程和结果
    
    Args:
        query: 查询文本
        chunks: 检索结果列表
        state: 状态对象
        filters: 过滤条件
        node_name: 节点名称
        level: 日志级别
        purpose: 检索目的描述（如"专科知识"，"历史记录"等）
    """
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    if not detail_logger:
        return
    
    # 记录检索请求
    detail_logger.info(f"\n📖 RAG {purpose}检索:")
    detail_logger.info(f"  🔍 查询: {query}")
    
    # 根据filters推断查询的目标数据库
    target_dbs = _infer_target_databases(filters, state)
    if target_dbs:
        detail_logger.info(f"  🗄️  目标库: {', '.join(target_dbs)}")
    
    # 记录过滤条件
    if filters:
        filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items() if v])
        if filter_desc:
            detail_logger.info(f"  🎯 过滤: {filter_desc}")
    
    # 记录检索结果统计
    if not chunks:
        detail_logger.info(f"  ℹ️  未检索到相关内容")
        return
    
    detail_logger.info(f"  ✅ 检索到 {len(chunks)} 个知识片段")
    
    # 统计各数据库来源
    db_sources = {}
    for chunk in chunks:
        meta = chunk.get('meta', {})
        source = meta.get('source', 'unknown')
        db_sources[source] = db_sources.get(source, 0) + 1
    
    if db_sources:
        detail_logger.info(f"  📊 数据来源:")
        db_name_map = {
            'MedicalGuide': '医学指南库 (MedicalGuide_db)',
            'ClinicalCase': '临床案例库 (ClinicalCase_db)',
            'HighQualityQA': '高质量问答库 (HighQualityQA_db)',
            'UserHistory': '患者历史库 (UserHistory_db)',
            'unknown': '未知来源'
        }
        for source, count in sorted(db_sources.items(), key=lambda x: -x[1]):
            source_name = db_name_map.get(source, f'{source}库')
            detail_logger.info(f"     • {source_name}: {count}条")
    else:
        # 如果没有source信息，记录警告
        detail_logger.info(f"  ⚠️  未能识别数据来源")
    
    # 记录前3条高质量结果的详细信息
    detail_logger.info(f"  📝 相关内容预览（前3条）:")
    for i, chunk in enumerate(chunks[:3], 1):
        score = chunk.get('score', 0.0)
        text = chunk.get('text', '')
        meta = chunk.get('meta', {})
        source = meta.get('source', 'unknown')
        
        # 截取文本预览（最多100字）
        preview = text[:100].replace('\n', ' ').strip()
        if len(text) > 100:
            preview += '...'
        
        # 格式化相关度显示
        relevance = "高" if score > 0.8 else "中" if score > 0.6 else "低"
        
        detail_logger.info(f"     [{i}] 相关度: {relevance} ({score:.3f})")
        detail_logger.info(f"         内容: {preview}")
        
        # 如果有特殊元数据，也记录
        if 'dept' in meta:
            detail_logger.info(f"         科室: {meta['dept']}")
        if 'type' in meta:
            detail_logger.info(f"         类型: {meta['type']}")
    
    # 如果检索结果超过3条，显示统计
    if len(chunks) > 3:
        detail_logger.info(f"     ... 及其他 {len(chunks) - 3} 条结果")


def _infer_target_databases(filters: dict[str, Any] | None, state: BaseState) -> list[str]:
    """根据过滤条件推断将要查询的目标数据库
    
    Args:
        filters: 过滤条件字典
        state: 状态对象
        
    Returns:
        目标数据库名称列表
    """
    if not filters:
        # 默认策略：提示用户应该指定 db_name
        return ["未指定数据库"]
    
    # 【优先策略】如果明确指定了 db_name，只返回该数据库
    db_name = filters.get("db_name")
    if db_name:
        db_name_map = {
            "HospitalProcess_db": "规则流程库",
            "MedicalGuide_db": "医学指南库",
            "ClinicalCase_db": "临床案例库",
            "HighQualityQA_db": "高质量问答库",
            "UserHistory_db": "患者历史库",
        }
        return [db_name_map.get(db_name, db_name)]
    
    # 如果没有指定 db_name，显示警告
    return ["⚠️ 未指定 db_name"]
