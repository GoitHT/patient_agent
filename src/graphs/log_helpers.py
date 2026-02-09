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
