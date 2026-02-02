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
    """统一的节点开始日志输出"""
    # 根据配置决定是否输出
    if should_log(1, "common_opd_graph", node_name):
        logger.info(f"{node_name}: {node_desc}")
    
    # 详细日志总是记录
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    if detail_logger:
        detail_logger.subsection(f"{node_name}: {node_desc}")


def _log_node_end(node_name: str, state: BaseState):
    """统一的节点结束日志输出"""
    if should_log(1, "common_opd_graph", node_name):
        logger.info(f"  ✅ {node_name}完成")


def _log_detail(message: str, state: BaseState, level: int = 2, node_name: str = ""):
    """记录详细信息（只在详细日志中）"""
    # 终端只在高详细级别时输出
    if should_log(level, "common_opd_graph", node_name):
        logger.info(message)
    
    # 详细日志总是记录
    detail_logger = state.patient_detail_logger if hasattr(state, 'patient_detail_logger') else None
    if detail_logger:
        detail_logger.info(message)


def _log_physical_state(state: BaseState, world: Any, node_name: str = "", level: int = 2):
    """统一的物理环境状态显示函数
    
    Args:
        state: 当前状态
        world: 物理世界对象
        node_name: 节点名称（用于日志标记）
        level: 日志级别
    """
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
    
    # 患者状态
    if state.patient_id in world.physical_states:
        ps = world.physical_states[state.patient_id]
        _log_detail(f"  👤 患者: 体力{ps.energy_level:.1f}/10 | 疼痛{ps.pain_level:.1f}/10", state, level, node_name)
    
    # 医生状态（如果已分配医生）
    if hasattr(state, 'assigned_doctor_id') and state.assigned_doctor_id:
        if state.assigned_doctor_id in world.physical_states:
            ds = world.physical_states[state.assigned_doctor_id]
            efficiency = ds.get_work_efficiency() * 100
            eff_icon = "🟢" if efficiency > 80 else ("🟡" if efficiency > 60 else "🔴")
            _log_detail(f"  👨‍⚕️ 医生: 体力{ds.energy_level:.1f}/10 | 负荷{ds.work_load:.1f}/10 | 效率{efficiency:.0f}% {eff_icon}", state, level, node_name)
