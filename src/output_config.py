"""
输出级别配置 - 控制终端输出的详细程度

使用方法：
1. 修改 DEFAULT_OUTPUT_LEVEL 来控制全局输出级别
2. 修改 NODE_OUTPUT_LEVELS 来控制特定节点的输出
3. 修改 SUPPRESS_UNCHECKED_LOGS 来控制是否抑制未检查的日志

输出级别说明：
- 0 = 静默（不显示）
- 1 = 最简（仅显示节点名称）
- 2 = 简洁（节点 + 关键状态）
- 3 = 详细（所有信息）
"""

import logging

# ========== 全局设置 ==========
# 推荐值：0（完全静默）或 1（仅节点名称）
DEFAULT_OUTPUT_LEVEL = 0  # 默认静默，所有详细信息在患者日志文件中

# 是否抑制未被should_log()检查的日志输出
# True: 只有经过should_log()检查的日志才会输出（推荐）
# False: 所有logger.info都会输出（会导致终端混乱）
SUPPRESS_UNCHECKED_LOGS = True

# ========== 模块级别设置 ==========
MODULE_OUTPUT_LEVELS = {
    "langgraph_multi_patient": 1,  # 患者执行器
    "specialty_subgraph": 1,  # 专科子图
    "common_opd_graph": 0,  # 通用门诊流程（静默）
}

# ========== 节点级别设置 ==========
# 可以为特定节点设置不同的输出级别
NODE_OUTPUT_LEVELS = {
    # === 通用流程节点 ===
    "C1": 0,  # 开始
    "C2": 0,  # 预约挂号
    "C3": 0,  # 签到候诊
    "C4": 0,  # 叫号入诊
    "C5": 0,  # 问诊准备
    "C6": 0,  # 专科流程调度
    "C7": 0,  # 路径决策
    "C8": 0,  # 开单说明
    "C9": 0,  # 缴费预约
    "C10a": 0,  # 获取报告
    "C10b": 0,  # LLM增强报告
    "C11": 0,  # 复诊查看报告
    "C12": 0,  # 综合分析诊断
    "C13": 0,  # 治疗方案
    "C14": 0,  # 文书记录
    "C15": 0,  # 患者宣教
    "C16": 0,  # 结束
    
    # === 专科节点 ===
    "S4": 1,  # 专科问诊（显示问诊轮数）
    "S5": 0,  # 体格检查
}


def get_output_level(module_name: str = "", node_name: str = "") -> int:
    """
    获取指定模块或节点的输出级别
    
    Args:
        module_name: 模块名称
        node_name: 节点名称
    
    Returns:
        输出级别 (0-3)
    """
    # 优先级：节点级别 > 模块级别 > 全局级别
    if node_name and node_name in NODE_OUTPUT_LEVELS:
        return NODE_OUTPUT_LEVELS[node_name]
    
    if module_name and module_name in MODULE_OUTPUT_LEVELS:
        return MODULE_OUTPUT_LEVELS[module_name]
    
    return DEFAULT_OUTPUT_LEVEL


def should_log(level_required: int, module_name: str = "", node_name: str = "") -> bool:
    """
    判断是否应该输出日志
    
    Args:
        level_required: 需要的最低输出级别
        module_name: 模块名称
        node_name: 节点名称
    
    Returns:
        是否应该输出
    """
    current_level = get_output_level(module_name, node_name)
    return current_level >= level_required


# ========== Logger过滤器 ==========
class OutputFilter(object):
    """
    日志过滤器 - 用于抑制common_opd_graph和specialty_subgraph中未被should_log包装的logger.info
    
    使用方法：
    1. 在common_opd_graph.py的开头添加：
       if SUPPRESS_UNCHECKED_LOGS:
           logger.addFilter(OutputFilter("common_opd_graph"))
    2. 在specialty_subgraph.py的开头添加：
       if SUPPRESS_UNCHECKED_LOGS:
           logger.addFilter(OutputFilter("specialty_subgraph"))
    """
    def __init__(self, module_name: str):
        self.module_name = module_name
    
    def filter(self, record):
        """
        过滤日志记录
        只允许以下日志通过：
        1. ERROR/WARNING级别的日志
        2. 来自langgraph_multi_patient_processor的日志
        3. 已经被should_log检查过的日志（通过特殊标记）
        """
        # 允许ERROR和WARNING级别的日志
        if record.levelno >= logging.WARNING:
            return True
        
        # 如果是来自患者执行器的日志，允许通过
        if "langgraph_multi_patient" in record.name:
            return True
        
        # 如果SUPPRESS_UNCHECKED_LOGS为False，允许所有日志
        if not SUPPRESS_UNCHECKED_LOGS:
            return True
        
        # 否则抑制该日志（common_opd_graph中未被should_log包装的日志）
        return False
