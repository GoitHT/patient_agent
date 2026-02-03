"""日志格式化工具 - 患者日志格式化和显示"""

# 患者颜色映射（用于终端显示区分）
PATIENT_COLORS = [
    "\033[96m",  # 青色
    "\033[93m",  # 黄色
    "\033[92m",  # 绿色
    "\033[95m",  # 紫色
    "\033[94m",  # 蓝色
    "\033[91m",  # 红色
    "\033[97m",  # 白色
    "\033[90m",  # 灰色
]
COLOR_RESET = "\033[0m"


def get_patient_color(patient_index: int) -> str:
    """获取患者的颜色代码
    
    Args:
        patient_index: 患者索引
    
    Returns:
        颜色代码字符串
    """
    return PATIENT_COLORS[patient_index % len(PATIENT_COLORS)]


def format_patient_log(patient_id: str, message: str, patient_index: int = 0) -> str:
    """格式化患者日志，添加颜色标识
    
    Args:
        patient_id: 患者ID
        message: 日志消息
        patient_index: 患者索引（用于选择颜色）
    
    Returns:
        格式化后的日志消息
    """
    color = get_patient_color(patient_index)
    return f"{color}[{patient_id}]{COLOR_RESET} {message}"
