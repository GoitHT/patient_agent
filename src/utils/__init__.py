"""工具函数模块 - 统一导出接口"""

# 从 common.py 导入所有常用工具函数
from .common import (
    # JSON 解析工具
    parse_json_with_retry,
    
    # 文本处理工具
    contains_positive,
    contains_any_positive,
    
    # 安全规则工具
    disclaimer_text,
    apply_safety_rules,
    
    # 时间工具
    now_iso,
    
    # Prompt 加载工具
    prompt_dir,
    load_prompt,
    
    # 日志工具
    ColoredFormatter,
    setup_console_logging,
    setup_dual_logging,
    get_logger,
    
    # ID 生成工具
    make_run_id,
)

__all__ = [
    # JSON 解析
    "parse_json_with_retry",
    
    # 文本处理
    "contains_positive",
    "contains_any_positive",
    
    # 安全规则
    "disclaimer_text",
    "apply_safety_rules",
    
    # 时间
    "now_iso",
    
    # Prompt
    "prompt_dir",
    "load_prompt",
    
    # 日志
    "ColoredFormatter",
    "setup_console_logging",
    "setup_dual_logging",
    "get_logger",
    
    # ID生成
    "make_run_id",
]
