"""显示模块 - 日志格式化和输出展示"""

from .log_formatter import get_patient_color, format_patient_log
from .output_formatter import (
    render_summary,
    display_startup_banner,
    display_mode_info,
    display_results_table,
    display_final_statistics,
    display_log_files
)

__all__ = [
    "get_patient_color",
    "format_patient_log",
    "render_summary",
    "display_startup_banner",
    "display_mode_info",
    "display_results_table",
    "display_final_statistics",
    "display_log_files"
]
