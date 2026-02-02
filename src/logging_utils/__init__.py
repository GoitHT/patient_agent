"""
日志工具模块 - Logging Utilities Module
患者详细日志、输出级别配置
"""

from .detail_logger import (
    PatientDetailLogger,
    get_patient_detail_logger,
    create_patient_detail_logger,
    close_patient_detail_logger,
    close_all_patient_detail_loggers,
    PATIENT_LOGS_DIR,
)

from .output_config import (
    should_log,
    get_output_level,
    OutputFilter,
    SUPPRESS_UNCHECKED_LOGS,
    DEFAULT_OUTPUT_LEVEL,
    NODE_OUTPUT_LEVELS,
    MODULE_OUTPUT_LEVELS,
)

__all__ = [
    # detail_logger
    'PatientDetailLogger',
    'get_patient_detail_logger',
    'create_patient_detail_logger',
    'close_patient_detail_logger',
    'close_all_patient_detail_loggers',
    'PATIENT_LOGS_DIR',
    # output_config
    'should_log',
    'get_output_level',
    'OutputFilter',
    'SUPPRESS_UNCHECKED_LOGS',
    'DEFAULT_OUTPUT_LEVEL',
    'NODE_OUTPUT_LEVELS',
    'MODULE_OUTPUT_LEVELS',
]
