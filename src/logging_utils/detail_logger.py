"""
患者详细日志记录器 - Patient Detail Logger
为每个患者创建独立的详细日志文件，记录完整的就诊过程
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# 患者详细日志存储目录
PATIENT_LOGS_DIR = Path("logs/patients")
PATIENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)


class PatientDetailLogger:
    """为每个患者创建独立的详细日志记录器"""
    
    def __init__(self, patient_id: str, case_id: int):
        """
        初始化患者详细日志记录器
        
        Args:
            patient_id: 患者ID
            case_id: 病例ID
        """
        self.patient_id = patient_id
        self.case_id = case_id
        
        # 创建日志文件路径：logs/patients/patient_<case_id>_<timestamp>.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = PATIENT_LOGS_DIR / f"patient_{case_id}_{timestamp}.log"
        
        # 创建独立的logger
        self.logger = logging.getLogger(f"patient_detail.{patient_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # 不传播到父logger（避免在终端显示）
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(
            self.log_file,
            mode='w',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 写入文件头信息
        self._write_header()
    
    def _write_header(self):
        """写入日志文件头信息"""
        self.logger.info("="*80)
        self.logger.info(f"患者就诊详细记录")
        self.logger.info("="*80)
        self.logger.info(f"患者ID: {self.patient_id}")
        self.logger.info(f"病例ID: {self.case_id}")
        self.logger.info(f"记录时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        self.logger.info("="*80)
        self.logger.info("")
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def section(self, title: str):
        """记录分节标题"""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info(f"【{title}】")
        self.logger.info("="*80)
        self.logger.info("")
    
    def subsection(self, title: str):
        """记录子节标题"""
        self.logger.info("")
        self.logger.info("-"*80)
        self.logger.info(f"【{title}】")
        self.logger.info("-"*80)
    
    def qa_round(self, round_num: int, question: str, answer: str):
        """记录问诊对话"""
        self.logger.info("")
        self.logger.info(f"📝 第 {round_num} 轮问诊:")
        self.logger.info(f"    🧑‍⚕️  医生问: {question}")
        self.logger.info(f"    👤 患者答: {answer}")
    
    def node_start(self, node_name: str, node_display_name: str = ""):
        """记录节点开始"""
        display = node_display_name if node_display_name else node_name
        self.logger.info("")
        self.logger.info("┌" + "─"*78 + "┐")
        self.logger.info(f"│ ▶️  开始执行: {display}" + " "*(78 - 14 - len(display.encode('utf-8').decode('utf-8', errors='ignore'))) + "│")
        self.logger.info("└" + "─"*78 + "┘")
    
    def node_end(self, node_name: str, node_display_name: str = ""):
        """记录节点结束"""
        display = node_display_name if node_display_name else node_name
        self.logger.info("")
        self.logger.info(f"✅ {display} 完成")
        self.logger.info("-"*80)
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)
    
    def close(self):
        """关闭日志记录器"""
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("就诊记录结束")
        self.logger.info("="*80)
        
        # 关闭所有处理器
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# 全局字典，用于存储每个患者的日志记录器
_patient_loggers = {}


def get_patient_detail_logger(patient_id: str, case_id: Optional[int] = None) -> Optional[PatientDetailLogger]:
    """
    获取患者的详细日志记录器
    
    Args:
        patient_id: 患者ID
        case_id: 病例ID（首次创建时需要）
    
    Returns:
        患者的详细日志记录器，如果不存在且未提供case_id则返回None
    """
    if patient_id not in _patient_loggers:
        if case_id is None:
            return None
        _patient_loggers[patient_id] = PatientDetailLogger(patient_id, case_id)
    
    return _patient_loggers[patient_id]


def create_patient_detail_logger(patient_id: str, case_id: int) -> PatientDetailLogger:
    """
    创建患者的详细日志记录器
    
    Args:
        patient_id: 患者ID
        case_id: 病例ID
    
    Returns:
        新创建的患者详细日志记录器
    """
    logger = PatientDetailLogger(patient_id, case_id)
    _patient_loggers[patient_id] = logger
    return logger


def close_patient_detail_logger(patient_id: str):
    """
    关闭并移除患者的详细日志记录器
    
    Args:
        patient_id: 患者ID
    """
    if patient_id in _patient_loggers:
        _patient_loggers[patient_id].close()
        del _patient_loggers[patient_id]


def close_all_patient_detail_loggers():
    """关闭所有患者的详细日志记录器"""
    for patient_id in list(_patient_loggers.keys()):
        close_patient_detail_logger(patient_id)
