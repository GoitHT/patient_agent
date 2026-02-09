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
        
        # 设置格式 - 移除时间戳，因为会导致日志冗长
        formatter = logging.Formatter(
            '%(message)s'  # 只记录消息内容，不含时间戳和级别
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 写入文件头信息
        self._write_header()
    
    def _write_header(self):
        """写入日志文件头信息"""
        self.logger.info("╔" + "═"*78 + "╗")
        self.logger.info("║" + " "*25 + "患者就诊详细记录" + " "*37 + "║")
        self.logger.info("╠" + "═"*78 + "╣")
        self.logger.info(f"║  患者ID: {self.patient_id:<67}║")
        self.logger.info(f"║  病例ID: {self.case_id:<67}║")
        self.logger.info(f"║  记录时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'):<64}║")
        self.logger.info("╚" + "═"*78 + "╝")
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
        self.logger.info("┏" + "━"*78 + "┓")
        self.logger.info(f"┃  {title:<74}  ┃")
        self.logger.info("┗" + "━"*78 + "┛")
        self.logger.info("")
    
    def subsection(self, title: str):
        """记录子节标题"""
        self.logger.info("")
        self.logger.info(f"┌─ {title} " + "─"*(74-len(title)))
        self.logger.info("")
    
    def qa_round(self, round_num: int, question: str, answer: str):
        """记录问诊对话"""
        self.logger.info("")
        self.logger.info(f"📝 第 {round_num} 轮问诊:")
        self.logger.info(f" � 第 {round_num} 轮问诊")
        self.logger.info(f"   ┌─ 医生问：")
        # 对长文本进行换行处理
        for line in self._wrap_text(question, 70):
            self.logger.info(f"   │  {line}")
        self.logger.info(f"   │")
        self.logger.info(f"   └─ 患者答：")
        for line in self._wrap_text(answer, 70):
            self.logger.info(f"      {line}")
        self.logger.info("")
    
    def _wrap_text(self, text: str, width: int) -> list:
        """将长文本按宽度换行"""
        if not text:
            return [""]
        lines = []
        current_line = ""
        for char in text:
            current_line += char
            if len(current_line) >= width and char in ['，', '。', '、', '！', '？', ' ', ',', '.', '!', '?']:
                lines.append(current_line.rstrip())
                current_line = ""
        if current_line:
            lines.append(current_line)
        return lines if lines else [""]
    def diagnosis_result(self, diagnosis: dict):
        """记录诊断结果"""
        self.logger.info("")
        self.logger.info("╭─ 🔬 诊断结果 " + "─"*63)
        if diagnosis.get('diagnoses'):
            self.logger.info(f"│  💊 诊断: {', '.join(diagnosis['diagnoses'])}")
        if diagnosis.get('confidence'):
            self.logger.info(f"│  📊 置信度: {diagnosis['confidence']}")
        if diagnosis.get('reasoning'):
            self.logger.info(f"│  💭 诊断依据:")
            for line in self._wrap_text(diagnosis['reasoning'], 70):
                self.logger.info(f"│     {line}")
        self.logger.info("╰" + "─"*78)
    
    def prescription(self, medications: list):
        """记录处方信息"""
        self.logger.info("")
        self.logger.info("╭─ 💊 处方药物 " + "─"*63)
        for i, med in enumerate(medications, 1):
            if isinstance(med, dict):
                name = med.get('name', med.get('药品', '未知'))
                dosage = med.get('dosage', med.get('剂量', ''))
                frequency = med.get('frequency', med.get('频次', ''))
                self.logger.info(f"│  {i}. {name}")
                if dosage:
                    self.logger.info(f"│     剂量: {dosage}")
                if frequency:
                    self.logger.info(f"│     频次: {frequency}")
            else:
                self.logger.info(f"│  {i}. {med}")
        self.logger.info("╰" + "─"*78)
        self.logger.info("")
    
    def lab_test(self, test_name: str, results: dict):
        """记录检验检查结果"""
        self.logger.info("")
        self.logger.info(f"╭─ 🔬 {test_name} " + "─"*(75-len(test_name)))
        if isinstance(results, dict):
            for key, value in results.items():
                # 对长值进行换行
                if isinstance(value, str) and len(str(value)) > 60:
                    self.logger.info(f"│  {key}:")
                    for line in self._wrap_text(str(value), 70):
                        self.logger.info(f"│    {line}")
                else:
                    self.logger.info(f"│  {key}: {value}")
        else:
            for line in self._wrap_text(str(results), 70):
                self.logger.info(f"│  {line}")
        self.logger.info("╰" + "─"*78)
        self.logger.info("")
    
    def staff_info(self, role: str, staff_id: str, staff_name: str):
        """记录医护人员信息"""
        self.logger.info(f"│  👨‍⚕️ {role}: {staff_name} ({staff_id})")
    
    def timing(self, stage: str, duration: float):
        """记录时间统计"""
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        self.logger.info(f"│  ⏱️  {stage} 耗时: {minutes}分{seconds}秒")
    
    def medical_advice(self, advice: str):
        """记录医嘱"""
        self.logger.info("")
        self.logger.info("╭─ 📋 医嘱 " + "─"*67)
        for line in advice.split('\n'):
            if line.strip():
                for wrapped_line in self._wrap_text(line.strip(), 70):
                    self.logger.info(f"│  • {wrapped_line}")
        self.logger.info("╰" + "─"*78)
        self.logger.info("")
    
    def followup_plan(self, plan: dict):
        """记录随访计划"""
        self.logger.info("")
        self.logger.info("╭─ 📅 随访计划 " + "─"*63)
        if plan.get('when'):
            self.logger.info(f"│  ⏰ 随访时间: {plan['when']}")
        if plan.get('what'):
            self.logger.info(f"│  📝 随访内容:")
            for line in self._wrap_text(plan['what'], 70):
                self.logger.info(f"│     {line}")
        if plan.get('why'):
            self.logger.info(f"│  💡 随访原因:")
            for line in self._wrap_text(plan['why'], 70):
                self.logger.info(f"│     {line}")
        self.logger.info("╰" + "─"*78)
        self.logger.info("")
    
    def node_start(self, node_name: str, node_display_name: str = ""):
        """记录节点开始"""
        display = node_display_name if node_display_name else node_name
        self.logger.info("")
        self.logger.info("┌─ ▶️  " + display + " " + "─"*(73 - len(display)))
    
    def node_end(self, node_name: str, node_display_name: str = ""):
        """记录节点结束"""
        display = node_display_name if node_display_name else node_name
        self.logger.info(f"└─ ✅ {display} 完成")
        self.logger.info("")
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)
    
    def close(self):
        """关闭日志记录器"""
        self.logger.info("")
        self.logger.info("")
        self.logger.info("╔" + "═"*78 + "╗")
        self.logger.info("║" + " "*28 + "就诊记录结束" + " "*38 + "║")
        self.logger.info("╚" + "═"*78 + "╝")
        
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
