"""
配置管理模块
支持多层级配置：默认值 < config.yaml < 环境变量 < CLI参数
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class LLMConfig:
    """LLM配置"""
    backend: str = "deepseek"
    enable_reports: bool = False


@dataclass
class AgentConfig:
    """智能体配置"""
    max_questions: int = 10  # 医生最多问几个问题（最底层默认值，优先级：CLI > 环境变量 > config.yaml > 此默认值）
    max_triage_questions: int = 3  # 护士分诊时最多问几个问题


@dataclass
class RAGConfig:
    """RAG配置"""
    persist_dir: Path = field(default_factory=lambda: Path(".chroma"))
    collection_name: str = "hospital_kb"
    skip_rag: bool = False


@dataclass
class ModeConfig:
    """运行模式配置"""
    multi_patient: bool = True
    num_patients: int = 1
    patient_interval: int = 0


@dataclass
class PhysicalConfig:
    """物理环境配置"""
    enable_simulation: bool = True
    interactive: bool = False


@dataclass
class SystemConfig:
    """系统配置"""
    verbose: bool = False
    log_file: Optional[str] = None
    save_trace: Path = field(default_factory=lambda: Path("agent_trace.json"))
    enable_trace: bool = False


@dataclass
class MicroservicesConfig:
    """微服务配置"""
    enabled: bool = False
    record_service_url: str = "http://localhost:8001"
    patient_service_url: str = "http://localhost:8002"
    doctor_service_url: str = "http://localhost:8003"
    notification_service_url: str = "http://localhost:8006"
    request_timeout: int = 30


@dataclass
class DatabaseConfig:
    """数据库配置"""
    enabled: bool = False
    connection_string: str = "mysql+pymysql://root:password@localhost:3306/hospital_db?charset=utf8mb4"
    backup_to_file: bool = True


@dataclass
class Config:
    """主配置类"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    microservices: MicroservicesConfig = field(default_factory=MicroservicesConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    @classmethod
    def load(cls, config_file: Optional[Path] = None, cli_args=None) -> Config:
        """
        加载配置，优先级：CLI参数 > 环境变量 > config.yaml > 默认值
        
        Args:
            config_file: 配置文件路径
            cli_args: argparse解析的命令行参数
        """
        config = cls()
        
        # 1. 从config.yaml加载（如果存在）
        if config_file and config_file.exists():
            config._load_from_yaml(config_file)
        elif Path("config.yaml").exists():
            config._load_from_yaml(Path("config.yaml"))
        elif Path("src/config.yaml").exists():
            config._load_from_yaml(Path("src/config.yaml"))
        
        # 2. 从环境变量加载
        config._load_from_env()
        
        # 3. 从CLI参数加载（最高优先级）
        if cli_args:
            config._load_from_args(cli_args)
        
        return config
    
    def _load_from_yaml(self, path: Path) -> None:
        """从YAML文件加载配置"""
        if not HAS_YAML:
            return
        
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # LLM配置
            if "llm" in data:
                llm_data = data["llm"]
                if "backend" in llm_data:
                    self.llm.backend = llm_data["backend"]
                if "enable_reports" in llm_data:
                    self.llm.enable_reports = llm_data["enable_reports"]
            
            # Agent配置
            if "agent" in data:
                agent_data = data["agent"]
                if "max_questions" in agent_data:
                    self.agent.max_questions = agent_data["max_questions"]
                if "max_triage_questions" in agent_data:
                    self.agent.max_triage_questions = agent_data["max_triage_questions"]
            
            # RAG配置
            if "rag" in data:
                rag_data = data["rag"]
                if "persist_dir" in rag_data:
                    self.rag.persist_dir = Path(rag_data["persist_dir"])
                if "collection_name" in rag_data:
                    self.rag.collection_name = rag_data["collection_name"]
                if "skip_rag" in rag_data:
                    self.rag.skip_rag = rag_data["skip_rag"]
            
            # Mode配置
            if "mode" in data:
                mode_data = data["mode"]
                if "multi_patient" in mode_data:
                    self.mode.multi_patient = mode_data["multi_patient"]
                if "num_patients" in mode_data:
                    self.mode.num_patients = mode_data["num_patients"]
                if "patient_interval" in mode_data:
                    self.mode.patient_interval = mode_data["patient_interval"]
            
            # Physical配置
            if "physical" in data:
                physical_data = data["physical"]
                if "enable_simulation" in physical_data:
                    self.physical.enable_simulation = physical_data["enable_simulation"]
                if "interactive" in physical_data:
                    self.physical.interactive = physical_data["interactive"]
            
            # 系统配置
            if "system" in data:
                system_data = data["system"]
                if "verbose" in system_data:
                    self.system.verbose = system_data["verbose"]
                if "log_file" in system_data and system_data["log_file"]:
                    self.system.log_file = system_data["log_file"]
                if "save_trace" in system_data:
                    self.system.save_trace = Path(system_data["save_trace"])
                if "enable_trace" in system_data:
                    self.system.enable_trace = system_data["enable_trace"]
            
            # 微服务配置
            if "microservices" in data:
                ms_data = data["microservices"]
                if "enabled" in ms_data:
                    self.microservices.enabled = ms_data["enabled"]
                if "record_service_url" in ms_data:
                    self.microservices.record_service_url = ms_data["record_service_url"]
                if "patient_service_url" in ms_data:
                    self.microservices.patient_service_url = ms_data["patient_service_url"]
                if "doctor_service_url" in ms_data:
                    self.microservices.doctor_service_url = ms_data["doctor_service_url"]
                if "notification_service_url" in ms_data:
                    self.microservices.notification_service_url = ms_data["notification_service_url"]
                if "request_timeout" in ms_data:
                    self.microservices.request_timeout = ms_data["request_timeout"]
            
            # 数据库配置
            if "database" in data:
                db_data = data["database"]
                if "enabled" in db_data:
                    self.database.enabled = db_data["enabled"]
                if "connection_string" in db_data:
                    self.database.connection_string = db_data["connection_string"]
                if "backup_to_file" in db_data:
                    self.database.backup_to_file = db_data["backup_to_file"]
                if "echo" in db_data:
                    self.database.echo = db_data["echo"]
                    
        except Exception as e:
            # 静默失败，使用默认值
            pass
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        # LLM配置
        if os.getenv("HOSPITAL_LLM_BACKEND"):
            self.llm.backend = os.getenv("HOSPITAL_LLM_BACKEND")
        if os.getenv("HOSPITAL_LLM_REPORTS"):
            self.llm.enable_reports = os.getenv("HOSPITAL_LLM_REPORTS").lower() in ("true", "1", "yes")
        
        # Agent配置
        if os.getenv("HOSPITAL_MAX_QUESTIONS"):
            self.agent.max_questions = int(os.getenv("HOSPITAL_MAX_QUESTIONS"))
        
        # RAG配置
        if os.getenv("HOSPITAL_CHROMA_DIR"):
            self.rag.persist_dir = Path(os.getenv("HOSPITAL_CHROMA_DIR"))
        if os.getenv("HOSPITAL_COLLECTION"):
            self.rag.collection_name = os.getenv("HOSPITAL_COLLECTION")
        
        # 系统配置
        if os.getenv("HOSPITAL_TRACE_FILE"):
            self.system.save_trace = Path(os.getenv("HOSPITAL_TRACE_FILE"))
        if os.getenv("HOSPITAL_ENABLE_TRACE"):
            self.system.enable_trace = os.getenv("HOSPITAL_ENABLE_TRACE").lower() in ("true", "1", "yes")
        
        # 微服务配置
        if os.getenv("MICROSERVICES_ENABLED"):
            self.microservices.enabled = os.getenv("MICROSERVICES_ENABLED").lower() in ("true", "1", "yes")
        if os.getenv("RECORD_SERVICE_URL"):
            self.microservices.record_service_url = os.getenv("RECORD_SERVICE_URL")
        if os.getenv("PATIENT_SERVICE_URL"):
            self.microservices.patient_service_url = os.getenv("PATIENT_SERVICE_URL")
        if os.getenv("DOCTOR_SERVICE_URL"):
            self.microservices.doctor_service_url = os.getenv("DOCTOR_SERVICE_URL")
        if os.getenv("NOTIFICATION_SERVICE_URL"):
            self.microservices.notification_service_url = os.getenv("NOTIFICATION_SERVICE_URL")
        if os.getenv("MICROSERVICES_TIMEOUT"):
            self.microservices.request_timeout = int(os.getenv("MICROSERVICES_TIMEOUT"))
    
    def _load_from_args(self, args) -> None:
        """从CLI参数加载配置（最高优先级）"""
        # LLM配置
        if hasattr(args, "llm") and args.llm:
            self.llm.backend = args.llm
        if hasattr(args, "llm_reports") and args.llm_reports:
            self.llm.enable_reports = args.llm_reports
        
        # Agent配置
        if hasattr(args, "max_questions") and args.max_questions is not None:
            self.agent.max_questions = args.max_questions
        
        # RAG配置
        if hasattr(args, "persist") and args.persist:
            self.rag.persist_dir = args.persist
        if hasattr(args, "collection") and args.collection:
            self.rag.collection_name = args.collection
        if hasattr(args, "skip_rag"):
            self.rag.skip_rag = args.skip_rag
        
        # Mode配置
        if hasattr(args, "multi_patient"):
            self.mode.multi_patient = args.multi_patient
        if hasattr(args, "num_patients") and args.num_patients is not None:
            self.mode.num_patients = args.num_patients
        if hasattr(args, "patient_interval") and args.patient_interval is not None:
            self.mode.patient_interval = args.patient_interval
        
        # Physical配置
        if hasattr(args, "physical_sim"):
            self.physical.enable_simulation = args.physical_sim
        if hasattr(args, "interactive"):
            self.physical.interactive = args.interactive
        
        # 系统配置
        if hasattr(args, "verbose"):
            self.system.verbose = args.verbose
        if hasattr(args, "log_file") and args.log_file:
            self.system.log_file = args.log_file
        if hasattr(args, "save_trace") and args.save_trace:
            self.system.save_trace = args.save_trace
            self.system.enable_trace = True
    
    def summary(self) -> str:
        """生成配置摘要"""
        mode = "多患者" if self.mode.multi_patient else "单例"
        lines = [
            "🔧 系统配置:",
            f"  - 运行模式: {mode}",
            f"  - LLM后端: {self.llm.backend}",
            f"  - 增强报告: {'是' if self.llm.enable_reports else '否'}",
            f"  - 最多问题数: {self.agent.max_questions}",
            f"  - 数据源: HuggingFace DiagnosisArena",
            f"  - RAG集合: {self.rag.collection_name}",
            f"  - 物理环境: {'启用' if self.physical.enable_simulation else '禁用'}",
        ]
        if self.mode.multi_patient:
            lines.append(f"  - 患者数量: {self.mode.num_patients}")
            lines.append(f"  - 进入间隔: {self.mode.patient_interval}秒")
        
        # 微服务配置
        if self.microservices.enabled:
            lines.append("\n🏢 架构模式: 微服务 (Microservices)")
            lines.append(f"  - 病例服务: {self.microservices.record_service_url}")
            lines.append(f"  - 患者服务: {self.microservices.patient_service_url}")
            lines.append(f"  - 医生服务: {self.microservices.doctor_service_url}")
            lines.append(f"  - 通知服务: {self.microservices.notification_service_url}")
        else:
            lines.append("\n🏢 架构模式: 单体 (Monolithic)")
        
        return "\n".join(lines)
