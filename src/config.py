"""
配置管理模块
支持多层级配置：默认值 < config.yaml < 环境变量
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


@dataclass
class AgentConfig:
    """智能体配置"""
    max_questions: int = 10  # 医生最多问几个问题（最底层默认值，优先级：环境变量 > config.yaml > 此默认值）


@dataclass
class RAGConfig:
    """RAG配置（Adaptive RAG 系统）"""
    # Adaptive RAG 配置
    spllm_root: Path = field(default_factory=lambda: Path("SPLLM-RAG1"))  # SPLLM-RAG1 项目路径
    adaptive_cache_folder: Optional[Path] = None  # 模型缓存目录（默认为 spllm_root/model_cache）
    adaptive_threshold: float = 0.3  # 余弦距离阈值（0-1，越小越严格）
    adaptive_embed_model: str = "BAAI/bge-large-zh-v1.5"  # 嵌入模型名称


@dataclass
class ModeConfig:
    """运行模式配置"""
    multi_patient: bool = True
    num_patients: int = 1
    patient_interval: int = 0


@dataclass
class PhysicalConfig:
    """物理环境配置"""
    interactive: bool = False


@dataclass
class SystemConfig:
    """系统配置"""
    verbose: bool = False


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
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> Config:
        """
        加载配置，优先级：环境变量 > config.yaml > 默认值
        
        Args:
            config_file: 配置文件路径
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
            
            # Agent配置
            if "agent" in data:
                agent_data = data["agent"]
                if "max_questions" in agent_data:
                    self.agent.max_questions = agent_data["max_questions"]
            
            # RAG配置
            if "rag" in data:
                rag_data = data["rag"]
                # Adaptive RAG 配置
                if "spllm_root" in rag_data:
                    self.rag.spllm_root = Path(rag_data["spllm_root"])
                if "adaptive_cache_folder" in rag_data:
                    self.rag.adaptive_cache_folder = Path(rag_data["adaptive_cache_folder"]) if rag_data["adaptive_cache_folder"] else None
                if "adaptive_threshold" in rag_data:
                    self.rag.adaptive_threshold = float(rag_data["adaptive_threshold"])
                if "adaptive_embed_model" in rag_data:
                    self.rag.adaptive_embed_model = rag_data["adaptive_embed_model"]
            
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
                if "interactive" in physical_data:
                    self.physical.interactive = physical_data["interactive"]
            
            # 系统配置
            if "system" in data:
                system_data = data["system"]
                if "verbose" in system_data:
                    self.system.verbose = system_data["verbose"]
            
            # 数据库配置
            if "database" in data:
                db_data = data["database"]
                if "enabled" in db_data:
                    self.database.enabled = db_data["enabled"]
                if "connection_string" in db_data:
                    self.database.connection_string = db_data["connection_string"]
                if "backup_to_file" in db_data:
                    self.database.backup_to_file = db_data["backup_to_file"]
                    
        except Exception as e:
            # 静默失败，使用默认值
            pass
    
    def _load_from_env(self) -> None:
        """从环境变量加载配置"""
        # LLM配置
        if os.getenv("HOSPITAL_LLM_BACKEND"):
            self.llm.backend = os.getenv("HOSPITAL_LLM_BACKEND")
        
        # Agent配置
        if os.getenv("HOSPITAL_MAX_QUESTIONS"):
            self.agent.max_questions = int(os.getenv("HOSPITAL_MAX_QUESTIONS"))
        

    
    def summary(self) -> str:
        """生成配置摘要"""
        mode = "多患者" if self.mode.multi_patient else "单例"
        lines = [
            "🔧 系统配置:",
            f"  - 运行模式: {mode}",
            f"  - LLM后端: {self.llm.backend}",
            f"  - 最多问题数: {self.agent.max_questions}",
            f"  - Adaptive RAG 阈值: {self.rag.adaptive_threshold}",                                                                        
        ]
        if self.mode.multi_patient:
            lines.append(f"  - 患者数量: {self.mode.num_patients}")
            lines.append(f"  - 进入间隔: {self.mode.patient_interval}秒")
        
        return "\n".join(lines)
