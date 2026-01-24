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
    max_questions: int = 10  # 医生最多问几个问题
    max_triage_questions: int = 3  # 护士分诊时最多问几个问题
    dataset_id: int = 15     # 数据集索引位置（从0开始），非病例本身的ID
    use_hf_data: bool = True # 是否从HuggingFace加载数据


@dataclass
class DatasetConfig:
    """数据集配置"""
    cache_dir: Path = field(default_factory=lambda: Path("./diagnosis_dataset"))  # 本地缓存目录
    use_local_cache: bool = True  # 是否使用本地缓存


@dataclass
class RAGConfig:
    """RAG配置"""
    persist_dir: Path = field(default_factory=lambda: Path(".chroma"))
    collection_name: str = "hospital_kb"


@dataclass
class SystemConfig:
    """系统配置"""
    seed: int = 42
    save_trace: Path = field(default_factory=lambda: Path("agent_trace.json"))
    enable_trace: bool = False  # 是否保存追踪


@dataclass
class Config:
    """主配置类"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
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
                if "dataset_id" in agent_data:
                    self.agent.dataset_id = agent_data["dataset_id"]
                if "use_hf_data" in agent_data:
                    self.agent.use_hf_data = agent_data["use_hf_data"]
            
            # Dataset配置
            if "dataset" in data:
                dataset_data = data["dataset"]
                if "cache_dir" in dataset_data:
                    self.dataset.cache_dir = Path(dataset_data["cache_dir"])
                if "use_local_cache" in dataset_data:
                    self.dataset.use_local_cache = dataset_data["use_local_cache"]
            
            # RAG配置
            if "rag" in data:
                rag_data = data["rag"]
                if "persist_dir" in rag_data:
                    self.rag.persist_dir = Path(rag_data["persist_dir"])
                if "collection_name" in rag_data:
                    self.rag.collection_name = rag_data["collection_name"]
            
            # 系统配置
            if "system" in data:
                system_data = data["system"]
                if "seed" in system_data:
                    self.system.seed = system_data["seed"]
                if "save_trace" in system_data:
                    self.system.save_trace = Path(system_data["save_trace"])
                if "enable_trace" in system_data:
                    self.system.enable_trace = system_data["enable_trace"]
                    
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
        if os.getenv("HOSPITAL_DATASET_ID"):
            self.agent.dataset_id = int(os.getenv("HOSPITAL_DATASET_ID"))
        if os.getenv("HOSPITAL_USE_HF_DATA"):
            self.agent.use_hf_data = os.getenv("HOSPITAL_USE_HF_DATA").lower() in ("true", "1", "yes")
        
        # Dataset配置
        if os.getenv("HOSPITAL_DATASET_CACHE_DIR"):
            self.dataset.cache_dir = Path(os.getenv("HOSPITAL_DATASET_CACHE_DIR"))
        if os.getenv("HOSPITAL_USE_LOCAL_CACHE"):
            self.dataset.use_local_cache = os.getenv("HOSPITAL_USE_LOCAL_CACHE").lower() in ("true", "1", "yes")
        
        # RAG配置
        if os.getenv("HOSPITAL_CHROMA_DIR"):
            self.rag.persist_dir = Path(os.getenv("HOSPITAL_CHROMA_DIR"))
        if os.getenv("HOSPITAL_COLLECTION"):
            self.rag.collection_name = os.getenv("HOSPITAL_COLLECTION")
        
        # 系统配置
        if os.getenv("HOSPITAL_SEED"):
            self.system.seed = int(os.getenv("HOSPITAL_SEED"))
        if os.getenv("HOSPITAL_TRACE_FILE"):
            self.system.save_trace = Path(os.getenv("HOSPITAL_TRACE_FILE"))
        if os.getenv("HOSPITAL_ENABLE_TRACE"):
            self.system.enable_trace = os.getenv("HOSPITAL_ENABLE_TRACE").lower() in ("true", "1", "yes")
    
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
        if hasattr(args, "dataset_id") and args.dataset_id is not None:
            self.agent.dataset_id = args.dataset_id
        if hasattr(args, "use_hf_data") and args.use_hf_data is not None:
            self.agent.use_hf_data = args.use_hf_data
        
        # RAG配置
        if hasattr(args, "persist") and args.persist:
            self.rag.persist_dir = args.persist
        if hasattr(args, "collection") and args.collection:
            self.rag.collection_name = args.collection
        
        # 系统配置
        if hasattr(args, "seed") and args.seed is not None:
            self.system.seed = args.seed
        if hasattr(args, "save_trace") and args.save_trace:
            self.system.save_trace = args.save_trace
            self.system.enable_trace = True
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            "🔧 系统配置:",
            f"  - 模式: 三智能体 (医生+患者+护士)",
            f"  - LLM后端: {self.llm.backend}",
            f"  - 增强报告: {'是' if self.llm.enable_reports else '否'}",
            f"  - 随机种子: {self.system.seed}",
            f"  - 数据集ID: {self.agent.dataset_id}",
            f"  - 最多问题数: {self.agent.max_questions}",
            f"  - 数据源: {'HuggingFace' if self.agent.use_hf_data else 'Mock'}",
            f"  - RAG集合: {self.rag.collection_name}",
            f"  - 保存追踪: {'是' if self.system.enable_trace else '否'}",
        ]
        return "\n".join(lines)
