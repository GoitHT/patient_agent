"""系统核心组件初始化器"""

import logging
from pathlib import Path
from typing import Dict, Any

from services.llm_client import build_llm_client
from graphs.router import default_retriever, build_services
from rag import DummyRetriever
from utils import get_logger
from config import Config
from integration import get_coordinator, get_medical_record_service


logger = get_logger("hospital_agent.initializer")


class SystemInitializer:
    """系统核心组件初始化器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.components: Dict[str, Any] = {}
    
    def initialize_logging(self) -> None:
        """初始化日志系统"""
        from utils import setup_console_logging
        
        console_level = logging.DEBUG if self.config.system.verbose else logging.INFO
        setup_console_logging(console_level=console_level)
        
        # 抑制第三方库日志
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    def initialize_llm(self) -> Any:
        """初始化大语言模型
        
        Returns:
            LLM客户端实例
        """
        logger.info(f"🤖 初始化 LLM ({self.config.llm.backend})")
        try:
            llm_client = build_llm_client(self.config.llm.backend)
            self.components['llm'] = llm_client
            return llm_client
        except Exception as e:
            logger.error(f"❌ 大语言模型初始化失败：{e}")
            raise
    
    def initialize_rag(self) -> Any:
        """初始化知识库检索器（Adaptive RAG 系统）
        
        Returns:
            检索器实例
        """
        logger.info("🚀 初始化 Adaptive RAG（SPLLM-RAG1）")
        try:
            from rag import AdaptiveRAGRetriever
            from pathlib import Path
            
            # 解析 SPLLM-RAG1 路径
            spllm_root = Path(self.config.rag.spllm_root)
            if not spllm_root.is_absolute():
                # 相对路径，相对于项目根目录
                from graphs.router import repo_root
                spllm_root = (repo_root() / spllm_root).resolve()
            
            # 检查路径是否存在
            if not spllm_root.exists():
                raise FileNotFoundError(
                    f"SPLLM-RAG1 路径不存在: {spllm_root}\n"
                    f"请检查 config.yaml 中的 spllm_root 配置"
                )
            
            chroma_path = spllm_root / "chroma"
            
            # 【自动初始化】检查并创建缺失的向量库
            if not chroma_path.exists():
                logger.warning(f"⚠️  chroma 目录不存在: {chroma_path}")
                logger.info("📦 自动创建向量库目录...")
                chroma_path.mkdir(parents=True, exist_ok=True)
            
            retriever = AdaptiveRAGRetriever(
                spllm_root=spllm_root,
                cache_folder=self.config.rag.adaptive_cache_folder,
                cosine_threshold=self.config.rag.adaptive_threshold,
                embed_model=self.config.rag.adaptive_embed_model,
            )
            logger.debug(f"   → SPLLM-RAG1: {spllm_root}")
            logger.debug(f"   → 阈值: {self.config.rag.adaptive_threshold}")
            self.components['retriever'] = retriever
            return retriever
        except Exception as e:
            logger.error(f"❌ Adaptive RAG 初始化失败：{e}")
            raise
    
    def initialize_business_services(self) -> Any:
        """初始化业务服务（预约、计费）
        
        Returns:
            业务服务集合
        """
        logger.info("💼 初始化业务服务")
        services = build_services()
        self.components['services'] = services
        return services
    
    def initialize_medical_record(self, storage_dir: Path) -> Any:
        """初始化病例库服务
        
        Args:
            storage_dir: 存储目录
        
        Returns:
            病例库服务实例
        """
        logger.info("📋 初始化病例库")
        medical_record_service = get_medical_record_service(
            config=self.config,
            storage_dir=storage_dir
        )
        
        if hasattr(self.config, 'database') and self.config.database.enabled:
            db_info = self.config.database.connection_string.split('@')[1] if '@' in self.config.database.connection_string else 'MySQL'
            logger.debug(f"   → 数据库: {db_info}")
        else:
            logger.debug(f"   → 文件: {storage_dir.absolute()}")
        
        self.components['medical_record_service'] = medical_record_service
        return medical_record_service
    
    def initialize_coordinator(self, medical_record_service: Any) -> Any:
        """初始化医院协调器
        
        Args:
            medical_record_service: 病例库服务实例
        
        Returns:
            协调器实例
        """
        logger.info("🏥 初始化协调器")
        coordinator = get_coordinator(medical_record_service=medical_record_service)
        self.components['coordinator'] = coordinator
        return coordinator
    
    def get_component(self, name: str) -> Any:
        """获取已初始化的组件
        
        Args:
            name: 组件名称
        
        Returns:
            组件实例
        """
        return self.components.get(name)
