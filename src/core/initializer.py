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
        logger.info(f"🤖 初始化大语言模型 ({self.config.llm.backend})...")
        try:
            llm_client = build_llm_client(self.config.llm.backend)
            logger.info("  ✅ 大语言模型初始化成功\n")
            self.components['llm'] = llm_client
            return llm_client
        except Exception as e:
            logger.error(f"❌ 大语言模型初始化失败：{e}")
            raise
    
    def initialize_rag(self) -> Any:
        """初始化知识库检索器
        
        Returns:
            检索器实例
        """
        if not self.config.rag.skip_rag:
            logger.info("📂 初始化知识库检索器...")
            try:
                retriever = default_retriever(
                    persist_dir=self.config.rag.persist_dir,
                    collection_name=self.config.rag.collection_name
                )
                logger.info("  ✅ 知识库检索器初始化成功\n")
                self.components['retriever'] = retriever
                return retriever
            except Exception as e:
                logger.error(f"❌ 知识库检索器初始化失败：{e}")
                raise
        else:
            logger.info("⏭️ 使用虚拟检索器（跳过RAG）\n")
            retriever = DummyRetriever()
            self.components['retriever'] = retriever
            return retriever
    
    def initialize_business_services(self) -> Any:
        """初始化业务服务（预约、计费）
        
        Returns:
            业务服务集合
        """
        logger.info("💼 初始化业务服务（预约、计费）...")
        services = build_services()
        logger.info("  ✅ 业务服务初始化完成\n")
        self.components['services'] = services
        return services
    
    def initialize_medical_record(self, storage_dir: Path) -> Any:
        """初始化病例库服务
        
        Args:
            storage_dir: 存储目录
        
        Returns:
            病例库服务实例
        """
        logger.info("📋 初始化病例库服务...")
        medical_record_service = get_medical_record_service(
            config=self.config,
            storage_dir=storage_dir
        )
        logger.info("  ✅ 病例库服务初始化完成")
        
        if hasattr(self.config, 'database') and self.config.database.enabled:
            logger.info(f"  🗄️  使用数据库存储: {self.config.database.connection_string.split('@')[1] if '@' in self.config.database.connection_string else 'MySQL'}")
            if self.config.database.backup_to_file:
                logger.info(f"  💾 同时备份到文件: {storage_dir.absolute()}\n")
            else:
                logger.info("")
        else:
            logger.info(f"  📁 病例存储目录: {storage_dir.absolute()}\n")
        
        self.components['medical_record_service'] = medical_record_service
        return medical_record_service
    
    def initialize_coordinator(self, medical_record_service: Any) -> Any:
        """初始化医院协调器
        
        Args:
            medical_record_service: 病例库服务实例
        
        Returns:
            协调器实例
        """
        logger.info("🏥 初始化医院协调器...")
        coordinator = get_coordinator(medical_record_service=medical_record_service)
        logger.info("  ✅ 协调器初始化完成\n")
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
