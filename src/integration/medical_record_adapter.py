"""
医疗记录服务适配器
Medical Record Service Adapter

提供单体模式、微服务模式和数据库模式的统一接口
"""

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from config import Config
    from services.medical_record import MedicalRecordService


def get_medical_record_service(
    config: 'Config',
    storage_dir: Path
) -> 'MedicalRecordService':
    """
    获取医疗记录服务实例
    
    根据配置返回单体模式、微服务模式或数据库模式的服务
    
    Args:
        config: 完整配置对象
        storage_dir: 存储目录（单体模式使用）
    
    Returns:
        MedicalRecordService实例
    """
    from services.medical_record import MedicalRecordService
    
    # 优先级：数据库模式 > 微服务模式 > 单体模式
    if hasattr(config, 'database') and config.database.enabled:
        # 数据库模式 - 使用MySQL存储
        from services.medical_record_db_service import DatabaseMedicalRecordService
        
        return DatabaseMedicalRecordService(
            connection_string=config.database.connection_string,
            storage_dir=storage_dir,
            backup_to_file=config.database.backup_to_file,
            echo=config.database.echo
        )
    
    elif config.microservices.enabled:
        # 微服务模式 - 通过HTTP调用远程服务
        # 目前先返回单体模式服务
        # TODO: 实现基于HTTP的远程服务客户端
        import warnings
        warnings.warn(
            "微服务模式尚未完全实现，将使用单体模式",
            UserWarning
        )
        return MedicalRecordService(storage_dir=storage_dir)
    else:
        # 单体模式 - 直接使用本地服务
        return MedicalRecordService(storage_dir=storage_dir)
