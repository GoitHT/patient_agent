"""
医疗记录服务适配器
Medical Record Service Adapter

提供本地文件存储或数据库存储的统一接口
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
    
    根据配置返回数据库模式或本地文件模式的服务
    
    Args:
        config: 完整配置对象
        storage_dir: 存储目录（本地模式使用）
    
    Returns:
        MedicalRecordService实例
    """
    from services.medical_record import MedicalRecordService
    
    # 如果启用数据库，使用DatabaseMedicalRecordService
    if hasattr(config, 'database') and config.database.enabled:
        from services.medical_record_db_service import DatabaseMedicalRecordService
        
        return DatabaseMedicalRecordService(
            connection_string=config.database.connection_string,
            storage_dir=storage_dir,
            backup_to_file=config.database.backup_to_file
        )
    else:
        # 否则使用本地文件存储
        return MedicalRecordService(storage_dir=storage_dir)
