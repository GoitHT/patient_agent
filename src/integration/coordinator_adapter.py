"""
医院协调器适配器
Hospital Coordinator Adapter

提供单体模式和微服务模式的统一接口
"""

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from config import MicroservicesConfig
    from hospital_coordinator import HospitalCoordinator
    from services.medical_record import MedicalRecordService


def get_coordinator(
    config: 'MicroservicesConfig',
    medical_record_service: 'MedicalRecordService'
) -> 'HospitalCoordinator':
    """
    获取医院协调器实例
    
    根据配置返回单体模式或微服务模式的协调器
    
    Args:
        config: 微服务配置
        medical_record_service: 医疗记录服务
    
    Returns:
        HospitalCoordinator实例
    """
    from hospital_coordinator import HospitalCoordinator
    
    if config.enabled:
        # 微服务模式 - 未来实现
        # 目前先返回单体模式协调器
        # TODO: 实现微服务模式的协调器
        import warnings
        warnings.warn(
            "微服务模式尚未完全实现，将使用单体模式",
            UserWarning
        )
        return HospitalCoordinator(medical_record_service=medical_record_service)
    else:
        # 单体模式
        return HospitalCoordinator(medical_record_service=medical_record_service)
