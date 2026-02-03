"""
医院协调器适配器
Hospital Coordinator Adapter

提供医院协调器实例
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coordination import HospitalCoordinator
    from services.medical_record import MedicalRecordService


def get_coordinator(
    medical_record_service: 'MedicalRecordService'
) -> 'HospitalCoordinator':
    """
    获取医院协调器实例
    
    Args:
        medical_record_service: 医疗记录服务
    
    Returns:
        HospitalCoordinator实例
    """
    from coordination import HospitalCoordinator
    return HospitalCoordinator(medical_record_service=medical_record_service)
