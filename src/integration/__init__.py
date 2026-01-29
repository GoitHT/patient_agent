"""
集成层 - 提供微服务和单体模式的统一接口
Integration Layer - Unified interface for microservices and monolithic mode
"""

from .coordinator_adapter import get_coordinator
from .medical_record_adapter import get_medical_record_service

__all__ = [
    'get_coordinator',
    'get_medical_record_service',
]
