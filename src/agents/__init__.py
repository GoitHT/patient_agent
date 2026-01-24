"""Agents module - 包含患者、医生、护士、检验科智能体"""
from __future__ import annotations

from .patient_agent import PatientAgent
from .doctor_agent import DoctorAgent
from .nurse_agent import NurseAgent
from .lab_agent import LabAgent

__all__ = ["PatientAgent", "DoctorAgent", "NurseAgent", "LabAgent"]
