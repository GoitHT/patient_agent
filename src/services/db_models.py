"""
数据库模型定义 - 简化版5表结构
Database Models - Simplified 5-table structure for medical records
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Patient(Base):
    """患者信息表"""
    __tablename__ = 'patients'
    
    patient_id = Column(String(50), primary_key=True)
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    phone = Column(String(20))
    id_card = Column(String(20))
    created_at = Column(DateTime, default=datetime.now)
    
    # 关系
    medical_records = relationship("MedicalRecord", back_populates="patient")
    
    # 索引
    __table_args__ = (
        Index('idx_name', 'name'),
        Index('idx_phone', 'phone'),
    )


class MedicalRecord(Base):
    """病历主表 - 合并主诉、分诊、诊断、治疗"""
    __tablename__ = 'medical_records'
    
    record_id = Column(String(50), primary_key=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False)
    visit_date = Column(DateTime, nullable=False, default=datetime.now)
    
    # 分诊信息
    dept = Column(String(50))
    dept_display_name = Column(String(100))
    triage_priority = Column(Integer, default=5)
    triage_reason = Column(Text)
    
    # 主诉与症状
    chief_complaint = Column(Text)
    original_complaint = Column(Text)
    
    # 诊断信息
    diagnosis_name = Column(String(200))
    diagnosis_code = Column(String(50))
    diagnosis_reasoning = Column(Text)
    differential_diagnoses = Column(JSON)  # 鉴别诊断列表
    
    # 治疗方案
    treatment_plan = Column(Text)
    medications = Column(JSON)  # [{name, dosage, frequency}...]
    precautions = Column(Text)
    
    # 随访
    followup_plan = Column(Text)
    followup_date = Column(DateTime)
    
    # 状态与元数据
    status = Column(String(20), default='ongoing')  # ongoing/completed/cancelled
    run_id = Column(String(100))
    dataset_id = Column(Integer)
    original_case_id = Column(String(50))
    
    # 物理环境数据（JSON存储）
    physical_info = Column(JSON)  # {start_time, end_time, total_minutes, energy_change...}
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    patient = relationship("Patient", back_populates="medical_records")
    consultations = relationship("Consultation", back_populates="medical_record", cascade="all, delete-orphan")
    examinations = relationship("Examination", back_populates="medical_record", cascade="all, delete-orphan")
    system_logs = relationship("SystemLog", back_populates="medical_record", cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        Index('idx_patient', 'patient_id'),
        Index('idx_visit_date', 'visit_date'),
        Index('idx_dept', 'dept'),
        Index('idx_status', 'status'),
        Index('idx_patient_visit', 'patient_id', 'visit_date'),
    )


class Consultation(Base):
    """问诊与交互记录表 - 合并问诊QA、分诊对话、医患交互"""
    __tablename__ = 'consultations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String(50), ForeignKey('medical_records.record_id'), nullable=False)
    
    # 交互类型：triage(分诊), doctor_qa(医生问诊), nurse_qa(护士问诊)
    interaction_type = Column(String(20))
    
    # 参与者
    staff_id = Column(String(50))  # doctor_id/nurse_id
    staff_role = Column(String(20))  # doctor/nurse
    
    # 对话内容
    question_order = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    
    # 元数据
    node_name = Column(String(50))
    asked_at = Column(DateTime, default=datetime.now)
    
    # 关系
    medical_record = relationship("MedicalRecord", back_populates="consultations")
    
    # 索引
    __table_args__ = (
        Index('idx_record', 'record_id'),
        Index('idx_type', 'interaction_type'),
        Index('idx_staff', 'staff_id'),
    )


class Examination(Base):
    """检查检验表 - 合并检查申请和结果"""
    __tablename__ = 'examinations'
    
    exam_id = Column(String(50), primary_key=True)
    record_id = Column(String(50), ForeignKey('medical_records.record_id'), nullable=False)
    
    # 检查基本信息
    exam_name = Column(String(200))
    exam_type = Column(String(50))  # lab/imaging/functional/endoscopy
    category = Column(String(100))
    
    # 申请信息
    ordered_by = Column(String(50))
    ordered_at = Column(DateTime, default=datetime.now)
    priority = Column(Integer, default=5)
    
    # 结果信息
    result_text = Column(Text)
    summary = Column(Text)
    is_abnormal = Column(Boolean, default=False)
    key_findings = Column(JSON)  # ["发现1", "发现2"...]
    clinical_significance = Column(Text)
    source = Column(String(50))  # dataset/llm_generated
    
    # 状态
    status = Column(String(20), default='pending')  # pending/completed/cancelled
    reported_at = Column(DateTime)
    
    # 关系
    medical_record = relationship("MedicalRecord", back_populates="examinations")
    
    # 索引
    __table_args__ = (
        Index('idx_record', 'record_id'),
        Index('idx_status', 'status'),
        Index('idx_type', 'exam_type'),
    )


class SystemLog(Base):
    """系统日志表 - 合并审计日志、物理状态、转诊记录"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String(50), ForeignKey('medical_records.record_id'), nullable=True)
    
    # 日志类型：audit(审计), physical(物理状态), referral(转诊), event(事件)
    log_type = Column(String(20))
    
    # 基本信息
    entity_id = Column(String(50))  # patient_id/doctor_id/nurse_id
    entity_type = Column(String(20))  # patient/doctor/nurse/system
    
    # 日志内容（JSON灵活存储）
    log_data = Column(JSON)
    
    # 时间戳
    log_time = Column(DateTime, default=datetime.now)
    
    # 关系
    medical_record = relationship("MedicalRecord", back_populates="system_logs")
    
    # 索引
    __table_args__ = (
        Index('idx_record', 'record_id'),
        Index('idx_type', 'log_type'),
        Index('idx_entity', 'entity_id'),
        Index('idx_time', 'log_time'),
    )
