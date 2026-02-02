"""
数据库模型定义 - 3表结构（以门诊号为主线）
Database Models - 3-table structure based on outpatient_no
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, DateTime, Date, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Patient(Base):
    """患者基本信息表 - 以门诊号为主键"""
    __tablename__ = 'patients'
    
    outpatient_no = Column(String(50), primary_key=True)  # 门诊号（业务主键）
    patient_id = Column(String(50))  # 患者唯一ID（可跨门诊）
    name = Column(String(100))
    age = Column(Integer)
    gender = Column(String(10))
    phone = Column(String(20))
    id_card = Column(String(20))
    created_at = Column(DateTime, default=datetime.now)
    
    # 关系
    medical_cases = relationship("MedicalCase", back_populates="patient")
    examinations = relationship("Examination", back_populates="patient")
    
    # 索引
    __table_args__ = (
        Index('idx_patient_id', 'patient_id'),
        Index('idx_name', 'name'),
        Index('idx_phone', 'phone'),
    )


class MedicalCase(Base):
    """病例表 - 完整就诊过程（集成医生问诊和系统日志）"""
    __tablename__ = 'medical_cases'
    
    case_id = Column(String(50), primary_key=True)
    outpatient_no = Column(String(50), ForeignKey('patients.outpatient_no'), nullable=False)
    
    # 就诊基本信息
    visit_date = Column(Date, nullable=False, default=datetime.now)
    dept = Column(String(50))
    
    # 主诉与现病史
    chief_complaint = Column(Text)
    present_illness = Column(Text)
    
    # 医生问诊记录（JSON集中存储）
    doctor_qa_records = Column(JSON)  # [{question_order, question, answer, asked_at}...]
    
    # 诊断信息（已删除 diagnosis_code）
    diagnosis_name = Column(String(200))
    diagnosis_reason = Column(Text)  # 诊断依据/推理过程
    
    # 治疗与处置
    treatment_plan = Column(Text)
    medications = Column(JSON)  # [{name, dosage, frequency}...]
    medical_advice = Column(Text)
    
    # 随访与转归
    followup_plan = Column(Text)
    followup_date = Column(Date)
    outcome = Column(String(50))
    
    # 状态
    status = Column(String(20), default='ongoing')  # ongoing/completed
    
    # 统一日志与过程记录（JSON集中存储）
    case_logs = Column(JSON)  # [{log_time, log_type, entity_id, entity_type, log_data}...]
    
    # 元数据
    run_id = Column(String(100))
    dataset_id = Column(Integer)
    original_case_id = Column(String(50))
    
    # 时间
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关系
    patient = relationship("Patient", back_populates="medical_cases")
    examinations = relationship("Examination", back_populates="medical_case", cascade="all, delete-orphan")
    
    # 索引
    __table_args__ = (
        Index('idx_outpatient_no', 'outpatient_no'),
        Index('idx_visit_date', 'visit_date'),
        Index('idx_dept', 'dept'),
        Index('idx_status', 'status'),
    )


class Examination(Base):
    """检查项目表 - 直接关联门诊号"""
    __tablename__ = 'examinations'
    
    exam_id = Column(String(100), primary_key=True)
    outpatient_no = Column(String(50), ForeignKey('patients.outpatient_no'), nullable=False)
    case_id = Column(String(50), ForeignKey('medical_cases.case_id'), nullable=True)  # 可选关联
    
    # 检查基本信息
    exam_name = Column(String(200))
    exam_type = Column(String(50))  # lab/imaging/functional
    
    # 时间
    ordered_at = Column(DateTime, default=datetime.now)
    reported_at = Column(DateTime)
    
    # 结果信息
    result_text = Column(Text)
    summary = Column(Text)
    is_abnormal = Column(Boolean, default=False)
    key_findings = Column(JSON)  # ["发现1", "发现2"...]
    
    # 状态
    status = Column(String(20), default='pending')  # pending/completed/cancelled
    
    # 关系
    patient = relationship("Patient", back_populates="examinations")
    medical_case = relationship("MedicalCase", back_populates="examinations")
    
    # 索引
    __table_args__ = (
        Index('idx_outpatient_no', 'outpatient_no'),
        Index('idx_case_id', 'case_id'),
        Index('idx_status', 'status'),
        Index('idx_type', 'exam_type'),
    )
