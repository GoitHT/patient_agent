"""
数据库模型定义 - 6表结构
Database Models - 6-table structure
  patients / visits / medical_cases / examinations / exam_items / case_qa_records
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, Numeric, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Patient(Base):
    """患者基本信息表"""
    __tablename__ = 'patients'

    patient_id = Column(String(50), primary_key=True, comment="患者唯一标识ID（主键）")
    name = Column(String(100), comment="患者姓名")
    gender = Column(String(10), comment="患者性别")
    age = Column(Integer, comment="患者年龄（冗余字段，可选存储）")
    ethnicity = Column(String(50), comment="民族")
    occupation = Column(String(100), comment="职业")
    phone = Column(String(20), comment="联系电话")
    id_card = Column(String(20), comment="身份证号")
    created_at = Column(DateTime, default=datetime.now, comment="记录创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="记录更新时间")

    # 关系
    visits = relationship("Visit", back_populates="patient")

    # 索引
    __table_args__ = (
        Index('idx_patients_name', 'name'),
        Index('idx_patients_phone', 'phone'),
    )


class Visit(Base):
    """就诊记录表"""
    __tablename__ = 'visits'

    visit_id = Column(String(50), primary_key=True, comment="就诊记录唯一ID（主键）")
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, comment="关联患者ID")
    outpatient_no = Column(String(50), comment="门诊号/就诊号")
    visit_date = Column(DateTime, nullable=False, default=datetime.now, comment="就诊日期时间")
    dept = Column(String(50), comment="就诊科室")
    attending_doctor_id = Column(String(50), comment="主治医生ID")
    attending_doctor_name = Column(String(100), comment="主治医生姓名")
    triage_nurse_id = Column(String(50), comment="分诊护士ID")
    triage_nurse_name = Column(String(100), comment="分诊护士姓名")
    status = Column(String(20), default='ongoing', comment="就诊状态（ongoing/completed）")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")

    # 关系
    patient = relationship("Patient", back_populates="visits")
    medical_cases = relationship("MedicalCase", back_populates="visit")

    # 索引
    __table_args__ = (
        Index('idx_visits_patient_id', 'patient_id'),
        Index('idx_visits_outpatient_no', 'outpatient_no'),
        Index('idx_visits_visit_date', 'visit_date'),
        Index('idx_visits_status', 'status'),
    )


class MedicalCase(Base):
    """病历表"""
    __tablename__ = 'medical_cases'

    case_id = Column(String(50), primary_key=True, comment="病历唯一编号")
    visit_id = Column(String(50), ForeignKey('visits.visit_id'), nullable=False, comment="关联就诊记录ID")

    history_narrator = Column(String(100), comment="病史陈述者（患者本人/家属等）")
    chief_complaint = Column(Text, comment="主诉")

    present_illness_detail = Column(Text, comment="现病史详细描述")
    present_illness_onset = Column(Text, comment="现病史-起病情况")
    present_illness_course = Column(Text, comment="现病史-病程")
    present_illness_progression = Column(Text, comment="现病史-病情发展")

    past_history_disease = Column(Text, comment="既往史-疾病史")
    past_history_surgery = Column(Text, comment="既往史-手术史")
    past_history_allergy = Column(Text, comment="既往史-过敏史")
    past_history_vaccination = Column(Text, comment="既往史-预防接种史")
    past_history_trauma = Column(Text, comment="既往史-外伤史")

    personal_history = Column(Text, comment="个人史综合描述")
    alcohol_history = Column(Text, comment="个人史-饮酒史")
    smoking_history = Column(Text, comment="个人史-吸烟史")
    menstrual_history = Column(Text, comment="个人史-月经史")
    marital_fertility_history = Column(Text, comment="婚育史")

    family_history_father = Column(Text, comment="家族史-父亲健康情况")
    family_history_mother = Column(Text, comment="家族史-母亲健康情况")
    family_history_siblings = Column(Text, comment="家族史-兄弟姐妹健康情况")
    family_history_disease = Column(Text, comment="家族史-遗传或家族疾病情况")

    pe_vital_signs = Column(Text, comment="体格检查-生命体征")
    pe_skin_mucosa = Column(Text, comment="体格检查-皮肤黏膜")
    pe_superficial_lymph_nodes = Column(Text, comment="体格检查-浅表淋巴结")
    pe_head_neck = Column(Text, comment="体格检查-头颈部")
    pe_cardiopulmonary_vascular = Column(Text, comment="体格检查-心肺血管系统")
    pe_abdomen = Column(Text, comment="体格检查-腹部")
    pe_spine_limbs = Column(Text, comment="体格检查-脊柱四肢")
    pe_nervous_system = Column(Text, comment="体格检查-神经系统")

    auxiliary_examination = Column(Text, comment="辅助检查结果汇总")

    preliminary_diagnosis = Column(Text, comment="初步诊断")
    diagnosis_basis = Column(Text, comment="诊断依据")
    treatment_principle = Column(Text, comment="治疗原则")
    treatment_plan = Column(Text, comment="治疗方案")
    medications = Column(Text, comment="治疗药物（文本）")
    medical_advice = Column(Text, comment="医嘱")
    followup_plan = Column(Text, comment="随访计划")

    doctor_patient_qa_ref_question = Column(Text, comment="医患问答参考-问题")
    doctor_patient_qa_ref_answer = Column(Text, comment="医患问答参考-答案")

    # 标准病历参考预留字段
    std_record_chief_complaint = Column(Text, comment="标准病历参考-主诉（预留字段）")
    std_record_present_illness = Column(Text, comment="标准病历参考-现病史（预留字段）")
    std_record_past_history = Column(Text, comment="标准病历参考-既往史（预留字段）")
    std_record_physical_exam = Column(Text, comment="标准病历参考-体格检查（预留字段）")
    std_record_aux_exam = Column(Text, comment="标准病历参考-辅助检查（预留字段）")
    std_record_diagnosis_result = Column(Text, comment="标准病历参考-诊断结果（预留字段）")

    # 教学扩展字段
    advanced_question = Column(Text, comment="高级问题题目（教学扩展字段）")
    advanced_answer = Column(Text, comment="高级问题答案（教学扩展字段）")
    exam_question = Column(Text, comment="考试题目（教学扩展字段）")
    exam_answer = Column(Text, comment="考试题目答案（教学扩展字段）")

    status = Column(String(20), default='draft', comment="病历状态（draft/final）")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    # 关系
    visit = relationship("Visit", back_populates="medical_cases")
    examinations = relationship("Examination", back_populates="medical_case", cascade="all, delete-orphan")
    qa_records = relationship("CaseQARecord", back_populates="medical_case", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_cases_visit_id', 'visit_id'),
        Index('idx_cases_status', 'status'),
    )


class Examination(Base):
    """检查记录表"""
    __tablename__ = 'examinations'

    exam_id = Column(String(50), primary_key=True, comment="检查记录ID")
    case_id = Column(String(50), ForeignKey('medical_cases.case_id'), nullable=False, comment="关联病历ID")

    exam_name = Column(String(200), comment="检查名称")
    exam_type = Column(String(50), comment="检查类型（lab/imaging等）")
    lab_doctor_id = Column(String(50), comment="检查医生ID")
    lab_doctor_name = Column(String(100), comment="检查医生姓名")
    ordered_at = Column(DateTime, default=datetime.now, comment="开单时间")
    reported_at = Column(DateTime, comment="出报告时间")
    result_text = Column(Text, comment="检查结果原始文本")
    summary = Column(Text, comment="检查结果总结")
    is_abnormal = Column(Boolean, default=False, comment="是否异常")
    status = Column(String(20), default='pending', comment="检查状态（pending/completed）")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")

    # 关系
    medical_case = relationship("MedicalCase", back_populates="examinations")
    items = relationship("ExamItem", back_populates="examination", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_exams_case_id', 'case_id'),
        Index('idx_exams_status', 'status'),
        Index('idx_exams_type', 'exam_type'),
    )


class ExamItem(Base):
    """检查项目明细表"""
    __tablename__ = 'exam_items'

    item_id = Column(String(50), primary_key=True, comment="检查项目ID")
    exam_id = Column(String(50), ForeignKey('examinations.exam_id'), nullable=False, comment="所属检查ID")

    item_name = Column(String(200), comment="检查项目名称")
    value_numeric = Column(Numeric(10, 2), comment="数值型结果（如血糖5.6）")
    value_text = Column(Text, comment="非数值型结果（阳性/阴性/描述性文本）")
    value_type = Column(String(20), comment="结果类型（numeric/qualitative/text/grade）")
    unit = Column(String(50), comment="单位（仅数值型有效）")
    ref_range = Column(String(100), comment="参考范围（仅数值型有效）")
    is_abnormal = Column(Boolean, default=False, comment="是否异常")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")

    # 关系
    examination = relationship("Examination", back_populates="items")

    # 索引
    __table_args__ = (
        Index('idx_items_exam_id', 'exam_id'),
    )


class CaseQARecord(Base):
    """医患问答记录表"""
    __tablename__ = 'case_qa_records'

    qa_id = Column(String(50), primary_key=True, comment="医患问答记录唯一ID（主键）")
    case_id = Column(String(50), ForeignKey('medical_cases.case_id'), nullable=False, comment="关联病历ID（外键）")
    role = Column(String(20), comment="角色（doctor/patient）")
    content = Column(Text, comment="对话内容")
    round_index = Column(Integer, comment="对话轮次序号")
    created_at = Column(DateTime, default=datetime.now, comment="记录创建时间")

    # 关系
    medical_case = relationship("MedicalCase", back_populates="qa_records")

    # 索引
    __table_args__ = (
        Index('idx_qa_case_id', 'case_id'),
    )
