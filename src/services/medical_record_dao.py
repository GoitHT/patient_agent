"""
数据库访问层 - 简化版DAO
Database Access Object - Simplified DAO for medical records
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .db_models import Base, Patient, MedicalRecord, Consultation, Examination, SystemLog
from utils import get_logger, now_iso

logger = get_logger("hospital_agent.dao")


class MedicalRecordDAO:
    """医疗记录数据访问对象"""
    
    def __init__(self, connection_string: str, echo: bool = False):
        """
        初始化数据库连接
        
        Args:
            connection_string: 数据库连接字符串
                MySQL: "mysql+pymysql://user:password@host:port/database?charset=utf8mb4"
                PostgreSQL: "postgresql://user:password@host:port/database"
            echo: 是否打印SQL语句（调试用）
        """
        self.connection_string = connection_string
        
        # 创建引擎（连接池配置）
        self.engine = create_engine(
            connection_string,
            echo=echo,
            pool_pre_ping=True,  # 连接前检查有效性
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
        )
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )
        
        # 创建所有表
        Base.metadata.create_all(self.engine)
        
        logger.info(f"数据库连接已建立: {connection_string.split('@')[1] if '@' in connection_string else 'local'}")
    
    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话（上下文管理器）"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()
    
    # ===== 患者操作 =====
    
    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """创建或更新患者记录"""
        with self.get_session() as session:
            patient_id = patient_data['patient_id']
            
            # 检查患者是否已存在
            existing = session.query(Patient).filter_by(patient_id=patient_id).first()
            if existing:
                # 更新患者信息
                for key, value in patient_data.items():
                    if hasattr(existing, key) and key != 'patient_id':
                        setattr(existing, key, value)
                logger.info(f"更新患者信息: {patient_id}")
            else:
                # 创建新患者
                patient = Patient(**patient_data)
                session.add(patient)
                logger.info(f"创建患者记录: {patient_id}")
            
            return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """获取患者信息"""
        with self.get_session() as session:
            patient = session.query(Patient).filter_by(patient_id=patient_id).first()
            if not patient:
                return None
            
            return {
                "patient_id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "phone": patient.phone,
                "id_card": patient.id_card,
                "created_at": patient.created_at.isoformat() if patient.created_at else None,
            }
    
    # ===== 病历操作 =====
    
    def create_medical_record(self, record_data: Dict[str, Any]) -> str:
        """创建病历记录"""
        with self.get_session() as session:
            # 确保患者存在
            patient_id = record_data.get('patient_id')
            patient = session.query(Patient).filter_by(patient_id=patient_id).first()
            if not patient:
                raise ValueError(f"患者不存在: {patient_id}")
            
            # 创建病历
            record = MedicalRecord(**record_data)
            session.add(record)
            
            logger.info(f"创建病历: {record.record_id} (患者: {patient_id})")
            
            return record.record_id
    
    def get_medical_record(self, record_id: str, include_relations: bool = True) -> Optional[Dict[str, Any]]:
        """获取完整病历（含关联数据）"""
        with self.get_session() as session:
            record = session.query(MedicalRecord).filter_by(record_id=record_id).first()
            if not record:
                return None
            
            result = {
                "record_id": record.record_id,
                "patient_id": record.patient_id,
                "visit_date": record.visit_date.isoformat() if record.visit_date else None,
                "dept": record.dept,
                "dept_display_name": record.dept_display_name,
                "triage_priority": record.triage_priority,
                "triage_reason": record.triage_reason,
                "chief_complaint": record.chief_complaint,
                "original_complaint": record.original_complaint,
                "diagnosis_name": record.diagnosis_name,
                "diagnosis_code": record.diagnosis_code,
                "diagnosis_reasoning": record.diagnosis_reasoning,
                "differential_diagnoses": record.differential_diagnoses,
                "treatment_plan": record.treatment_plan,
                "medications": record.medications,
                "precautions": record.precautions,
                "followup_plan": record.followup_plan,
                "followup_date": record.followup_date.isoformat() if record.followup_date else None,
                "status": record.status,
                "run_id": record.run_id,
                "dataset_id": record.dataset_id,
                "original_case_id": record.original_case_id,
                "physical_info": record.physical_info,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            }
            
            # 包含关联数据
            if include_relations:
                result["consultations"] = [
                    {
                        "id": c.id,
                        "interaction_type": c.interaction_type,
                        "staff_id": c.staff_id,
                        "staff_role": c.staff_role,
                        "question_order": c.question_order,
                        "question": c.question,
                        "answer": c.answer,
                        "node_name": c.node_name,
                        "asked_at": c.asked_at.isoformat() if c.asked_at else None,
                    }
                    for c in record.consultations
                ]
                
                result["examinations"] = [
                    {
                        "exam_id": e.exam_id,
                        "exam_name": e.exam_name,
                        "exam_type": e.exam_type,
                        "category": e.category,
                        "ordered_by": e.ordered_by,
                        "ordered_at": e.ordered_at.isoformat() if e.ordered_at else None,
                        "priority": e.priority,
                        "result_text": e.result_text,
                        "summary": e.summary,
                        "is_abnormal": e.is_abnormal,
                        "key_findings": e.key_findings,
                        "clinical_significance": e.clinical_significance,
                        "source": e.source,
                        "status": e.status,
                        "reported_at": e.reported_at.isoformat() if e.reported_at else None,
                    }
                    for e in record.examinations
                ]
            
            return result
    
    def update_medical_record(self, record_id: str, update_data: Dict[str, Any]) -> bool:
        """更新病历信息"""
        with self.get_session() as session:
            record = session.query(MedicalRecord).filter_by(record_id=record_id).first()
            if not record:
                logger.warning(f"病历不存在: {record_id}")
                return False
            
            # 更新字段
            for key, value in update_data.items():
                if hasattr(record, key) and key not in ['record_id', 'created_at']:
                    setattr(record, key, value)
            
            record.updated_at = datetime.now()
            
            logger.info(f"更新病历: {record_id}")
            return True
    
    def get_patient_records(self, patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取患者的所有病历记录"""
        with self.get_session() as session:
            records = session.query(MedicalRecord)\
                .filter_by(patient_id=patient_id)\
                .order_by(MedicalRecord.visit_date.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "record_id": r.record_id,
                    "visit_date": r.visit_date.isoformat() if r.visit_date else None,
                    "dept": r.dept,
                    "chief_complaint": r.chief_complaint,
                    "diagnosis_name": r.diagnosis_name,
                    "status": r.status,
                }
                for r in records
            ]
    
    # ===== 问诊记录操作 =====
    
    def add_consultation(self, consultation_data: Dict[str, Any]) -> int:
        """添加问诊记录"""
        with self.get_session() as session:
            consultation = Consultation(**consultation_data)
            session.add(consultation)
            session.flush()  # 获取自动生成的ID
            
            logger.debug(f"添加问诊记录: {consultation.record_id}")
            return consultation.id
    
    def get_consultations(self, record_id: str) -> List[Dict[str, Any]]:
        """获取病历的所有问诊记录"""
        with self.get_session() as session:
            consultations = session.query(Consultation)\
                .filter_by(record_id=record_id)\
                .order_by(Consultation.question_order)\
                .all()
            
            return [
                {
                    "id": c.id,
                    "interaction_type": c.interaction_type,
                    "question": c.question,
                    "answer": c.answer,
                    "staff_role": c.staff_role,
                    "asked_at": c.asked_at.isoformat() if c.asked_at else None,
                }
                for c in consultations
            ]
    
    # ===== 检查检验操作 =====
    
    def add_examination(self, exam_data: Dict[str, Any]) -> str:
        """添加检查记录"""
        with self.get_session() as session:
            examination = Examination(**exam_data)
            session.add(examination)
            
            logger.debug(f"添加检查记录: {examination.exam_id}")
            return examination.exam_id
    
    def update_examination_result(self, exam_id: str, result_data: Dict[str, Any]) -> bool:
        """更新检查结果"""
        with self.get_session() as session:
            exam = session.query(Examination).filter_by(exam_id=exam_id).first()
            if not exam:
                return False
            
            for key, value in result_data.items():
                if hasattr(exam, key):
                    setattr(exam, key, value)
            
            exam.status = 'completed'
            exam.reported_at = datetime.now()
            
            logger.debug(f"更新检查结果: {exam_id}")
            return True
    
    def get_examinations(self, record_id: str) -> List[Dict[str, Any]]:
        """获取病历的所有检查记录"""
        with self.get_session() as session:
            exams = session.query(Examination)\
                .filter_by(record_id=record_id)\
                .order_by(Examination.ordered_at)\
                .all()
            
            return [
                {
                    "exam_id": e.exam_id,
                    "exam_name": e.exam_name,
                    "exam_type": e.exam_type,
                    "result_text": e.result_text,
                    "summary": e.summary,
                    "is_abnormal": e.is_abnormal,
                    "status": e.status,
                }
                for e in exams
            ]
    
    # ===== 系统日志操作 =====
    
    def log_event(self, log_data: Dict[str, Any]) -> int:
        """记录系统事件"""
        with self.get_session() as session:
            log = SystemLog(**log_data)
            session.add(log)
            session.flush()
            
            return log.id
    
    def get_logs(self, record_id: str = None, log_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """查询系统日志"""
        with self.get_session() as session:
            query = session.query(SystemLog)
            
            if record_id:
                query = query.filter_by(record_id=record_id)
            if log_type:
                query = query.filter_by(log_type=log_type)
            
            logs = query.order_by(SystemLog.log_time.desc()).limit(limit).all()
            
            return [
                {
                    "id": log.id,
                    "record_id": log.record_id,
                    "log_type": log.log_type,
                    "entity_id": log.entity_id,
                    "entity_type": log.entity_type,
                    "log_data": log.log_data,
                    "log_time": log.log_time.isoformat() if log.log_time else None,
                }
                for log in logs
            ]
    
    # ===== 统计查询 =====
    
    def get_daily_statistics(self, date: datetime = None) -> Dict[str, Any]:
        """获取每日统计数据"""
        if date is None:
            date = datetime.now()
        
        with self.get_session() as session:
            # 查询当天的病历
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            records = session.query(MedicalRecord)\
                .filter(MedicalRecord.visit_date >= start_of_day)\
                .filter(MedicalRecord.visit_date <= end_of_day)\
                .all()
            
            # 统计
            total_count = len(records)
            completed_count = sum(1 for r in records if r.status == 'completed')
            
            dept_stats = {}
            for record in records:
                dept = record.dept or 'unknown'
                if dept not in dept_stats:
                    dept_stats[dept] = 0
                dept_stats[dept] += 1
            
            return {
                "date": date.strftime("%Y-%m-%d"),
                "total_visits": total_count,
                "completed_visits": completed_count,
                "by_department": dept_stats,
            }
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
        logger.info("数据库连接已关闭")
