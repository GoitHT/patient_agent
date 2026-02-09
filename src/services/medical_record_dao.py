"""
数据库访问层 - 3表结构（基于门诊号）
Database Access Object - 3-table structure based on outpatient_no
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .db_models import Base, Patient, MedicalCase, Examination
from utils import get_logger, now_iso

logger = get_logger("hospital_agent.dao")


class MedicalRecordDAO:
    """医疗记录数据访问对象"""
    
    def __init__(self, connection_string: str):
        """
        初始化数据库连接
        
        Args:
            connection_string: 数据库连接字符串
                MySQL: "mysql+pymysql://user:password@host:port/database?charset=utf8mb4"
                PostgreSQL: "postgresql://user:password@host:port/database"
        """
        self.connection_string = connection_string
        
        # 创建引擎（连接池配置）
        self.engine = create_engine(
            connection_string,
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
        
        # 不显示连接提示，由initializer统一管理
    
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
        """创建患者记录（以门诊号为主键）"""
        with self.get_session() as session:
            outpatient_no = patient_data['outpatient_no']
            
            # 检查门诊号是否已存在
            existing = session.query(Patient).filter_by(outpatient_no=outpatient_no).first()
            if existing:
                logger.warning(f"门诊号已存在: {outpatient_no}")
                # 更新患者信息
                for key, value in patient_data.items():
                    if hasattr(existing, key) and key != 'outpatient_no':
                        setattr(existing, key, value)
                logger.info(f"更新患者信息: {outpatient_no}")
            else:
                # 创建新患者
                patient = Patient(**patient_data)
                session.add(patient)
                logger.info(f"创建患者记录: {outpatient_no}")
            
            return outpatient_no
    
    def get_patient(self, outpatient_no: str) -> Optional[Dict[str, Any]]:
        """根据门诊号获取患者信息"""
        with self.get_session() as session:
            patient = session.query(Patient).filter_by(outpatient_no=outpatient_no).first()
            if not patient:
                return None
            
            return {
                "outpatient_no": patient.outpatient_no,
                "patient_id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "phone": patient.phone,
                "id_card": patient.id_card,
                "created_at": patient.created_at.isoformat() if patient.created_at else None,
            }
    
    def get_patient_by_id(self, patient_id: str) -> List[Dict[str, Any]]:
        """根据患者ID获取所有门诊记录"""
        with self.get_session() as session:
            patients = session.query(Patient).filter_by(patient_id=patient_id).all()
            
            return [
                {
                    "outpatient_no": p.outpatient_no,
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in patients
            ]
    
    # ===== 病例操作 =====
    
    def create_medical_case(self, case_data: Dict[str, Any]) -> str:
        """创建病例记录（每次就诊创建新病例，但检查case_id是否重复）"""
        with self.get_session() as session:
            # 确保患者（门诊号）存在
            outpatient_no = case_data.get('outpatient_no')
            patient = session.query(Patient).filter_by(outpatient_no=outpatient_no).first()
            if not patient:
                raise ValueError(f"门诊号不存在: {outpatient_no}")
            
            # 按case_id检查是否已存在（避免重复创建同一病例）
            case_id = case_data.get('case_id')
            existing_case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if existing_case:
                logger.info(f"该病例已存在，跳过创建: {case_id} (门诊号: {outpatient_no})")
                return existing_case.case_id
            
            # 创建新病例（同一门诊号可以有多个病例，对应多次就诊）
            case = MedicalCase(**case_data)
            session.add(case)
            
            logger.debug(f"创建病例: {case.case_id} (门诊号: {outpatient_no})")
            
            return case.case_id
    
    def get_medical_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """获取完整病例"""
        with self.get_session() as session:
            case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if not case:
                return None
            
            result = {
                "case_id": case.case_id,
                "outpatient_no": case.outpatient_no,
                "visit_date": case.visit_date.isoformat() if case.visit_date else None,
                "dept": case.dept,
                "chief_complaint": case.chief_complaint,
                "present_illness": case.present_illness,
                "doctor_qa_records": case.doctor_qa_records or [],
                "diagnosis_name": case.diagnosis_name,
                "diagnosis_reason": case.diagnosis_reason,
                "treatment_plan": case.treatment_plan,
                "medications": case.medications or [],
                "medical_advice": case.medical_advice,
                "followup_plan": case.followup_plan,
                "followup_date": case.followup_date.isoformat() if case.followup_date else None,
                "outcome": case.outcome,
                "status": case.status,
                "case_logs": case.case_logs or [],
                "run_id": case.run_id,
                "dataset_id": case.dataset_id,
                "original_case_id": case.original_case_id,
                "created_at": case.created_at.isoformat() if case.created_at else None,
                "updated_at": case.updated_at.isoformat() if case.updated_at else None,
            }
            
            # 获取关联的检查记录
            examinations = session.query(Examination).filter_by(case_id=case_id).all()
            result["examinations"] = [
                {
                    "exam_id": e.exam_id,
                    "exam_name": e.exam_name,
                    "exam_type": e.exam_type,
                    "ordered_at": e.ordered_at.isoformat() if e.ordered_at else None,
                    "reported_at": e.reported_at.isoformat() if e.reported_at else None,
                    "result_text": e.result_text,
                    "summary": e.summary,
                    "is_abnormal": e.is_abnormal,
                    "key_findings": e.key_findings or [],
                    "status": e.status,
                }
                for e in examinations
            ]
            
            return result
    
    def update_medical_case(self, case_id: str, update_data: Dict[str, Any]) -> bool:
        """更新病例信息"""
        with self.get_session() as session:
            case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if not case:
                logger.warning(f"病例不存在: {case_id}")
                return False
            
            # 更新字段
            for key, value in update_data.items():
                if hasattr(case, key) and key not in ['case_id', 'created_at']:
                    setattr(case, key, value)
            
            case.updated_at = datetime.now()
            
            logger.debug(f"更新病例: {case_id}")
            return True
    
    def get_cases_by_outpatient_no(self, outpatient_no: str) -> List[Dict[str, Any]]:
        """获取某门诊号的所有病例"""
        with self.get_session() as session:
            cases = session.query(MedicalCase)\
                .filter_by(outpatient_no=outpatient_no)\
                .order_by(MedicalCase.visit_date.desc())\
                .all()
            
            return [
                {
                    "case_id": c.case_id,
                    "visit_date": c.visit_date.isoformat() if c.visit_date else None,
                    "dept": c.dept,
                    "chief_complaint": c.chief_complaint,
                    "diagnosis_name": c.diagnosis_name,
                    "status": c.status,
                }
                for c in cases
            ]
    
    # ===== 医生问诊记录操作 =====
    
    def add_doctor_qa(self, case_id: str, qa_record: Dict[str, Any]) -> bool:
        """添加医生问诊记录到病例的JSON字段"""
        with self.get_session() as session:
            case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if not case:
                logger.warning(f"病例不存在: {case_id}")
                return False
            
            # 初始化JSON数组
            if case.doctor_qa_records is None:
                case.doctor_qa_records = []
            
            # 添加问诊记录
            case.doctor_qa_records.append(qa_record)
            case.updated_at = datetime.now()
            
            # 标记为已修改（SQLAlchemy需要）
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(case, "doctor_qa_records")
            
            logger.debug(f"添加问诊记录到病例: {case_id}")
            return True
    
    def add_case_log(self, case_id: str, log_entry: Dict[str, Any]) -> bool:
        """添加日志到病例的JSON字段"""
        with self.get_session() as session:
            case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if not case:
                logger.warning(f"病例不存在: {case_id}")
                return False
            
            # 初始化JSON数组
            if case.case_logs is None:
                case.case_logs = []
            
            # 添加日志
            log_entry['log_time'] = log_entry.get('log_time', now_iso())
            case.case_logs.append(log_entry)
            case.updated_at = datetime.now()
            
            # 标记为已修改
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(case, "case_logs")
            
            logger.debug(f"添加日志到病例: {case_id}")
            return True
    
    # ===== 检查检验操作 =====
    
    def add_examination(self, exam_data: Dict[str, Any]) -> str:
        """添加检查记录（如果已存在则更新）"""
        with self.get_session() as session:
            exam_id = exam_data.get('exam_id')
            
            # 检查是否已存在
            existing_exam = session.query(Examination).filter_by(exam_id=exam_id).first()
            
            if existing_exam:
                # 如果已存在，更新记录
                logger.debug(f"检查记录已存在，更新: {exam_id}")
                for key, value in exam_data.items():
                    if hasattr(existing_exam, key) and key != 'exam_id':
                        setattr(existing_exam, key, value)
                return exam_id
            else:
                # 如果不存在，插入新记录
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
    
    def get_examinations_by_outpatient_no(self, outpatient_no: str) -> List[Dict[str, Any]]:
        """获取某门诊号的所有检查记录"""
        with self.get_session() as session:
            exams = session.query(Examination)\
                .filter_by(outpatient_no=outpatient_no)\
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
                    "ordered_at": e.ordered_at.isoformat() if e.ordered_at else None,
                }
                for e in exams
            ]
    
    def get_examinations_by_case_id(self, case_id: str) -> List[Dict[str, Any]]:
        """获取某病例的所有检查记录"""
        with self.get_session() as session:
            exams = session.query(Examination)\
                .filter_by(case_id=case_id)\
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
                    "key_findings": e.key_findings or [],
                    "status": e.status,
                }
                for e in exams
            ]
    
    # ===== 统计查询 =====
    
    def get_daily_statistics(self, target_date: date = None) -> Dict[str, Any]:
        """获取每日统计数据"""
        if target_date is None:
            target_date = date.today()
        
        with self.get_session() as session:
            # 查询当天的病例
            cases = session.query(MedicalCase)\
                .filter(MedicalCase.visit_date == target_date)\
                .all()
            
            # 统计
            total_count = len(cases)
            completed_count = sum(1 for c in cases if c.status == 'completed')
            
            dept_stats = {}
            for case in cases:
                dept = case.dept or 'unknown'
                if dept not in dept_stats:
                    dept_stats[dept] = 0
                dept_stats[dept] += 1
            
            return {
                "date": target_date.strftime("%Y-%m-%d"),
                "total_visits": total_count,
                "completed_visits": completed_count,
                "by_department": dept_stats,
            }
    
    def get_all_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取所有患者列表"""
        with self.get_session() as session:
            patients = session.query(Patient)\
                .order_by(Patient.created_at.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "outpatient_no": p.outpatient_no,
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in patients
            ]
    
    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
        logger.info("数据库连接已关闭")

