"""
数据库访问层 - 6表结构
Database Access Object - 6-table structure
  patients / visits / medical_cases / examinations / exam_items / case_qa_records
"""
from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .db_models import Base, Patient, Visit, MedicalCase, Examination, ExamItem, CaseQARecord
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

        self.engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
        )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

        Base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self) -> Iterator[Session]:
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
        """创建或更新患者记录（以 patient_id 为主键）"""
        with self.get_session() as session:
            patient_id = patient_data['patient_id']

            existing = session.query(Patient).filter_by(patient_id=patient_id).first()
            if existing:
                for key, value in patient_data.items():
                    if hasattr(existing, key) and key != 'patient_id':
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
                logger.info(f"更新患者信息: {patient_id}")
            else:
                patient = Patient(**patient_data)
                session.add(patient)
                logger.info(f"创建患者: {patient_id}")

            return patient_id

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """根据 patient_id 获取患者信息"""
        with self.get_session() as session:
            patient = session.query(Patient).filter_by(patient_id=patient_id).first()
            if not patient:
                return None
            return {
                "patient_id": patient.patient_id,
                "name": patient.name,
                "gender": patient.gender,
                "age": patient.age,
                "ethnicity": patient.ethnicity,
                "occupation": patient.occupation,
                "phone": patient.phone,
                "id_card": patient.id_card,
                "created_at": patient.created_at.isoformat() if patient.created_at else None,
                "updated_at": patient.updated_at.isoformat() if patient.updated_at else None,
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
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "gender": p.gender,
                    "age": p.age,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in patients
            ]

    # ===== 就诊记录操作 =====

    def create_visit(self, visit_data: Dict[str, Any]) -> str:
        """创建就诊记录"""
        with self.get_session() as session:
            visit_id = visit_data.get('visit_id') or str(uuid.uuid4())
            visit_data['visit_id'] = visit_id

            # 确认患者存在
            patient_id = visit_data.get('patient_id')
            patient = session.query(Patient).filter_by(patient_id=patient_id).first()
            if not patient:
                raise ValueError(f"患者不存在: {patient_id}")

            existing = session.query(Visit).filter_by(visit_id=visit_id).first()
            if existing:
                for key, value in visit_data.items():
                    if hasattr(existing, key) and key != 'visit_id':
                        setattr(existing, key, value)
                logger.info(f"更新就诊记录: {visit_id}")
            else:
                visit = Visit(**visit_data)
                session.add(visit)
                logger.info(f"创建就诊记录: {visit_id}")

            return visit_id

    def update_visit(self, visit_id: str, update_data: Dict[str, Any]) -> bool:
        """更新就诊记录"""
        with self.get_session() as session:
            visit = session.query(Visit).filter_by(visit_id=visit_id).first()
            if not visit:
                logger.warning(f"就诊记录不存在: {visit_id}")
                return False
            for key, value in update_data.items():
                if hasattr(visit, key) and key != 'visit_id':
                    setattr(visit, key, value)
            logger.debug(f"更新就诊记录: {visit_id}")
            return True

    def get_visit(self, visit_id: str) -> Optional[Dict[str, Any]]:
        """获取就诊记录"""
        with self.get_session() as session:
            visit = session.query(Visit).filter_by(visit_id=visit_id).first()
            if not visit:
                return None
            return {
                "visit_id": visit.visit_id,
                "patient_id": visit.patient_id,
                "outpatient_no": visit.outpatient_no,
                "visit_date": visit.visit_date.isoformat() if visit.visit_date else None,
                "dept": visit.dept,
                "attending_doctor_id": visit.attending_doctor_id,
                "attending_doctor_name": visit.attending_doctor_name,
                "triage_nurse_id": visit.triage_nurse_id,
                "triage_nurse_name": visit.triage_nurse_name,
                "status": visit.status,
                "created_at": visit.created_at.isoformat() if visit.created_at else None,
            }

    def get_visits_by_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """获取某患者的所有就诊记录"""
        with self.get_session() as session:
            visits = session.query(Visit)\
                .filter_by(patient_id=patient_id)\
                .order_by(Visit.visit_date.desc())\
                .all()
            return [
                {
                    "visit_id": v.visit_id,
                    "outpatient_no": v.outpatient_no,
                    "visit_date": v.visit_date.isoformat() if v.visit_date else None,
                    "dept": v.dept,
                    "status": v.status,
                }
                for v in visits
            ]

    def get_next_visit_sequence(self, patient_id: str, date_str: str) -> int:
        """获取某患者某日期的下一个就诊流水号（从001开始）"""
        prefix = f"OPD-{patient_id}-{date_str}-"
        with self.get_session() as session:
            records = session.query(Visit.outpatient_no).filter(
                Visit.patient_id == patient_id,
                Visit.outpatient_no.like(f"{prefix}%")
            ).all()

            if not records:
                return 1

            max_seq = 0
            for (outpatient_no,) in records:
                try:
                    seq = int(str(outpatient_no).split("-")[-1])
                    if seq > max_seq:
                        max_seq = seq
                except (ValueError, TypeError, IndexError):
                    continue

            return max_seq + 1

    # ===== 病历操作 =====

    def create_medical_case(self, case_data: Dict[str, Any]) -> str:
        """创建或更新病历记录"""
        with self.get_session() as session:
            # 确认就诊记录存在
            visit_id = case_data.get('visit_id')
            visit = session.query(Visit).filter_by(visit_id=visit_id).first()
            if not visit:
                raise ValueError(f"就诊记录不存在: {visit_id}")

            case_id = case_data.get('case_id')
            existing_case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if existing_case:
                for key, value in case_data.items():
                    if key in ('case_id', 'created_at'):
                        continue
                    if hasattr(existing_case, key):
                        setattr(existing_case, key, value)
                existing_case.updated_at = datetime.now()
                logger.info(f"更新病历: {case_id}")
                return existing_case.case_id

            case = MedicalCase(**case_data)
            session.add(case)
            logger.debug(f"创建病历: {case.case_id}")
            return case.case_id

    def get_medical_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """获取完整病历"""
        with self.get_session() as session:
            case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if not case:
                return None

            result = {
                "case_id": case.case_id,
                "visit_id": case.visit_id,
                "history_narrator": case.history_narrator,
                "chief_complaint": case.chief_complaint,
                "present_illness_detail": case.present_illness_detail,
                "present_illness_onset": case.present_illness_onset,
                "present_illness_course": case.present_illness_course,
                "present_illness_progression": case.present_illness_progression,
                "past_history_disease": case.past_history_disease,
                "past_history_surgery": case.past_history_surgery,
                "past_history_allergy": case.past_history_allergy,
                "past_history_vaccination": case.past_history_vaccination,
                "past_history_trauma": case.past_history_trauma,
                "personal_history": case.personal_history,
                "alcohol_history": case.alcohol_history,
                "smoking_history": case.smoking_history,
                "menstrual_history": case.menstrual_history,
                "marital_fertility_history": case.marital_fertility_history,
                "family_history_father": case.family_history_father,
                "family_history_mother": case.family_history_mother,
                "family_history_siblings": case.family_history_siblings,
                "family_history_disease": case.family_history_disease,
                "pe_vital_signs": case.pe_vital_signs,
                "pe_skin_mucosa": case.pe_skin_mucosa,
                "pe_superficial_lymph_nodes": case.pe_superficial_lymph_nodes,
                "pe_head_neck": case.pe_head_neck,
                "pe_cardiopulmonary_vascular": case.pe_cardiopulmonary_vascular,
                "pe_abdomen": case.pe_abdomen,
                "pe_spine_limbs": case.pe_spine_limbs,
                "pe_nervous_system": case.pe_nervous_system,
                "auxiliary_examination": case.auxiliary_examination,
                "preliminary_diagnosis": case.preliminary_diagnosis,
                "diagnosis_basis": case.diagnosis_basis,
                "treatment_principle": case.treatment_principle,
                "treatment_plan": case.treatment_plan,
                "medications": case.medications,
                "medical_advice": case.medical_advice,
                "followup_plan": case.followup_plan,
                "doctor_patient_qa_ref_question": case.doctor_patient_qa_ref_question,
                "doctor_patient_qa_ref_answer": case.doctor_patient_qa_ref_answer,
                "std_record_chief_complaint": case.std_record_chief_complaint,
                "std_record_present_illness": case.std_record_present_illness,
                "std_record_past_history": case.std_record_past_history,
                "std_record_physical_exam": case.std_record_physical_exam,
                "std_record_aux_exam": case.std_record_aux_exam,
                "std_record_diagnosis_result": case.std_record_diagnosis_result,
                "advanced_question": case.advanced_question,
                "advanced_answer": case.advanced_answer,
                "exam_question": case.exam_question,
                "exam_answer": case.exam_answer,
                "status": case.status,
                "created_at": case.created_at.isoformat() if case.created_at else None,
                "updated_at": case.updated_at.isoformat() if case.updated_at else None,
            }

            # 关联检查记录
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
                    "status": e.status,
                }
                for e in examinations
            ]

            return result

    def update_medical_case(self, case_id: str, update_data: Dict[str, Any]) -> bool:
        """更新病历信息"""
        with self.get_session() as session:
            case = session.query(MedicalCase).filter_by(case_id=case_id).first()
            if not case:
                logger.warning(f"病历不存在: {case_id}")
                return False
            for key, value in update_data.items():
                if hasattr(case, key) and key not in ['case_id', 'created_at']:
                    setattr(case, key, value)
            case.updated_at = datetime.now()
            logger.debug(f"更新病历: {case_id}")
            return True

    def get_cases_by_visit_id(self, visit_id: str) -> List[Dict[str, Any]]:
        """获取某就诊记录下的所有病历"""
        with self.get_session() as session:
            cases = session.query(MedicalCase)\
                .filter_by(visit_id=visit_id)\
                .all()
            return [
                {
                    "case_id": c.case_id,
                    "chief_complaint": c.chief_complaint,
                    "preliminary_diagnosis": c.preliminary_diagnosis,
                    "status": c.status,
                }
                for c in cases
            ]

    # ===== 检查记录操作 =====

    def add_examination(self, exam_data: Dict[str, Any]) -> str:
        """添加或更新检查记录"""
        with self.get_session() as session:
            exam_id = exam_data.get('exam_id')

            existing_exam = session.query(Examination).filter_by(exam_id=exam_id).first()
            if existing_exam:
                for key, value in exam_data.items():
                    if hasattr(existing_exam, key) and key != 'exam_id':
                        setattr(existing_exam, key, value)
                logger.debug(f"更新检查记录: {exam_id}")
                return exam_id
            else:
                examination = Examination(**exam_data)
                session.add(examination)
                logger.debug(f"添加检查记录: {exam_id}")
                return exam_id

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

    def get_examinations_by_case_id(self, case_id: str) -> List[Dict[str, Any]]:
        """获取某病历的所有检查记录"""
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
                    "status": e.status,
                    "ordered_at": e.ordered_at.isoformat() if e.ordered_at else None,
                }
                for e in exams
            ]

    # ===== 检查项目明细操作 =====

    def add_exam_item(self, item_data: Dict[str, Any]) -> str:
        """添加检查项目明细"""
        with self.get_session() as session:
            item_id = item_data.get('item_id') or str(uuid.uuid4())
            item_data['item_id'] = item_id
            item = ExamItem(**item_data)
            session.add(item)
            logger.debug(f"添加检查项目: {item_id}")
            return item_id

    def get_exam_items(self, exam_id: str) -> List[Dict[str, Any]]:
        """获取某检查的所有项目明细"""
        with self.get_session() as session:
            items = session.query(ExamItem).filter_by(exam_id=exam_id).all()
            return [
                {
                    "item_id": i.item_id,
                    "item_name": i.item_name,
                    "value_numeric": float(i.value_numeric) if i.value_numeric is not None else None,
                    "value_text": i.value_text,
                    "value_type": i.value_type,
                    "unit": i.unit,
                    "ref_range": i.ref_range,
                    "is_abnormal": i.is_abnormal,
                }
                for i in items
            ]

    # ===== 医患问答记录操作 =====

    def add_case_qa_record(self, qa_data: Dict[str, Any]) -> str:
        """添加医患问答记录"""
        with self.get_session() as session:
            qa_id = qa_data.get('qa_id') or str(uuid.uuid4())
            qa_data['qa_id'] = qa_id
            record = CaseQARecord(**qa_data)
            session.add(record)
            logger.debug(f"添加问答记录: {qa_id}")
            return qa_id

    def get_qa_records_by_case(self, case_id: str) -> List[Dict[str, Any]]:
        """获取某病历的所有问答记录"""
        with self.get_session() as session:
            records = session.query(CaseQARecord)\
                .filter_by(case_id=case_id)\
                .order_by(CaseQARecord.round_index, CaseQARecord.created_at)\
                .all()
            return [
                {
                    "qa_id": r.qa_id,
                    "role": r.role,
                    "content": r.content,
                    "round_index": r.round_index,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]

    # ===== 统计查询 =====

    def get_daily_statistics(self, target_date: date = None) -> Dict[str, Any]:
        """获取每日统计数据"""
        if target_date is None:
            target_date = date.today()

        with self.get_session() as session:
            visits = session.query(Visit)\
                .filter(Visit.visit_date >= datetime.combine(target_date, datetime.min.time()),
                        Visit.visit_date < datetime.combine(target_date, datetime.max.time()))\
                .all()

            total_count = len(visits)
            completed_count = sum(1 for v in visits if v.status == 'completed')

            dept_stats: Dict[str, int] = {}
            for v in visits:
                dept = v.dept or 'unknown'
                dept_stats[dept] = dept_stats.get(dept, 0) + 1

            return {
                "date": target_date.strftime("%Y-%m-%d"),
                "total_visits": total_count,
                "completed_visits": completed_count,
                "by_department": dept_stats,
            }

    def close(self):
        """关闭数据库连接"""
        self.engine.dispose()
        logger.info("数据库连接已关闭")

