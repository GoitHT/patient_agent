"""
基于数据库的医疗记录服务 - 继承原有服务接口
Database-backed Medical Record Service - Inherits from original service
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .medical_record import MedicalRecordService, MedicalRecord, MedicalRecordEntry
from .medical_record_dao import MedicalRecordDAO
from utils import now_iso, get_logger

logger = get_logger("hospital_agent.medical_record_db")


class DatabaseMedicalRecordService(MedicalRecordService):
    """基于数据库的医疗记录服务（双写模式：数据库 + 文件备份）"""
    
    def __init__(self, connection_string: str, storage_dir: Optional[Path] = None, 
                 backup_to_file: bool = True, echo: bool = False):
        """
        初始化数据库医疗记录服务
        
        Args:
            connection_string: 数据库连接字符串
            storage_dir: 文件备份目录（可选）
            backup_to_file: 是否同时备份到文件（双保险）
            echo: 是否打印SQL语句
        """
        # 初始化父类（文件存储）
        super().__init__(storage_dir or Path("./medical_records"))
        
        # 初始化数据库DAO
        self.dao = MedicalRecordDAO(connection_string, echo=echo)
        self.backup_to_file = backup_to_file
        
        logger.info(f"数据库医疗记录服务已初始化 (备份到文件: {backup_to_file})")
    
    def create_record(self, patient_id: str, patient_profile: Dict[str, Any]) -> MedicalRecord:
        """创建病历（写入数据库 + 可选文件备份）"""
        # 调用父类方法创建内存对象
        record = super().create_record(patient_id, patient_profile)
        
        try:
            # 创建患者记录
            self.dao.create_patient({
                "patient_id": patient_id,
                "name": patient_profile.get("name"),
                "age": patient_profile.get("age"),
                "gender": patient_profile.get("gender"),
                "phone": patient_profile.get("phone"),
            })
            
            # 创建病历记录
            self.dao.create_medical_record({
                "record_id": record.record_id,
                "patient_id": patient_id,
                "visit_date": datetime.now(),
                "dataset_id": patient_profile.get("dataset_id"),
                "original_case_id": patient_profile.get("case_id"),
                "run_id": patient_profile.get("run_id"),
            })
            
            logger.info(f"病历已写入数据库: {record.record_id}")
            
        except Exception as e:
            logger.error(f"写入数据库失败: {e}，已保留文件备份")
        
        return record
    
    def get_record(self, patient_id: str) -> Optional[MedicalRecord]:
        """获取病历（优先从内存缓存，否则从数据库或文件）"""
        # 先查内存缓存
        if patient_id in self._active_records:
            return self._active_records[patient_id]
        
        # 尝试从数据库加载
        try:
            records = self.dao.get_patient_records(patient_id, limit=1)
            if records:
                record_id = records[0]['record_id']
                db_record = self.dao.get_medical_record(record_id, include_relations=True)
                
                if db_record:
                    # 转换为内存对象
                    record = self._convert_from_db(db_record)
                    self._active_records[patient_id] = record
                    return record
        except Exception as e:
            logger.warning(f"从数据库加载失败: {e}，尝试从文件加载")
        
        # 降级到文件加载
        return super().get_record(patient_id)
    
    def add_triage(self, patient_id: str, dept: str, chief_complaint: str, 
                   nurse_id: str = "nurse_001", location: str = "triage") -> bool:
        """添加分诊记录（数据库 + 文件）"""
        # 先更新内存和文件
        success = super().add_triage(patient_id, dept, chief_complaint, nurse_id, location)
        
        if not success:
            return False
        
        # 写入数据库
        try:
            record = self._active_records.get(patient_id)
            if record:
                self.dao.update_medical_record(record.record_id, {
                    "dept": dept,
                    "dept_display_name": dept,
                    "chief_complaint": chief_complaint,
                    "original_complaint": chief_complaint,
                    "triage_reason": f"主诉: {chief_complaint}",
                })
                
                # 记录分诊对话
                self.dao.add_consultation({
                    "record_id": record.record_id,
                    "interaction_type": "triage",
                    "staff_id": nurse_id,
                    "staff_role": "nurse",
                    "question_order": 0,
                    "question": "请问您有什么不适？",
                    "answer": chief_complaint,
                })
                
        except Exception as e:
            logger.error(f"分诊信息写入数据库失败: {e}")
        
        return True
    
    def add_consultation(self, patient_id: str, doctor_id: str, 
                        conversation: List[Dict[str, str]], 
                        history: Dict[str, Any],
                        exam_findings: Dict[str, Any],
                        location: str = "internal_medicine") -> bool:
        """添加问诊记录（数据库 + 文件）"""
        # 先更新文件
        success = super().add_consultation(patient_id, doctor_id, conversation, 
                                          history, exam_findings, location)
        
        if not success:
            return False
        
        # 写入数据库
        try:
            record = self._active_records.get(patient_id)
            if record:
                # 保存每一轮问答
                for i, qa in enumerate(conversation, start=1):
                    self.dao.add_consultation({
                        "record_id": record.record_id,
                        "interaction_type": "doctor_qa",
                        "staff_id": doctor_id,
                        "staff_role": "doctor",
                        "question_order": i,
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "node_name": location,
                    })
                
        except Exception as e:
            logger.error(f"问诊记录写入数据库失败: {e}")
        
        return True
    
    def add_lab_test(self, patient_id: str, test_name: str, 
                    test_results: Dict[str, Any], operator: str = "lab_tech_001") -> bool:
        """添加检验结果（数据库 + 文件）"""
        success = super().add_lab_test(patient_id, test_name, test_results, operator)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                exam_id = f"LAB_{record.record_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                self.dao.add_examination({
                    "exam_id": exam_id,
                    "record_id": record.record_id,
                    "exam_name": test_name,
                    "exam_type": "lab",
                    "ordered_by": operator,
                    "result_text": json.dumps(test_results, ensure_ascii=False),
                    "summary": test_results.get("summary", ""),
                    "is_abnormal": test_results.get("abnormal", False),
                    "key_findings": test_results.get("findings", []),
                    "source": "dataset",
                    "status": "completed",
                })
                
        except Exception as e:
            logger.error(f"检验结果写入数据库失败: {e}")
        
        return True
    
    def add_imaging(self, patient_id: str, imaging_type: str,
                   imaging_results: Dict[str, Any], operator: str = "radiology_tech_001") -> bool:
        """添加影像结果（数据库 + 文件）"""
        success = super().add_imaging(patient_id, imaging_type, imaging_results, operator)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                exam_id = f"IMG_{record.record_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                self.dao.add_examination({
                    "exam_id": exam_id,
                    "record_id": record.record_id,
                    "exam_name": imaging_type,
                    "exam_type": "imaging",
                    "ordered_by": operator,
                    "result_text": json.dumps(imaging_results, ensure_ascii=False),
                    "summary": imaging_results.get("summary", ""),
                    "is_abnormal": imaging_results.get("abnormal", False),
                    "key_findings": imaging_results.get("findings", []),
                    "source": "dataset",
                    "status": "completed",
                })
                
        except Exception as e:
            logger.error(f"影像结果写入数据库失败: {e}")
        
        return True
    
    def add_diagnosis(self, patient_id: str, doctor_id: str,
                     diagnosis: Dict[str, Any], location: str = "internal_medicine") -> bool:
        """添加诊断记录（数据库 + 文件）"""
        success = super().add_diagnosis(patient_id, doctor_id, diagnosis, location)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                self.dao.update_medical_record(record.record_id, {
                    "diagnosis_name": diagnosis.get("name"),
                    "diagnosis_code": diagnosis.get("code"),
                    "diagnosis_reasoning": diagnosis.get("reasoning", ""),
                    "differential_diagnoses": diagnosis.get("differential_diagnoses", []),
                })
                
        except Exception as e:
            logger.error(f"诊断记录写入数据库失败: {e}")
        
        return True
    
    def add_prescription(self, patient_id: str, doctor_id: str,
                        medications: List[Dict[str, Any]], location: str = "internal_medicine") -> bool:
        """添加处方记录（数据库 + 文件）"""
        success = super().add_prescription(patient_id, doctor_id, medications, location)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                self.dao.update_medical_record(record.record_id, {
                    "medications": medications,
                    "treatment_plan": f"处方药物：{', '.join([m.get('name', '') for m in medications])}",
                })
                
        except Exception as e:
            logger.error(f"处方记录写入数据库失败: {e}")
        
        return True
    
    def discharge_patient(self, patient_id: str, discharge_docs: List[Dict[str, Any]],
                         doctor_id: str = "doctor_001") -> bool:
        """患者出院（数据库 + 文件）"""
        success = super().discharge_patient(patient_id, discharge_docs, doctor_id)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                # 计算物理环境信息
                physical_info = {
                    "start_time": record.created_at,
                    "end_time": now_iso(),
                    "total_entries": len(record.entries),
                    "discharge_docs": discharge_docs,
                }
                
                self.dao.update_medical_record(record.record_id, {
                    "status": "completed",
                    "physical_info": physical_info,
                })
                
                # 记录出院日志
                self.dao.log_event({
                    "record_id": record.record_id,
                    "log_type": "discharge",
                    "entity_id": patient_id,
                    "entity_type": "patient",
                    "log_data": {
                        "doctor_id": doctor_id,
                        "discharge_docs": discharge_docs,
                    }
                })
                
        except Exception as e:
            logger.error(f"出院信息写入数据库失败: {e}")
        
        return True
    
    def _convert_from_db(self, db_record: Dict[str, Any]) -> MedicalRecord:
        """从数据库记录转换为内存对象"""
        record = MedicalRecord(
            patient_id=db_record['patient_id'],
            record_id=db_record['record_id'],
            created_at=db_record['created_at'],
            last_updated=db_record['updated_at'],
        )
        
        # 基本信息
        record.current_dept = db_record['dept']
        record.current_status = "active" if db_record['status'] == 'ongoing' else "discharged"
        
        # 主诉
        if db_record['chief_complaint']:
            record.chief_complaints.append({
                "timestamp": db_record['created_at'],
                "complaint": db_record['chief_complaint'],
                "dept": db_record['dept'],
            })
        
        # 诊断
        if db_record['diagnosis_name']:
            record.diagnoses.append({
                "timestamp": db_record['updated_at'],
                "diagnosis": {
                    "name": db_record['diagnosis_name'],
                    "code": db_record['diagnosis_code'],
                    "reasoning": db_record['diagnosis_reasoning'],
                }
            })
        
        # 处方
        if db_record.get('medications'):
            record.prescriptions.append({
                "timestamp": db_record['updated_at'],
                "medications": db_record['medications'],
            })
        
        # 检查结果
        for exam in db_record.get('examinations', []):
            if exam['exam_type'] == 'lab':
                record.lab_results.append({
                    "timestamp": exam['reported_at'],
                    "test_name": exam['exam_name'],
                    "results": json.loads(exam['result_text']) if exam['result_text'] else {},
                })
            elif exam['exam_type'] == 'imaging':
                record.imaging_results.append({
                    "timestamp": exam['reported_at'],
                    "imaging_type": exam['exam_name'],
                    "results": json.loads(exam['result_text']) if exam['result_text'] else {},
                })
        
        return record
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计数据"""
        try:
            return self.dao.get_daily_statistics()
        except Exception as e:
            logger.error(f"获取统计数据失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        self.dao.close()
