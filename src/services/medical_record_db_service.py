"""
基于数据库的医疗记录服务 - 3表结构（门诊号为主线）
Database-backed Medical Record Service - 3-table structure based on outpatient_no
"""
from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .medical_record import MedicalRecordService, MedicalRecord, MedicalRecordEntry
from .medical_record_dao import MedicalRecordDAO
from utils import now_iso, get_logger

logger = get_logger("hospital_agent.medical_record_db")


class DatabaseMedicalRecordService(MedicalRecordService):
    """基于数据库的医疗记录服务（双写模式：数据库 + 文件备份）"""
    
    def __init__(self, connection_string: str, storage_dir: Optional[Path] = None, 
                 backup_to_file: bool = True):
        """
        初始化数据库医疗记录服务
        
        Args:
            connection_string: 数据库连接字符串
            storage_dir: 文件备份目录（可选）
            backup_to_file: 是否同时备份到文件（双保险）
        """
        # 初始化父类（文件存储）
        super().__init__(storage_dir or Path("./medical_records"))
        
        # 初始化数据库DAO
        self.dao = MedicalRecordDAO(connection_string)
        self.backup_to_file = backup_to_file
        
        # 门诊号映射 (patient_id -> outpatient_no)
        self._outpatient_map: Dict[str, str] = {}
        
        logger.info(f"数据库医疗记录服务已初始化 (备份到文件: {backup_to_file})")
    
    def _save_record(self, record: MedicalRecord):
        """覆盖父类方法：仅在启用备份时才保存到文件"""
        if self.backup_to_file:
            # 调用父类的文件保存方法
            super()._save_record(record)
        # 如果不备份，则什么都不做（数据已在数据库中）
    
    def _get_outpatient_no(self, patient_id: str) -> str:
        """生成或获取门诊号"""
        if patient_id not in self._outpatient_map:
            # 生成门诊号：格式 OPD_患者ID_日期时间
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            self._outpatient_map[patient_id] = f"OPD_{patient_id}_{timestamp}"
        return self._outpatient_map[patient_id]
    
    def create_record(self, patient_id: str, patient_profile: Dict[str, Any]) -> MedicalRecord:
        """创建病历（写入数据库 + 可选文件备份）"""
        # 调用父类方法创建内存对象
        record = super().create_record(patient_id, patient_profile)
        
        try:
            # 生成门诊号
            outpatient_no = self._get_outpatient_no(patient_id)
            
            # 创建患者记录
            self.dao.create_patient({
                "outpatient_no": outpatient_no,
                "patient_id": patient_id,
                "name": patient_profile.get("name"),
                "age": patient_profile.get("age"),
                "gender": patient_profile.get("gender"),
                "phone": patient_profile.get("phone"),
            })
            
            # 创建病例记录
            actual_case_id = self.dao.create_medical_case({
                "case_id": record.record_id,
                "outpatient_no": outpatient_no,
                "visit_date": date.today(),
                "dataset_id": patient_profile.get("dataset_id"),
                "original_case_id": patient_profile.get("case_id"),
                "run_id": patient_profile.get("run_id"),
                "status": "ongoing",
            })
            
            # 如果返回的case_id与record.record_id不同（已存在情况）
            if actual_case_id != record.record_id:
                logger.warning(f"⚠️  [数据库] 病例已存在，使用已有case_id: {actual_case_id}")
                # 更新record的record_id以匹配数据库
                record.record_id = actual_case_id
            
        except Exception as e:
            logger.error(f"❌ [数据库] 写入数据库失败: {e}，已保留文件备份")
            import traceback
            logger.error(traceback.format_exc())
        
        return record
    
    def get_record(self, patient_id: str) -> Optional[MedicalRecord]:
        """获取病历（优先从内存缓存）"""
        # 先查内存缓存
        if patient_id in self._active_records:
            return self._active_records[patient_id]
        
        # 尝试从数据库加载（暂不实现，降级到文件）
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
                # 从record中获取最新的chief_complaint（如果有的话）
                latest_complaint = chief_complaint
                if record.chief_complaints:
                    latest_complaint = record.chief_complaints[-1].get('complaint', chief_complaint)
                
                self.dao.update_medical_case(record.record_id, {
                    "dept": dept,
                    "chief_complaint": latest_complaint,
                })
                
                # 记录分诊日志到case_logs
                self.dao.add_case_log(record.record_id, {
                    "log_type": "triage",
                    "entity_id": nurse_id,
                    "entity_type": "nurse",
                    "log_data": {
                        "dept": dept,
                        "chief_complaint": latest_complaint,
                        "location": location,
                    }
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 分诊信息写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
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
                # conversation格式: [{"role": "doctor", "content": "..."}, {"role": "patient", "content": "..."}]
                # 需要成对处理
                qa_pairs = []
                for i in range(0, len(conversation), 2):
                    if i + 1 < len(conversation):
                        doctor_msg = conversation[i]
                        patient_msg = conversation[i + 1]
                        qa_pairs.append({
                            "question": doctor_msg.get("content", ""),
                            "answer": patient_msg.get("content", "")
                        })
                
                # 保存每一轮问答到doctor_qa_records JSON字段
                for i, qa in enumerate(qa_pairs, start=1):
                    qa_record = {
                        "question_order": i,
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "staff_id": doctor_id,
                        "staff_role": "doctor",
                        "asked_at": now_iso(),
                        "location": location,
                    }
                    self.dao.add_doctor_qa(record.record_id, qa_record)
                
                # 更新现病史和主诉
                update_data = {}
                if history:
                    update_data["present_illness"] = json.dumps(history, ensure_ascii=False)
                
                # 从第一轮患者回答中提取主诉（如果chief_complaint为空）
                if qa_pairs and not record.chief_complaints:
                    first_answer = qa_pairs[0].get("answer", "")
                    if first_answer and len(first_answer) > 10:
                        # 截取前200字符作为主诉
                        update_data["chief_complaint"] = first_answer[:200]
                
                if update_data:
                    self.dao.update_medical_case(record.record_id, update_data)
                
        except Exception as e:
            logger.error(f"❌ [数据库] 问诊记录写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def add_lab_test(self, patient_id: str, test_name: str, 
                    test_results: Dict[str, Any], operator: str = "lab_tech_001") -> bool:
        """添加检验结果（数据库 + 文件）"""
        success = super().add_lab_test(patient_id, test_name, test_results, operator)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            outpatient_no = self._get_outpatient_no(patient_id)
            if record:
                # 生成检查ID
                import random
                now = datetime.now()
                timestamp_sec = now.strftime('%y%m%d%H%M%S')
                microsec = str(now.microsecond)[:4]
                random_suffix = random.randint(10, 99)
                exam_id = f"L_{patient_id}_{timestamp_sec}{microsec}{random_suffix}"
                
                # 验证case_id是否存在于数据库
                db_case = self.dao.get_medical_case(record.record_id)
                case_id_to_use = record.record_id if db_case else None
                
                if not db_case:
                    logger.warning(f"⚠️  [数据库] 病例 {record.record_id} 在数据库中不存在，case_id设为NULL")
                
                self.dao.add_examination({
                    "exam_id": exam_id,
                    "outpatient_no": outpatient_no,
                    "case_id": case_id_to_use,
                    "exam_name": test_name,
                    "exam_type": "lab",
                    "result_text": json.dumps(test_results, ensure_ascii=False),
                    "summary": test_results.get("summary", ""),
                    "is_abnormal": test_results.get("abnormal", False),
                    "key_findings": test_results.get("findings", []),
                    "status": "completed",
                    "reported_at": datetime.now(),
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 检验结果写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def add_imaging(self, patient_id: str, imaging_type: str,
                   imaging_results: Dict[str, Any], operator: str = "radiology_tech_001") -> bool:
        """添加影像结果（数据库 + 文件）"""
        success = super().add_imaging(patient_id, imaging_type, imaging_results, operator)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            outpatient_no = self._get_outpatient_no(patient_id)
            if record:
                # 生成检查ID
                import random
                now = datetime.now()
                timestamp_sec = now.strftime('%y%m%d%H%M%S')
                microsec = str(now.microsecond)[:4]
                random_suffix = random.randint(10, 99)
                exam_id = f"I_{patient_id}_{timestamp_sec}{microsec}{random_suffix}"
                
                # 验证case_id是否存在于数据库
                db_case = self.dao.get_medical_case(record.record_id)
                case_id_to_use = record.record_id if db_case else None
                
                if not db_case:
                    logger.warning(f"⚠️  [数据库] 病例 {record.record_id} 在数据库中不存在，case_id设为NULL")
                
                self.dao.add_examination({
                    "exam_id": exam_id,
                    "outpatient_no": outpatient_no,
                    "case_id": case_id_to_use,
                    "exam_name": imaging_type,
                    "exam_type": "imaging",
                    "result_text": json.dumps(imaging_results, ensure_ascii=False),
                    "summary": imaging_results.get("summary", ""),
                    "is_abnormal": imaging_results.get("abnormal", False),
                    "key_findings": imaging_results.get("findings", []),
                    "status": "completed",
                    "reported_at": datetime.now(),
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 影像结果写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def add_diagnosis(self, patient_id: str, doctor_id: str,
                     diagnosis: Dict[str, Any], location: str = "internal_medicine") -> bool:
        """添加诊断记录（数据库 + 文件）
        
        方案一优化：将完整诊断信息存储为JSON格式到diagnosis_reason字段
        """
        success = super().add_diagnosis(patient_id, doctor_id, diagnosis, location)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                # 【方案一】构建完整诊断信息JSON
                diagnosis_full_info = {
                    "reasoning": diagnosis.get("reasoning", ""),
                    "evidence": diagnosis.get("evidence", []),
                    "rule_out": diagnosis.get("rule_out", []),
                    "uncertainty": diagnosis.get("uncertainty", "unknown"),
                    "disclaimer": diagnosis.get("disclaimer", ""),
                }
                
                update_data = {
                    "diagnosis_name": diagnosis.get("name"),
                    # 存储完整诊断信息（JSON字符串）
                    "diagnosis_reason": json.dumps(diagnosis_full_info, ensure_ascii=False),
                }
                
                # 如果诊断中包含治疗计划，一并保存
                if diagnosis.get("treatment_plan"):
                    update_data["treatment_plan"] = diagnosis.get("treatment_plan")
                elif diagnosis.get("recommended_treatment"):
                    update_data["treatment_plan"] = diagnosis.get("recommended_treatment")
                
                # 如果诊断中包含医嘱，一并保存
                if diagnosis.get("medical_advice"):
                    update_data["medical_advice"] = diagnosis.get("medical_advice")
                
                self.dao.update_medical_case(record.record_id, update_data)
                
                # 记录诊断日志到case_logs
                self.dao.add_case_log(record.record_id, {
                    "log_type": "diagnosis",
                    "entity_id": doctor_id,
                    "entity_type": "doctor",
                    "log_data": {
                        "location": location,
                        "diagnosis_name": diagnosis.get("name"),
                        "timestamp": now_iso(),
                    }
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 诊断记录写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
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
                # 生成详细的治疗计划
                med_details = []
                for med in medications:
                    med_name = med.get('name', '未知药物')
                    dosage = med.get('dosage', '')
                    frequency = med.get('frequency', '')
                    duration = med.get('duration', '')
                    
                    detail = f"{med_name}"
                    if dosage:
                        detail += f" {dosage}"
                    if frequency:
                        detail += f" {frequency}"
                    if duration:
                        detail += f" {duration}"
                    
                    med_details.append(detail)
                
                treatment_plan = "药物治疗：\n" + "\n".join([f"{i+1}. {d}" for i, d in enumerate(med_details)])
                
                self.dao.update_medical_case(record.record_id, {
                    "medications": medications,
                    "treatment_plan": treatment_plan,
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 处方记录写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def discharge_patient(self, patient_id: str, discharge_docs: List[Dict[str, Any]],
                         doctor_id: str = "doctor_001") -> bool:
        """患者出院（数据库 + 文件）
        
        方案一优化：出院文档完整存储到case_logs的log_data字段
        """
        success = super().discharge_patient(patient_id, discharge_docs, doctor_id)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                # 更新病例状态
                self.dao.update_medical_case(record.record_id, {
                    "status": "completed",
                    "outcome": "discharged",
                })
                
                # 【方案一】将完整出院文档存入case_logs
                self.dao.add_case_log(record.record_id, {
                    "log_type": "discharge",
                    "entity_id": patient_id,
                    "entity_type": "patient",
                    "log_data": {
                        "doctor_id": doctor_id,
                        "discharge_docs": discharge_docs,  # 完整的出院文档数组
                        "total_entries": len(record.entries),
                        "discharge_time": now_iso(),
                    }
                })
                
        except Exception as e:
            logger.error(f"出院信息写入数据库失败: {e}")
        
        return True
    
    def add_treatment(self, patient_id: str, treatment_type: str,
                     treatment_details: Dict[str, Any], operator: str, location: str) -> bool:
        """添加治疗记录（数据库 + 文件）"""
        success = super().add_treatment(patient_id, treatment_type, treatment_details, operator, location)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                # 提取或构造treatment_plan
                treatment_plan = treatment_details.get("plan", "")
                if not treatment_plan:
                    treatment_plan = f"{treatment_type}：{treatment_details.get('description', '详见病历')}"
                
                self.dao.update_medical_case(record.record_id, {
                    "treatment_plan": treatment_plan,
                })
                
                # 记录治疗日志
                self.dao.add_case_log(record.record_id, {
                    "log_type": "treatment",
                    "entity_id": operator,
                    "entity_type": "staff",
                    "log_data": {
                        "treatment_type": treatment_type,
                        "details": treatment_details,
                        "location": location,
                    }
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 治疗记录写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def add_medical_advice(self, patient_id: str, advice: str, doctor_id: str = "doctor_001") -> bool:
        """添加医嘱（数据库专用方法）"""
        record = self.get_record(patient_id)
        if not record:
            return False
        
        try:
            self.dao.update_medical_case(record.record_id, {
                "medical_advice": advice,
            })
            
            # 记录日志
            self.dao.add_case_log(record.record_id, {
                "log_type": "medical_advice",
                "entity_id": doctor_id,
                "entity_type": "doctor",
                "log_data": {
                    "advice": advice,
                }
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ [数据库] 医嘱写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def add_followup(self, patient_id: str, followup_plan: str, 
                    followup_date: Optional[str] = None, doctor_id: str = "doctor_001") -> bool:
        """添加随访计划（数据库专用方法）"""
        record = self.get_record(patient_id)
        if not record:
            return False
        
        try:
            update_data = {"followup_plan": followup_plan}
            if followup_date:
                # 解析日期字符串为date对象
                from datetime import datetime
                if isinstance(followup_date, str):
                    try:
                        followup_date_obj = datetime.fromisoformat(followup_date.replace('Z', '+00:00')).date()
                        update_data["followup_date"] = followup_date_obj
                    except:
                        pass
            
            self.dao.update_medical_case(record.record_id, update_data)
            
            # 记录日志
            self.dao.add_case_log(record.record_id, {
                "log_type": "followup",
                "entity_id": doctor_id,
                "entity_type": "doctor",
                "log_data": {
                    "followup_plan": followup_plan,
                    "followup_date": followup_date,
                }
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ [数据库] 随访计划写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
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
