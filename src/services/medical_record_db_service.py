"""
基于数据库的医疗记录服务 - 6表结构
Database-backed Medical Record Service - 6-table structure
  patients / visits / medical_cases / examinations / exam_items / case_qa_records
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
        
        # 就诊流水号缓存（键：patient_id:YYYYMMDD）
        self._visit_daily_seq_cache: Dict[str, int] = {}
        # 当前就诊 visit_id 缓存（键：patient_id）
        self._current_visit_id: Dict[str, str] = {}
        
        # 不显示初始化提示，由initializer统一管理
    
    def _save_record(self, record: MedicalRecord):
        """覆盖父类方法：仅在启用备份时才保存到文件"""
        if self.backup_to_file:
            # 调用父类的文件保存方法
            super()._save_record(record)
        # 如果不备份，则什么都不做（数据已在数据库中）
    
    def _get_visit_id_and_outpatient_no(self, patient_id: str) -> tuple:
        """生成就诊 visit_id 与门诊号（每次就诊唯一）

        visit_id      格式：VISIT-{patient_id}-{YYYYMMDD}-{NNN}
        outpatient_no 格式：OPD-{patient_id}-{YYYYMMDD}-{NNN}
        """
        date_str = datetime.now().strftime('%Y%m%d')
        cache_key = f"{patient_id}:{date_str}"

        if cache_key not in self._visit_daily_seq_cache:
            next_seq = self.dao.get_next_visit_sequence(patient_id, date_str)
            self._visit_daily_seq_cache[cache_key] = next_seq
        else:
            self._visit_daily_seq_cache[cache_key] += 1

        seq = self._visit_daily_seq_cache[cache_key]
        visit_id = f"VISIT-{patient_id}-{date_str}-{seq:03d}"
        outpatient_no = f"OPD-{patient_id}-{date_str}-{seq:03d}"
        return visit_id, outpatient_no

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """安全转换为 int，失败返回 None"""
        if value is None:
            return None
        try:
            text = str(value).replace("岁", "").strip()
            return int(text) if text else None
        except Exception:
            return None

    def _build_structured_case_payload(self, patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """将 known_case/case_data 字段映射为数据库 MedicalCase 列。"""
        case_data = patient_profile.get("case_data", {}) or {}

        def _pick(*keys: str, default: str = "") -> Any:
            for key in keys:
                if key in case_data and case_data.get(key) not in (None, ""):
                    return case_data.get(key)
                if key in patient_profile and patient_profile.get(key) not in (None, ""):
                    return patient_profile.get(key)
            return default

        payload: Dict[str, Any] = {
            "history_narrator": _pick("病史陈述者"),

            "chief_complaint": _pick("主诉"),
            "present_illness_detail": _pick("现病史_详细描述"),
            "present_illness_onset": _pick("现病史_起病情况"),
            "present_illness_course": _pick("现病史_病程"),
            "present_illness_progression": _pick("现病史_病情发展"),

            "past_history_disease": _pick("既往史_疾病史"),
            "past_history_surgery": _pick("既往史_手术史"),
            "past_history_allergy": _pick("既往史_过敏史"),
            "past_history_vaccination": _pick("既往史_预防接种史"),
            "past_history_trauma": _pick("既往史_外伤史"),

            "personal_history": _pick("个人史"),
            "alcohol_history": _pick("个人史_饮酒史"),
            "smoking_history": _pick("个人史_抽烟史"),
            "menstrual_history": _pick("个人史_月经史"),
            "marital_fertility_history": _pick("婚育史"),

            "family_history_father": _pick("家族史_父亲"),
            "family_history_mother": _pick("家族史_母亲"),
            "family_history_siblings": _pick("家族史_兄弟姐妹"),
            "family_history_disease": _pick("家族史_疾病"),

            "pe_vital_signs": _pick("体格检查_生命体征"),
            "pe_skin_mucosa": _pick("体格检查_皮肤黏膜"),
            "pe_superficial_lymph_nodes": _pick("体格检查_浅表淋巴结"),
            "pe_head_neck": _pick("体格检查_头颈部"),
            "pe_cardiopulmonary_vascular": _pick("体格检查_心肺血管"),
            "pe_abdomen": _pick("体格检查_腹部"),
            "pe_spine_limbs": _pick("体格检查_脊柱四肢"),
            "pe_nervous_system": _pick("体格检查_神经系统"),

            "auxiliary_examination": _pick("辅助检查"),
            "preliminary_diagnosis": _pick("初步诊断"),
            "diagnosis_basis": _pick("诊断依据"),
            "treatment_principle": _pick("治疗原则"),
            "doctor_patient_qa_ref_question": _pick("医患问答参考_问"),
            "doctor_patient_qa_ref_answer": _pick("医患问答参考_答"),

            # 预留字段（仅初始化）
            "std_record_chief_complaint": "",
            "std_record_present_illness": "",
            "std_record_past_history": "",
            "std_record_physical_exam": "",
            "std_record_aux_exam": "",
            "std_record_diagnosis_result": "",
            "advanced_question": "",
            "advanced_answer": "",
            "exam_question": "",
            "exam_answer": "",
        }
        return payload
    
    def create_record(self, patient_id: str, patient_profile: Dict[str, Any]) -> MedicalRecord:
        """创建病历（写入数据库 + 可选文件备份）"""
        # 复诊场景：若已有病例对象则复用，避免重置历史记录
        existing_record = super().get_record(patient_id)
        if existing_record:
            record = existing_record
            record.patient_profile.update(patient_profile)
            record.last_updated = now_iso()
            self._add_entry(
                record=record,
                entry_type="visit_started",
                location="lobby",
                operator="system",
                content={"patient_profile": patient_profile},
                notes="患者复诊，创建新门诊号"
            )
            self._save_record(record)
        else:
            # 首诊场景：创建新病例对象
            record = super().create_record(patient_id, patient_profile)
        
        try:
            # 生成就诊 visit_id 与门诊号
            visit_id, outpatient_no = self._get_visit_id_and_outpatient_no(patient_id)
            self._current_visit_id[patient_id] = visit_id
            
            # 创建/更新患者记录（patient_id 为主键）
            self.dao.create_patient({
                "patient_id": patient_id,
                "name": patient_profile.get("name"),
                "age": self._safe_int(patient_profile.get("age")),
                "gender": patient_profile.get("gender"),
                "ethnicity": patient_profile.get("case_data", {}).get("民族", ""),
                "occupation": patient_profile.get("case_data", {}).get("职业", ""),
                "phone": patient_profile.get("phone"),
            })

            # 创建就诊记录
            self.dao.create_visit({
                "visit_id": visit_id,
                "patient_id": patient_id,
                "outpatient_no": outpatient_no,
                "visit_date": datetime.now(),
                "status": "ongoing",
            })

            # 保存当前就诊 visit_id 到病例对象
            record.patient_profile["visit_id"] = visit_id
            record.patient_profile["outpatient_no"] = outpatient_no
            self._save_record(record)
            
            # 创建病历记录
            structured_case_payload = self._build_structured_case_payload(patient_profile)
            actual_case_id = self.dao.create_medical_case({
                "case_id": record.record_id,
                "visit_id": visit_id,
                "status": "draft",
                **structured_case_payload,
            })
            
            # 如果返回的case_id与record.record_id不同（已存在情况）
            if actual_case_id != record.record_id:
                logger.warning(f"⚠️  [数据库] 病例已存在，使用已有case_id: {actual_case_id}")
                # 更新record的record_id以匹配数据库
                record.record_id = actual_case_id
            else:
                logger.info(f"创建病历号: {actual_case_id}")
            
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
                   nurse_id: str = "nurse_001", location: str = "triage",
                   nurse_name: str = "分诊护士") -> bool:
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
                
                # 更新就诊记录中的科室和分诊护士信息
                visit_id = self._current_visit_id.get(patient_id, "")
                if visit_id:
                    self.dao.update_visit(visit_id, {
                        "dept": dept,
                        "triage_nurse_id": nurse_id,
                        "triage_nurse_name": nurse_name,
                    })

                # 更新病历中的现病史描述
                self.dao.update_medical_case(record.record_id, {
                    "present_illness_detail": latest_complaint,
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
                # 按轮次写入 case_qa_records
                round_idx = 1
                i = 0
                while i < len(conversation):
                    msg = conversation[i]
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    # 按成对计算轮次（doctor+patient 为一轮）
                    if role == "doctor":
                        self.dao.add_case_qa_record({
                            "case_id": record.record_id,
                            "role": "doctor",
                            "content": content,
                            "round_index": round_idx,
                        })
                        if i + 1 < len(conversation):
                            patient_msg = conversation[i + 1]
                            self.dao.add_case_qa_record({
                                "case_id": record.record_id,
                                "role": "patient",
                                "content": patient_msg.get("content", ""),
                                "round_index": round_idx,
                            })
                            i += 2
                        else:
                            i += 1
                        round_idx += 1
                    else:
                        self.dao.add_case_qa_record({
                            "case_id": record.record_id,
                            "role": role or "patient",
                            "content": content,
                            "round_index": round_idx,
                        })
                        i += 1
                        round_idx += 1
                
                # 更新主诉（如果为空）
                update_data = {}
                if conversation and not record.chief_complaints:
                    # 取第一个 patient 回答作为主诉参考
                    for msg in conversation:
                        if msg.get("role") == "patient":
                            first_answer = msg.get("content", "")
                            if first_answer and len(first_answer) > 10:
                                update_data["chief_complaint"] = first_answer[:200]
                            break
                
                if update_data:
                    self.dao.update_medical_case(record.record_id, update_data)
                
        except Exception as e:
            logger.error(f"❌ [数据库] 问诊记录写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def add_lab_test(self, patient_id: str, test_name: str, 
                    test_results: Dict[str, Any], operator: str = "lab_tech_001", 
                    operator_name: str = "检验科医生") -> bool:
        """添加检验结果（数据库 + 文件）"""
        success = super().add_lab_test(patient_id, test_name, test_results, operator)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
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
                if not db_case:
                    logger.warning(f"⚠️  [数据库] 病历 {record.record_id} 在数据库中不存在，跳过写入检查记录")
                    return True
                
                self.dao.add_examination({
                    "exam_id": exam_id,
                    "case_id": record.record_id,
                    "exam_name": test_name,
                    "exam_type": "lab",
                    "lab_doctor_id": operator,
                    "lab_doctor_name": operator_name,
                    "result_text": json.dumps(test_results, ensure_ascii=False),
                    "summary": test_results.get("summary", ""),
                    "is_abnormal": test_results.get("abnormal", False),
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
                if not db_case:
                    logger.warning(f"⚠️  [数据库] 病历 {record.record_id} 在数据库中不存在，跳过写入影像记录")
                    return True
                
                self.dao.add_examination({
                    "exam_id": exam_id,
                    "case_id": record.record_id,
                    "exam_name": imaging_type,
                    "exam_type": "imaging",
                    "result_text": json.dumps(imaging_results, ensure_ascii=False),
                    "summary": imaging_results.get("summary", ""),
                    "is_abnormal": imaging_results.get("abnormal", False),
                    "status": "completed",
                    "reported_at": datetime.now(),
                })
                
        except Exception as e:
            logger.error(f"❌ [数据库] 影像结果写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return True
    
    def add_diagnosis(self, patient_id: str, doctor_id: str,
                     diagnosis: Dict[str, Any], location: str = "internal_medicine",
                     doctor_name: str = "主治医生") -> bool:
        """添加诊断记录（数据库 + 文件）
        
        将诊断名写入 preliminary_diagnosis，诊断依据/推理写入 diagnosis_basis（JSON），
        并更新 visits 表中的主治医生信息。
        """
        success = super().add_diagnosis(patient_id, doctor_id, diagnosis, location)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                update_data = {
                    "preliminary_diagnosis": diagnosis.get("name"),
                    "diagnosis_basis": json.dumps({
                        "reasoning": diagnosis.get("reasoning", ""),
                        "evidence": diagnosis.get("evidence", []),
                        "rule_out": diagnosis.get("rule_out", []),
                        "uncertainty": diagnosis.get("uncertainty", "unknown"),
                    }, ensure_ascii=False),
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

                # 更新就诊记录中的主治医生信息
                visit_id = self._current_visit_id.get(patient_id, "")
                if visit_id:
                    self.dao.update_visit(visit_id, {
                        "attending_doctor_id": doctor_id,
                        "attending_doctor_name": doctor_name,
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
                    "medications": json.dumps(medications, ensure_ascii=False),
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
        
        更新病历状态为 final，就诊状态为 completed。
        """
        success = super().discharge_patient(patient_id, discharge_docs, doctor_id)
        
        if not success:
            return False
        
        try:
            record = self._active_records.get(patient_id)
            if record:
                # 更新病历状态为 final
                self.dao.update_medical_case(record.record_id, {
                    "status": "final",
                })
                
                # 更新就诊状态为 completed
                visit_id = self._current_visit_id.get(patient_id, "")
                if visit_id:
                    self.dao.update_visit(visit_id, {
                        "status": "completed",
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
            self.dao.update_medical_case(record.record_id, update_data)
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
