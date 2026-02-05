"""
医疗病例库服务 - 管理患者完整就医记录
Medical Record Service - Patient Electronic Medical Record Management
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from utils import now_iso, get_logger

logger = get_logger("hospital_agent.medical_record")


@dataclass
class MedicalRecordEntry:
    """单次就医记录条目"""
    timestamp: str
    entry_type: str  # 'triage', 'consultation', 'exam', 'lab_test', 'diagnosis', 'treatment', 'prescription', etc.
    location: str
    operator: str  # 操作人员ID（医生、护士等）
    content: Dict[str, Any]
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MedicalRecord:
    """患者完整病例记录"""
    patient_id: str
    record_id: str  # 病例号
    created_at: str
    last_updated: str
    
    # 基本信息
    patient_profile: Dict[str, Any] = field(default_factory=dict)
    
    # 就医流程记录（按时间顺序）
    entries: List[MedicalRecordEntry] = field(default_factory=list)
    
    # 分类整理的医疗数据
    chief_complaints: List[Dict[str, Any]] = field(default_factory=list)  # 历次主诉
    vital_signs_history: List[Dict[str, Any]] = field(default_factory=list)  # 生命体征记录
    diagnoses: List[Dict[str, Any]] = field(default_factory=list)  # 诊断记录
    prescriptions: List[Dict[str, Any]] = field(default_factory=list)  # 处方记录
    lab_results: List[Dict[str, Any]] = field(default_factory=list)  # 检验结果
    imaging_results: List[Dict[str, Any]] = field(default_factory=list)  # 影像结果
    treatments: List[Dict[str, Any]] = field(default_factory=list)  # 治疗记录
    
    # 当前状态
    current_status: str = "active"  # active, discharged, transferred
    current_location: str = "lobby"
    current_dept: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "patient_id": self.patient_id,
            "record_id": self.record_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "patient_profile": self.patient_profile,
            "entries": [entry.to_dict() for entry in self.entries],
            "chief_complaints": self.chief_complaints,
            "vital_signs_history": self.vital_signs_history,
            "diagnoses": self.diagnoses,
            "prescriptions": self.prescriptions,
            "lab_results": self.lab_results,
            "imaging_results": self.imaging_results,
            "treatments": self.treatments,
            "current_status": self.current_status,
            "current_location": self.current_location,
            "current_dept": self.current_dept,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MedicalRecord':
        """从字典恢复对象"""
        entries = [
            MedicalRecordEntry(**entry_data) 
            for entry_data in data.get("entries", [])
        ]
        
        record = cls(
            patient_id=data["patient_id"],
            record_id=data["record_id"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            patient_profile=data.get("patient_profile", {}),
            entries=entries,
            chief_complaints=data.get("chief_complaints", []),
            vital_signs_history=data.get("vital_signs_history", []),
            diagnoses=data.get("diagnoses", []),
            prescriptions=data.get("prescriptions", []),
            lab_results=data.get("lab_results", []),
            imaging_results=data.get("imaging_results", []),
            treatments=data.get("treatments", []),
            current_status=data.get("current_status", "active"),
            current_location=data.get("current_location", "lobby"),
            current_dept=data.get("current_dept"),
        )
        
        return record


class MedicalRecordService:
    """医疗病例库服务"""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        初始化病例库服务
        
        Args:
            storage_dir: 病例存储目录，默认为 ./medical_records/
        """
        self.storage_dir = storage_dir or Path("./medical_records")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存（当前会话的活跃病例）
        self._active_records: Dict[str, MedicalRecord] = {}
        
        logger.info(f"医疗病例库服务已初始化，存储目录: {self.storage_dir.absolute()}")
    
    def create_record(self, patient_id: str, patient_profile: Dict[str, Any]) -> MedicalRecord:
        """
        为新患者创建病例
        
        Args:
            patient_id: 患者ID
            patient_profile: 患者基本信息（姓名、性别、年龄等）
            
        Returns:
            新创建的病例对象
        """
        record_id = self._generate_record_id(patient_id)
        now = now_iso()
        
        record = MedicalRecord(
            patient_id=patient_id,
            record_id=record_id,
            created_at=now,
            last_updated=now,
            patient_profile=patient_profile,
        )
        
        # 添加创建记录
        self._add_entry(
            record=record,
            entry_type="record_created",
            location="system",
            operator="system",
            content={"patient_profile": patient_profile},
            notes="患者病例创建"
        )
        
        # 加入活跃缓存
        self._active_records[patient_id] = record
        
        # 持久化
        self._save_record(record)
        
        logger.info(f"创建病例: {record_id} (患者: {patient_id})")
        
        return record
    
    def get_record(self, patient_id: str) -> Optional[MedicalRecord]:
        """
        获取患者病例
        
        Args:
            patient_id: 患者ID
            
        Returns:
            病例对象，如果不存在则返回None
        """
        # 先查缓存
        if patient_id in self._active_records:
            return self._active_records[patient_id]
        
        # 从磁盘加载
        record = self._load_record(patient_id)
        if record:
            self._active_records[patient_id] = record
        
        return record
    
    def add_triage(self, patient_id: str, dept: str, chief_complaint: str, 
                   nurse_id: str = "nurse_001", location: str = "triage") -> bool:
        """
        添加分诊记录
        
        Args:
            patient_id: 患者ID
            dept: 分诊科室
            chief_complaint: 主诉
            nurse_id: 分诊护士ID
            location: 分诊位置
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            logger.warning(f"患者 {patient_id} 病例不存在")
            return False
        
        # 添加主诉记录
        record.chief_complaints.append({
            "timestamp": now_iso(),
            "complaint": chief_complaint,
            "dept": dept,
            "nurse": nurse_id,
        })
        
        # 更新当前科室
        record.current_dept = dept
        
        # 添加流程记录
        self._add_entry(
            record=record,
            entry_type="triage",
            location=location,
            operator=nurse_id,
            content={
                "dept": dept,
                "chief_complaint": chief_complaint,
            },
            notes=f"分诊至{dept}科"
        )
        
        self._save_record(record)
        
        logger.info(f"添加分诊记录: {patient_id} -> {dept}")
        
        return True
    
    def add_vital_signs(self, patient_id: str, vital_signs: Dict[str, float],
                       location: str, operator: str = "nurse_001") -> bool:
        """
        添加生命体征记录
        
        Args:
            patient_id: 患者ID
            vital_signs: 生命体征数据（心率、血压等）
            location: 测量位置
            operator: 操作人员
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        vital_signs_entry = {
            "timestamp": now_iso(),
            "data": vital_signs,
            "location": location,
            "operator": operator,
        }
        
        record.vital_signs_history.append(vital_signs_entry)
        
        self._add_entry(
            record=record,
            entry_type="vital_signs",
            location=location,
            operator=operator,
            content=vital_signs,
            notes="生命体征测量"
        )
        
        self._save_record(record)
        
        return True
    
    def add_consultation(self, patient_id: str, doctor_id: str, 
                        conversation: List[Dict[str, str]], 
                        history: Dict[str, Any],
                        exam_findings: Dict[str, Any],
                        location: str = "internal_medicine") -> bool:
        """
        添加问诊记录
        
        Args:
            patient_id: 患者ID
            doctor_id: 医生ID
            conversation: 医患对话记录
            history: 病史信息
            exam_findings: 体格检查结果
            location: 诊室位置
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        self._add_entry(
            record=record,
            entry_type="consultation",
            location=location,
            operator=doctor_id,
            content={
                "conversation": conversation,
                "history": history,
                "exam_findings": exam_findings,
            },
            notes="医生问诊"
        )
        
        self._save_record(record)
        
        return True
    
    def add_lab_test(self, patient_id: str, test_name: str, 
                    test_results: Dict[str, Any], operator: str = "lab_tech_001") -> bool:
        """
        添加检验结果
        
        Args:
            patient_id: 患者ID
            test_name: 检验项目名称
            test_results: 检验结果
            operator: 检验技师ID
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        lab_result = {
            "timestamp": now_iso(),
            "test_name": test_name,
            "results": test_results,
            "operator": operator,
        }
        
        record.lab_results.append(lab_result)
        
        self._add_entry(
            record=record,
            entry_type="lab_test",
            location="lab",
            operator=operator,
            content={
                "test_name": test_name,
                "results": test_results,
            },
            notes=f"检验: {test_name}"
        )
        
        self._save_record(record)
        
        return True
    
    def add_imaging(self, patient_id: str, imaging_type: str,
                   imaging_results: Dict[str, Any], operator: str = "radiology_tech_001") -> bool:
        """
        添加影像结果
        
        Args:
            patient_id: 患者ID
            imaging_type: 影像类型（X光、CT、MRI等）
            imaging_results: 影像结果
            operator: 影像技师ID
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        imaging_result = {
            "timestamp": now_iso(),
            "imaging_type": imaging_type,
            "results": imaging_results,
            "operator": operator,
        }
        
        record.imaging_results.append(imaging_result)
        
        self._add_entry(
            record=record,
            entry_type="imaging",
            location="imaging",
            operator=operator,
            content={
                "imaging_type": imaging_type,
                "results": imaging_results,
            },
            notes=f"影像检查: {imaging_type}"
        )
        
        self._save_record(record)
        
        return True
    
    def add_diagnosis(self, patient_id: str, doctor_id: str,
                     diagnosis: Dict[str, Any], doctor_name: str = "主治医生", 
                     location: str = "internal_medicine") -> bool:
        """
        添加诊断记录
        
        Args:
            patient_id: 患者ID
            doctor_id: 医生ID
            diagnosis: 诊断信息
            doctor_name: 医生姓名
            location: 诊室位置
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        diagnosis_entry = {
            "timestamp": now_iso(),
            "diagnosis": diagnosis,
            "doctor": doctor_id,
            "doctor_name": doctor_name,
            "location": location,
        }
        
        record.diagnoses.append(diagnosis_entry)
        
        self._add_entry(
            record=record,
            entry_type="diagnosis",
            location=location,
            operator=doctor_id,
            content=diagnosis,
            notes=f"诊断: {diagnosis.get('name', '未知')}"
        )
        
        self._save_record(record)
        
        return True
    
    def add_prescription(self, patient_id: str, doctor_id: str,
                        medications: List[Dict[str, Any]], location: str = "internal_medicine") -> bool:
        """
        添加处方记录
        
        Args:
            patient_id: 患者ID
            doctor_id: 医生ID
            medications: 药物列表
            location: 开处方位置
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        prescription = {
            "timestamp": now_iso(),
            "doctor": doctor_id,
            "medications": medications,
            "location": location,
        }
        
        record.prescriptions.append(prescription)
        
        self._add_entry(
            record=record,
            entry_type="prescription",
            location=location,
            operator=doctor_id,
            content={
                "medications": medications,
            },
            notes=f"开具处方({len(medications)}种药物)"
        )
        
        self._save_record(record)
        
        return True
    
    def add_treatment(self, patient_id: str, treatment_type: str,
                     treatment_details: Dict[str, Any], operator: str, location: str) -> bool:
        """
        添加治疗记录
        
        Args:
            patient_id: 患者ID
            treatment_type: 治疗类型
            treatment_details: 治疗详情
            operator: 操作人员
            location: 治疗位置
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        treatment = {
            "timestamp": now_iso(),
            "treatment_type": treatment_type,
            "details": treatment_details,
            "operator": operator,
            "location": location,
        }
        
        record.treatments.append(treatment)
        
        self._add_entry(
            record=record,
            entry_type="treatment",
            location=location,
            operator=operator,
            content=treatment_details,
            notes=f"治疗: {treatment_type}"
        )
        
        self._save_record(record)
        
        return True
    
    def update_location(self, patient_id: str, new_location: str) -> bool:
        """
        更新患者位置
        
        Args:
            patient_id: 患者ID
            new_location: 新位置
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        old_location = record.current_location
        record.current_location = new_location
        
        self._add_entry(
            record=record,
            entry_type="location_change",
            location=new_location,
            operator="system",
            content={
                "from": old_location,
                "to": new_location,
            },
            notes=f"位置变更: {old_location} -> {new_location}"
        )
        
        self._save_record(record)
        
        return True
    
    def discharge_patient(self, patient_id: str, discharge_docs: List[Dict[str, Any]],
                         doctor_id: str = "doctor_001") -> bool:
        """
        患者出院
        
        Args:
            patient_id: 患者ID
            discharge_docs: 出院文档（病历摘要、诊断证明、病假单等）
            doctor_id: 主治医生ID
            
        Returns:
            是否成功
        """
        record = self.get_record(patient_id)
        if not record:
            return False
        
        record.current_status = "discharged"
        
        self._add_entry(
            record=record,
            entry_type="discharge",
            location=record.current_location,
            operator=doctor_id,
            content={
                "discharge_docs": discharge_docs,
                "diagnoses": record.diagnoses,
                "treatments": record.treatments,
            },
            notes="患者出院"
        )
        
        self._save_record(record)
        
        logger.info(f"患者出院: {patient_id}")
        
        return True
    
    def get_patient_summary(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        获取患者病例摘要
        
        Args:
            patient_id: 患者ID
            
        Returns:
            病例摘要字典
        """
        record = self.get_record(patient_id)
        if not record:
            return None
        
        return {
            "patient_id": patient_id,
            "record_id": record.record_id,
            "patient_profile": record.patient_profile,
            "created_at": record.created_at,
            "current_status": record.current_status,
            "current_location": record.current_location,
            "current_dept": record.current_dept,
            "chief_complaints_count": len(record.chief_complaints),
            "diagnoses_count": len(record.diagnoses),
            "lab_tests_count": len(record.lab_results),
            "imaging_count": len(record.imaging_results),
            "prescriptions_count": len(record.prescriptions),
            "latest_diagnosis": record.diagnoses[-1] if record.diagnoses else None,
            "total_entries": len(record.entries),
        }
    
    def search_records(self, **criteria) -> List[MedicalRecord]:
        """
        搜索病例（简单实现，可扩展）
        
        Args:
            **criteria: 搜索条件（如 dept='neurology', status='active'）
            
        Returns:
            符合条件的病例列表
        """
        results = []
        
        # 遍历所有病例文件
        for record_file in self.storage_dir.glob("*.json"):
            try:
                record = self._load_record_from_file(record_file)
                
                # 检查搜索条件
                match = True
                for key, value in criteria.items():
                    if key == "dept" and record.current_dept != value:
                        match = False
                        break
                    elif key == "status" and record.current_status != value:
                        match = False
                        break
                
                if match:
                    results.append(record)
                    
            except Exception as e:
                logger.error(f"加载病例文件失败: {record_file}, 错误: {e}")
        
        return results
    
    # ===== 内部辅助方法 =====
    
    def _generate_record_id(self, patient_id: str) -> str:
        """生成病例号"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"MR-{patient_id}-{timestamp}"
    
    def _add_entry(self, record: MedicalRecord, entry_type: str,
                   location: str, operator: str, content: Dict[str, Any], notes: str = ""):
        """添加流程记录条目"""
        entry = MedicalRecordEntry(
            timestamp=now_iso(),
            entry_type=entry_type,
            location=location,
            operator=operator,
            content=content,
            notes=notes,
        )
        
        record.entries.append(entry)
        record.last_updated = now_iso()
    
    def _save_record(self, record: MedicalRecord):
        """保存病例到磁盘"""
        file_path = self.storage_dir / f"{record.patient_id}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
                
            logger.debug(f"病例已保存: {file_path}")
            
        except Exception as e:
            logger.error(f"保存病例失败: {file_path}, 错误: {e}")
    
    def _load_record(self, patient_id: str) -> Optional[MedicalRecord]:
        """从磁盘加载病例"""
        file_path = self.storage_dir / f"{patient_id}.json"
        
        if not file_path.exists():
            return None
        
        return self._load_record_from_file(file_path)
    
    def _load_record_from_file(self, file_path: Path) -> Optional[MedicalRecord]:
        """从文件加载病例"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            record = MedicalRecord.from_dict(data)
            logger.debug(f"病例已加载: {file_path}")
            
            return record
            
        except Exception as e:
            logger.error(f"加载病例失败: {file_path}, 错误: {e}")
            return None
