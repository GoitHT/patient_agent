"""
病例库与物理环境集成模块
Medical Record and Physical Environment Integration
"""
from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from environment import HospitalWorld
    from services.medical_record import MedicalRecordService
    from state.schema import BaseState


class MedicalRecordIntegration:
    """病例库与物理环境集成器"""
    
    def __init__(self, medical_record_service: 'MedicalRecordService', 
                 hospital_world: Optional['HospitalWorld'] = None):
        """
        初始化集成器
        
        Args:
            medical_record_service: 病例库服务
            hospital_world: 医院物理环境（可选）
        """
        self.mrs = medical_record_service
        self.world = hospital_world
    
    def on_patient_entry(self, patient_id: str, patient_profile: Dict[str, Any]) -> str:
        """
        患者进入医院 - 创建病例
        
        Args:
            patient_id: 患者ID
            patient_profile: 患者基本信息
            
        Returns:
            病例号
        """
        # 检查是否已有病例
        existing_record = self.mrs.get_record(patient_id)
        if existing_record:
            # 已有病例，更新位置
            self.mrs.update_location(patient_id, "lobby")
            # 确保在物理环境中
            if self.world and patient_id not in self.world.agents:
                self.world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
            return existing_record.record_id
        
        # 创建新病例
        record = self.mrs.create_record(patient_id, patient_profile)
        
        # 同步到物理环境
        if self.world and patient_id not in self.world.agents:
            success = self.world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
            if not success:
                # 如果添加失败，可能是已存在，尝试更新位置
                if patient_id in self.world.agents:
                    self.world.move_agent(patient_id, "lobby")
        
        return record.record_id
    
    def on_triage(self, state: 'BaseState', nurse_id: str = "nurse_001"):
        """
        分诊节点 - 更新病例
        
        Args:
            state: 图状态
            nurse_id: 护士ID
        """
        patient_id = state.patient_id
        
        # 记录分诊信息
        self.mrs.add_triage(
            patient_id=patient_id,
            dept=state.dept,
            chief_complaint=state.chief_complaint,
            nurse_id=nurse_id,
            location="triage"
        )
        
        # 记录生命体征（如果有）
        if self.world and patient_id in self.world.physical_states:
            physical_state = self.world.physical_states[patient_id]
            vital_signs = {
                name: vs.value 
                for name, vs in physical_state.vital_signs.items()
            }
            
            self.mrs.add_vital_signs(
                patient_id=patient_id,
                vital_signs=vital_signs,
                location="triage",
                operator=nurse_id
            )
        
        # 更新位置
        self.mrs.update_location(patient_id, "triage")
    
    def on_doctor_consultation(self, state: 'BaseState', doctor_id: str = "doctor_001"):
        """
        医生问诊节点 - 更新病例
        
        Args:
            state: 图状态
            doctor_id: 医生ID
        """
        patient_id = state.patient_id
        
        # 提取对话记录
        conversation = []
        if "doctor_patient" in state.agent_interactions:
            qa_pairs = state.agent_interactions["doctor_patient"]
            for qa in qa_pairs:
                conversation.append({
                    "role": "doctor",
                    "content": qa.get("question", "")
                })
                conversation.append({
                    "role": "patient",
                    "content": qa.get("answer", "")
                })
        
        # 记录问诊
        self.mrs.add_consultation(
            patient_id=patient_id,
            doctor_id=doctor_id,
            conversation=conversation,
            history=state.history,
            exam_findings=state.exam_findings,
            location=state.dept
        )
        
        # 更新位置
        dept_location = self._map_dept_to_location(state.dept)
        self.mrs.update_location(patient_id, dept_location)
    
    def on_lab_test_ordered(self, state: 'BaseState', doctor_id: str = "doctor_001"):
        """
        检验申请节点 - 更新病例
        
        Args:
            state: 图状态
            doctor_id: 医生ID
        """
        patient_id = state.patient_id
        
        # 记录每项检验申请
        for test in state.ordered_tests:
            self.mrs._add_entry(
                record=self.mrs.get_record(patient_id),
                entry_type="test_ordered",
                location=state.dept,
                operator=doctor_id,
                content={
                    "test_name": test.get("name"),
                    "test_type": test.get("type"),
                    "indication": test.get("indication", "")
                },
                notes=f"申请检验: {test.get('name')}"
            )
        
        self.mrs._save_record(self.mrs.get_record(patient_id))
    
    def on_lab_test_completed(self, state: 'BaseState', lab_tech_id: str = "lab_tech_001"):
        """
        检验完成节点 - 更新病例
        
        Args:
            state: 图状态
            lab_tech_id: 检验技师ID
        """
        patient_id = state.patient_id
        
        # 记录每项检验结果
        for result in state.test_results:
            self.mrs.add_lab_test(
                patient_id=patient_id,
                test_name=result.get("name", ""),
                test_results=result,
                operator=lab_tech_id
            )
    
    def on_imaging_completed(self, state: 'BaseState', radiology_tech_id: str = "radiology_tech_001"):
        """
        影像检查完成节点 - 更新病例
        
        Args:
            state: 图状态
            radiology_tech_id: 影像技师ID
        """
        patient_id = state.patient_id
        
        # 从检验结果中筛选影像结果
        for result in state.test_results:
            if result.get("type") in ["imaging", "xray", "ct", "mri", "ultrasound"]:
                self.mrs.add_imaging(
                    patient_id=patient_id,
                    imaging_type=result.get("name", ""),
                    imaging_results=result,
                    operator=radiology_tech_id
                )
    
    def on_diagnosis(self, state: 'BaseState', doctor_id: str = "doctor_001"):
        """
        诊断节点 - 更新病例
        
        Args:
            state: 图状态
            doctor_id: 医生ID
        """
        patient_id = state.patient_id
        
        # 记录诊断
        self.mrs.add_diagnosis(
            patient_id=patient_id,
            doctor_id=doctor_id,
            diagnosis=state.diagnosis,
            location=state.dept
        )
    
    def on_prescription(self, state: 'BaseState', doctor_id: str = "doctor_001"):
        """
        开处方节点 - 更新病例
        
        Args:
            state: 图状态
            doctor_id: 医生ID
        """
        patient_id = state.patient_id
        
        # 从治疗计划中提取药物
        medications = []
        if "medications" in state.treatment_plan:
            meds = state.treatment_plan["medications"]
            if isinstance(meds, list):
                medications = meds
            elif isinstance(meds, str):
                # 简单解析字符串
                medications = [{"name": med.strip()} for med in meds.split(",")]
        
        if medications:
            self.mrs.add_prescription(
                patient_id=patient_id,
                doctor_id=doctor_id,
                medications=medications,
                location=state.dept
            )
    
    def on_treatment(self, state: 'BaseState', treatment_type: str, 
                    treatment_details: Dict[str, Any], operator: str):
        """
        治疗节点 - 更新病例
        
        Args:
            state: 图状态
            treatment_type: 治疗类型
            treatment_details: 治疗详情
            operator: 操作人员
        """
        patient_id = state.patient_id
        
        self.mrs.add_treatment(
            patient_id=patient_id,
            treatment_type=treatment_type,
            treatment_details=treatment_details,
            operator=operator,
            location=state.current_location
        )
    
    def on_discharge(self, state: 'BaseState', doctor_id: str = "doctor_001"):
        """
        出院节点 - 更新病例
        
        Args:
            state: 图状态
            doctor_id: 主治医生ID
        """
        patient_id = state.patient_id
        
        # 记录出院
        self.mrs.discharge_patient(
            patient_id=patient_id,
            discharge_docs=state.discharge_docs,
            doctor_id=doctor_id
        )
        
        # 更新位置
        self.mrs.update_location(patient_id, "discharged")
    
    def sync_physical_state(self, state: 'BaseState'):
        """
        同步物理状态到病例库
        
        Args:
            state: 图状态
        """
        if not self.world:
            return
        
        patient_id = state.patient_id
        
        # 同步位置
        if patient_id in self.world.agents:
            location = self.world.agents[patient_id]
            self.mrs.update_location(patient_id, location)
        
        # 同步生命体征
        if patient_id in self.world.physical_states:
            physical_state = self.world.physical_states[patient_id]
            
            if physical_state.vital_signs:
                vital_signs = {
                    name: vs.value 
                    for name, vs in physical_state.vital_signs.items()
                }
                
                self.mrs.add_vital_signs(
                    patient_id=patient_id,
                    vital_signs=vital_signs,
                    location=state.current_location,
                    operator="system"
                )
    
    def get_patient_history(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        获取患者历史病例摘要
        
        Args:
            patient_id: 患者ID
            
        Returns:
            病例摘要
        """
        return self.mrs.get_patient_summary(patient_id)
    
    def _map_dept_to_location(self, dept: str) -> str:
        """
        将科室映射到物理位置
        
        Args:
            dept: 科室代码
            
        Returns:
            位置ID
        """
        dept_location_map = {
            "internal_medicine": "internal_medicine",
            "surgery": "surgery",
            "neurology": "neuro",
            "gastroenterology": "gastro",
            "emergency": "emergency",
        }
        
        return dept_location_map.get(dept, "internal_medicine")
