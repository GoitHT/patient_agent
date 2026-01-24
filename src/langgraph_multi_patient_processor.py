"""
LangGraph 多患者处理器 - 与 LangGraph 流程深度集成
LangGraph Multi-Patient Processor - Deep integration with LangGraph workflows

功能：
1. 为每个患者执行完整的 LangGraph 诊断流程
2. 支持多医生并发接诊
3. 医生资源通过 HospitalCoordinator 统一调度
4. 物理环境模拟与 LangGraph 节点集成
"""

import concurrent.futures
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from agents import PatientAgent, DoctorAgent, NurseAgent, LabAgent
from environment import HospitalWorld
from graphs.router import build_common_graph, build_dept_subgraphs, build_services
from hospital_coordinator import HospitalCoordinator, PatientStatus
from loaders import load_diagnosis_arena_case
from rag import ChromaRetriever
from services.llm_client import LLMClient
from services.medical_record import MedicalRecordService
from services.medical_record_integration import MedicalRecordIntegration
from state.schema import BaseState
from utils import get_logger, make_run_id, make_rng

logger = get_logger("hospital_agent.langgraph_multi_patient")


class LangGraphPatientExecutor:
    """为单个患者执行完整的 LangGraph 流程"""
    
    def __init__(
        self,
        patient_id: str,
        case_id: int,
        dept: str,
        priority: int,
        coordinator: HospitalCoordinator,
        retriever: ChromaRetriever,
        llm: LLMClient,
        services: Any,
        medical_record_service: MedicalRecordService,
        seed: int,
        max_questions: int = 3,
        use_hf_data: bool = False,
        shared_world: HospitalWorld = None,  # 新增：共享物理环境
        shared_nurse_agent: NurseAgent = None,  # 新增：共享护士
        shared_lab_agent: LabAgent = None,  # 新增：共享检验科
        doctor_agents: Dict[str, DoctorAgent] = None,  # 新增：医生agents字典
    ):
        self.patient_id = patient_id
        self.case_id = case_id
        self.dept = dept
        self.priority = priority
        self.coordinator = coordinator
        self.retriever = retriever
        self.llm = llm
        self.services = services
        self.medical_record_service = medical_record_service
        self.seed = seed
        self.max_questions = max_questions
        self.use_hf_data = use_hf_data
        self.logger = get_logger(f"patient.{patient_id}")
        
        # 使用共享资源
        self.world = shared_world
        self.nurse_agent = shared_nurse_agent
        self.lab_agent = shared_lab_agent
        self.doctor_agents = doctor_agents or {}
    
    def _wait_for_doctor_assignment(self, timeout: int = 300) -> Optional[str]:
        """
        等待 coordinator 分配医生
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            分配的医生ID，超时返回 None
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            session = self.coordinator.get_patient(self.patient_id)
            if session and session.assigned_doctor:
                return session.assigned_doctor
            time.sleep(0.5)  # 每 0.5 秒检查一次
        
        self.logger.error(f"等待医生分配超时 ({timeout}秒)")
        return None
    
    def execute(self) -> Dict[str, Any]:
        """执行完整的患者诊断流程"""
        try:
            self.logger.info(f"{'='*80}")
            self.logger.info(f"开始执行患者 {self.patient_id} 的 LangGraph 诊断流程")
            self.logger.info(f"{'='*80}")
            
            # 1. 加载病例数据
            self.logger.info(f"📚 加载病例数据 (案例ID: {self.case_id})...")
            case_bundle = load_diagnosis_arena_case(
                self.case_id, 
                use_mock=not self.use_hf_data,
                local_cache_dir="./diagnosis_dataset"  # 默认使用本地缓存
            )
            known_case = case_bundle["known_case"]
            ground_truth = case_bundle["ground_truth"]
            
            # 提取原始主诉
            case_info = known_case.get("Case Information", "")
            if "主诉：" in case_info:
                start_idx = case_info.find("主诉：") + 3
                remaining = case_info[start_idx:]
                end_markers = ["现病史：", "既往史：", "个人史：", "家族史：", "体格检查：", "\n\n"]
                end_idx = len(remaining)
                for marker in end_markers:
                    pos = remaining.find(marker)
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
                original_chief_complaint = remaining[:end_idx].strip()
            else:
                original_chief_complaint = case_info[:200].strip()
            
            self.logger.info(f"  原始主诉: {original_chief_complaint[:50]}...")
            
            # 2. 使用共享物理环境（不再创建新的）
            self.logger.info(f"🏥 使用共享物理环境...")
            world = self.world  # 使用传入的共享 world
            
            # 患者已在 submit_patient 时添加到 world，无需重复添加
            # world.add_agent(self.patient_id, ...)  # ❌ 删除
            
            # 3. 初始化 State
            run_id = make_run_id(self.seed, self.dept)
            state = BaseState(
                run_id=run_id,
                dept=self.dept,
                patient_profile={"case_text": case_info},
                appointment={"channel": "APP", "timeslot": "上午"},
                original_chief_complaint=original_chief_complaint,
                chief_complaint="",
                case_data=known_case,
                ground_truth=ground_truth,
                patient_id=self.patient_id,
                current_location="lobby",
                agent_config={
                    "max_questions": self.max_questions,
                    "use_agents": True,
                },
            )
            
            # 集成物理环境和病例库
            state.world_context = world
            medical_record_integration = MedicalRecordIntegration(self.medical_record_service, world)
            state.medical_record_integration = medical_record_integration
            
            # 创建病例
            patient_profile = {
                "name": state.case_data.get("name", f"患者{self.patient_id}"),
                "age": state.case_data.get("age", 0),
                "gender": state.case_data.get("gender", "未知"),
                "case_id": self.case_id,
            }
            record_id = medical_record_integration.on_patient_entry(self.patient_id, patient_profile)
            self.logger.info(f"  病例已创建: {record_id}")
            
            # 4. 初始化 Agents（使用共享的）
            self.logger.info(f"🤖 使用共享智能体...")
            patient_agent = PatientAgent(
                known_case=state.case_data,
                llm=self.llm,
                chief_complaint=original_chief_complaint
            )
            
            # 使用共享的 nurse 和 lab agent
            nurse_agent = self.nurse_agent
            lab_agent = self.lab_agent
            
            # nurse 和 lab_tech 已在初始化时添加到 world，无需重复添加
            
            # ===== 5. 执行护士分诊 =====
            self.logger.info(f"👩‍⚕️ 执行护士分诊...")
            world.move_agent(self.patient_id, "triage")
            
            patient_description = patient_agent.describe_to_nurse()
            triaged_dept = nurse_agent.triage(
                patient_description=patient_description
            )
            
            # 更新科室和 run_id
            state.dept = triaged_dept
            run_id = make_run_id(self.seed, triaged_dept)
            state.run_id = run_id
            state.chief_complaint = patient_description
            
            triage_summary = nurse_agent.get_triage_summary()
            state.agent_interactions["nurse_triage"] = triage_summary
            
            # 从分诊历史中获取分诊理由
            triage_reason = ""
            if triage_summary.get("history"):
                latest_triage = triage_summary["history"][-1]
                triage_reason = latest_triage.get("reason", "")
            
            if state.medical_record_integration:
                state.medical_record_integration.on_triage(state, nurse_id="nurse_001")
            
            self.logger.info(f"  分诊结果: {triaged_dept} (理由: {triage_reason})")
            
            # ===== 6. 通过 Coordinator 注册患者并等待医生分配 =====
            self.logger.info(f"📋 通过 Coordinator 注册患者...")
            
            # 准备患者数据
            patient_data = {
                "name": state.case_data.get("name", f"患者{self.patient_id}"),
                "age": state.case_data.get("age", 0),
                "gender": state.case_data.get("gender", "未知"),
                "case_id": self.case_id,
            }
            
            # 注册患者到 coordinator
            self.coordinator.register_patient(
                patient_id=self.patient_id,
                patient_data=patient_data,
                dept=triaged_dept,
                priority=self.priority
            )
            
            # 加入等候队列（这会触发自动分配）
            self.coordinator.enqueue_patient(self.patient_id)
            
            # 等待医生分配
            self.logger.info(f"⏳ 等待医生分配...")
            assigned_doctor_id = self._wait_for_doctor_assignment()
            
            if not assigned_doctor_id:
                raise Exception("医生分配超时")
            
            doctor = self.coordinator.get_doctor(assigned_doctor_id)
            self.logger.info(f"  ✅ 已分配医生: {doctor.name} ({doctor.dept})")
            
            # ===== 7. 使用分配的医生 Agent =====
            self.logger.info(f"👨‍⚕️ 使用分配的医生 Agent...")
            
            # 从共享的 doctor_agents 获取对应的 DoctorAgent
            doctor_agent = self.doctor_agents.get(assigned_doctor_id)
            if not doctor_agent:
                # 如果没有预创建，动态创建（理论上不应该发生）
                self.logger.warning(f"⚠️  未找到预创建的 DoctorAgent {assigned_doctor_id}，动态创建")
                doctor_agent = DoctorAgent(
                    dept=triaged_dept,
                    retriever=self.retriever,
                    llm=self.llm,
                    max_questions=self.max_questions
                )
            
            # 医生已在初始化时添加到 world，无需重复添加
            # world.add_agent("doctor_001", ...)  # ❌ 删除
            
            # 7. 构建 LangGraph
            self.logger.info(f"🕸️ 构建 LangGraph 执行图...")
            rng = make_rng(self.seed)
            
            dept_subgraphs = build_dept_subgraphs(
                retriever=self.retriever,
                rng=rng,
                llm=self.llm,
                doctor_agent=doctor_agent,
                patient_agent=patient_agent,
                max_questions=self.max_questions
            )
            
            graph = build_common_graph(
                dept_subgraphs,
                retriever=self.retriever,
                services=self.services,
                rng=rng,
                llm=self.llm,
                llm_reports=False,
                use_agents=True,
                patient_agent=patient_agent,
                doctor_agent=doctor_agent,
                nurse_agent=self.nurse_agent,
                lab_agent=self.lab_agent,
                max_questions=self.max_questions,
                world=self.world,  # 使用共享的 world
            )
            
            # 8. 执行 LangGraph 流程
            self.logger.info(f"🚀 开始执行门诊流程...")
            self.logger.info(f"{'─'*80}")
            
            node_count = 0
            out = None
            
            for chunk in graph.stream(state):
                node_count += 1
                if isinstance(chunk, dict) and len(chunk) > 0:
                    out = chunk[list(chunk.keys())[0]]
            
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"✅ 诊断流程完成 (共执行 {node_count} 个节点)")
            
            # ===== 9. 释放医生资源 =====
            self.logger.info(f"🔓 释放医生资源...")
            self.coordinator.release_doctor(assigned_doctor_id)
            self.logger.info(f"  ✅ 医生 {assigned_doctor_id} 已释放")
            
            # 10. 提取结果
            final_diagnosis = out.diagnosis.get("name", "未明确") if out and hasattr(out, 'diagnosis') else "未明确"
            ground_truth_diagnosis = ground_truth.get('Final Diagnosis', 'N/A')
            
            result = {
                "status": "completed",
                "patient_id": self.patient_id,
                "case_id": self.case_id,
                "dept": triaged_dept,
                "diagnosis": final_diagnosis,
                "ground_truth": ground_truth_diagnosis,
                "node_count": node_count,
                "record_id": record_id,
            }
            
            self.logger.info(f"✅ 患者 {self.patient_id} 诊断完成")
            self.logger.info(f"  诊断结果: {final_diagnosis}")
            self.logger.info(f"  标准诊断: {ground_truth_diagnosis}")
            self.logger.info(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 患者 {self.patient_id} 执行失败: {e}", exc_info=True)
            
            # 如果已分配医生，需要释放
            try:
                session = self.coordinator.get_patient(self.patient_id)
                if session and session.assigned_doctor:
                    self.coordinator.release_doctor(session.assigned_doctor)
                    self.logger.info(f"🔓 异常处理：已释放医生 {session.assigned_doctor}")
            except Exception as release_error:
                self.logger.error(f"释放医生资源失败: {release_error}")
            
            return {
                "status": "failed",
                "patient_id": self.patient_id,
                "case_id": self.case_id,
                "error": str(e),
            }


class LangGraphMultiPatientProcessor:
    """LangGraph 多患者并发处理器"""
    
    def __init__(
        self,
        coordinator: HospitalCoordinator,
        retriever: ChromaRetriever,
        llm: LLMClient,
        services: Any,
        medical_record_service: MedicalRecordService,
        seed: int,
        max_questions: int = 3,
        use_hf_data: bool = False,
        max_workers: int = 10,
    ):
        """
        初始化处理器
        
        Args:
            coordinator: 医院协调器
            retriever: RAG 检索器
            llm: LLM 客户端
            services: 服务组件
            medical_record_service: 病例库服务
            seed: 随机种子
            max_questions: 最大问题数
            use_hf_data: 是否使用 HuggingFace 数据
            max_workers: 最大并发数
        """
        self.coordinator = coordinator
        self.retriever = retriever
        self.llm = llm
        self.services = services
        self.medical_record_service = medical_record_service
        self.seed = seed
        self.max_questions = max_questions
        self.use_hf_data = use_hf_data
        self.max_workers = max_workers
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        
        # ===== 创建共享的物理环境（核心改动）=====
        logger.info("🏥 初始化共享物理环境...")
        self.shared_world = HospitalWorld(start_time=None)
        
        # 添加共享的医护人员到 world
        logger.info("  添加共享医护人员...")
        self.shared_world.add_agent("nurse_001", agent_type="nurse", initial_location="triage")
        self.shared_world.add_agent("lab_tech_001", agent_type="lab_technician", initial_location="lab")
        
        # 根据 coordinator 中注册的医生添加到 world
        for doctor_id, doctor in self.coordinator.doctors.items():
            dept_location = self._get_dept_location(doctor.dept)
            self.shared_world.add_agent(doctor_id, agent_type="doctor", initial_location=dept_location)
            logger.info(f"  添加医生: {doctor.name} ({doctor.dept}) -> {dept_location}")
        
        # 初始化共享设备
        self._setup_shared_equipment()
        
        # 创建共享的 Nurse 和 Lab Agent（所有患者共用）
        self.shared_nurse_agent = NurseAgent(llm=self.llm, max_triage_questions=3)
        self.shared_lab_agent = LabAgent(llm=self.llm)
        
        # 为每个医生创建 DoctorAgent 实例（映射到 coordinator 的医生）
        self.doctor_agents: Dict[str, DoctorAgent] = {}
        for doctor_id, doctor in self.coordinator.doctors.items():
            self.doctor_agents[doctor_id] = DoctorAgent(
                dept=doctor.dept,
                retriever=self.retriever,
                llm=self.llm,
                max_questions=self.max_questions
            )
            logger.info(f"  创建 DoctorAgent: {doctor_id} ({doctor.dept})")
        
        logger.info(f"✅ LangGraph 多患者处理器已启动 (最大并发: {max_workers})")
        logger.info(f"  共享环境: 1个 World, {len(self.coordinator.doctors)}名医生")
    
    def _get_dept_location(self, dept: str) -> str:
        """获取科室对应的物理位置
        
        Args:
            dept: 科室代码
        
        Returns:
            位置ID
        """
        dept_location_map = {
            "internal_medicine": "internal_medicine",
            "surgery": "surgery",
            "gastro": "gastro",
            "neuro": "neuro",
            "emergency": "emergency",
            "dermatology_std": "internal_medicine",  # 皮肤科使用内科诊室
            "orthopedics": "surgery",  # 骨科使用外科诊室
            "urology": "surgery",  # 泌尿外科使用外科诊室
            "obstetrics_gynecology": "internal_medicine",  # 妇产科使用内科诊室
            "pediatrics": "internal_medicine",  # 儿科使用内科诊室
            "neurology": "neuro",  # 神经医学使用神经内科诊室
            "oncology": "internal_medicine",  # 肿瘤科使用内科诊室
            "infectious_disease": "internal_medicine",  # 感染科使用内科诊室
            "ent_ophthalmology_stomatology": "internal_medicine",  # 五官科使用内科诊室
            "psychiatry": "internal_medicine",  # 精神心理科使用内科诊室
            "rehabilitation_pain": "internal_medicine",  # 康复疼痛科使用内科诊室
            "traditional_chinese_medicine": "internal_medicine",  # 中医科使用内科诊室
        }
        return dept_location_map.get(dept, "internal_medicine")
    
    def _setup_shared_equipment(self):
        """设置共享设备（可选，暂时简化实现）"""
        # 这里可以添加共享设备的初始化逻辑
        # 例如：限制检验设备数量、配置队列等
        logger.info("  共享设备初始化完成（使用 world 默认配置）")
    
    def submit_patient(
        self,
        patient_id: str,
        case_id: int,
        dept: str,
        priority: int = 5,
    ) -> str:
        """
        提交患者任务
        
        Args:
            patient_id: 患者ID
            case_id: 病例ID
            dept: 就诊科室
            priority: 优先级
        
        Returns:
            任务ID
        """
        # 先将患者添加到共享 world
        success = self.shared_world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
        if not success:
            logger.warning(f"⚠️  患者 {patient_id} 已在 world 中，跳过添加")
        
        # 创建执行器，传入共享 world 和共享 agents
        executor = LangGraphPatientExecutor(
            patient_id=patient_id,
            case_id=case_id,
            dept=dept,
            priority=priority,
            coordinator=self.coordinator,
            retriever=self.retriever,
            llm=self.llm,
            services=self.services,
            medical_record_service=self.medical_record_service,
            seed=self.seed + hash(patient_id) % 1000,  # 每个患者不同的种子
            max_questions=self.max_questions,
            use_hf_data=self.use_hf_data,
            shared_world=self.shared_world,  # 传入共享 world
            shared_nurse_agent=self.shared_nurse_agent,  # 传入共享 nurse
            shared_lab_agent=self.shared_lab_agent,  # 传入共享 lab agent
            doctor_agents=self.doctor_agents,  # 传入医生 agents 字典
        )
        
        # 提交任务
        with self._lock:
            future = self.executor.submit(executor.execute)
            self.active_tasks[patient_id] = future
        
        logger.info(f"✅ 任务已提交: 患者 {patient_id} (案例 {case_id}, 科室 {dept}, 优先级 {priority})")
        
        return patient_id
    
    def submit_batch(self, patients: List[Dict[str, Any]]) -> List[str]:
        """
        批量提交患者任务
        
        Args:
            patients: 患者列表，每个元素包含:
                - patient_id: 患者ID
                - case_id: 病例ID
                - dept: 科室（可选，默认为 "internal_medicine"，会被护士分诊覆盖）
                - priority: 优先级（可选，默认为 5）
        
        Returns:
            任务ID列表
        """
        task_ids = []
        
        for patient_info in patients:
            patient_id = patient_info["patient_id"]
            case_id = patient_info["case_id"]
            dept = patient_info.get("dept", "internal_medicine")  # 默认科室，会被护士分诊覆盖
            priority = patient_info.get("priority", 5)
            
            task_id = self.submit_patient(patient_id, case_id, dept, priority)
            task_ids.append(task_id)
            
            # 稍微错开提交时间，避免资源竞争
            time.sleep(0.1)
        
        logger.info(f"✅ 批量提交完成: {len(task_ids)} 个患者")
        
        return task_ids
    
    def wait_for_patient(self, patient_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """等待单个患者任务完成"""
        with self._lock:
            future = self.active_tasks.get(patient_id)
        
        if not future:
            return {"status": "not_found", "patient_id": patient_id}
        
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            return {"status": "timeout", "patient_id": patient_id}
        except Exception as e:
            logger.error(f"任务执行失败 ({patient_id}): {e}")
            return {"status": "error", "patient_id": patient_id, "error": str(e)}
    
    def wait_all(self, timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """等待所有任务完成"""
        results = []
        
        with self._lock:
            futures = list(self.active_tasks.items())
        
        for patient_id, future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                logger.warning(f"任务超时: {patient_id}")
                results.append({"status": "timeout", "patient_id": patient_id})
            except Exception as e:
                logger.error(f"任务执行失败 ({patient_id}): {e}")
                results.append({"status": "error", "patient_id": patient_id, "error": str(e)})
        
        logger.info(f"✅ 所有任务完成: {len(results)} 个")
        
        return results
    
    def get_active_count(self) -> int:
        """获取活跃任务数"""
        with self._lock:
            return len([f for f in self.active_tasks.values() if not f.done()])
    
    def shutdown(self, wait: bool = True):
        """关闭处理器"""
        logger.info("关闭 LangGraph 多患者处理器...")
        self.executor.shutdown(wait=wait)
        logger.info("✅ 处理器已关闭")
