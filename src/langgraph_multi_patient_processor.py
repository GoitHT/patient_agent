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
from patient_detail_logger import create_patient_detail_logger, close_patient_detail_logger, get_patient_detail_logger
from rag import ChromaRetriever
from services.llm_client import LLMClient
from services.medical_record import MedicalRecordService
from services.medical_record_integration import MedicalRecordIntegration
from state.schema import BaseState
from utils import get_logger, make_run_id

logger = get_logger("hospital_agent.langgraph_multi_patient")


# ANSI颜色代码 - 用于区分不同患者的输出
class Colors:
    """终端颜色代码"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 前景色
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    
    # 背景色（用于高亮患者ID）
    BG_CYAN = '\033[46m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    
    @staticmethod
    def get_patient_color(patient_index: int) -> tuple:
        """根据患者索引获取颜色（循环使用）"""
        colors = [
            (Colors.CYAN, Colors.BG_CYAN),
            (Colors.GREEN, Colors.BG_GREEN),
            (Colors.YELLOW, Colors.BG_YELLOW),
            (Colors.MAGENTA, Colors.BG_MAGENTA),
            (Colors.BLUE, Colors.BG_BLUE),
        ]
        return colors[patient_index % len(colors)]


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
        max_questions: int = 3,  # 最底层默认值，通常从config传入
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
        self.max_questions = max_questions
        self.logger = get_logger(f"patient.{patient_id}")
        
        # 使用共享资源
        self.world = shared_world
        self.nurse_agent = shared_nurse_agent
        self.lab_agent = shared_lab_agent
        self.doctor_agents = doctor_agents or {}
        
        # 创建患者详细日志记录器
        self.detail_logger = None  # 延迟到execute时创建（需要case_id）
    
    def _wait_for_doctor_assignment(self, timeout: int = 600) -> Optional[str]:
        """
        等待 coordinator 分配医生（优化：主动重试）
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            分配的医生ID，超时返回 None
        """
        import time
        start_time = time.time()
        check_interval = 0.5  # 检查间隔（秒）
        retry_interval = 5  # 重试间隔（秒）
        last_warning_time = start_time
        last_retry_time = start_time
        
        while time.time() - start_time < timeout:
            session = self.coordinator.get_patient(self.patient_id)
            if session and session.assigned_doctor:
                elapsed = time.time() - start_time
                self.logger.info(f"✅ 医生分配成功（等待 {elapsed:.1f}秒）")
                return session.assigned_doctor
            
            # 每5秒主动重试一次分配
            current_time = time.time()
            if current_time - last_retry_time >= retry_interval:
                if session:
                    self.coordinator._try_assign_doctor(session.dept)
                last_retry_time = current_time
            
            # 每30秒输出一次等待提示
            current_time = time.time()
            if current_time - last_warning_time > 30:
                elapsed = current_time - start_time
                self.logger.info(f"⏳ 仍在等待医生分配... (已等待 {elapsed:.0f}秒)")
                last_warning_time = current_time
            
            time.sleep(check_interval)
        
        # 超时，输出详细的资源状态
        session = self.coordinator.get_patient(self.patient_id)
        if session:
            dept = session.dept
            queue_size = self.coordinator.get_queue_size(dept)
            available_doctors = len(self.coordinator.get_available_doctors(dept))
            total_doctors = len([d for d in self.coordinator.doctors.values() if d.dept == dept])
            
            self.logger.error(f"❌ 等待医生分配超时 ({timeout}秒)")
            self.logger.error(f"   科室: {dept}")
            self.logger.error(f"   队列长度: {queue_size}")
            self.logger.error(f"   可用/总医生: {available_doctors}/{total_doctors}")
        else:
            self.logger.error(f"❌ 等待医生分配超时 ({timeout}秒)")
        
        return None
    
    def execute(self) -> Dict[str, Any]:
        """执行完整的患者诊断流程"""
        try:
            # 创建患者详细日志记录器
            self.detail_logger = create_patient_detail_logger(self.patient_id, self.case_id)
            
            # 为患者分配颜色（基于case_id的哈希）
            patient_hash = hash(str(self.case_id)) % 5
            fg_color, bg_color = Colors.get_patient_color(patient_hash)
            
            # 终端显示带颜色的开始信息
            separator = f"{fg_color}{'='*70}{Colors.RESET}"
            patient_tag = f"{bg_color}{Colors.BOLD} 患者 {self.case_id} {Colors.RESET}"
            
            self.logger.info(f"\n{separator}")
            self.logger.info(f"{fg_color}▶️  {patient_tag} {fg_color}| 开始就诊{Colors.RESET}")
            self.logger.info(f"{separator}")
            
            # 记录开始时间
            import time
            start_time = time.time()
            
            # 详细日志中记录完整信息
            self.detail_logger.section("开始诊断流程")
            self.detail_logger.info(f"案例ID: {self.case_id}")
            self.detail_logger.info(f"患者ID: {self.patient_id}")
            self.detail_logger.info(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 1. 加载病例数据
            self.detail_logger.subsection("加载病例数据")
            case_bundle = load_diagnosis_arena_case(self.case_id)
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
            
            # 详细日志中记录完整病例信息
            self.detail_logger.info(f"原始主诉: {original_chief_complaint}")
            if ground_truth.get('treatment_plan'):
                self.detail_logger.info(f"参考治疗方案: {ground_truth['treatment_plan'][:100]}...")
            self.detail_logger.info("")
            
            # 2. 使用共享物理环境
            world = self.world  # 使用传入的共享 world
            
            # 患者已在 submit_patient 时添加到 world，无需重复添加
            # world.add_agent(self.patient_id, ...)  # ❌ 删除
            
            # 3. 初始化 State
            run_id = make_run_id(self.dept)
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
            
            # 注入患者详细日志记录器到 state
            state.patient_detail_logger = self.detail_logger
            
            # 创建病例
            patient_profile = {
                "name": state.case_data.get("name", f"患者{self.patient_id}"),
                "age": state.case_data.get("age", 0),
                "gender": state.case_data.get("gender", "未知"),
                "case_id": self.case_id,
            }
            record_id = medical_record_integration.on_patient_entry(self.patient_id, patient_profile)
            
            # 详细日志记录病例创建
            self.detail_logger.info(f"病例已创建: {record_id}")
            self.detail_logger.info(f"患者信息: {patient_profile['name']}, {patient_profile['age']}岁, {patient_profile['gender']}")
            
            # 4. 初始化 Agents（患者Agent每次新建，护士和检验科Agent共享需要reset）
            # PatientAgent: 每个患者单独创建新实例，天然隔离状态
            patient_agent = PatientAgent(
                known_case=state.case_data,
                llm=self.llm,
                chief_complaint=original_chief_complaint
            )
            
            # 使用共享的 nurse 和 lab agent（多患者共用）
            nurse_agent = self.nurse_agent
            lab_agent = self.lab_agent
            
            # ⚠️ 重要：重置护士分诊状态（清空历史记录，避免患者之间状态污染）
            nurse_agent.reset()
            self.logger.debug(f"  🔄 护士Agent已重置（处理新患者）")
            
            # nurse 和 lab_tech 已在初始化时添加到 world，无需重复添加
            
            # ===== 5. 执行护士分诊 =====
            self.logger.info(f"{fg_color}👩‍⚕️  {patient_tag} {fg_color}| 护士分诊{Colors.RESET}")
            
            self.detail_logger.section("护士分诊")
            world.move_agent(self.patient_id, "triage")
            
            patient_description = patient_agent.describe_to_nurse()
            
            
            # 调用分诊（使用LLM判断）
            triaged_dept = nurse_agent.triage(
                patient_description=patient_description
            )
            
            # 更新科室和 run_id
            state.dept = triaged_dept
            run_id = make_run_id(triaged_dept)
            state.run_id = run_id
            state.chief_complaint = patient_description
            
            triage_summary = nurse_agent.get_triage_summary()
            state.agent_interactions["nurse_triage"] = triage_summary
            
            # 从分诊历史中获取分诊理由
            triage_reason = ""
            if triage_summary.get("history"):
                latest_triage = triage_summary["history"][-1]
                triage_reason = latest_triage.get("reason", "")
                # 详细日志：记录分诊理由
                self.detail_logger.info(f"LLM分诊分析: {triage_reason}")
            
            if state.medical_record_integration:
                state.medical_record_integration.on_triage(state, nurse_id="nurse_001")
            
            # 终端显示分诊结果（包括理由）
            dept_cn_names = {
                "neurology": "神经医学科",
            }
            dept_display = dept_cn_names.get(triaged_dept, triaged_dept)
            self.logger.info(f"{fg_color}  ✅ 分诊至: {dept_display} ({triaged_dept}){Colors.RESET}")
            if triage_reason:
                # 截取前50个字符避免输出过长
                reason_short = triage_reason[:50] + "..." if len(triage_reason) > 50 else triage_reason
                self.logger.info(f"{fg_color}  💡 理由: {reason_short}{Colors.RESET}")
            
            # 详细日志记录分诊信息
            self.detail_logger.info(f"患者描述: {patient_description}")
            self.detail_logger.info(f"分诊科室: {triaged_dept}")
            if triage_reason:
                self.detail_logger.info(f"分诊理由: {triage_reason}")
            
            # ===== 6. 通过 Coordinator 注册患者并等待医生分配 =====
            
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
            self.logger.info(f"{fg_color}⏳ {patient_tag} {fg_color}| 等待医生分配{Colors.RESET}")
            self.detail_logger.subsection("等待医生分配")
            
            assigned_doctor_id = self._wait_for_doctor_assignment()
            
            if not assigned_doctor_id:
                raise Exception("医生分配超时")
            
            doctor = self.coordinator.get_doctor(assigned_doctor_id)
            self.logger.info(f"{fg_color}  ✅ 医生: {doctor.name}{Colors.RESET}")
            
            self.detail_logger.info(f"分配医生: {doctor.name} (ID: {assigned_doctor_id})")
            self.detail_logger.info(f"医生科室: {doctor.dept}")
            
            # ===== 7. 使用分配的医生 Agent =====
            
            # 从共享的 doctor_agents 获取对应的 DoctorAgent（多个医生，按科室或ID分配）
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
            
            # ⚠️ 重要：重置医生状态（清空上一个患者的问诊历史，包括已问问题列表）
            # 确保每个新患者都从零开始问诊，不会受上一患者的问题影响
            doctor_agent.reset()
            self.logger.debug(f"  🔄 医生Agent已重置（清空问诊历史: collected_info + questions_asked）")
            
            # 医生已在初始化时添加到 world，无需重复添加
            # world.add_agent("doctor_001", ...)  # ❌ 删除
            
            # 7. 构建 LangGraph（静默）
            self.detail_logger.subsection("构建执行图")
            
            dept_subgraphs = build_dept_subgraphs(
                retriever=self.retriever,
                llm=self.llm,
                doctor_agent=doctor_agent,
                patient_agent=patient_agent,
                max_questions=self.max_questions
            )
            
            graph = build_common_graph(
                dept_subgraphs,
                retriever=self.retriever,
                services=self.services,
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
            self.logger.info(f"{fg_color}🏥 {patient_tag} {fg_color}| 门诊流程开始{Colors.RESET}")
            
            self.detail_logger.section("执行门诊流程")
            
            node_count = 0
            node_names = []  # 记录节点名称
            out = None
            
            for chunk in graph.stream(state):
                node_count += 1
                if isinstance(chunk, dict) and len(chunk) > 0:
                    node_name = list(chunk.keys())[0]
                    node_names.append(node_name)
                    out = chunk[node_name]
            
            self.logger.info(f"{fg_color}  ✅ 流程完成{Colors.RESET}")
            
            # 计算总耗时
            import time
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            
            self.detail_logger.section("诊断完成")
            self.detail_logger.info("")
            self.detail_logger.info("📋 执行概要:")
            self.detail_logger.info(f"  • 总节点数: {node_count}个")
            self.detail_logger.info(f"  • 总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
            self.detail_logger.info(f"  • 平均每节点: {total_time/node_count:.1f}秒" if node_count > 0 else "")
            self.detail_logger.info("")
            self.detail_logger.info("📍 完整节点路径:")
            self.detail_logger.info(f"  {' → '.join(node_names)}")
            self.detail_logger.info("")
            
            # 终端显示简化的节点路径（只显示关键节点）
            key_nodes = [n for n in node_names if n in ['triage', 'register', 'consultation', 'physical_exam', 'lab_test', 'diagnosis', 'discharge']]
            if key_nodes:
                self.logger.info(f"{fg_color}📍 {patient_tag} {fg_color}| 关键节点: {' → '.join(key_nodes)}{Colors.RESET}")
            
            # ===== 9. 释放医生资源 =====
            self.coordinator.release_doctor(assigned_doctor_id)
            
            # 10. 提取结果
            # 安全提取诊断结果（检查out是否存在，以及diagnosis是否为有效字典）
            final_diagnosis = "未明确"
            if out and hasattr(out, 'diagnosis'):
                if isinstance(out.diagnosis, dict) and out.diagnosis:
                    final_diagnosis = out.diagnosis.get("name", "未明确")
                    self.detail_logger.debug(f"📋 诊断提取: {final_diagnosis} (来自 out.diagnosis)")
                else:
                    self.detail_logger.warning(f"⚠️  out.diagnosis 为空字典或无效: {out.diagnosis}")
            else:
                self.detail_logger.warning(f"⚠️  out 为 None 或没有 diagnosis 属性")
            
            result = {
                "status": "completed",
                "patient_id": self.patient_id,
                "case_id": self.case_id,
                "dept": triaged_dept,
                "diagnosis": final_diagnosis,
                "node_count": node_count,
                "node_names": node_names,  # 添加节点名称列表
                "record_id": record_id,
                "detail_log_file": self.detail_logger.get_log_file_path() if self.detail_logger else "",  # 添加详细日志路径
            }
            
            self.logger.info(f"{fg_color}🎯 {patient_tag} {fg_color}| 诊断: {final_diagnosis}{Colors.RESET}")
            self.logger.info(f"{separator}\n")
            
            # 详细日志记录完整诊断结果
            self.detail_logger.info("🎯 诊断结果:")
            self.detail_logger.info(f"  • AI诊断: {final_diagnosis}")
            self.detail_logger.info("")
            
            # 问诊质量评估
            if hasattr(out, 'collected_info'):
                info_items = len([k for k, v in out.collected_info.items() if v])
                self.detail_logger.info("📊 问诊质量评估:")
                self.detail_logger.info(f"  • 收集信息项: {info_items}项")
                if hasattr(out, 'test_results'):
                    self.detail_logger.info(f"  • 完成检查: {len(out.test_results)}项")
                self.detail_logger.info("")
            
            # 关键决策点
            self.detail_logger.info("📌 关键决策点:")
            if hasattr(out, 'ordered_tests') and out.ordered_tests:
                self.detail_logger.info(f"  • 开单检查: {len(out.ordered_tests)}项")
                for test in out.ordered_tests[:5]:  # 最多显示5项
                    self.detail_logger.info(f"    - {test.get('name', '未知')} ({test.get('type', '未知')})")
            if hasattr(out, 'escalations') and out.escalations:
                self.detail_logger.info(f"  • 升级建议: {len(out.escalations)}项")
                for esc in out.escalations[:3]:
                    self.detail_logger.info(f"    - {esc}")
            self.detail_logger.info("")
            
            self.detail_logger.info("📋 病例记录:")
            self.detail_logger.info(f"  • 记录ID: {record_id}")
            self.detail_logger.info(f"  • 详细日志: {self.detail_logger.get_log_file_path()}")
            self.detail_logger.info("")
            
            # 添加诊疗流程总结
            self.detail_logger.section("诊疗流程总结")
            self.detail_logger.info("")
            self.detail_logger.info("📋 就诊流程回顾:")
            self.detail_logger.info(f"  1️⃣  患者到达 → 护士分诊 → {triaged_dept}")
            self.detail_logger.info(f"  2️⃣  问诊收集信息 → {node_count}个节点")
            if hasattr(out, 'ordered_tests') and out.ordered_tests:
                self.detail_logger.info(f"  3️⃣  开单检查 → {len(out.ordered_tests)}项检查")
            if hasattr(out, 'test_results') and out.test_results:
                self.detail_logger.info(f"  4️⃣  检查结果 → {len(out.test_results)}份报告")
            self.detail_logger.info(f"  5️⃣  诊断结论 → {final_diagnosis}")
            self.detail_logger.info("")
            
            # 质量指标
            self.detail_logger.info("📊 质量指标:")
            self.detail_logger.info(f"  • 流程效率: {total_time:.1f}秒 / {node_count}节点")
            if hasattr(out, 'ordered_tests'):
                test_coverage = "充分" if len(out.ordered_tests) >= 3 else "一般" if len(out.ordered_tests) >= 1 else "不足"
                self.detail_logger.info(f"  • 检查覆盖: {test_coverage} ({len(out.ordered_tests)}项)")
            self.detail_logger.info("")
            
            # 改进建议
            self.detail_logger.info("💡 流程改进建议:")
            if hasattr(out, 'ordered_tests') and len(out.ordered_tests) == 0:
                self.detail_logger.info("  ⚠️  未开具任何检查，可能影响诊断准确性")
            if node_count > 20:
                self.detail_logger.info("  ℹ️  流程节点较多，考虑优化诊疗路径")
            if total_time > 300:  # 5分钟
                self.detail_logger.info("  ℹ️  就诊时间较长，考虑优化响应速度")
            if hasattr(out, 'ordered_tests') and len(out.ordered_tests) >= 3:
                self.detail_logger.info("  ✅ 诊疗流程规范，质量良好")
            self.detail_logger.info("")
            
            # 最后输出详细日志路径
            self.logger.info(f"{fg_color}📋 详细日志: {self.detail_logger.get_log_file_path()}{Colors.RESET}")
            
            return result
            
        except Exception as e:
            # 使用红色显示错误
            self.logger.error(f"{Colors.RED}❌ 患者 {self.patient_id} 执行失败: {e}{Colors.RESET}", exc_info=True)
            
            # 如果已分配医生，需要释放（改进：使用 finally 确保清理）
            return self._cleanup_and_return_error(str(e))
        finally:
            # 确保资源清理（即使在异常情况下）
            try:
                # 关闭患者详细日志记录器
                if self.detail_logger:
                    from patient_detail_logger import close_patient_detail_logger
                    close_patient_detail_logger(self.patient_id)
                
                session = self.coordinator.get_patient(self.patient_id)
                if session and session.assigned_doctor:
                    doctor_id = session.assigned_doctor
                    # 检查医生是否仍在接诊该患者
                    doctor = self.coordinator.get_doctor(doctor_id)
                    if doctor and doctor.current_patient == self.patient_id:
                        self.coordinator.release_doctor(doctor_id)
                        # 资源清理日志移到详细日志中
                        if self.detail_logger:
                            self.detail_logger.info(f"清理资源：已释放医生 {doctor_id}")
            except Exception as cleanup_error:
                self.logger.error(f"⚠️ 资源清理失败: {cleanup_error}")
    
    def _cleanup_and_return_error(self, error_msg: str) -> Dict[str, Any]:
        """清理资源并返回错误结果"""
        return {
            "status": "failed",
            "patient_id": self.patient_id,
            "case_id": self.case_id,
            "error": error_msg,
            "detail_log_file": self.detail_logger.get_log_file_path() if self.detail_logger else "",  # 即使失败也返回日志路径
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
        max_questions: int = 3,
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
            max_questions: 最大问题数
            max_workers: 最大并发数
        """
        self.coordinator = coordinator
        self.retriever = retriever
        self.llm = llm
        self.services = services
        self.medical_record_service = medical_record_service
        self.max_questions = max_questions
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
        logger.info(f"  总计 {len(self.coordinator.doctors)} 名医生待添加到物理环境")
        for doctor_id, doctor in self.coordinator.doctors.items():
            dept_location = self._get_dept_location(doctor.dept)
            self.shared_world.add_agent(doctor_id, agent_type="doctor", initial_location=dept_location)
            logger.info(f"    ✓ {doctor.name} (ID:{doctor_id}, 科室:{doctor.dept}) -> 物理位置:{dept_location}")
        
        # 初始化共享设备
        self._setup_shared_equipment()
        
        # 创建共享的 Nurse 和 Lab Agent（所有患者共用）
        self.shared_nurse_agent = NurseAgent(llm=self.llm, max_triage_questions=3)
        self.shared_lab_agent = LabAgent(llm=self.llm)
        
        # 为每个医生创建 DoctorAgent 实例（映射到 coordinator 的医生）
        self.doctor_agents: Dict[str, DoctorAgent] = {}
        logger.info(f"  为 {len(self.coordinator.doctors)} 名医生创建 DoctorAgent...")
        for doctor_id, doctor in self.coordinator.doctors.items():
            self.doctor_agents[doctor_id] = DoctorAgent(
                dept=doctor.dept,
                retriever=self.retriever,
                llm=self.llm,
                max_questions=self.max_questions
            )
            logger.info(f"    ✓ DoctorAgent[{doctor_id}]: 科室={doctor.dept}, 最大问诊={self.max_questions}轮")
            
            # 【资源管理】注册医生到物理世界的资源池
            if self.shared_world:
                self.shared_world.register_doctor(doctor_id, doctor.dept)
                logger.debug(f"      → 已注册到物理世界资源池: {doctor.dept}")
        
        logger.info(f"✅ LangGraph 多患者处理器已启动 (最大并发: {max_workers})")
        logger.info(f"  📊 资源配置: {len(self.coordinator.doctors)}名医生, 1个共享World")
        logger.info(f"  🏥 神经内科医生: {', '.join([d.name for d in self.coordinator.doctors.values()])}")
        logger.info("")
    
    def _get_dept_location(self, dept: str) -> str:
        """获取科室对应的物理位置
        
        Args:
            dept: 科室代码
        
        Returns:
            位置ID
        """
        dept_location_map = {
            "neurology": "neuro",  # 神经医学使用神经内科诊室
        }
        return dept_location_map.get(dept, "neuro")
    
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

            max_questions=self.max_questions,
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
