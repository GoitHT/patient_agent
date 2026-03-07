from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from utils import now_iso

if TYPE_CHECKING:
    from environment import HospitalWorld, PhysicalState


def _as_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for c in chunks:
        citations.append(
            {
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
            }
        )
    return citations


def make_audit_entry(
    *,
    node_name: str,
    inputs_summary: dict[str, Any],
    outputs_summary: dict[str, Any],
    decision: str,
    chunks: list[dict[str, Any]] | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    """Create an audit entry that follows the project traceability spec.

    Notes:
    - inputs_summary / outputs_summary should avoid sensitive details.
    - chunks are the retrieved RAG chunks; only doc_id/chunk_id/score are stored as citations.
    """

    return {
        "ts": now_iso(),
        "node_name": node_name,
        "inputs_summary": inputs_summary,
        "outputs_summary": outputs_summary,
        "decision": decision,
        "citations": _as_citations(chunks or []),
        "flags": flags or [],
    }


class BaseState(BaseModel):
    """Common state shared across all departments and graph nodes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dept: Literal[
        "internal_medicine", "surgery", "orthopedics", "urology",
        "obstetrics_gynecology", "pediatrics", "neurology", "oncology",
        "infectious_disease", "dermatology_std", "ent_ophthalmology_stomatology",
        "psychiatry", "emergency", "rehabilitation_pain", "traditional_chinese_medicine"
    ]
    patient_profile: dict[str, Any] = Field(default_factory=dict)
    appointment: dict[str, Any] = Field(default_factory=dict)
    original_chief_complaint: str = ""  # 从数据集读取的原始主诉（仅患者智能体可见）
    chief_complaint: str = ""  # 医生通过问诊总结得出的主诉（医生可见）
    history: dict[str, Any] = Field(default_factory=dict)
    exam_findings: dict[str, Any] = Field(default_factory=dict)
    dept_payload: dict[str, Any] = Field(default_factory=dict)
    specialty_summary: dict[str, Any] = Field(default_factory=dict)

    preliminary_assessment: dict[str, Any] = Field(default_factory=dict)
    need_aux_tests: bool = False
    ordered_tests: list[dict[str, Any]] = Field(default_factory=list)
    test_prep: list[dict[str, Any]] = Field(default_factory=list)
    test_results: list[dict[str, Any]] = Field(default_factory=list)

    diagnosis: dict[str, Any] = Field(default_factory=dict)
    treatment_plan: dict[str, Any] = Field(default_factory=dict)
    discharge_docs: list[dict[str, Any]] = Field(default_factory=list)
    followup_plan: dict[str, Any] = Field(default_factory=dict)

    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    escalations: list[str] = Field(default_factory=list)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    
    # 多智能体系统新增字段
    agent_interactions: dict[str, Any] = Field(default_factory=dict)  # 医患护对话记录
    agent_config: dict[str, Any] = Field(default_factory=dict)  # Agent配置（max_questions等）
    ground_truth: dict[str, Any] = Field(default_factory=dict)  # 仅含初步诊断，用于后期评估
    medical_data: dict[str, Any] = Field(default_factory=dict)  # 患者不可见的医疗数据（所有体格检查+辅助检查）供医生/系统参考
    case_data: dict[str, Any] = Field(default_factory=dict)  # 完整病例数据
    node_qa_counts: dict[str, int] = Field(default_factory=dict)  # 每个节点的问答轮数计数
    interaction_state: list[dict[str, Any]] = Field(default_factory=list)  # 问诊对话记录（与 case_qa_records 表字段完全对应），用于后期评估
    
    # 物理环境集成字段
    world_context: Optional[Any] = Field(default=None, exclude=True)  # HospitalWorld实例（不序列化）
    patient_id: str = "patient_001"  # 患者在物理环境中的ID
    current_location: str = "lobby"  # 当前物理位置
    physical_state_snapshot: dict[str, Any] = Field(default_factory=dict)  # 物理状态快照（可序列化）
    movement_history: list[dict[str, Any]] = Field(default_factory=list)  # 移动轨迹记录
    
    # 病例库集成字段
    medical_record_integration: Optional[Any] = Field(default=None, exclude=True)  # 病例库集成器（不序列化）
    dept_display_name: str = ""  # 科室中文显示名称（用于输出显示）
    
    # 患者详细日志记录器（不序列化）
    patient_detail_logger: Optional[Any] = Field(default=None, exclude=True)
    
    # 多智能体系统：Agent实例（不序列化）
    coordinator: Optional[Any] = Field(default=None, exclude=True)  # HospitalCoordinator实例
    doctor_agents: dict[str, Any] = Field(default_factory=dict, exclude=True)  # 医生Agent字典
    doctor_agent: Optional[Any] = Field(default=None, exclude=True)  # 当前分配的医生Agent
    patient_agent: Optional[Any] = Field(default=None, exclude=True)  # 患者Agent实例
    assigned_doctor_id: str = ""  # 分配的医生ID（可序列化）
    assigned_doctor_name: str = ""  # 分配的医生姓名（可序列化）
    
    @property
    def world(self) -> Optional[Any]:
        """简洁的world访问器，指向world_context"""
        return self.world_context

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        # LangGraph may serialize Pydantic state with `exclude_unset=True`.
        # In-place mutations (e.g., list.append / dict.__setitem__) do not mark fields as "set" in Pydantic v2,
        # which can cause those fields to be omitted from the serialized state and effectively lost between nodes.
        #
        # We proactively mark all mutable containers as "set" to guarantee persistence across graph steps.
        fields = {
            "patient_profile",
            "appointment",
            "history",
            "exam_findings",
            "dept_payload",
            "specialty_summary",
            "preliminary_assessment",
            "ordered_tests",
            "test_prep",
            "test_results",
            "diagnosis",
            "treatment_plan",
            "discharge_docs",
            "followup_plan",
            "retrieved_chunks",
            "escalations",
            "audit_trail",
            "agent_interactions",
            "agent_config",
            "ground_truth",
            "medical_data",
            "case_data",
            "node_qa_counts",
            "interaction_state",
            "physical_state_snapshot",
            "movement_history",
        }
        try:
            self.__pydantic_fields_set__.update(fields)
        except Exception:
            # If internal field tracking changes, we prefer correctness at runtime over strict tracking.
            pass

    def add_retrieved_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self.retrieved_chunks = [*self.retrieved_chunks, *chunks]

    def add_audit(self, entry: dict[str, Any]) -> None:
        self.audit_trail = [*self.audit_trail, entry]
    
    def sync_physical_state(self) -> None:
        """从HospitalWorld同步物理状态到快照"""
        if self.world_context and self.patient_id in self.world_context.physical_states:
            physical_state = self.world_context.physical_states[self.patient_id]
            self.physical_state_snapshot = {
                "energy_level": physical_state.energy_level,
                "pain_level": physical_state.pain_level,
                "consciousness_level": physical_state.consciousness_level,
                "vital_signs": physical_state.get_vital_signs_dict(),
                "symptoms": physical_state.get_symptom_severity_dict(),
                "last_update": physical_state.last_update.isoformat() if physical_state.last_update else None,
                "current_time": self.world_context.current_time.isoformat() if self.world_context else None,
            }
            # 更新当前位置
            if self.patient_id in self.world_context.agents:
                self.current_location = self.world_context.agents[self.patient_id]
    
    def update_physical_world(self, action: str, duration_minutes: int = 0, **kwargs) -> dict[str, Any]:
        """更新物理世界状态并返回观察结果
        
        Args:
            action: 执行的动作（如'consult', 'wait', 'test'）
            duration_minutes: 动作持续时间（分钟）
            **kwargs: 其他参数，如severity变化、症状更新等
        
        Returns:
            包含物理状态变化的字典
        """
        if not self.world_context:
            return {"status": "no_world", "message": "物理环境未初始化"}
        
        result = {"action": action, "duration": duration_minutes}
        
        # 推进时间
        if duration_minutes > 0:
            self.world_context.advance_time(duration_minutes)
        
        # 更新物理状态
        if self.patient_id in self.world_context.physical_states:
            physical_state = self.world_context.physical_states[self.patient_id]
            
            # 更新生理状态（基于时间流逝）
            physical_state.update_physiology(self.world_context.current_time)
            
            # 根据动作类型应用额外效果
            if action == "consult":
                # 问诊消耗体力
                energy_cost = kwargs.get("energy_cost", 0.5)
                physical_state.energy_level = max(0, physical_state.energy_level - energy_cost)
                result["energy_cost"] = energy_cost
                
            elif action == "wait":
                # 等待时症状可能恶化
                stress_increase = kwargs.get("stress_increase", 0.1)
                for symptom in physical_state.symptoms.values():
                    if not symptom.treated:
                        symptom.severity = min(10.0, symptom.severity + stress_increase)
                result["stress_increase"] = stress_increase
                
            elif action == "test":
                # 检查可能增加不适
                discomfort = kwargs.get("discomfort", 0.3)
                physical_state.energy_level = max(0, physical_state.energy_level - discomfort)
                result["discomfort"] = discomfort
                
            elif action == "treatment":
                # 治疗缓解症状
                effectiveness = kwargs.get("effectiveness", 0.8)
                physical_state.apply_medication(
                    kwargs.get("medication", "治疗"),
                    effectiveness
                )
                result["effectiveness"] = effectiveness
            
            # 检查危急状态
            is_critical = physical_state.check_critical_condition()
            if is_critical:
                result["critical_warning"] = True
                result["consciousness"] = physical_state.consciousness_level
                
                # 记录危急警告到audit_trail
                from utils import now_iso
                self.audit_trail.append({
                    "ts": now_iso(),
                    "event": "PHYSICAL_CRITICAL_WARNING",
                    "consciousness": physical_state.consciousness_level,
                    "energy": physical_state.energy_level,
                    "pain": physical_state.pain_level,
                    "message": "⚠️ 患者物理状态异常，需关注",
                })
        
        # 同步状态到快照
        self.sync_physical_state()
        
        result["status"] = "success"
        result["current_time"] = self.physical_state_snapshot.get("current_time")
        result["physical_state"] = self.physical_state_snapshot
        
        # 记录物理状态变化到audit_trail（简化版）
        from utils import now_iso
        self.audit_trail.append({
            "ts": now_iso(),
            "event": "PHYSICAL_UPDATE",
            "action": action,
            "duration": duration_minutes,
            "location": self.current_location,
            "energy": self.physical_state_snapshot.get("energy_level"),
            "pain": self.physical_state_snapshot.get("pain_level"),
            "time": self.physical_state_snapshot.get("current_time"),
        })
        
        return result
    
    def get_physical_impact_on_diagnosis(self) -> dict[str, Any]:
        """获取物理状态对诊断的影响
        
        Returns:
            包含影响因素的字典，可用于调整问诊策略
        """
        if not self.physical_state_snapshot:
            return {"has_impact": False}
        
        impact = {
            "has_impact": True,
            "energy_level": self.physical_state_snapshot.get("energy_level", 10),
            "pain_level": self.physical_state_snapshot.get("pain_level", 0),
            "consciousness": self.physical_state_snapshot.get("consciousness_level", "alert"),
        }
        
        # 根据状态给出建议
        suggestions = []
        warnings = []  # 严重警告
        impact["max_questions"] = 10  # 默认最大问诊轮数
        
        # 只保留意识异常的严重警告
        if impact["consciousness"] != "alert":
            warnings.append("🚨🚨 患者意识异常")
            suggestions.append(f"患者意识状态异常（{impact['consciousness']}），需立即紧急处理！")
            impact["emergency"] = True
            impact["max_questions"] = 0  # 不应继续常规问诊
        
        impact["suggestions"] = suggestions
        impact["warnings"] = warnings  # 严重警告单独列出
        
        return impact
