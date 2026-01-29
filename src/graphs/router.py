from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


from graphs.common_opd_graph import CommonOPDGraph, Services
from graphs.dept_subgraphs.common_specialty_subgraph import build_common_specialty_subgraph
from rag import ChromaRetriever
from services.appointment import AppointmentService
from services.billing import BillingService
from services.llm_client import LLMClient


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_services() -> Services:
    """构建必要的服务（预约和计费）"""
    return Services(
        appointment=AppointmentService(),
        billing=BillingService(),
    )


def build_dept_subgraphs(
    *, retriever: ChromaRetriever, llm: LLMClient | None = None,
    doctor_agent: Any | None = None, patient_agent: Any | None = None, 
    max_questions: int = 3  # 最底层默认值，通常从 config传入
) -> dict[str, Any]:
    """为各科室构建通用子图，返回 {dept_key: compiled_graph}
    
    注意：所有科室共享同一个通用专科子图结构，
    医生智能体的科室属性会在子图执行时动态设置
    """
    
    # 构建通用专科子图（所有科室共用）
    common_graph = build_common_specialty_subgraph(
        retriever=retriever,
        llm=llm,
        doctor_agent=doctor_agent,
        patient_agent=patient_agent,
        max_questions=max_questions,
    )
    
    # 只保留神经医学科
    return {
        "neurology": common_graph,
    }


def build_common_graph(
    dept_subgraphs: dict[str, Any],
    *,
    retriever: ChromaRetriever,
    services: Services,
    llm: LLMClient | None = None,
    llm_reports: bool = False,
    use_agents: bool = True,  # 总是使用多智能体模式
    patient_agent: Any | None = None,
    doctor_agent: Any | None = None,
    nurse_agent: Any | None = None,
    lab_agent: Any | None = None,
    max_questions: int = 3,  # 最底层默认值，通常从config传入
    world: Any | None = None,  # 新增：HospitalWorld实例
):
    return CommonOPDGraph(
        retriever=retriever,
        dept_subgraphs=dept_subgraphs,
        services=services,
        llm=llm,
        llm_reports=llm_reports,
        use_agents=use_agents,
        patient_agent=patient_agent,
        doctor_agent=doctor_agent,
        nurse_agent=nurse_agent,
        lab_agent=lab_agent,
        max_questions=max_questions,
        world=world,  # 传递world实例
    ).build()


def default_retriever(
    *, persist_dir: Path | None = None, collection_name: str = "hospital_kb"
) -> ChromaRetriever:
    root = repo_root()
    if persist_dir is None:
        persist_dir = root / ".chroma"
    persist_dir = Path(persist_dir)
    if not persist_dir.is_absolute():
        persist_dir = (root / persist_dir).resolve()
    return ChromaRetriever(persist_dir=persist_dir, collection_name=collection_name)
