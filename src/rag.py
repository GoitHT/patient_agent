"""RAG (检索增强生成) 模块 - Adaptive RAG 系统

本模块完全采用 SPLLM-RAG1 的 Adaptive RAG 系统，提供：
- 真实语义嵌入（text2vec-base-chinese）
- 多向量库协同检索（医学指南、临床案例、高质量问答、患者历史）
- 余弦相似度匹配
"""
from __future__ import annotations

from typing import Any

from rag.adaptive_rag_retriever import AdaptiveRAGRetriever


class DummyRetriever:
    """虚拟检索器 - 用于测试或跳过 RAG 功能时"""
    
    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = 4,
    ) -> list[dict[str, Any]]:
        """返回空检索结果"""
        return []


# 主要导出
__all__ = [
    "AdaptiveRAGRetriever",
    "DummyRetriever",
]
