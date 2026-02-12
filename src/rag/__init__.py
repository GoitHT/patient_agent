"""RAG (检索增强生成) 模块 - 增强版 RAG 系统

本模块提供多种 RAG 检索器：
- AdaptiveRAGRetriever: 基础自适应检索器（向量检索）
- HybridRetriever: 混合检索器（BM25 + 向量检索）
- EnhancedRAGRetriever: 增强版检索器（完整功能）

特性：
- 真实语义嵌入（BAAI/bge-large-zh-v1.5）
- 四大向量库协同检索（医学指南、临床案例、高质量问答、患者历史）
- 融合检索（BM25 + 向量检索 + RRF 融合）
- 动态分块策略
- 分层检索策略
- 自进化机制
"""
from __future__ import annotations

from typing import Any

from .adaptive_rag_retriever import AdaptiveRAGRetriever
from .hybrid_retriever import HybridRetriever
from .enhanced_rag_retriever import EnhancedRAGRetriever, QueryType
from .dynamic_chunker import (
    DynamicChunker,
    ChunkConfig,
    ChunkStrategy,
    create_chunker_for_medical_documents,
)
from .qa_evaluator import (
    DialogueQualityEvaluator,
    DialogueQualityScore,
    PatientAnswerMetrics,
    DoctorQuestionMetrics,
)
from .query_optimizer import RAGQueryOptimizer, QueryContext, get_query_optimizer
from .keyword_generator import RAGKeywordGenerator, NodeContext


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
    # 检索器
    "AdaptiveRAGRetriever",
    "HybridRetriever",
    "EnhancedRAGRetriever",
    "DummyRetriever",
    # 分块器
    "DynamicChunker",
    "ChunkConfig",
    "ChunkStrategy",
    "create_chunker_for_medical_documents",
    # 查询优化器
    "RAGQueryOptimizer",
    "QueryContext",
    "get_query_optimizer",
    # 关键词生成器
    "RAGKeywordGenerator",
    "NodeContext",
    # 其他
    "QueryType",
    "DialogueQualityEvaluator",
]
