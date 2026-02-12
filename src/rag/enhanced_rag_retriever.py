"""增强版 RAG 检索器 - 整合所有高级特性
融合：混合检索 + 动态分块 + 分层检索策略
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, List, Dict
from enum import Enum

# 强制使用离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 禁用不必要的警告
logging.getLogger("chromadb").setLevel(logging.ERROR)


class QueryType(Enum):
    """查询类型（用于分层检索）"""
    FACTUAL = "factual"  # 事实查询（疾病、症状）
    PROCEDURAL = "procedural"  # 流程查询（如何诊断、治疗）
    CASE_BASED = "case_based"  # 案例查询（类似病例）
    HISTORICAL = "historical"  # 历史回顾（患者历史）
    HOSPITAL_PROCESS = "hospital_process"  # 医院流程查询（挂号、缴费、表单模板等）


class EnhancedRAGRetriever:
    """增强版 RAG 检索器 - 完整解决方案
    
    特性：
        1. 混合检索：BM25 + 向量检索融合
        2. 动态分块：智能文档分块
        3. 分层检索：根据查询类型选择检索策略
        4. 多知识库：医学指南、临床案例、问答、历史记忆
        5. 自进化机制：高质量问答库持续更新
    """
    
    def __init__(
        self,
        *,
        spllm_root: Path | str,
        cache_folder: Path | str | None = None,
        enable_hybrid: bool = True,  # 是否启用混合检索
        enable_rerank: bool = False,  # 是否启用重排序
        cosine_threshold: float = 0.3,
        embed_model: str = "BAAI/bge-large-zh-v1.5",
    ):
        """
        Args:
            spllm_root: SPLLM-RAG1 项目根目录
            cache_folder: 模型缓存目录
            enable_hybrid: 是否启用混合检索（BM25 + 向量）
            enable_rerank: 是否启用重排序（需要重排序模型）
            cosine_threshold: 余弦距离阈值
            embed_model: 嵌入模型名称
        """
        self.spllm_root = Path(spllm_root).resolve()
        self.cache_folder = Path(cache_folder) if cache_folder else self.spllm_root / "model_cache"
        self.enable_hybrid = enable_hybrid
        self.enable_rerank = enable_rerank
        self.cosine_threshold = cosine_threshold
        self.embed_model = embed_model
        
        # 设置缓存路径
        os.environ['HF_HOME'] = str(self.cache_folder)
        
        # 初始化检索器
        self._hybrid_retriever = None
        self._simple_retriever = None
        
        # 日志
        self._logger = logging.getLogger("hospital_agent.enhanced_rag")
        self._logger.info(
            f"🚀 增强版 RAG 初始化: hybrid={enable_hybrid}, rerank={enable_rerank}"
        )
    
    def _get_retriever(self):
        """获取合适的检索器"""
        if self.enable_hybrid:
            if self._hybrid_retriever is None:
                from .hybrid_retriever import HybridRetriever
                self._hybrid_retriever = HybridRetriever(
                    spllm_root=self.spllm_root,
                    cache_folder=self.cache_folder,
                    cosine_threshold=self.cosine_threshold,
                    embed_model=self.embed_model,
                )
            return self._hybrid_retriever
        else:
            if self._simple_retriever is None:
                from .adaptive_rag_retriever import AdaptiveRAGRetriever
                self._simple_retriever = AdaptiveRAGRetriever(
                    spllm_root=self.spllm_root,
                    cache_folder=self.cache_folder,
                    cosine_threshold=self.cosine_threshold,
                    embed_model=self.embed_model,
                )
            return self._simple_retriever
    
    def _classify_query(self, query: str, context: Dict[str, Any] = None) -> QueryType:
        """分类查询类型（用于分层检索）
        
        Args:
            query: 查询文本
            context: 上下文信息（如患者信息、对话历史）
            
        Returns:
            查询类型
        """
        query_lower = query.lower()
        
        # 医院流程查询（优先级最高）
        if any(kw in query for kw in ["流程", "模板", "证明", "病假", "病历", "表单", "SOP", "缴费", "预约", "挂号", "诊断书", "宣教"]):
            return QueryType.HOSPITAL_PROCESS
        
        # 历史查询
        if any(kw in query for kw in ["上次", "之前", "历史", "以前", "记录"]):
            return QueryType.HISTORICAL
        
        # 流程查询（医学流程）
        if any(kw in query for kw in ["如何", "怎么", "步骤", "检查", "治疗"]):
            return QueryType.PROCEDURAL
        
        # 案例查询
        if any(kw in query for kw in ["类似", "案例", "病例", "参考"]):
            return QueryType.CASE_BASED
        
        # 默认为事实查询
        return QueryType.FACTUAL
    
    def _hierarchical_retrieve(
        self,
        query: str,
        query_type: QueryType,
        filters: Dict[str, Any],
        k: int
    ) -> List[Dict[str, Any]]:
        """分层检索策略
        
        根据查询类型，调整检索库的优先级和权重
        """
        retriever = self._get_retriever()
        
        # 定义不同查询类型的检索策略
        strategies = {
            QueryType.FACTUAL: {
                "libraries": ["MedicalGuide_db", "HighQualityQA_db"],
                "weights": [0.6, 0.4],
                "k_per_lib": [k, k]
            },
            QueryType.PROCEDURAL: {
                "libraries": ["MedicalGuide_db", "ClinicalCase_db", "HighQualityQA_db"],
                "weights": [0.5, 0.3, 0.2],
                "k_per_lib": [k, k//2, k//2]
            },
            QueryType.CASE_BASED: {
                "libraries": ["ClinicalCase_db", "HighQualityQA_db", "MedicalGuide_db"],
                "weights": [0.5, 0.3, 0.2],
                "k_per_lib": [k, k//2, k//2]
            },
            QueryType.HISTORICAL: {
                "libraries": ["UserHistory_db", "HighQualityQA_db"],
                "weights": [0.7, 0.3],
                "k_per_lib": [k//2, k//2]
            },
            QueryType.HOSPITAL_PROCESS: {
                "libraries": ["HospitalProcess_db", "HighQualityQA_db"],
                "weights": [0.8, 0.2],
                "k_per_lib": [k, k//2]
            },
        }
        
        strategy = strategies.get(query_type, strategies[QueryType.FACTUAL])
        
        all_results = []
        
        # 按策略检索各个库
        for lib, weight, k_lib in zip(
            strategy["libraries"],
            strategy["weights"],
            strategy["k_per_lib"]
        ):
            try:
                if lib == "UserHistory_db" and not filters.get("patient_id"):
                    continue  # 跳过需要 patient_id 的库
                
                if self.enable_hybrid and hasattr(retriever, 'hybrid_retrieve'):
                    results = retriever.hybrid_retrieve(query, lib, k=k_lib)
                else:
                    # 简单向量检索（需要适配）
                    results = self._simple_vector_retrieve(retriever, query, lib, k=k_lib)
                
                # 调整分数权重
                for r in results:
                    r["score"] = r.get("score", 0) * weight
                    r["meta"]["query_type"] = query_type.value
                    r["meta"]["library"] = lib
                
                all_results.extend(results)
            
            except Exception as e:
                self._logger.warning(f"⚠️  检索 {lib} 失败: {e}")
        
        # 按分数排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return all_results[:k * 2]
    
    def _simple_vector_retrieve(
        self,
        retriever,
        query: str,
        db_name: str,
        k: int
    ) -> List[Dict[str, Any]]:
        """简单向量检索（用于非混合模式）"""
        # 这里调用简单检索器的内部方法
        if hasattr(retriever, '_vector_search'):
            return retriever._vector_search(query, db_name, k)
        
        # 降级方案
        return []
    
    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = 4,
        enable_hierarchical: bool = True,  # 是否启用分层检索
    ) -> list[dict[str, Any]]:
        """统一检索接口
        
        Args:
            query: 查询文本
            filters: 过滤条件（patient_id, dept, etc.）
            k: 返回结果数量
            enable_hierarchical: 是否启用分层检索
            
        Returns:
            检索结果列表
        """
        if not query or not query.strip():
            return []
        
        filters = filters or {}
        
        # 是否启用分层检索
        if enable_hierarchical:
            # 分析查询类型
            query_type = self._classify_query(query, filters)
            self._logger.debug(f"🔍 查询类型: {query_type.value}")
            
            # 分层检索
            results = self._hierarchical_retrieve(query, query_type, filters, k)
        else:
            # 标准多库检索
            retriever = self._get_retriever()
            results = retriever.retrieve(query, filters=filters, k=k)
        
        # 可选：重排序
        if self.enable_rerank and results:
            results = self._rerank(query, results)
        
        self._logger.info(f"✅ 检索完成: 找到 {len(results)} 条结果")
        return results
    
    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序（可选功能，需要重排序模型）
        
        使用交叉编码器对检索结果重新排序
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # 加载重排序模型（需要预先下载）
            model = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
            
            # 构建查询-文档对
            pairs = [[query, r["text"]] for r in results]
            
            # 计算重排序分数
            scores = model.predict(pairs)
            
            # 更新分数
            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)
                results[i]["original_score"] = results[i].get("score", 0)
                results[i]["score"] = float(score)
            
            # 按新分数排序
            results.sort(key=lambda x: x["score"], reverse=True)
            
            self._logger.debug("✨ 重排序完成")
            return results
        
        except Exception as e:
            self._logger.warning(f"⚠️  重排序失败，使用原始排序: {e}")
            return results
    
    def update_history(
        self,
        patient_id: str,
        dialogue_summary: str,
        diagnosis: str = None,
        treatment: str = None
    ):
        """更新患者历史记忆库
        
        Args:
            patient_id: 患者 ID
            dialogue_summary: 对话摘要
            diagnosis: 诊断结果（可选）
            treatment: 治疗方案（可选）
        """
        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            
            retriever = self._get_retriever()
            if hasattr(retriever, '_init_embeddings'):
                retriever._init_embeddings()
            
            db_path = self.spllm_root / "chroma" / "UserHistory_db"
            db = Chroma(
                persist_directory=str(db_path),
                embedding_function=retriever._embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # 构建摘要文档
            summary_text = f"患者 {patient_id} 就诊记录：\n{dialogue_summary}"
            if diagnosis:
                summary_text += f"\n诊断：{diagnosis}"
            if treatment:
                summary_text += f"\n治疗：{treatment}"
            
            doc = Document(
                page_content=summary_text,
                metadata={
                    "patient_id": patient_id,
                    "type": "dialogue_summary",
                    "diagnosis": diagnosis or "",
                    "treatment": treatment or ""
                }
            )
            
            db.add_documents([doc])
            self._logger.info(f"✅ 患者 {patient_id} 历史记忆已更新")
        
        except Exception as e:
            self._logger.error(f"❌ 更新历史记忆失败: {e}")
    
    def update_high_quality_qa(
        self,
        question: str,
        answer: str,
        quality_score: float = 1.0
    ):
        """更新高质量问答库（自进化机制）
        
        Args:
            question: 问题
            answer: 答案
            quality_score: 质量评分（0-1）
        """
        if quality_score < 0.7:  # 只保存高质量问答
            self._logger.debug(f"⚠️  问答质量较低（{quality_score:.2f}），跳过")
            return
        
        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            
            retriever = self._get_retriever()
            if hasattr(retriever, '_init_embeddings'):
                retriever._init_embeddings()
            
            db_path = self.spllm_root / "chroma" / "HighQualityQA_db"
            db = Chroma(
                persist_directory=str(db_path),
                embedding_function=retriever._embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # 使用问题作为嵌入内容（便于检索）
            doc = Document(
                page_content=question,
                metadata={
                    "question": question,
                    "answer": answer,
                    "quality_score": quality_score,
                    "type": "qa_pair"
                }
            )
            
            db.add_documents([doc])
            self._logger.info(f"✅ 高质量问答已添加（质量={quality_score:.2f}）")
        
        except Exception as e:
            self._logger.error(f"❌ 更新问答库失败: {e}")


__all__ = ["EnhancedRAGRetriever", "QueryType"]
