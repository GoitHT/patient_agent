"""混合检索器 - 融合 BM25 和向量检索
实现融合检索策略，结合关键词匹配和语义检索的优势
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, List, Dict
from collections import defaultdict

# 强制使用离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 禁用不必要的警告
logging.getLogger("chromadb").setLevel(logging.ERROR)


class HybridRetriever:
    """混合检索器 - BM25 + 向量检索融合
    
    特性：
        - BM25 关键词匹配（处理专业术语、疾病名称等）
        - 向量语义检索（理解语义相似性）
        - 倒数排序融合（Reciprocal Rank Fusion）
        - 动态权重调节
        - 多知识库支持：医学指南、临床案例、问答、历史、医院流程
    
    知识库说明：
        - MedicalGuide_db: 医学专业指南、诊疗规范、检查指征
        - HospitalProcess_db: 医院通用流程、表单模板、SOP文档
        - ClinicalCase_db: 临床案例库
        - HighQualityQA_db: 高质量问答库
        - UserHistory_db: 患者历史记录
    """
    
    def __init__(
        self,
        *,
        spllm_root: Path | str,
        cache_folder: Path | str | None = None,
        cosine_threshold: float = 0.3,
        embed_model: str = "BAAI/bge-large-zh-v1.5",
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        k1: float = 1.5,  # BM25 参数
        b: float = 0.75,  # BM25 参数
    ):
        """
        Args:
            spllm_root: SPLLM-RAG1 项目根目录
            cache_folder: 模型缓存目录
            cosine_threshold: 余弦距离阈值
            embed_model: 嵌入模型名称
            bm25_weight: BM25 检索权重（0-1）
            vector_weight: 向量检索权重（0-1）
            k1: BM25 词频饱和参数
            b: BM25 文档长度归一化参数
        """
        self.spllm_root = Path(spllm_root).resolve()
        self.cache_folder = Path(cache_folder) if cache_folder else self.spllm_root / "model_cache"
        self.cosine_threshold = cosine_threshold
        self.embed_model = embed_model
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.k1 = k1
        self.b = b
        
        # 设置缓存路径
        os.environ['HF_HOME'] = str(self.cache_folder)
        
        # 延迟导入
        self._embeddings = None
        self._dbs = {}
        self._bm25_indices = {}  # BM25 索引缓存
        
        # 日志
        self._logger = logging.getLogger("hospital_agent.hybrid_rag")
        self._logger.info(f"🔄 混合检索器初始化: BM25={bm25_weight}, Vector={vector_weight}")
    
    def _init_embeddings(self):
        """延迟初始化嵌入模型"""
        if self._embeddings is not None:
            return
        
        try:
            # 临时屏蔽嵌入模型的加载日志
            import logging as std_logging
            sentence_transformers_logger = std_logging.getLogger('sentence_transformers')
            transformers_logger = std_logging.getLogger('transformers')
            old_st_level = sentence_transformers_logger.level
            old_tf_level = transformers_logger.level
            sentence_transformers_logger.setLevel(std_logging.WARNING)
            transformers_logger.setLevel(std_logging.WARNING)
            
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32
                },
                cache_folder=str(self.cache_folder)
            )
            
            # 恢复日志级别
            sentence_transformers_logger.setLevel(old_st_level)
            transformers_logger.setLevel(old_tf_level)
            
            test_vec = self._embeddings.embed_query("测试")
            self._logger.info(f"✅ 嵌入模型加载成功（维度={len(test_vec)}）")
        except Exception as e:
            # 恢复日志级别（即使出错）
            try:
                sentence_transformers_logger.setLevel(old_st_level)
                transformers_logger.setLevel(old_tf_level)
            except:
                pass
            self._logger.error(f"❌ 嵌入模型初始化失败: {e}")
            raise RuntimeError(f"无法初始化嵌入模型: {e}")
    
    def _get_db(self, db_name: str):
        """获取或加载向量库"""
        if db_name in self._dbs:
            return self._dbs[db_name]
        
        self._init_embeddings()
        
        try:
            from langchain_chroma import Chroma
            
            db_path = self.spllm_root / "chroma" / db_name
            if not db_path.exists():
                self._logger.warning(f"⚠️  向量库路径不存在: {db_path}")
                return None
            
            db = Chroma(
                persist_directory=str(db_path),
                embedding_function=self._embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            self._dbs[db_name] = db
            self._logger.debug(f"✅ 向量库加载成功: {db_name}")
            return db
        except Exception as e:
            self._logger.error(f"❌ 向量库 {db_name} 加载失败: {e}")
            return None
    
    def _get_bm25_index(self, db_name: str):
        """获取或构建 BM25 索引"""
        if db_name in self._bm25_indices:
            return self._bm25_indices[db_name]
        
        db = self._get_db(db_name)
        if not db:
            return None
        
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            # 获取向量库中的所有文档
            collection = db._collection
            all_docs = collection.get(include=["documents", "metadatas"])
            
            if not all_docs or not all_docs.get("documents"):
                self._logger.warning(f"⚠️  向量库 {db_name} 无文档")
                return None
            
            # 分词构建 BM25 索引
            documents = all_docs["documents"]
            metadatas = all_docs.get("metadatas", [])
            
            tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            
            # 缓存索引和原始文档
            self._bm25_indices[db_name] = {
                "bm25": bm25,
                "documents": documents,
                "metadatas": metadatas,
                "tokenized_corpus": tokenized_corpus
            }
            
            self._logger.debug(f"✅ BM25 索引构建成功: {db_name}（{len(documents)} 文档）")
            return self._bm25_indices[db_name]
        
        except ImportError:
            self._logger.error("❌ 缺少依赖：pip install rank-bm25 jieba")
            return None
        except Exception as e:
            self._logger.error(f"❌ BM25 索引构建失败: {e}")
            return None
    
    def _bm25_search(self, query: str, db_name: str, k: int = 10) -> List[Dict[str, Any]]:
        """BM25 关键词检索"""
        index_data = self._get_bm25_index(db_name)
        if not index_data:
            return []
        
        try:
            import jieba
            
            bm25 = index_data["bm25"]
            documents = index_data["documents"]
            metadatas = index_data["metadatas"]
            
            # 查询分词
            query_tokens = list(jieba.cut(query))
            scores = bm25.get_scores(query_tokens)
            
            # 获取 top-k 结果
            top_indices = scores.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只保留有得分的结果
                    results.append({
                        "text": documents[idx],
                        "meta": metadatas[idx] if idx < len(metadatas) else {},
                        "score": float(scores[idx]),
                        "source": "bm25"
                    })
            
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  BM25 检索失败: {e}")
            return []
    
    def _vector_search(self, query: str, db_name: str, k: int = 10) -> List[Dict[str, Any]]:
        """向量语义检索"""
        db = self._get_db(db_name)
        if not db:
            return []
        
        try:
            docs_and_distances = db.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, distance in docs_and_distances:
                if distance < self.cosine_threshold * 2:  # 放宽阈值以便融合
                    similarity = max(0, 1 - distance)
                    results.append({
                        "text": doc.page_content,
                        "meta": doc.metadata,
                        "score": float(similarity),
                        "source": "vector"
                    })
            
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  向量检索失败: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """倒数排序融合（RRF）算法
        
        RRF 分数 = sum(1 / (k + rank_i))，其中 k 是常数（通常为 60）
        """
        # 为每个文档计算 RRF 分数
        rrf_scores = defaultdict(float)
        doc_info = {}  # 存储文档信息
        
        # BM25 结果
        for rank, result in enumerate(bm25_results, start=1):
            doc_key = result["text"][:100]  # 使用前100字符作为唯一标识
            rrf_scores[doc_key] += self.bm25_weight / (k + rank)
            if doc_key not in doc_info:
                doc_info[doc_key] = result
        
        # 向量检索结果
        for rank, result in enumerate(vector_results, start=1):
            doc_key = result["text"][:100]
            rrf_scores[doc_key] += self.vector_weight / (k + rank)
            if doc_key not in doc_info:
                doc_info[doc_key] = result
        
        # 按 RRF 分数排序
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建最终结果
        final_results = []
        for doc_key, rrf_score in sorted_docs:
            result = doc_info[doc_key].copy()
            result["rrf_score"] = float(rrf_score)
            result["fusion_method"] = "RRF"
            final_results.append(result)
        
        return final_results
    
    def hybrid_retrieve(
        self,
        query: str,
        db_name: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """混合检索（单个库）
        
        Args:
            query: 查询文本
            db_name: 数据库名称
            k: 返回结果数量
            
        Returns:
            融合后的检索结果
        """
        # 1. BM25 检索
        bm25_results = self._bm25_search(query, db_name, k=k*2)
        
        # 2. 向量检索
        vector_results = self._vector_search(query, db_name, k=k*2)
        
        # 3. 融合排序
        fused_results = self._reciprocal_rank_fusion(bm25_results, vector_results)
        
        self._logger.debug(
            f"🔄 混合检索 {db_name}: BM25={len(bm25_results)}, "
            f"Vector={len(vector_results)}, Fused={len(fused_results)}"
        )
        
        return fused_results[:k]
    
    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = 4,
    ) -> list[dict[str, Any]]:
        """多库混合检索接口（兼容原有接口）
        
        Args:
            query: 查询文本
            filters: 过滤条件（支持 db_name 参数指定单一数据库）
            k: 每个库返回的结果数量
            
        Returns:
            统一格式的检索结果
        """
        filters = filters or {}
        patient_id = filters.get("patient_id")
        db_name = filters.get("db_name")  # 如果指定了db_name，只查询该数据库
        
        all_results = []
        
        # 如果指定了单一数据库，只查询该库
        if db_name:
            source_map = {
                "HospitalProcess_db": "HospitalProcess",
                "MedicalGuide_db": "MedicalGuide",
                "ClinicalCase_db": "ClinicalCase",
                "HighQualityQA_db": "HighQualityQA",
                "UserHistory_db": "UserHistory",
            }
            results = self.hybrid_retrieve(query, db_name, k=k)
            for r in results:
                r["meta"]["source"] = source_map.get(db_name, db_name)
                if patient_id and db_name == "UserHistory_db":
                    r["meta"]["patient_id"] = patient_id
            return self._format_results(results, k)
        
        # 否则按照原有逻辑查询多个库
        
        # 1. 患者历史记忆（如果有 patient_id）
        if patient_id:
            history_results = self.hybrid_retrieve(query, "UserHistory_db", k=2)
            for r in history_results:
                r["meta"]["patient_id"] = patient_id
                r["meta"]["source"] = "UserHistory"
            all_results.extend(history_results)
        
        # 2. 高质量问答库（核心）
        qa_results = self.hybrid_retrieve(query, "HighQualityQA_db", k=k)
        for r in qa_results:
            r["meta"]["source"] = "HighQualityQA"
        all_results.extend(qa_results)
        
        # 3. 医学指南库（医学专业知识）
        guide_results = self.hybrid_retrieve(query, "MedicalGuide_db", k=k)
        for r in guide_results:
            r["meta"]["source"] = "MedicalGuide"
        all_results.extend(guide_results)
        
        # 4. 医院流程库（医院通用流程、模板等）
        # 仅在查询包含流程相关关键词时检索
        if any(kw in query for kw in ["流程", "模板", "证明", "病假", "病历", "表单", "SOP", "缴费", "预约", "挂号"]):
            process_results = self.hybrid_retrieve(query, "HospitalProcess_db", k=k)
            for r in process_results:
                r["meta"]["source"] = "HospitalProcess"
            all_results.extend(process_results)
        
        # 5. 临床案例库（可选）
        # case_results = self.hybrid_retrieve(query, "ClinicalCase_db", k=k)
        # for r in case_results:
        #     r["meta"]["source"] = "ClinicalCase"
        # all_results.extend(case_results)
        
        # 去重并格式化
        return self._format_results(all_results, k * 2)
    
    def _format_results(
        self,
        results: List[Dict[str, Any]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """格式化并去重结果"""
        seen = set()
        formatted = []
        
        for r in results:
            # 去重键
            key = r.get("text", "")[:50]
            if key in seen:
                continue
            seen.add(key)
            
            # 统一格式
            formatted.append({
                "doc_id": r["meta"].get("source", "unknown"),
                "chunk_id": r["meta"].get("chunk_id", "0"),
                "score": r.get("rrf_score", r.get("score", 0)),
                "text": r["text"],
                "meta": {
                    **r["meta"],
                    "retrieval_method": r.get("fusion_method", r.get("source", "unknown"))
                }
            })
        
        # 按分数排序
        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted[:max_results]


__all__ = ["HybridRetriever"]
