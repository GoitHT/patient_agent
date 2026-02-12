"""Adaptive RAG 检索器 - 完全替换原有 RAG 系统
整合 SPLLM-RAG1 的多向量库检索、Adaptive RAG 流程
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import logging

# 强制使用离线模式（在导入 HuggingFace 库之前设置）
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 禁用不必要的警告
logging.getLogger("chromadb").setLevel(logging.ERROR)


class AdaptiveRAGRetriever:
    """Adaptive RAG 检索器 - 接口适配器
    
    本类实现与 ChromaRetriever 相同的接口，但底层使用：
    1. 真实语义嵌入（text2vec-base-chinese）
    2. 多向量库检索（医学指南、临床案例、高质量问答、用户历史）
    3. 余弦相似度匹配
    
    特性：
        - 兼容原有 retrieve() 接口
        - 支持患者历史记忆检索
        - 支持高质量问答参考
        - 支持医学指南和临床案例检索
    """
    
    def __init__(
        self,
        *,
        spllm_root: Path | str,
        cache_folder: Path | str | None = None,
        cosine_threshold: float = 0.3,
        embed_model: str = "BAAI/bge-large-zh-v1.5",
    ):
        """
        Args:
            spllm_root: SPLLM-RAG1 项目根目录（包含 chroma/ 文件夹）
            cache_folder: 模型缓存目录（默认为 spllm_root/model_cache）
            cosine_threshold: 余弦距离阈值（0-1，越小越严格）
            embed_model: 嵌入模型名称
        """
        self.spllm_root = Path(spllm_root).resolve()
        self.cache_folder = Path(cache_folder) if cache_folder else self.spllm_root / "model_cache"
        self.cosine_threshold = cosine_threshold
        self.embed_model = embed_model
        
        # 设置缓存路径
        os.environ['HF_HOME'] = str(self.cache_folder)
        
        # 延迟导入（避免启动时加载模型）
        self._embeddings = None
        self._dbs = {}
        
        # 日志
        self._logger = logging.getLogger("hospital_agent.adaptive_rag")
        self._logger.info(f"📦 AdaptiveRAG 初始化: spllm_root={self.spllm_root}")
    
    def _init_embeddings(self):
        """延迟初始化嵌入模型（首次调用 retrieve 时触发）"""
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
            
            # 测试嵌入
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
        """获取或加载向量库（带缓存）"""
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
    
    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = 4,
    ) -> list[dict[str, Any]]:
        """检索接口（兼容 ChromaRetriever）
        
        Args:
            query: 查询文本
            filters: 过滤条件（可选，包含 dept/type/patient_id/scenario/db_name）
            k: 返回结果数量
            
        Returns:
            统一格式的检索结果: [{doc_id, chunk_id, score, text, meta}, ...]
        """
        filters = filters or {}
        patient_id = filters.get("patient_id")
        dept = filters.get("dept")
        scenario = filters.get("scenario")
        db_name = filters.get("db_name")  # 如果指定了db_name，只查询该数据库
        
        results = []
        
        # 【优先策略】如果指定了 db_name，强制只查询该单一数据库
        if db_name:
            db_method_map = {
                "HospitalProcess_db": self._retrieve_hospital_process,
                "MedicalGuide_db": self._retrieve_guide,
                "ClinicalCase_db": self._retrieve_case,
                "HighQualityQA_db": self._retrieve_high_quality_qa,
                "UserHistory_db": lambda q, k: self._retrieve_history(q, patient_id, k) if patient_id else [],
            }
            method = db_method_map.get(db_name)
            if method:
                results = method(query, k=k)
            else:
                self._logger.warning(f"⚠️  未知的数据库名称: {db_name}")
            return results[:k]  # 强制返回，不走后续逻辑
        
        # 根据场景选择检索策略
        if scenario == "patient_history":
            # 专注于患者历史（C5/C8/C14）
            if patient_id:
                history_results = self._retrieve_history(query, patient_id, k=k)
                results.extend(history_results)
            guide_results = self._retrieve_guide(query, k=k//2)
            results.extend(guide_results)
        
        elif scenario == "clinical_case":
            # 专注于临床案例（C11/C12）- 只查询临床案例库
            case_results = self._retrieve_case(query, k=k)
            results.extend(case_results)
        
        elif scenario == "quality_qa":
            # 专注于高质量问答（S4）- 只查询高质量问答库
            qa_results = self._retrieve_high_quality_qa(query, k=k)
            results.extend(qa_results)
        
        elif scenario == "hospital_process":
            # 专注于规则流程库（C5/C8/C14/C15）- 只查询流程规则库
            process_results = self._retrieve_hospital_process(query, k=k)
            results.extend(process_results)
        
        else:
            # 默认策略：均衡检索所有库
            # 1. 患者历史记忆（如果有 patient_id）
            if patient_id:
                history_results = self._retrieve_history(query, patient_id, k=2)
                results.extend(history_results)
            
            # 2. 高质量问答库（核心）
            qa_results = self._retrieve_high_quality_qa(query, k=k)
            results.extend(qa_results)
            
            # 3. 医学指南库（补充专业知识）
            guide_results = self._retrieve_guide(query, k=k)
            results.extend(guide_results)
            
            # 4. 临床案例库（已启用）
            case_results = self._retrieve_case(query, k=k//2)
            results.extend(case_results)
            results.extend(case_results)
        
        # 去重并按分数排序
        unique_results = self._deduplicate_and_sort(results)
        
        # 限制返回数量
        return unique_results[:k * 2]  # 返回最多 2k 个结果
    
    def _retrieve_history(
        self,
        query: str,
        patient_id: str,
        k: int = 2
    ) -> list[dict[str, Any]]:
        """检索患者历史记忆"""
        db = self._get_db("UserHistory_db")
        if not db:
            return []
        
        try:
            # similarity_search_with_score 返回 (doc, distance)
            docs_and_distances = db.similarity_search_with_score(
                query,
                k=k,
                filter={"patient_id": patient_id}
            )
            
            results = []
            for doc, distance in docs_and_distances:
                if distance < self.cosine_threshold:
                    similarity = max(0, 1 - distance)
                    results.append({
                        "doc_id": f"history_{patient_id}",
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "score": float(similarity),
                        "text": doc.page_content,
                        "meta": {
                            "source": "UserHistory",
                            "patient_id": patient_id,
                            **doc.metadata
                        }
                    })
            
            if results:
                self._logger.debug(f"📜 历史记忆检索: 找到 {len(results)} 条")
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  历史记忆检索失败: {e}")
            return []
    
    def retrieve_patient_test_history(
        self,
        patient_id: str,
        test_keywords: list[str],
        k: int = 5
    ) -> list[dict[str, Any]]:
        """检索患者历史检查记录（用于避免重复开单）
        
        Args:
            patient_id: 患者ID
            test_keywords: 检查关键词列表（如 ["CT", "MRI", "血常规"]）
            k: 返回结果数量
            
        Returns:
            患者历史检查记录列表
        """
        if not patient_id or not test_keywords:
            return []
        
        # 构建查询语句
        query = f"检查 检验 开单 {' '.join(test_keywords)}"
        
        db = self._get_db("UserHistory_db")
        if not db:
            return []
        
        try:
            docs_and_distances = db.similarity_search_with_score(
                query,
                k=k,
                filter={"patient_id": patient_id}
            )
            
            results = []
            for doc, distance in docs_and_distances:
                # 历史检查记录阈值放宽
                if distance < self.cosine_threshold * 1.5:
                    similarity = max(0, 1 - distance)
                    results.append({
                        "doc_id": f"test_history_{patient_id}",
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "score": float(similarity),
                        "text": doc.page_content,
                        "meta": {
                            "source": "UserHistory",
                            "patient_id": patient_id,
                            "keywords": test_keywords,
                            **doc.metadata
                        }
                    })
            
            if results:
                self._logger.info(f"🔍 历史检查记录: 找到 {len(results)} 条相关记录")
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  历史检查记录检索失败: {e}")
            return []
    
    def _retrieve_high_quality_qa(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """检索高质量问答（核心知识库）"""
        db = self._get_db("HighQualityQA_db")
        if not db:
            return []
        
        try:
            docs_and_distances = db.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, distance in docs_and_distances:
                if distance < self.cosine_threshold:
                    similarity = max(0, 1 - distance)
                    question = doc.metadata.get("question", "")
                    answer = doc.metadata.get("answer", "")
                    
                    # 格式化为问答对
                    text = f"【历史问答】\n问：{question}\n答：{answer[:300]}..."
                    
                    results.append({
                        "doc_id": "high_quality_qa",
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "score": float(similarity),
                        "text": text,
                        "meta": {
                            "source": "HighQualityQA",
                            "question": question,
                            "answer": answer,
                            "distance": distance,
                        }
                    })
            
            if results:
                self._logger.debug(f"💎 高质量问答: 找到 {len(results)} 条")
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  高质量问答检索失败: {e}")
            return []
    
    def _retrieve_guide(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """检索医学指南"""
        db = self._get_db("MedicalGuide_db")
        if not db:
            return []
        
        try:
            docs_and_distances = db.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, distance in docs_and_distances:
                if distance < self.cosine_threshold * 1.5:  # 指南库阈值放宽一些
                    similarity = max(0, 1 - distance)
                    results.append({
                        "doc_id": "medical_guide",
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "score": float(similarity),
                        "text": doc.page_content,
                        "meta": {
                            "source": "MedicalGuide",
                            **doc.metadata
                        }
                    })
            
            if results:
                self._logger.debug(f"📚 医学指南: 找到 {len(results)} 条")
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  医学指南检索失败: {e}")
            return []
    
    def _retrieve_case(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """检索临床案例"""
        db = self._get_db("ClinicalCase_db")
        if not db:
            return []
        
        try:
            docs_and_distances = db.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, distance in docs_and_distances:
                if distance < self.cosine_threshold * 1.2:
                    similarity = max(0, 1 - distance)
                    results.append({
                        "doc_id": "clinical_case",
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "score": float(similarity),
                        "text": doc.page_content,
                        "meta": {
                            "source": "ClinicalCase",
                            **doc.metadata
                        }
                    })
            
            if results:
                self._logger.debug(f"🏥 临床案例: 找到 {len(results)} 条")
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  临床案例检索失败: {e}")
            return []
    
    def _retrieve_hospital_process(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """检索医院规则流程（SOP、文书模板、宣教材料等）
        
        用于：
        - C5: 通用就诊流程SOP
        - C8: 检查/检验前准备事项
        - C14: 门诊病历/诊断证明/病假条模板
        - C15: 疾病科普材料、健康宣教和随访计划模板
        """
        db = self._get_db("HospitalProcess_db")
        if not db:
            return []
        
        try:
            docs_and_distances = db.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, distance in docs_and_distances:
                # 规则流程库要求精准匹配
                if distance < self.cosine_threshold:
                    similarity = max(0, 1 - distance)
                    results.append({
                        "doc_id": "hospital_process",
                        "chunk_id": doc.metadata.get("chunk_id", "0"),
                        "score": float(similarity),
                        "text": doc.page_content,
                        "meta": {
                            "source": "HospitalProcess",
                            **doc.metadata
                        }
                    })
            
            if results:
                self._logger.debug(f"📋 规则流程: 找到 {len(results)} 条")
            return results
        except Exception as e:
            self._logger.warning(f"⚠️  规则流程检索失败: {e}")
            return []
    
    def _deduplicate_and_sort(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """去重并按分数排序"""
        seen = set()
        unique = []
        
        for r in results:
            # 使用文本前50个字符作为去重键
            key = (r.get("doc_id"), r.get("text", "")[:50])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        # 按分数降序排序
        unique.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique


__all__ = ["AdaptiveRAGRetriever"]
