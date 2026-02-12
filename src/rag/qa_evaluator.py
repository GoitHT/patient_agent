"""对话质量评估模块 - 评估医患问答质量并存储高质量对话

本模块实现以下功能：
1. 患者回答评估：相关性、忠实性、鲁棒性
2. 医生提问评估：具体性、针对性、专业性
3. 高质量对话存储到向量库
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


@dataclass
class PatientAnswerMetrics:
    """患者回答评估指标"""
    relevance: float  # α: 相关性 [0,1]
    faithfulness: float  # β: 忠实性 [0,1]
    robustness: float  # γ: 鲁棒性 [0,1]
    
    @property
    def ability(self) -> float:
        """能力：综合指标"""
        return (self.relevance + self.faithfulness + self.robustness) / 3.0
    
    def to_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "faithfulness": self.faithfulness,
            "robustness": self.robustness,
            "ability": self.ability
        }


@dataclass
class DoctorQuestionMetrics:
    """医生提问评估指标"""
    specificity: float  # δ: 具体性 [0,1]
    targetedness: float  # ε: 针对性 [0,1]
    professionalism: float  # ζ: 专业性 [0,1]
    
    @property
    def quality(self) -> float:
        """质量：综合指标"""
        return (self.specificity + self.targetedness + self.professionalism) / 3.0
    
    def to_dict(self) -> dict:
        return {
            "specificity": self.specificity,
            "targetedness": self.targetedness,
            "professionalism": self.professionalism,
            "quality": self.quality
        }


@dataclass
class DialogueQualityScore:
    """对话质量综合评分"""
    question: str
    answer: str
    patient_metrics: PatientAnswerMetrics
    doctor_metrics: DoctorQuestionMetrics
    
    @property
    def overall_score(self) -> float:
        """综合得分：医生质量 + 患者能力 / 2"""
        return (self.doctor_metrics.quality + self.patient_metrics.ability) / 2.0
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """判断是否为高质量对话"""
        return self.overall_score >= threshold
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "patient_metrics": self.patient_metrics.to_dict(),
            "doctor_metrics": self.doctor_metrics.to_dict(),
            "overall_score": self.overall_score
        }


class DialogueQualityEvaluator:
    """对话质量评估器"""
    
    def __init__(
        self,
        *,
        llm = None,
        spllm_root: Path | str | None = None,
        embed_model: str = "BAAI/bge-large-zh-v1.5",
        high_quality_threshold: float = 0.7,
    ):
        """
        Args:
            llm: LLM客户端（用于评估）
            spllm_root: SPLLM-RAG1项目根目录
            embed_model: 嵌入模型名称
            high_quality_threshold: 高质量对话阈值
        """
        self.llm = llm
        self.spllm_root = Path(spllm_root).resolve() if spllm_root else None
        self.embed_model = embed_model
        self.high_quality_threshold = high_quality_threshold
        
        self._embeddings = None
        self._logger = logging.getLogger("hospital_agent.qa_evaluator")
        
    def _init_embeddings(self):
        """延迟初始化嵌入模型"""
        if self._embeddings is not None:
            return
        
        if not self.spllm_root:
            self._logger.warning("⚠️  未配置spllm_root，无法初始化嵌入模型")
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
            
            cache_folder = self.spllm_root / "model_cache"
            os.environ['HF_HOME'] = str(cache_folder)
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                cache_folder=str(cache_folder)
            )
            
            # 恢复日志级别
            sentence_transformers_logger.setLevel(old_st_level)
            transformers_logger.setLevel(old_tf_level)
            
            self._logger.debug("✅ 嵌入模型加载成功")
        except Exception as e:
            # 恢复日志级别（即使出错）
            try:
                sentence_transformers_logger.setLevel(old_st_level)
                transformers_logger.setLevel(old_tf_level)
            except:
                pass
            self._logger.error(f"❌ 嵌入模型初始化失败: {e}")
    
    def evaluate_patient_answer(
        self,
        question: str,
        answer: str,
        patient_info: dict[str, Any] | None = None
    ) -> PatientAnswerMetrics:
        """评估患者回答质量
        
        Args:
            question: 医生提问
            answer: 患者回答
            patient_info: 患者病历信息（可选）
            
        Returns:
            患者回答评估指标
        """
        # 1. 相关性评估：基于语义相似度
        relevance = self._evaluate_relevance(question, answer)
        
        # 2. 忠实性评估：基于病历信息一致性
        faithfulness = self._evaluate_faithfulness(answer, patient_info)
        
        # 3. 鲁棒性评估：检测信息泄露
        robustness = self._evaluate_robustness(answer)
        
        return PatientAnswerMetrics(
            relevance=relevance,
            faithfulness=faithfulness,
            robustness=robustness
        )
    
    def evaluate_doctor_question(
        self,
        question: str,
        context: dict[str, Any] | None = None
    ) -> DoctorQuestionMetrics:
        """评估医生提问质量
        
        Args:
            question: 医生提问
            context: 问诊上下文（科室、已收集信息等）
            
        Returns:
            医生提问评估指标
        """
        # 1. 具体性评估：问题明确性
        specificity = self._evaluate_specificity(question)
        
        # 2. 针对性评估：是否有助于诊断
        targetedness = self._evaluate_targetedness(question, context)
        
        # 3. 专业性评估：医学术语和临床规范
        professionalism = self._evaluate_professionalism(question)
        
        return DoctorQuestionMetrics(
            specificity=specificity,
            targetedness=targetedness,
            professionalism=professionalism
        )
    
    def evaluate_dialogue(
        self,
        question: str,
        answer: str,
        patient_info: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None
    ) -> DialogueQualityScore:
        """评估完整对话质量
        
        Args:
            question: 医生提问
            answer: 患者回答
            patient_info: 患者病历信息
            context: 问诊上下文
            
        Returns:
            对话质量综合评分
        """
        patient_metrics = self.evaluate_patient_answer(question, answer, patient_info)
        doctor_metrics = self.evaluate_doctor_question(question, context)
        
        return DialogueQualityScore(
            question=question,
            answer=answer,
            patient_metrics=patient_metrics,
            doctor_metrics=doctor_metrics
        )
    
    # ========== 具体评估方法 ==========
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """评估相关性：基于语义相似度（余弦距离）"""
        if not question or not answer:
            return 0.0
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        if not self._embeddings:
            # 如果没有嵌入模型，使用简单的关键词匹配
            return self._simple_relevance_check(question, answer)
        
        try:
            # 计算问答语义相似度
            q_embedding = self._embeddings.embed_query(question)
            a_embedding = self._embeddings.embed_query(answer)
            
            # 余弦相似度（已归一化向量，直接点积）
            import numpy as np
            similarity = float(np.dot(q_embedding, a_embedding))
            
            # 将[-1,1]映射到[0,1]
            relevance = (similarity + 1) / 2
            
            # 检查回答是否完整（长度惩罚过短回答）
            if len(answer) < 5:
                relevance *= 0.5
            
            return max(0.0, min(1.0, relevance))
        except Exception as e:
            self._logger.warning(f"⚠️  相关性评估失败: {e}")
            return self._simple_relevance_check(question, answer)
    
    def _simple_relevance_check(self, question: str, answer: str) -> float:
        """简单的相关性检查（fallback）"""
        # 提取问题关键词
        q_keywords = set(question.lower().split())
        a_keywords = set(answer.lower().split())
        
        if not q_keywords:
            return 0.5
        
        # 计算关键词重叠率
        overlap = len(q_keywords & a_keywords) / len(q_keywords)
        
        # 长度惩罚
        if len(answer) < 5:
            overlap *= 0.5
        
        return max(0.0, min(1.0, overlap))
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        patient_info: dict[str, Any] | None
    ) -> float:
        """评估忠实性：回答是否符合病历信息"""
        if not patient_info:
            # 无病历信息，默认中等分数
            return 0.7
        
        # 使用LLM评估忠实性
        if self.llm:
            return self._llm_evaluate_faithfulness(answer, patient_info)
        
        # 简单规则评估
        # 1. 检查是否包含病历中的关键信息
        # 2. 检查是否有明显矛盾
        
        # 提取病历关键信息
        chief_complaint = patient_info.get("chief_complaint", "")
        history = patient_info.get("history", {})
        
        # 粗略检查：回答中提到的症状是否在病历中
        # 这里可以扩展更复杂的逻辑
        
        return 0.75  # 默认较高分数（假设SP遵循病历）
    
    def _llm_evaluate_faithfulness(
        self,
        answer: str,
        patient_info: dict[str, Any]
    ) -> float:
        """使用LLM评估忠实性"""
        try:
            prompt = f"""请评估患者的回答是否忠实于病历信息。

【病历信息】
{json.dumps(patient_info, ensure_ascii=False, indent=2)}

【患者回答】
{answer}

【评估标准】
1. 回答是否基于病历中的信息？
2. 回答是否与病历信息矛盾？
3. 回答是否符合标准化病人要求？

请给出0-1之间的分数（0=完全不忠实，1=完全忠实），仅输出数字。
"""
            # 调用LLM
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
        except Exception as e:
            self._logger.warning(f"⚠️  LLM忠实性评估失败: {e}")
            return 0.75
    
    def _evaluate_robustness(self, answer: str) -> float:
        """评估鲁棒性：检测信息泄露"""
        # 关键信息泄露检测
        leak_keywords = [
            # 疾病名称（不应直接说出）
            "诊断", "确诊", "癫痫", "脑梗", "肿瘤", "炎症",
            # 详细病历描述
            "病历", "检查报告", "CT显示", "MRI显示",
            # 医学术语（患者一般不会说）
            "症状学", "体征", "鉴别诊断", "病理",
        ]
        
        # 统计泄露关键词数量
        leak_count = sum(1 for keyword in leak_keywords if keyword in answer)
        
        # 检查是否一次回答提供过多细节
        answer_length = len(answer)
        is_too_detailed = answer_length > 200
        
        # 计算鲁棒性分数
        leak_penalty = leak_count * 0.1
        detail_penalty = 0.1 if is_too_detailed else 0.0
        
        robustness = 1.0 - leak_penalty - detail_penalty
        
        return max(0.0, min(1.0, robustness))
    
    def _evaluate_specificity(self, question: str) -> float:
        """评估具体性：问题明确性"""
        # 检查问题是否具体
        
        # 1. 开放式问题得分较低
        open_keywords = ["怎么样", "如何", "情况", "有没有", "是否"]
        
        # 2. 具体的时间、地点、程度描述得分较高
        specific_keywords = [
            "什么时候", "多久", "多长时间", "几次", "多少",
            "哪里", "什么部位", "什么性质", "什么程度",
            "伴随", "诱因", "缓解", "加重"
        ]
        
        open_count = sum(1 for k in open_keywords if k in question)
        specific_count = sum(1 for k in specific_keywords if k in question)
        
        # 具体问题得高分，开放问题得低分
        specificity = 0.5 + (specific_count * 0.1) - (open_count * 0.05)
        
        return max(0.0, min(1.0, specificity))
    
    def _evaluate_targetedness(
        self,
        question: str,
        context: dict[str, Any] | None
    ) -> float:
        """评估针对性：是否有助于诊断"""
        # 检查问题是否与诊断相关
        
        # 核心诊断要素
        diagnostic_keywords = [
            # 症状特征
            "症状", "疼痛", "不适", "感觉", "表现",
            # 时间特征
            "开始", "持续", "频率", "发作",
            # 诱因和缓解
            "诱因", "缓解", "加重", "因素",
            # 伴随症状
            "伴随", "同时", "还有",
            # 既往史
            "以前", "曾经", "病史", "治疗",
            # 系统回顾
            "头痛", "眩晕", "恶心", "呕吐", "发热"
        ]
        
        keyword_count = sum(1 for k in diagnostic_keywords if k in question)
        
        # 针对性分数
        targetedness = min(1.0, 0.4 + keyword_count * 0.1)
        
        return targetedness
    
    def _evaluate_professionalism(self, question: str) -> float:
        """评估专业性：医学术语和临床规范"""
        # 检查医学术语使用
        
        # 专业术语
        professional_terms = [
            # 症状描述术语
            "性质", "部位", "持续时间", "程度", "频率",
            # 医学术语
            "伴随症状", "诱发因素", "缓解因素", "加重因素",
            "既往史", "家族史", "过敏史", "用药史",
            # 专科术语（神经内科）
            "意识", "肢体", "感觉", "运动", "反射",
            "头痛", "眩晕", "肢体麻木", "肢体无力"
        ]
        
        term_count = sum(1 for term in professional_terms if term in question)
        
        # 专业性分数
        professionalism = min(1.0, 0.5 + term_count * 0.08)
        
        return professionalism
    
    def store_high_quality_dialogue(
        self,
        dialogue_score: DialogueQualityScore,
        patient_id: str,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """存储高质量对话到向量库
        
        Args:
            dialogue_score: 对话质量评分
            patient_id: 患者ID
            metadata: 额外元数据
            
        Returns:
            是否存储成功
        """
        if not dialogue_score.is_high_quality(self.high_quality_threshold):
            self._logger.debug(f"⏭️  对话得分 {dialogue_score.overall_score:.3f} 低于阈值，跳过存储")
            return False
        
        if not self.spllm_root:
            self._logger.warning("⚠️  未配置spllm_root，无法存储对话")
            return False
        
        try:
            from langchain_chroma import Chroma
            from langchain.docstore.document import Document
            
            # 初始化嵌入模型
            self._init_embeddings()
            
            if not self._embeddings:
                self._logger.error("❌ 嵌入模型未初始化，无法存储对话")
                return False
            
            # 准备文档内容
            doc_content = f"【问】{dialogue_score.question}\n【答】{dialogue_score.answer}"
            
            # 准备元数据
            doc_metadata = {
                "patient_id": patient_id,
                "question": dialogue_score.question,
                "answer": dialogue_score.answer,
                "overall_score": dialogue_score.overall_score,
                "doctor_quality": dialogue_score.doctor_metrics.quality,
                "patient_ability": dialogue_score.patient_metrics.ability,
                **(metadata or {})
            }
            
            # 存储到HighQualityQA_db
            db_path = self.spllm_root / "chroma" / "HighQualityQA_db"
            db_path.mkdir(parents=True, exist_ok=True)
            
            db = Chroma(
                persist_directory=str(db_path),
                embedding_function=self._embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # 添加文档
            db.add_documents([
                Document(page_content=doc_content, metadata=doc_metadata)
            ])
            
            self._logger.info(
                f"✅ 高质量对话已存储 (得分: {dialogue_score.overall_score:.3f})"
            )
            return True
            
        except Exception as e:
            self._logger.error(f"❌ 存储高质量对话失败: {e}")
            return False


__all__ = [
    "PatientAnswerMetrics",
    "DoctorQuestionMetrics",
    "DialogueQualityScore",
    "DialogueQualityEvaluator",
]
