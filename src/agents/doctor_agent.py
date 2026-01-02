"""医生智能体：基于RAG知识库进行问诊、开单、诊断"""
from __future__ import annotations

import json
from typing import Any

from rag import ChromaRetriever
from services.llm_client import LLMClient


class DoctorAgent:
    """医生智能体：初始不知道任何病例信息，通过问诊获取"""
    
    def __init__(
        self,
        dept: str,
        retriever: ChromaRetriever,
        llm: LLMClient | None = None,
        max_questions: int = 10
    ):
        """
        Args:
            dept: 科室 (gastro/neuro)
            retriever: RAG检索器
            llm: 语言模型客户端
            max_questions: 最多问题数
        """
        self.dept = dept
        self.retriever = retriever
        self.llm = llm
        self.max_questions = max_questions
        self.collected_info: dict[str, Any] = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "history": {},
            "exam_findings": {}
        }
        self.questions_asked: list[str] = []
        self.patient_answers: list[str] = []
    
    def reset(self) -> None:
        """重置医生状态（用于处理新患者）"""
        self.collected_info = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "history": {},
            "exam_findings": {}
        }
        self.questions_asked = []
        self.patient_answers = []
    
    def generate_one_question(
        self,
        chief_complaint: str,
        context: str = "",
        rag_chunks: list[dict[str, Any]] | None = None
    ) -> str:
        """逐步生成单个问题（一问一答模式）
        
        Args:
            chief_complaint: 主诉
            context: 当前问诊上下文（如"消化内科专科问诊"）
            rag_chunks: RAG检索到的知识片段
            
        Returns:
            单个问题字符串，如果不需要继续问则返回空字符串
        """
        # 检查是否已问足够问题
        if len(self.questions_asked) >= self.max_questions:
            return ""
        
        if self.llm is None:
            # 规则模式：依次返回规则问题
            all_questions = self._rule_based_questions(chief_complaint)
            remaining = [q for q in all_questions if q not in self.questions_asked]
            return remaining[0] if remaining else ""
        
        # 构建RAG上下文
        kb_context = ""
        if rag_chunks:
            kb_context = self._format_chunks(rag_chunks)
        elif context:
            # 如果没有提供chunks，尝试检索
            chunks = self.retriever.retrieve(
                f"{self.dept} {context} 问诊要点",
                filters={"dept": self.dept},
                k=3
            )
            kb_context = self._format_chunks(chunks)
        
        # 构建已问问题列表
        asked_summary = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(self.questions_asked)])
        answers_summary = "\n".join([f"A{i+1}: {a[:50]}..." for i, a in enumerate(self.patient_answers)])
        
        system_prompt = f"""你是一名经验丰富的{self._dept_name()}医生，正在进行{context}。

【主诉】
{chief_complaint}

【已收集信息】
{json.dumps(self.collected_info, ensure_ascii=False, indent=2)}

【已问问题】
{asked_summary if asked_summary else "（尚未提问）"}

【患者回答】
{answers_summary if answers_summary else "（尚无回答）"}

【参考知识】
{kb_context}

【任务】
基于以上信息，判断是否需要继续提问：
1. 如果已经收集了足够的信息（症状、病史、持续时间、严重程度等核心信息）可以进行初步评估，输出空字符串表示不需要继续提问
2. 如果还缺少关键信息，生成下一个最重要的问题来帮助明确诊断

要求：
- 问题要简洁直接，患者容易理解
- 避免重复已问过的问题
- 针对{context}的重点进行询问
- 有助于鉴别诊断或评估病情严重程度
- 优先询问影响诊断的关键信息

【重要】如果核心信息已经充足，请返回空字符串 "" 表示无需继续提问。
"""
        
        user_prompt = '请生成一个问题，输出JSON格式：{"question": "问题内容"} 或 {"question": ""} 表示无需继续提问'
        
        try:
            obj, _, _ = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {"question": ""},
                temperature=0.3
            )
            return str(obj.get("question", ""))
        except Exception:
            # 发生异常时返回规则问题
            all_questions = self._rule_based_questions(chief_complaint)
            remaining = [q for q in all_questions if q not in self.questions_asked]
            return remaining[0] if remaining else ""
    
    def generate_question_based_on_tests(
        self,
        test_results: list[dict[str, Any]],
        chief_complaint: str,
        collected_info: dict[str, Any]
    ) -> str:
        """基于检查结果生成进一步问诊问题
        
        Args:
            test_results: 检查结果列表
            chief_complaint: 主诉
            collected_info: 已收集的信息
            
        Returns:
            单个问题字符串，如果不需要继续问则返回空字符串
        
        注意：问题数量限制由调用方（如C11节点）通过全局计数器控制
        """
        
        if self.llm is None:
            # 规则模式：基于异常结果生成简单问题
            abnormal = [r for r in test_results if r.get("abnormal")]
            if not abnormal:
                return ""
            
            first_abnormal = abnormal[0]
            test_name = first_abnormal.get("test", "")
            
            if "贫血" in str(test_name) or "血红蛋白" in str(test_name):
                return "最近有没有感觉特别疲劳或者头晕？"
            elif "炎症" in str(test_name) or "白细胞" in str(test_name):
                return "有没有发热或者身体哪里特别不舒服？"
            else:
                return "看到检查结果有些异常，您最近身体还有其他不适吗？"
        
        # LLM模式：智能生成问题
        # 提取异常结果
        abnormal_results = [r for r in test_results if r.get("abnormal")]
        normal_results = [r for r in test_results if not r.get("abnormal")]
        
        # 如果没有异常，可能不需要继续问
        if not abnormal_results:
            return ""
        
        # 构建已问问题列表
        asked_summary = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(self.questions_asked)])
        
        system_prompt = f"""你是一名经验丰富的{self._dept_name()}医生，刚刚收到患者的检查报告。

【主诉】
{chief_complaint}

【已收集的病史信息】
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

【检查结果摘要】
异常结果 ({len(abnormal_results)}项):
{json.dumps(abnormal_results, ensure_ascii=False, indent=2)}

正常结果 ({len(normal_results)}项):
{json.dumps([r.get('test') for r in normal_results], ensure_ascii=False)}

【已问过的问题】
{asked_summary if asked_summary else "（之前尚未基于检查结果提问）"}

【任务】
基于检查结果（特别是异常结果），生成一个重要的补充问题，帮助：
1. 了解异常结果相关的症状或病史
2. 评估病情的严重程度和进展
3. 排除可能的并发症
4. 明确诊断

要求：
- 问题要简洁、直接，患者容易理解
- 针对最重要的异常结果进行询问
- 避免重复已问过的问题
- 如果异常轻微且不需要进一步询问，可以返回空字符串
"""
        
        user_prompt = '请生成一个问题，输出JSON格式：{"question": "问题内容"} 或 {"question": ""} 表示无需继续提问'
        
        try:
            obj, _, _ = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {"question": ""},
                temperature=0.3
            )
            return str(obj.get("question", ""))
        except Exception:
            # 发生异常时返回规则问题
            abnormal = [r for r in test_results if r.get("abnormal")]
            if abnormal:
                return "看到检查结果有些异常，您最近身体还有其他不适吗？"
            return ""
    
    def _rule_based_questions(self, chief_complaint: str) -> list[str]:
        """基于规则的问题生成（当没有LLM时）"""
        base_questions = [
            "这个症状有多久了？",
            "有什么能让症状加重或缓解的因素吗？",
            "还有其他不舒服的地方吗？"
        ]
        
        if self.dept == "gastro":
            specific = [
                "有没有恶心、呕吐？",
                "大便颜色正常吗？有没有黑便？",
                "吃饭后症状会加重吗？"
            ]
        else:  # neuro
            specific = [
                "头痛的话，是哪个部位痛？",
                "有没有视物模糊或者复视？",
                "四肢活动有没有受影响？"
            ]
        
        return (base_questions + specific)[:3]
    
    def process_patient_answer(self, question: str, answer: str) -> None:
        """处理患者的回答，更新收集的信息"""
        self.questions_asked.append(question)
        self.patient_answers.append(answer)
        
        # 如果有LLM，使用LLM提取结构化信息
        if self.llm is not None:
            try:
                system_prompt = f"""你是{self._dept_name()}医生，正在分析患者回答并提取关键信息。

【问题】
{question}

【患者回答】
{answer}

【任务】
从回答中提取结构化信息，如：
- duration: 持续时间
- severity: 严重程度
- frequency: 频率
- aggravating_factors: 加重因素
- relieving_factors: 缓解因素
- associated_symptoms: 伴随症状
- medical_history: 相关病史
- other_info: 其他信息

只提取回答中明确包含的信息，不要推测。
"""
                
                user_prompt = '输出JSON格式：{"extracted_info": {"duration": "...", "severity": "...", ...}}'
                
                obj, _, _ = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=lambda: {"extracted_info": {}},
                    temperature=0.1
                )
                
                extracted = obj.get("extracted_info", {})
                # 合并到history
                for key, value in extracted.items():
                    if value and value != "N/A" and value != "不详":
                        if key in ["associated_symptoms", "aggravating_factors", "relieving_factors"]:
                            # 列表类型，追加
                            if key not in self.collected_info["history"]:
                                self.collected_info["history"][key] = []
                            if isinstance(value, list):
                                self.collected_info["history"][key].extend(value)
                            else:
                                self.collected_info["history"][key].append(value)
                        else:
                            # 单值类型，覆盖
                            self.collected_info["history"][key] = value
            except Exception:
                # LLM失败，使用简单规则
                self._simple_extract(question, answer)
        else:
            # 无LLM，使用简单规则
            self._simple_extract(question, answer)
    
    def _simple_extract(self, question: str, answer: str) -> None:
        """简单的规则提取（备用）"""
        if "久" in question or "时间" in question:
            self.collected_info["history"]["duration"] = answer
        elif "加重" in question or "缓解" in question:
            if "aggravating_factors" not in self.collected_info["history"]:
                self.collected_info["history"]["aggravating_factors"] = []
            self.collected_info["history"]["aggravating_factors"].append(answer)
        elif "还有" in question or "其他" in question:
            if answer and "没" not in answer and "不" not in answer:
                self.collected_info["symptoms"].append(answer)
    
    def decide_tests(self) -> list[dict[str, Any]]:
        """根据收集的信息决定需要做哪些检查"""
        if self.llm is None:
            return self._rule_based_tests()
        
        # 检索检查指南
        chunks = self.retriever.retrieve(
            f"{self.dept} 检查 适应症 {self.collected_info.get('chief_complaint', '')}",
            filters={"dept": self.dept, "type": "plan"},
            k=4
        )
        
        kb_context = self._format_chunks(chunks)
        
        system_prompt = f"""你是{self._dept_name()}医生，需要根据问诊结果决定检查项目。

【收集的信息】
{json.dumps(self.collected_info, ensure_ascii=False, indent=2)}

【检查指南】
{kb_context}

【任务】
根据上述信息，选择必要的检查项目。每项检查需要说明理由。
"""
        
        user_prompt = '''输出JSON格式：
{
  "ordered_tests": [
    {"name": "检查名称", "type": "lab/imaging/endoscopy", "reason": "开单理由", "priority": "routine/urgent"}
  ]
}'''
        
        try:
            obj, _, _ = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {"ordered_tests": self._rule_based_tests()},
                temperature=0.2
            )
            return obj.get("ordered_tests", [])
        except Exception:
            return self._rule_based_tests()
    
    def _rule_based_tests(self) -> list[dict[str, Any]]:
        """基于规则的检查开单"""
        tests = []
        
        if self.dept == "gastro":
            tests = [
                {
                    "dept": "gastro",
                    "type": "endoscopy",
                    "name": "胃镜",
                    "reason": "上消化道症状评估",
                    "priority": "routine",
                    "need_prep": True,
                    "need_schedule": True
                },
                {
                    "dept": "gastro",
                    "type": "lab",
                    "name": "血常规",
                    "reason": "评估炎症/贫血",
                    "priority": "routine",
                    "need_prep": False,
                    "need_schedule": False
                }
            ]
        else:  # neuro
            tests = [
                {
                    "dept": "neuro",
                    "type": "imaging",
                    "name": "头颅CT",
                    "reason": "排除颅内病变",
                    "priority": "urgent",
                    "need_prep": False,
                    "need_schedule": False
                }
            ]
        
        return tests
    
    def make_diagnosis(self, test_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """综合分析后做出诊断"""
        if self.llm is None:
            return self._rule_based_diagnosis()
        
        # 检索诊疗指南
        chunks = self.retriever.retrieve(
            f"{self.dept} 诊断 治疗方案 {self.collected_info.get('chief_complaint', '')}",
            filters={"dept": self.dept, "type": "plan"},
            k=4
        )
        
        kb_context = self._format_chunks(chunks)
        
        test_summary = json.dumps(test_results, ensure_ascii=False) if test_results else "尚无检查结果"
        
        system_prompt = f"""你是{self._dept_name()}医生，需要做出诊断并制定治疗方案。

【问诊信息】
{json.dumps(self.collected_info, ensure_ascii=False, indent=2)}

【检查结果】
{test_summary}

【诊疗指南】
{kb_context}

【任务】
综合分析并给出诊断和治疗建议。
"""
        
        user_prompt = '''输出JSON格式：
{
  "diagnosis": {"name": "诊断名称", "confidence": "high/medium/low", "differential": ["鉴别诊断1", "鉴别诊断2"]},
  "treatment_plan": {"medications": ["用药1", "用药2"], "lifestyle": ["生活建议1"], "followup": "随访计划"}
}'''
        
        try:
            obj, _, _ = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: self._rule_based_diagnosis(),
                temperature=0.2
            )
            return obj
        except Exception:
            return self._rule_based_diagnosis()
    
    def _rule_based_diagnosis(self) -> dict[str, Any]:
        """基于规则的诊断"""
        if self.dept == "gastro":
            diagnosis_name = "消化不良"
        else:
            diagnosis_name = "神经系统症状待查"
        
        return {
            "diagnosis": {
                "name": diagnosis_name,
                "confidence": "medium",
                "differential": ["需进一步检查明确"]
            },
            "treatment_plan": {
                "medications": ["对症治疗"],
                "lifestyle": ["注意休息", "清淡饮食"],
                "followup": "1-2周复诊"
            }
        }
    
    def _dept_name(self) -> str:
        """根据科室代码返回科室中文名"""
        dept_names = {
            "gastro": "消化内科",
            "neuro": "神经内科",
            "internal_medicine": "内科",
            "surgery": "外科",
            "orthopedics": "骨科",
            "urology": "泌尿外科",
            "obstetrics_gynecology": "妇产科",
            "pediatrics": "儿科",
            "neurology": "神经医学",
            "oncology": "肿瘤科",
            "infectious_disease": "感染性疾病科",
            "dermatology_std": "皮肤性病科",
            "ent_ophthalmology_stomatology": "眼耳鼻喉口腔科",
            "psychiatry": "精神心理科",
            "emergency": "急诊医学科",
            "rehabilitation_pain": "康复疼痛科",
            "traditional_chinese_medicine": "中医科",
        }
        return dept_names.get(self.dept, "通用科室")
    
    def _format_chunks(self, chunks: list[dict[str, Any]]) -> str:
        """格式化RAG检索结果"""
        lines = []
        for i, c in enumerate(chunks[:4], 1):
            text = str(c.get("text", "")).strip()
            # 保留完整信息，只在过长时截断（500字符）
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(f"{i}. [{c.get('doc_id')}] {text}")
        return "\n".join(lines)
    
    def get_interaction_summary(self) -> dict[str, Any]:
        """获取医生问诊摘要"""
        return {
            "questions_count": len(self.questions_asked),
            "qa_pairs": [
                {"question": q, "answer": a}
                for q, a in zip(self.questions_asked, self.patient_answers)
            ],
            "collected_info": self.collected_info
        }
