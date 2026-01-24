"""护士智能体：负责预检分诊，根据主诉分配科室"""
from __future__ import annotations

from services.llm_client import LLMClient


class NurseAgent:
    """护士智能体：根据患者主诉进行分诊"""
    
    # 15个标准科室代码
    VALID_DEPTS = [
        "internal_medicine", "surgery", "orthopedics", "urology",
        "obstetrics_gynecology", "pediatrics", "neurology", "oncology",
        "infectious_disease", "dermatology_std", "ent_ophthalmology_stomatology",
        "psychiatry", "emergency", "rehabilitation_pain", "traditional_chinese_medicine"
    ]
    
    def __init__(self, llm: LLMClient, max_triage_questions: int = 3):
        """
        Args:
            llm: 语言模型客户端（必需，用于智能分诊）
            max_triage_questions: 分诊时最多可以问患者的问题数（默认3个）
        """
        self._llm = llm
        self._max_triage_questions = max_triage_questions
        self._triage_history: list[dict[str, str]] = []
        self._triage_qa: list[dict[str, str]] = []  # 分诊对话记录
    
    def reset(self) -> None:
        """重置分诊历史（用于处理新的就诊流程）"""
        self._triage_history = []
        self._triage_qa = []
    
    def triage(self, patient_description: str) -> str:
        """
        根据患者描述进行分诊到15个标准科室之一
        
        Args:
            patient_description: 患者描述的症状（来自患者智能体）
            
        Returns:
            科室代码（internal_medicine, surgery, orthopedics等）
        """
        # 参数验证
        if not patient_description or not patient_description.strip():
            raise ValueError("患者描述不能为空")
        
        patient_description = patient_description.strip()
        
        # 使用LLM进行智能分诊
        system_prompt = """你是一名经验丰富的分诊护士。

【可选科室】（必须从以下15个科室中选择）
1. internal_medicine（内科）：发热、咳嗽、胸闷、高血压、糖尿病、消化道症状等
2. surgery（外科）：外伤、肿块、阑尾炎、疝气、体表手术等
3. orthopedics（骨科）：骨折、关节疼痛、扭伤、腰腿痛、骨关节疾病等
4. urology（泌尿外科）：泌尿系统结石、血尿、排尿困难、前列腺疾病等
5. obstetrics_gynecology（妇产科）：妇科疾病、孕产检查、月经异常、妇科肿瘤等
6. pediatrics（儿科）：儿童疾病、生长发育问题、小儿感染等
7. neurology（神经医学）：头痛、头晕、肢体无力、癫痫、帕金森、脑血管病等
8. oncology（肿瘤科）：恶性肿瘤诊治、化疗、放疗等
9. infectious_disease（感染性疾病科）：发热待查、传染病、寄生虫病、HIV等
10. dermatology_std（皮肤性病科）：皮疹、瘙痒、性传播疾病等
11. ent_ophthalmology_stomatology（眼耳鼻喉口腔科）：视力下降、耳鸣、鼻塞、咽喉痛、牙痛等
12. psychiatry（精神心理科）：抑郁、焦虑、精神分裂、失眠、心理障碍等
13. emergency（急诊医学科）：急性危重症、创伤、中毒、休克等
14. rehabilitation_pain（康复疼痛科）：慢性疼痛、康复治疗、运动损伤康复等
15. traditional_chinese_medicine（中医科）：中医诊疗、针灸、推拿、中药调理等

【任务】
根据患者主诉，判断应该挂哪个科室。优先考虑最相关和最紧急的科室。
"""
        
        user_prompt = f"""患者描述：{patient_description}

请判断应该挂哪个科室，输出JSON格式：
{{
  "dept": "科室代码（如internal_medicine）",
  "reason": "分诊理由"
}}"""
        
        try:
            obj, _, _ = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {
                    "dept": "internal_medicine",  # LLM失败时默认内科
                    "reason": "LLM解析失败，默认分诊至内科"
                },
                temperature=0.1  # 低温度保证一致性
            )
            
            dept = obj.get("dept", "internal_medicine")  # 默认内科
            reason = obj.get("reason", "")
            
            # 验证结果（确保在15个科室范围内）
            if dept not in self.VALID_DEPTS:
                print(f"⚠️  警告：LLM返回的科室'{dept}'不在标准列表中，默认分诊至内科")
                dept = "internal_medicine"
                reason = "LLM返回无效科室，默认分诊至内科"
            
            # 记录分诊
            self._triage_history.append({
                "patient_description": patient_description,
                "dept": dept,
                "reason": reason
            })
            
            return dept
            
        except Exception as e:
            print(f"⚠️  分诊异常: {str(e)}，默认分诊至内科")
            dept = "internal_medicine"
            self._triage_history.append({
                "patient_description": patient_description,
                "dept": dept,
                "reason": f"异常回退：{str(e)}，默认内科"
            })
            return dept
    
    def get_triage_summary(self) -> dict[str, int | list[dict[str, str]]]:
        """获取分诊摘要"""
        return {
            "total_triages": len(self._triage_history),
            "history": self._triage_history,
            "triage_qa": self._triage_qa,  # 包含分诊对话记录
            "questions_asked": len(self._triage_qa),
        }
    
    def needs_more_info(self, patient_description: str, conversation_history: list[dict[str, str]] | None = None) -> dict[str, bool | str]:
        """判断当前信息是否足够进行分诊
        
        Args:
            patient_description: 患者描述
            conversation_history: 之前的对话历史（避免重复提问）
            
        Returns:
            dict: {"needs_more": bool, "question": str, "reason": str}
        """
        if not self._llm:
            # 无LLM时，简单规则判断
            if len(patient_description) < 10:
                return {
                    "needs_more": True,
                    "question": "能详细说说您哪里不舒服吗？",
                    "reason": "描述过于简短"
                }
            return {"needs_more": False, "question": "", "reason": "信息充足"}
        
        # 使用LLM判断
        system_prompt = """你是一名经验丰富的分诊护士。你需要判断患者的描述是否足够进行科室分诊。

【判断标准】
信息充足的描述应包含：
1. 主要症状是什么（如头痛、腹痛、咳嗽等）
2. 症状的基本特征（部位、性质、程度等至少一项）

信息不足的情况：
- 描述过于模糊（如"不舒服"、"难受"）
- 缺少症状的具体部位
- 缺少主要症状描述
- 多个系统症状但无主次

【重要提醒】
- 不要重复问已经问过的问题
- 如果患者已经回答过但不清楚，可以换个角度问
- 如果患者明确表示"不知道"、"记不清"，不要继续追问同一问题"""

        # 构建用户提示，包含对话历史
        user_prompt = f"""患者描述：{patient_description}"""
        
        if conversation_history:
            user_prompt += f"\n\n已经问过的问题和回答：\n"
            for qa in conversation_history:
                user_prompt += f"Q{qa['round']}: {qa['question']}\nA{qa['round']}: {qa['answer']}\n"
        
        user_prompt += """

请判断：
1. 这个描述是否足够准确分诊到合适的科室？
2. 如果不够，你需要问患者什么问题来获取关键信息？（一次只问一个最关键的问题，不要重复已问过的问题）

输出JSON格式：
{{
  "needs_more": true/false,
  "question": "如果需要更多信息，问患者的问题（口语化、简洁）",
  "reason": "为什么需要/不需要更多信息"
}}"""

        try:
            obj, _, _ = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {"needs_more": False, "question": "", "reason": "默认认为信息充足"},
                temperature=0.3
            )
            
            return {
                "needs_more": obj.get("needs_more", False),
                "question": obj.get("question", ""),
                "reason": obj.get("reason", "")
            }
        except Exception:
            # 异常时认为信息充足，直接分诊
            return {"needs_more": False, "question": "", "reason": "LLM判断失败，直接分诊"}
    
    def triage_with_conversation(self, patient_agent, initial_description: str) -> str:
        """通过多轮对话进行分诊
        
        Args:
            patient_agent: 患者智能体（用于获取更多信息）
            initial_description: 患者初始描述
            
        Returns:
            科室代码
        """
        # 初始化对话历史
        self._triage_qa = []
        current_info = initial_description
        
        # 首次评估：判断初始描述是否已足够
        initial_assessment = self.needs_more_info(current_info, conversation_history=self._triage_qa)
        if not initial_assessment["needs_more"]:
            # 初始信息已充足，无需提问，直接分诊
            print(f"  ✅ 初始描述已充分，无需追问（理由：{initial_assessment['reason']}）")
            return self.triage(current_info)
        
        # 最多问max_triage_questions个问题
        for i in range(self._max_triage_questions):
            # 判断是否需要更多信息（传入对话历史以避免重复提问）
            assessment = self.needs_more_info(current_info, conversation_history=self._triage_qa)
            
            if not assessment["needs_more"]:
                # 信息充足，提前结束对话
                print(f"  ✅ 信息已充分，结束追问（理由：{assessment['reason']}）")
                break
            
            # 需要更多信息，向患者提问
            question = assessment["question"]
            if not question:
                # LLM判断需要更多信息但未生成问题，结束对话
                print(f"  ⚠️  未能生成有效问题，结束追问")
                break
            
            # 检查是否与之前的问题过于相似（额外保护机制）
            if self._is_duplicate_question(question, self._triage_qa):
                print(f"  ⚠️  检测到重复问题，结束追问")
                break
            
            # 记录问题
            print(f"  👩‍⚕️ 护士问（第{i+1}轮）: {question}")
            
            # 患者回答
            answer = patient_agent.respond_to_doctor(question)
            print(f"  👤 患者答: {answer}")
            
            # 记录对话
            self._triage_qa.append({
                "question": question,
                "answer": answer,
                "round": i + 1
            })
            
            # 更新当前信息（合并之前的描述和新回答）
            current_info = f"{current_info}\n补充信息：{answer}"
            
            # 每轮问答后立即重新评估信息充足性
            # 这样可以在获得关键信息后立即结束，而不是机械地问满所有轮次
        
        # 基于所有收集的信息进行分诊
        return self.triage(current_info)

    def _is_duplicate_question(self, new_question: str, conversation_history: list[dict[str, str]]) -> bool:
        """检查新问题是否与之前的问题重复（简单的字符串相似度检查）
        
        Args:
            new_question: 新问题
            conversation_history: 对话历史
            
        Returns:
            bool: 是否重复
        """
        if not conversation_history:
            return False
        
        # 简单的关键词检查
        new_q_clean = new_question.lower().strip("？?。.！!")
        for qa in conversation_history:
            old_q_clean = qa["question"].lower().strip("？?。.！!")
            
            # 如果新问题和旧问题有80%以上的相似度，认为是重复
            if new_q_clean == old_q_clean:
                return True
            
            # 检查是否包含相同的关键词组
            new_words = set(new_q_clean.split())
            old_words = set(old_q_clean.split())
            if len(new_words) > 2 and len(old_words) > 2:
                overlap = len(new_words & old_words)
                similarity = overlap / min(len(new_words), len(old_words))
                if similarity > 0.7:  # 70%以上重叠认为是相似问题
                    return True
        
        return False
