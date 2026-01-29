"""护士智能体：负责预检分诊，根据主诉分配科室"""
from __future__ import annotations

from services.llm_client import LLMClient


class NurseAgent:
    """护士智能体：根据患者主诉进行分诊"""
    
    # 只保留神经内科
    VALID_DEPTS = [
        "neurology"
    ]
    
    def __init__(self, llm: LLMClient, max_triage_questions: int = 3):
        """
        Args:
            llm: 语言模型客户端（必需，用于智能分诊）
            max_triage_questions: 分诊时最多可以问患者的问题数（默认3个）
        """
        self._llm = llm
        self._max_triage_questions = max_triage_questions
    
    def reset(self) -> None:
        """重置分诊历史（用于处理新患者）
        
        NurseAgent目前为无状态设计，每次分诊独立处理，不保存历史记录。
        提供此方法是为了保持接口一致性，未来如果需要添加状态管理时方便扩展。
        
        多患者处理时会自动调用此方法确保状态隔离。
        """
        pass
    
    def triage(self, patient_description: str) -> str:
        """
        根据患者描述进行分诊（当前系统只有神经医学科一个科室）
        
        Args:
            patient_description: 患者描述的症状（来自患者智能体）
            
        Returns:
            科室代码（neurology）
        """
        # 参数验证
        if not patient_description or not patient_description.strip():
            raise ValueError("患者描述不能为空")
        
        # 系统当前只有神经医学科，直接返回
        return "neurology"
    
    def get_triage_summary(self) -> dict[str, int | list[dict[str, str]]]:
        """获取分诊摘要"""
        return {
            "total_triages": 1,
            "history": [],
            "triage_qa": [],
            "questions_asked": 0,
        }
    
    def needs_more_info(self, patient_description: str, conversation_history: list[dict[str, str]] | None = None) -> dict[str, bool | str]:
        """判断当前信息是否足够进行分诊
        
        Args:
            patient_description: 患者描述
            conversation_history: 之前的对话历史（避免重复提问）
            
        Returns:
            dict: {"needs_more": bool, "question": str, "reason": str}
        """
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
        
        由于系统当前只有神经医学科一个科室，此方法直接返回neurology，
        保留方法接口是为了保持与系统其他部分的兼容性。
        
        Args:
            patient_agent: 患者智能体（用于获取更多信息）
            initial_description: 患者初始描述
            
        Returns:
            科室代码（neurology）
        """
        # 系统当前只有神经医学科，直接返回
        return "neurology"

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
