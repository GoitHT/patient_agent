"""患者智能体：模拟真实患者根据病例信息回答医生提问"""
from __future__ import annotations

import json
from typing import Any

from services.llm_client import LLMClient


class PatientAgent:
    """患者智能体：只知道自己的症状和基本信息，不知道检查结果"""
    
    def __init__(self, known_case: dict[str, Any], llm: LLMClient):
        """
        Args:
            known_case: 患者可见的病例信息（仅 Case Information）
            llm: 语言模型客户端（必需，用于生成真实回答）
        """
        if not known_case:
            raise ValueError("known_case不能为空")
        
        self.known_case = known_case
        self.case_info = known_case.get("Case Information", "")
        
        if not self.case_info:
            raise ValueError("Case Information为空，无法创建患者智能体")
        
        self.conversation_history: list[dict[str, str]] = []
        self.llm = llm
    
    def reset(self, new_case: dict[str, Any] | None = None) -> None:
        """重置患者状态（用于处理新病例）"""
        if new_case:
            if not new_case.get("Case Information"):
                raise ValueError("新病例的Case Information为空")
            self.known_case = new_case
            self.case_info = new_case.get("Case Information", "")
        
        self.conversation_history = []
    
    def respond_to_doctor(self, doctor_question: str) -> str:
        """根据病例信息回答医生的问题（完全使用LLM）"""
        # 构建对话历史上下文
        history_context = ""
        if self.conversation_history:
            recent = self.conversation_history[-5:]  # 保留最近5轮
            history_lines = []
            for turn in recent:
                history_lines.append(f"医生：{turn['doctor']}")
                history_lines.append(f"患者：{turn['patient']}")
            history_context = "\n".join(history_lines)
        else:
            history_context = "（首次问诊）"
        
        system_prompt = f"""你是一位患者，正在接受医生问诊。

【你的病情】
{self.case_info}

【对话历史】
{history_context}

【回答要求】
1. 只回答病情描述中明确提到的信息
2. 没提到的就说"不太清楚"或"不记得了"
3. 用口语化、简短回答（1-2句话）
4. 不要编造或猜测
5. 可以表现出真实情绪（担心、焦虑等）
"""
        
        user_prompt = f"医生问：{doctor_question}\n\n患者回答："
        
        # 使用LLM生成回答
        response = self.llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=150
        )
        answer = response.strip()
        
        # 记录对话
        self.conversation_history.append({
            "doctor": doctor_question,
            "patient": answer
        })
        
        return answer
    
    def get_conversation_summary(self) -> dict[str, Any]:
        """获取对话摘要"""
        return {
            "total_turns": len(self.conversation_history),
            "conversation": self.conversation_history,
            "case_info": self.case_info
        }
