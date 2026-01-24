"""患者智能体：模拟真实患者根据病例信息回答医生提问"""
from __future__ import annotations

import json
from typing import Any

from services.llm_client import LLMClient


class PatientAgent:
    """患者智能体：只知道自己的症状和基本信息，不知道检查结果"""
    
    def __init__(self, known_case: dict[str, Any], llm: LLMClient, chief_complaint: str = ""):
        """
        Args:
            known_case: 患者可见的病例信息（仅 Case Information）
            llm: 语言模型客户端（必需，用于生成真实回答）
            chief_complaint: 从病例中提取的主诉（患者会基于此向护士/医生描述）
        """
        if not known_case:
            raise ValueError("known_case不能为空")
        
        self._known_case = known_case  # 私有：完整病例信息
        self.case_info = known_case.get("Case Information", "")  # 公开：供外部访问
        self._chief_complaint = chief_complaint  # 私有：患者自己的主诉
        
        if not self.case_info:
            raise ValueError("Case Information为空，无法创建患者智能体")
        
        self.conversation_history: list[dict[str, str]] = []  # 公开：对话历史
        self.llm = llm
    
    def describe_to_nurse(self) -> str:
        """患者向护士描述自己的症状（基于主诉，用自然语言）
        
        Returns:
            患者的症状描述（口语化表达）
        """
        if not self.llm:
            # 如果没有LLM，直接返回主诉
            return self._chief_complaint if self._chief_complaint else "我感觉不太舒服"
        
        system_prompt = f"""你是一位感到不适的患者，刚来到医院分诊台。

【你的病情】
{self.case_info}

【你的主诉】
{self._chief_complaint}

【角色要求】
1. **核心信息优先**：必须包含主诉中的关键医疗信息
   - 如果有具体体温，要说出来（比如"烧到38度多"）
   - 如果有重要既往病史，要提及（比如"我之前得过XX病"、"我正在化疗"）
   - 如果有明确的病程时长，要说清楚（比如"已经10天了"）
   - 如果有特殊用药史，可以提及（比如"我在吃XX药"）

2. **语言表达自然**：用普通人的口语，避免医学术语
   - "发烧"而不是"发热"
   - "疙瘩"、"红点"而不是"皮损"
   - "疼"而不是"触痛"
   - 但重要的病名可以直接说（如"骨髓瘤"、"化疗"）

3. **结构清晰完整**：
   - 先说主要症状（1-2个最困扰的）
   - 补充重要背景（既往病史、近期治疗）
   - 表达情绪（担心、焦虑）
   - 控制在3-5句话内

4. **情绪真实自然**：
   - 有严重病史的患者会更担心复发或并发症
   - 可以表达焦虑："不知道是不是和XX有关"
   - 可以表达期待："希望医生帮我看看"

【示例风格】
好的示例1（有既往史）：
"医生，我这几天一直发烧，烧到38度多，脖子和胸前还起了很多红疙瘩，又疼又痒。我之前有骨髓瘤，上个月刚做完一次化疗，不知道是不是有关系，挺担心的。"

好的示例2（症状明确）：
"护士，我肚子疼得厉害，右下腹这块（手指位置），疼了两天了，还发烧到38.5度，吐了好几次，走路都疼得直不起腰。"

不好的示例（信息不完整）：
"我最近一直发烧，身上还起了好多疙瘩，挺难受的。"（缺少体温、病程、既往史等关键信息）
"""
        
        user_prompt = "请向护士描述你的症状（3-5句话，必须包含主诉中的关键信息如体温、病程、既往病史等）："
        
        try:
            description = self.llm.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=150
            ).strip()
            return description
        except Exception as e:
            # LLM失败时返回主诉
            return self._chief_complaint if self._chief_complaint else "我感觉不太舒服"
    
    def respond_to_doctor(self, doctor_question: str, physical_state: dict[str, Any] | None = None) -> str:
        """根据病例信息回答医生的问题（完全使用LLM，受物理状态影响）
        
        Args:
            doctor_question: 医生的问题
            physical_state: 患者当前物理状态快照（energy_level, pain_level等）
        """
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
        
        # 根据物理状态调整回答策略
        state_instruction = ""
        response_style = "正常、完整地"
        if physical_state:
            pain_level = physical_state.get("pain_level", 0)
            energy_level = physical_state.get("energy_level", 10)
            
            if pain_level > 7:
                state_instruction = f"\n\n【身体状态】你正感到剧烈疼痛（{pain_level}/10），很难集中注意力，说话时会不自觉地皱眉或捂住疼痛部位。回答要简短，可能会说一半停下来喘口气。表达时可以说'哎呦...好疼'、'我...我说不太清楚'等。"
                response_style = "简短、断断续续地"
            elif pain_level > 4:
                state_instruction = f"\n\n【身体状态】你感到明显的疼痛或不适（{pain_level}/10），有些难受，说话时可能会皱眉，回答相对简短。"
                response_style = "稍微简短地"
            
            if energy_level < 3:
                state_instruction += f"\n你非常疲惫虚弱（体力{energy_level}/10），说话有气无力，回答要尽量简短，可能会说'累...说不动了'、'医生，我好累'。"
                response_style = "非常简短、有气无力地"
            elif energy_level < 6:
                state_instruction += f"\n你感到疲劳（体力{energy_level}/10），不太想多说话，回答倾向简洁。"
        
        system_prompt = f"""你是一位真实的患者，正在医院接受医生问诊。你对自己的病情感到担心。

【你的真实病情】
{self.case_info}

【之前的对话】
{history_context}
{state_instruction}

【角色要求】
1. **信息真实性**：只说病情描述中明确提到的信息，没提到的就说'不太清楚'、'不记得了'、'应该没有吧'
2. **表达方式**：用普通人的口语表达，不要用医学术语（比如说'肚子疼'而不是'腹痛'）
3. **情绪表现**：适度表现出担心、焦虑、不安等真实情绪
   - 初次提到严重症状时：'医生，我很担心...'、'是不是很严重啊？'
   - 症状困扰很久：'已经好几天了，一直不见好'
   - 影响生活：'疼得都睡不着觉'、'都没法正常吃饭了'
4. **回答长度**：{response_style}回答（一般1-2句话，状态不好时更短）
5. **不确定性**：对时间、程度等细节可能记不清楚：'大概...'、'好像是...'、'记不太清了'
6. **主动补充**：如果医生问到相关的，可以自然地补充其他明显症状

【示例风格】
❌ 不好：'我出现上腹部疼痛，呈持续性钝痛，进食后加重。'
✅ 好：'肚子疼，这里（指上腹），吃完饭就更疼了。'

❌ 不好：'是的，确实存在该症状。'
✅ 好：'对对对，就是这样。'或'嗯，有的。'
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
