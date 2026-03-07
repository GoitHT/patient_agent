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
            known_case: 患者可见的结构化病例信息
            llm: 语言模型客户端（必需，用于生成真实回答）
            chief_complaint: 从病例中提取的主诉（若为空则自动从结构化字段中读取）
        """
        if not known_case:
            raise ValueError("known_case不能为空")
        
        self._known_case = known_case  # 私有：完整病例信息（含结构化字段）

        # ── 判断病史陈述者身份 ──────────────────────────────────────
        narrator_raw: str = known_case.get("病史陈述者", "").strip()
        # 认定为患者本人的关键词
        _SELF_KEYWORDS = ("患者本人", "本人", "自述", "患者自述")
        self._is_patient_self: bool = (
            not narrator_raw  # 空字段默认视为本人
            or any(kw in narrator_raw for kw in _SELF_KEYWORDS)
        )
        # 陈述者描述（用于提示词，如 "患者母亲"、"患者家属" 等）
        self._narrator_role: str = narrator_raw if narrator_raw else "患者本人"
        # 患者性别代词，供家属代述时以第三人称引用
        patient_gender: str = known_case.get("性别", "")
        self._he_she: str = "她" if "女" in patient_gender else "他"
        patient_name: str = known_case.get("姓名", "患者")
        self._patient_ref: str = patient_name if patient_name else "患者"
        # ────────────────────────────────────────────────────────────

        # 从结构化字段拼接病情文本
        parts: list[str] = []
        for label, key in [("姓名", "姓名"), ("性别", "性别"), ("年龄", "年龄")]:
            val = known_case.get(key, "")
            if val:
                parts.append(f"{label}：{val}")
        chief = known_case.get("主诉", "")
        if chief:
            parts.append(f"主诉：{chief}")
        present = known_case.get("现病史_详细描述", "")
        if present:
            parts.append(f"现病史：{present}")
        case_info = "，".join(parts) if parts else ""
        
        self.case_info = case_info  # 公开：供外部访问
        
        # 主诉优先级：传入参数 > 结构化字段 "主诉" > 空字符串
        if chief_complaint:
            self._chief_complaint = chief_complaint
        else:
            self._chief_complaint = known_case.get("主诉", "")
        
        if not self.case_info:
            raise ValueError("结构化病例字段为空，无法创建患者智能体")
        
        self.llm = llm
        self._recent_context: list[dict[str, str]] = []  # 仅保留最近5轮用于生成自然回答
    
    def reset(self) -> None:
        """重置患者对话状态（用于处理新对话或重新开始问诊）
        
        清空对话历史上下文，确保不会将之前的对话记忆带入新的问诊。
        
        注意：在多患者处理系统中，每个患者都会创建新的PatientAgent实例，
        因此通常不需要手动调用reset。此方法主要用于测试或单患者多轮对话场景。
        """
        self._recent_context = []
    
    def describe_to_nurse(self) -> str:
        """向护士描述患者症状（基于主诉，用自然语言）

        若病史陈述者为患者本人，以第一人称描述；
        否则以家属/陪同人身份、用第三人称描述患者症状。

        Returns:
            症状描述（口语化表达）
        """
        if self._is_patient_self:
            identity_line = "你是一位感到不适的患者，刚来到医院分诊台。"
            narrator_note = '用第一人称（"我"）描述自身症状'
            extra_rule = ""
        else:
            identity_line = (
                f"你是{self._narrator_role}，陪同患者{self._patient_ref}来到医院分诊台。"
            )
            narrator_note = (
                f'以{self._narrator_role}的口吻、用"{self._he_she}"或'
                f'"{self._patient_ref}"（第三人称）描述患者的症状，不要说"我的症状"'
            )
            extra_rule = (
                f'\n   - 以家属视角描述，可以说"他/她跟我说..."或"我看到他/她..."'
                f'\n   - 不要用"我感觉..."描述患者的主观感受'
            )

        system_prompt = f"""{identity_line}

【患者病情】
{self.case_info}

【主诉】
{self._chief_complaint}

【角色要求】
1. **核心信息优先**：必须包含主诉中的关键医疗信息
   - 如果有具体体温，要说出来（比如"烧到38度多"）
   - 如果有重要既往病史，要提及（比如"之前得过XX病"、"正在化疗"）
   - 如果有明确的病程时长，要说清楚（比如"已经10天了"）
   - 如果有特殊用药史，可以提及{extra_rule}

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
   - 有严重病史时会更担心复发或并发症
   - 可以表达焦虑："不知道是不是和XX有关"

⚠️ 注意：直接描述症状即可，不要加"医生"、"护士"等称呼。
"""

        user_prompt = f"请{narrator_note}（3-5句话，必须包含主诉中的关键信息如体温、病程、既往病史等，不要加称呼语）："

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
        # 构建对话历史上下文（仅最近5轮）
        history_context = ""
        if self._recent_context:
            history_lines = []
            for turn in self._recent_context:
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
        
        system_prompt = f"""你是一位真实的患者，正在医院接受医生问诊。你对自己的病情感到担心。你没有医学专业知识，是一个普通人。

【你的真实病情】
{self.case_info}

【之前的对话】
{history_context}
{state_instruction}

【回答原则】
1. **非专业特征**：
   - 你没有医学知识，不知道专业医学术语
   - 当医生问到具体身体部位的专业名词时（如"淋巴"、"胱骨"、"颈椎"），应回答“医生，我不太懂这个”或“这个专业名词我不太清楚”
   - 用口语化表达描述症状，不使用医学术语

2. **口语化特征**：
   - 使用口语语气词：“唔...”、“我觉得...”、“应该是...”、“我记得...”
   - 可以用“我也不确定”、“可能是吧”等不确定表达
   - 有时会用手势指位置：“就这儿（指着XX部位）”
   - 回答时像在聊天，不是在背书

3. **信息依赖性**：
   - 仅根据【真实病情】中的内容回答
   - 如果病情信息中没有提及，说“这个我不太清楚”或“没注意过”
   - 不自己编造信息

4. **检查结果回答策略**：
   - 如果医生直接询问一项完整的医院检查结果（如MRI、免疫组化、CT等）：
     * 如果病情信息中没有写明存在该检查，回答“我还没做过这个检查”
     * 如果有，直接如实详细回答，可以使用医学术语，回答应当准确简明
   - 如果医生问的是具体身体部位的专业名词，你应该不回答

5. **回答风格**：
   - {response_style}回答医生的问题
   - 符合第一人称患者口吻，给人交流感
   - 不要一次说太多，像真实聊天一样一句一句说
   - 如果不确定，可以表现出犹豫：“唔...这个...”、“我想想...”

6. **特殊情况处理**：
   - 当医生的问题非常宽泛，内涵多于3个问题时，可以说“医生，你问的太多了，我有点乱，能一个一个来吗？”
   - 如果医生要求“介绍你的病情情况”这种极度开放的问题，可以说“医生，我也不知道从哪说起，你问吧”

【示例】
医生：你的颈椎有没有问题？
患者：唔...颈椎？医生，这个专业名词我不太清楞。你是说脖子吗？脖子倒是没什么不舒服的。

医生：你做过MRI吗？结果怎么样？
患者：做过，就上个月做的。（然后详细描述MRI结果，可以用专业术语）

医生：你的血细胞比容多少？
患者：医生，这个我不太懂。是说血常规吗？我有单子，但上面的数据我也看不懂...

医生：请介绍你的病情情况。
患者：唔...医生，我也不知道从哪儿说起，你问吧，我跟你说。

现在医生问：{doctor_question}

请{response_style}回答医生的问题，记住：你是患者，不是医生。"""
        
        user_prompt = f"医生问：{doctor_question}\n\n患者回答："
        
        # 使用LLM生成回答
        response = self.llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_tokens=150
        )
        answer = response.strip()
        
        # 更新最近上下文（仅保留最近5轮）
        self._recent_context.append({
            "doctor": doctor_question,
            "patient": answer
        })
        if len(self._recent_context) > 5:
            self._recent_context.pop(0)
        
        return answer
    
    def get_conversation_summary(self) -> dict[str, Any]:
        """获取对话摘要（简化版）"""
        return {
            "total_turns": len(self._recent_context),
            "conversation": [],  # 不再保存完整对话历史
            "case_info": self.case_info
        }
