"""医生智能体：基于RAG知识库进行问诊、开单、诊断"""
from __future__ import annotations

import json
from typing import Any

from rag import AdaptiveRAGRetriever
from services.llm_client import LLMClient
from utils import get_logger


class DoctorAgent:
    """医生智能体：初始不知道任何病例信息，通过问诊获取"""
    
    def __init__(
        self,
        dept: str,
        retriever: AdaptiveRAGRetriever,
        llm: LLMClient | None = None,
        max_questions: int = 10  # 最底层默认值，通常由config.yaml覆盖
    ):
        """
        Args:
            dept: 科室代码 (例如: neurology)
            retriever: RAG检索器
            llm: 语言模型客户端
            max_questions: 最多问题数（通常从config.yaml读取）
        """
        self.dept = dept
        self._retriever = retriever
        self._llm = llm
        self._max_questions = max_questions
        self._logger = get_logger("hospital_agent.doctor")
        self.collected_info: dict[str, Any] = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "history": {},
            "exam_findings": {},
            "conversation_history": []  # 新增：完整的对话历史（问题+回答）
        }
        self.questions_asked: list[str] = []
    
    def reset(self) -> None:
        """重置医生状态（用于处理新患者）
        
        清空所有与上一个患者相关的状态，确保每个新患者都从零开始问诊：
        - collected_info: 清空已收集的患者信息
        - questions_asked: 清空已问问题列表（避免重复问题检查时误判）
        
        ⚠️ 多患者处理时必须调用此方法，否则会出现：
          1. 问题重复检查失效（以为已经问过，实际是上个患者问的）
          2. 问诊历史污染（显示错误的历史记录）
        """
        self.collected_info = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "history": {},
            "exam_findings": {},
            "conversation_history": []  # 确保重置时清空历史对话
        }
        # ⚠️ 关键：清空已问问题列表
        self.questions_asked = []
        self._logger.debug(f"医生 Agent 已重置：collected_info + questions_asked 已清空")
    
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
        if len(self.questions_asked) >= self._max_questions:
            return ""
        
        # 构建RAG上下文
        kb_context = ""
        if rag_chunks:
            kb_context = self._format_chunks(rag_chunks)
        elif context:
            # 如果没有提供chunks，尝试检索
            chunks = self._retriever.retrieve(
                f"{self.dept} {context} 问诊要点",
                filters={"dept": self.dept},
                k=3
            )
            kb_context = self._format_chunks(chunks)
        
        # 构建历史对话展示 - 包括问题和患者的回答
        is_first_question = len(self.questions_asked) == 0

        # Python层面早停检查：核心信息充分收集后，跳过LLM调用直接终止
        if not is_first_question and self._should_stop_early():
            self._logger.info("  ✅ [早停] 核心信息已充分收集，提前结束问诊")
            return ""

        if self.questions_asked:
            # 显示完整的对话历史（问题 + 回答）
            conversation_history = self.collected_info.get("conversation_history", [])
            
            if conversation_history:
                # 如果有完整的对话记录，显示问题和回答
                history_lines = []
                for i, conv in enumerate(conversation_history, 1):
                    history_lines.append(f"👨‍⚕️ 医生第{i}问: {conv.get('question', '')}")
                    history_lines.append(f"👤 患者回答: {conv.get('answer', '')[:100]}{'...' if len(conv.get('answer', '')) > 100 else ''}")
                    history_lines.append("")  # 空行分隔
                history_display = "\n".join(history_lines)
            else:
                # 如果没有对话记录，只显示问题列表
                history_display = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(self.questions_asked)])
            
            asked_summary = f"""⛔ **已问过的所有问题及患者回答（共{len(self.questions_asked)}个）**

{history_display}

⚠️ **绝对禁止重复！**
- 以上每一个问题都已经问过，绝对不能再问！
- 不能问语义相同或相似的问题（即使换个说法也不行）
- **仅当患者的回答不完整时，才可以针对回答中的具体内容深入追问**
- 如果患者已经回答了该问题，就不要再问相关内容！
- 如果想不出新问题，就返回空字符串停止问诊！

💡 **如何判断是否重复**：
1. 检查你想问的内容是否已经在患者的回答中出现
2. 检查是否已经有相似的问题（即使表述不同）
3. 如果患者已经提到过某个症状，就不要再问“有没有这个症状”

📝 **深入追问示例**：
- 已问：“有哪里不舒服？” 患者答：“头疼”
  ✅ 允许：“头哪个位置疼？前额、后脑勺还是两侧？”
  ❌ 禁止：“还有别的不舒服吗？”（这是新问题，不是深入追问）

- 已问：“疼痛会扩散到其他地方吗？” 患者答：“会，有时候会到后背”
  ✅ 允许：“后背的哪个位置？是一直疼还是偶尔疼？”
  ❌ 禁止：“疼痛的范围大吗？”（与已问问题重复）
"""
        else:
            # 首次问诊：使用独立最简提示词，直接返回开场问候，避免被后续阶段策略干扰
            _fq_prompt = (
                f"你是{self._dept_name()}医生，患者刚进入诊室，这是本次就诊的第一句话。\n"
                "唯一任务：生成一个开放式问候，让患者自己描述来意，不得假设任何症状。\n\n"
                "✅ 直接使用以下之一：\n"
                '  "您好，哪里不舒服？"\n'
                '  "您好，今天来看什么问题？"\n'
                '  "您好，跟我说说您的情况吧。"\n\n'
                "❌ 绝对禁止（患者尚未陈述任何症状！）：\n"
                '  问具体细节，如"疼多久了"、"有放射痛吗"\n'
                '  使用专业术语，如"放射痛"、"压榨感"\n'
                "  一次问多个问题\n\n"
                f"【分诊护士主诉记录（仅做背景参考，不要据此假设具体症状）】\n{chief_complaint}\n\n"
                '输出JSON：{"question": "问候语", "reason": "首次开放式问诊", "duplicate_check": "首次问诊无需检查"}'
            )
            try:
                _fq_obj, _, _ = self._llm.generate_json(
                    system_prompt=_fq_prompt,
                    user_prompt="请生成首次问诊的开场问候语。",
                    fallback=lambda: {"question": "您好，哪里不舒服？", "reason": "首次", "duplicate_check": "首次"},
                    temperature=0.2,
                )
                return str(_fq_obj.get("question", "")).strip() or "您好，哪里不舒服？"
            except Exception as _fq_err:
                self._logger.error(f"  ❌ 首次问诊生成失败: {_fq_err}")
                return "您好，哪里不舒服？"

        # ── 后续轮次：构建含话题覆盖图的结构化提示词 ──
        topic_map = self._build_topic_coverage_map()
        system_prompt = f"""你是一名经验丰富的{self._dept_name()}医生，正在进行{context}（第{len(self.questions_asked)+1}问/上限{self._max_questions}问）。你需要通过系统的问诊收集关键信息，做出准确诊断。

⚠️ 患者状态感知
- 观察患者的回答：如果简短、含糊或表现痛苦，说明状态不佳，应优先问最关键的问题
- 疼痛剧烈或极度疲劳时：立即转向核心问题（症状性质、持续时间、危险征象），避免细枝末节
- 意识异常或病情危重：停止常规问诊，建议紧急处理

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【主诉】
{chief_complaint}

【已收集的信息】
{json.dumps(self.collected_info, ensure_ascii=False, indent=2)}

【问诊历史】({len(self.questions_asked)}/{self._max_questions}）
{asked_summary}

【临床知识参考】
{kb_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📊 已覆盖话题（优先阅读，决定停止前必看）
{topic_map}
（✅=已收集到信息  ❌=仍空缺）
⭐ 当 ❌缺失维度 ≤2 时，信息已充分，应主动停止提问。

🚫 **重复问题检测 - 核心原则（最高优先级）**

⚠️ 在生成新问题前，你必须执行以下严格检查流程：

**步骤1：逐一核对已问问题**
仔细阅读【问诊历史】中列出的每一个问题，逐条检查你想生成的问题是否与任何一个已问问题在以下维度重复：
- 语义重复：询问的内容实质相同（如"是否扩散"和"有无扩散现象"）
- 意图重复：目的相同但表述不同（如"疼痛程度"和"疼痛打几分"）
- 部分重复：询问的某个方面已包含在之前的问题中

**步骤2：判断是否为深入追问**
如果你想问的内容与已问问题相关，必须确保是基于已有答案的深入追问：
- ✅ 允许：患者已回答"疼痛会扩散"，追问"具体扩散到哪些部位？"
- ❌ 禁止：患者已回答"疼痛会扩散"，再问"疼痛有没有放射到其他地方？"（语义重复）

**步骤3：重复问题识别规则**
以下情况都属于重复，严禁生成：
1. **换词重述**：用不同词汇表达相同问题
   - 已问"是否扩散到其他部位？" ❌ "有没有扩散现象？" ❌ "会不会放射到别的地方？"
   
2. **量化方式重复**：已经问过定量或定性评估
   - 已问"疼痛程度如何？" ❌ "能给疼痛打个分吗？" ❌ "疼痛厉害吗？"
   
3. **因素重复**：已经询问过加重/缓解/诱发因素
   - 已问"接触什么会加重？" ❌ "有什么因素会诱发或加重吗？" ❌ "什么情况下会更严重？"
   
4. **时间重复**：已经问过持续时间、发作频率、时间规律
   - 已问"持续多久了？" ❌ "什么时候开始的？" ❌ "症状出现多长时间？"

5. **伴随症状重复**：已经全面询问过其他症状
   - 已问"还有其他不舒服吗？" ❌ "伴随什么症状？" ❌ "有别的异常吗？"

**执行要求**：
- 生成问题前，必须在心中逐条检查上述5类重复模式
- 如果存在任何疑似重复，立即放弃该问题，生成完全不同方向的问题
- 当无法确定新问题是否重复时，宁可选择其他信息缺口，绝不冒险生成

❌ 重复示例（绝对禁止）：
```
已问："疼痛会扩散吗？"
禁止："有没有放射到其他部位？"  # 语义完全相同
禁止："疼痛范围有变化吗？"      # 意图相同

已问："什么因素会加重疼痛？"
禁止："有什么诱因吗？"          # 询问相同的因素
禁止："做什么动作会更痛？"      # 仍然是加重因素

已问："疼痛程度如何？"
禁止："能打几分吗？"            # 都是量化疼痛
禁止："很痛吗？"                # 仍然问程度
```

✅ 正确做法（深入追问）：
```
已问："疼痛会扩散吗？" 患者答："会"
允许："具体扩散到哪些部位？左肩？右肩？还是背部？"  # 基于答案深入细节

已问："什么因素会加重？" 患者答："吃饭后"
允许："是吃完饭立即加重，还是过一段时间？大概多久？"  # 基于答案细化时间

已问："疼痛程度如何？" 患者答："很痛，8分"
允许："这8分的疼痛会影响您的日常活动吗？比如睡眠、工作？"  # 转向功能影响
```

💡 **防重复策略**：
- 优先询问不同维度的信息（症状→危险信号→病史→暴露史）
- 当某个维度已充分询问时，立即转向其他维度
- 使用【已收集的信息】判断哪些方面仍然空白

【问诊策略 - 逐步深入原则】

🔰 **第一阶段：开放式问诊**（第1-2个问题）
   - **如果是首次问诊**：必须使用开放式问题
     * "您好，哪里不舒服？" 或 "今天来看什么问题？"
     * 让患者自己描述，不要假设患者的症状
   - **避免**：直接问很具体的问题、使用专业术语、一次问多个问题
   - **目的**：了解患者的主要症状和主诉

1️⃣ **第二阶段：核心信息收集**（第3-5个问题）
   - **基于患者的初步描述**，追问关键细节：
     * 症状特点：性质、部位、程度（用患者能理解的方式问，如"有多痛？能给个分数吗？1-10分"）
     * 时间信息："什么时候开始的？持续多久了？"（避免说"病程"这种专业词）
     * 伴随症状："除了XX，还有哪里不舒服吗？"（基于患者已说的症状）
   - **问题要求**：口语化、具体、一次一个问题

2️⃣ **危险信号**（排除急危重症）
   - 红旗征象：突发剧烈、进行性加重、意识改变、生命体征异常
   - 需要立即处理的情况
   - **优先筛查**：发热、体重下降、夜间盗汗、呼吸困难、意识障碍

3️⃣ **深度追问**（深入挖掘关键细节）
   - 对主要症状进行SOCRATES分析：
     * Site(部位)：具体位置，是否放射
     * Onset(起病)：急性/慢性，诱因
     * Character(性质)：描述性质（刺痛/钝痛/胀痛/绞痛）
     * Radiation(放射)：是否扩散
     * Associated(伴随)：相关症状
     * Timing(时间)：持续/间歇，昼夜节律
     * Exacerbating/Relieving(加重/缓解)：什么因素影响
     * Severity(严重度)：量化评分
   - 对异常症状追问到具体细节（不满足于模糊回答）

4️⃣ **鉴别诊断**（缩小诊断范围）
   - 加重/缓解因素（饮食、体位、活动、情绪）
   - 既往类似发作（频率、治疗史、效果）
   - 相关疾病史（系统疾病、手术史）
   - **关键鉴别点**：根据初步怀疑的诊断，询问区分性特征

5️⃣ **病史补充**（完善诊断依据）
   - 既往病史、用药史（慢性病、长期用药）
   - 家族史（遗传性疾病）
   - **暴露史筛查**（重要！常被忽视的诊断线索）：
     * 吸烟史（香烟/电子烟，每天支数，吸烟年数）
     * 饮酒史（种类、频率、饮酒年数）
     * 职业暴露（粉尘、化学品、辐射等）
     * 环境接触（新装修、宠物、霉菌）
     * 药物/保健品（新用药物、中草药、补充剂）
   - 其他系统症状（全身性疾病筛查）

【你的任务】
**① 优先判断：是否停止提问？**（满足任一条件即立即停止，返回 question=""）
- 上方已覆盖话题中 ❌缺失维度 ≤2 个（信息已充分）
- 已收集信息足以制定初步检查计划或诊断方向
- 已达问诊上限 {len(self.questions_asked)}/{self._max_questions} 问

⭐ 质量原则：5个精准问题 > 机械问满{self._max_questions}问。信息够用时主动停止。

**② 若继续：从【❌ 仍缺失】维度选方向，生成前确认：**
- 不与已问问题语义重复（换词重述同样算重复）
  · "疼痛会扩散吗" ＝ "有放射感吗" ＝ "痛会跑别处吗" → 都是同一问题
  · "加重因素" ＝ "什么情况更痛" ＝ "诱发因素" → 都是同一维度
- 不问患者回答中已主动提及的内容
- 无新角度时，返回 question="" 停止（不要凑问题）

**③ 问题质量要求**
- 口语化（"有多久了"而非"病程多长"）
- 一次只问一个问题
- 从患者已有答案出发具体追问，避免"还有什么不舒服"类泛问

**输出JSON格式**：
{{
  "question": "问题内容（停止则为空字符串）",
  "reason": "决策理由（新问题需说明：针对哪个缺失维度 + 与哪些已问问题不重复）",
  "duplicate_check": "已核对第N、M问，确认不重复"
}}


【良好问题示例 - 按问诊阶段分类】

🔰 **首次问诊（开放式）**：
  ✅ "您好，哪里不舒服？"
  ✅ "您好，今天来看什么问题？"
  ✅ "您好，跟我说说您的情况吧。"
  ❌ "疼痛持续多久了？"（患者还没说疼痛）
  ❌ "您的放射痛如何？"（太专业，患者不懂）

✅ **量化追问**（要用口语化方式）：
  "有多痛？能给个分数吗？1分是最轻，10分是最痛。"
  "一天发作几次？每次大概多长时间？"
  "什么时候开始的？有多久了？"
  
✅ **深度挖掘**（避免专业术语）：
  "疼起来是什么感觉？是闷痛、刺痛、还是像针扎一样？"
  "痛的地方会跑到别的地方吗？比如后背或者肩膀？"
  "疼痛的位置能指给我看看吗？"
  
✅ **鉴别关键**（基于患者已有描述）：
  "吃饭后会更痛还是好一点？"
  "躺下时会不会加重？弯腰或者咳嗽的时候呢？"
  "什么情况下会更难受？"
  
✅ **危险筛查**（用患者能懂的话）：
  "最近有没有发烧？"
  "体重有没有掉？"
  "有没有觉得心跳快、喘不上气、或者晕倒过？"
  
✅ **病史关联**：
  "以前有没有这样过？当时怎么好的？"
  "有没有什么慢性病？比如高血压、糖尿病？"

❌ **避免的问题类型**：
1. **首次问诊就问具体细节**（患者还没描述主要症状）
   - ❌ "疼痛持续多久了？" → 应先问"哪里不舒服？"
   - ❌ "有放射痛吗？" → 应先让患者描述症状
   
2. **使用专业术语**（患者可能不理解）
   - ❌ "触痛明显吗？" → ✅ "我按这里会痛吗？"
   - ❌ "有夜间盗汗吗？" → ✅ "晚上睡觉会出很多汗吗？"
   - ❌ "放射性疼痛" → ✅ "痛会扩散到其他地方吗？"
   
3. **一次问多个问题**（患者难以回答）
   - ❌ "请描述一下疼痛的性质、部位、持续时间和加重因素。" 
   - ✅ 一次问一个："是什么样的痛？闷痛还是刺痛？"
   
4. **过度开放的问题**（无法引导患者）
   - ❌ "您还有什么其他症状吗？"
   - ✅ "除了头痛，肚子有没有不舒服？恶心想吐吗？"
   
5. **假设性问题**（患者还没提到的情况）
   - ❌ "您的胸痛会放射到左臂吗？"（患者可能只是头痛）
   - ✅ 先确认症状，再追问细节

💡 **口语化原则**：
- 用"痛"而不是"疼痛"、"触痛"
- 用"有没有"而不是"是否存在"
- 用"什么时候开始的"而不是"发病时间"
- 用"晚上"而不是"夜间"
- 用"拉肚子"而不是"腹泻"

💡 **逐步引导原则**：
- 第1问：开放式，让患者自己说
- 第2-3问：针对患者提到的症状，问时间、程度
- 第4-5问：追问伴随症状、加重缓解因素
- 第6-7问：危险信号筛查
- 第8-10问：既往史、暴露史
❌ 避免：与已问问题意思相同或高度相似的问题
❌ 避免："最近怎么样？"（过于开放，缺乏目的性）
"""
        
        base_user_prompt = "请根据以上信息，决定是否继续问诊。如果需要继续，生成下一个问题；如果信息已足够，返回空字符串。"

        # 最多重试2次：当检测到重复时，携带明确禁止的问题重新请求LLM
        max_retries = 2
        banned_questions: list[str] = []  # 本轮被判为重复的问题，用于重试时明确告知LLM

        for attempt in range(max_retries + 1):
            # 如果有被拒绝的问题，追加到user_prompt中
            if banned_questions:
                banned_block = "\n\n".join(
                    f"  ❌ 禁止生成（已判定为与已问问题重复）：{q}" for q in banned_questions
                )
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    f"【注意】以下问题已被系统判定为重复，本次必须生成完全不同方向的新问题：\n"
                    f"{banned_block}\n"
                    f"请换一个完全不同的问诊方向，绝对不能再问与以上禁止问题语义相同或相似的内容。"
                )
            else:
                user_prompt = base_user_prompt

            try:
                obj, _, _ = self._llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=lambda: {"question": "", "reason": "", "duplicate_check": ""},
                    temperature=0.3
                )
                question = str(obj.get("question", "")).strip()
                reason = str(obj.get("reason", "")).strip()
                duplicate_check = str(obj.get("duplicate_check", "")).strip()

                # 如果为空，直接返回
                if not question:
                    if reason:
                        self._logger.debug(f"  💡 停止问诊: {reason}")
                    return ""

                # 【安全网】检查是否仍然重复
                if self._is_duplicate_question(question):
                    self._logger.warning(
                        f"  ⚠️  LLM生成了重复问题（第{attempt + 1}次尝试）\n"
                        f"     生成的问题: {question}\n"
                        f"     LLM自己的检查: {duplicate_check if duplicate_check else '未填写'}\n"
                        f"     当前已问问题数: {len(self.questions_asked)}"
                    )
                    self._logger.warning(
                        f"     已问问题参考: {self.questions_asked[-3:] if len(self.questions_asked) >= 3 else self.questions_asked}"
                    )
                    banned_questions.append(question)  # 记录本次被拒绝的问题
                    if attempt < max_retries:
                        self._logger.warning(
                            f"     携带禁止问题重试（第{attempt + 2}次）..."
                        )
                        continue  # 重试
                    else:
                        self._logger.warning(f"     已达最大重试次数，跳过本轮提问")
                        return ""

                # 成功得到非重复问题
                if reason:
                    self._logger.debug(f"  💡 问题目的: {reason}")
                if duplicate_check:
                    self._logger.debug(f"  ✓ 重复检查: {duplicate_check}")
                return question

            except Exception as e:
                self._logger.error(f"  ❌ 生成问题时出错: {e}")
                return ""

        return ""
    
    def _should_stop_early(self) -> bool:
        """Python层面判断核心维度覆盖是否充分，提前终止问诊以避免凑问题。

        检查7个关键维度中已覆盖的数量（≥5个且已问≥5问时返回True）。
        仅用于后续问诊（首次问诊不受此限制）。
        """
        if len(self.questions_asked) < 3:
            return False

        all_text = " ".join([
            str(self.collected_info.get("chief_complaint", "")),
            str(self.collected_info.get("duration", "")),
            json.dumps(self.collected_info.get("symptoms", []), ensure_ascii=False),
            json.dumps(self.collected_info.get("history", {}), ensure_ascii=False),
            " ".join(c.get("answer", "") for c in self.collected_info.get("conversation_history", []))
        ]).lower()

        dimension_keywords = [
            ["症状", "不舒服", "痛", "疼", "胀", "麻", "晕"],           # 主诉/症状
            ["多久", "天", "周", "月", "年", "开始", "时候"],            # 发病时间
            ["程度", "分", "严重", "厉害", "轻", "重"],                  # 症状程度
            ["加重", "缓解", "诱发", "什么情况", "好转", "减轻"],        # 加重/缓解因素
            ["还有", "其他", "伴随", "发烧", "恶心", "头晕", "呕吐"],   # 伴随症状
            ["发烧", "体重", "晕倒", "意识", "呼吸", "黑矇", "大汗"],   # 危险征象
            ["以前", "历史", "既往", "药", "手术", "过敏", "慢性"],     # 既往史
        ]
        covered = sum(1 for kws in dimension_keywords if any(kw in all_text for kw in kws))
        return covered >= 5 and len(self.questions_asked) >= 5

    def _build_topic_coverage_map(self) -> str:
        """生成话题覆盖状态图，帮助LLM直观判断哪些维度已覆盖、哪些仍缺失。"""
        all_text = " ".join([
            str(self.collected_info.get("chief_complaint", "")),
            str(self.collected_info.get("duration", "")),
            json.dumps(self.collected_info.get("symptoms", []), ensure_ascii=False),
            json.dumps(self.collected_info.get("history", {}), ensure_ascii=False),
            " ".join(c.get("answer", "") for c in self.collected_info.get("conversation_history", []))
        ]).lower()

        dimensions = [
            ("主诉/症状描述",     ["症状", "不舒服", "痛", "疼", "胀", "麻", "晕"]),
            ("发病时间/持续时长", ["多久", "天", "周", "月", "开始", "什么时候"]),
            ("症状程度",          ["分", "程度", "严重", "厉害", "轻", "重"]),
            ("加重/缓解因素",     ["加重", "缓解", "诱发", "什么情况", "好转"]),
            ("伴随症状",          ["还有", "其他", "伴随", "发烧", "恶心", "头晕"]),
            ("危险征象",          ["发烧", "体重", "晕倒", "意识", "呼吸困难", "黑矇"]),
            ("既往史/用药史",     ["以前", "历史", "既往", "药物", "手术", "过敏"]),
        ]
        lines = []
        for dim, kws in dimensions:
            covered = any(kw in all_text for kw in kws)
            lines.append(f"  {'✅' if covered else '❌'} {dim}")
        return "\n".join(lines)

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
        
        # LLM模式：智能生成问题
        # 提取异常结果
        abnormal_results = [r for r in test_results if r.get("abnormal")]
        normal_results = [r for r in test_results if not r.get("abnormal")]
        
        # 如果没有异常，可能不需要继续问
        if not abnormal_results:
            return ""
        
        # 构建已问问题列表
        asked_summary = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(self.questions_asked)])
        
        system_prompt = f"""你是一名经验丰富的{self._dept_name()}医生，刚刚收到患者的检查报告。你需要将检查结果与临床表现相互印证。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【患者主诉】
{chief_complaint}

【问诊已收集的信息】
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

【检查结果分析】
🔴 异常项目 ({len(abnormal_results)}项)：
{json.dumps(abnormal_results, ensure_ascii=False, indent=2)}

✅ 正常项目 ({len(normal_results)}项)：
{json.dumps([r.get('test') for r in normal_results], ensure_ascii=False)}

【之前基于检查结果的提问】
{asked_summary if asked_summary else "（尚未基于检查结果提问）"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【临床思维流程】

1️⃣ **异常结果解读**
   - 这个异常指标提示什么病理状态？
   - 与患者主诉是否相符？
   - 是否需要追问相关症状来佐证？

2️⃣ **症状-体征-检查三角印证**
   - 检查异常 → 应该有对应的临床表现
   - 询问患者是否有相关症状但之前未提及
   - 评估症状的严重程度与检查结果是否匹配

3️⃣ **病因追溯**
   - 这个异常是急性还是慢性？
   - 是否有诱发因素或基础疾病？
   - 既往是否有类似异常？

4️⃣ **并发症筛查**
   - 这个异常可能导致什么并发症？
   - 患者是否已出现早期征象？

【决策指南】

✅ **需要追问**（满足以下任一条件）：
- 异常结果显著，但患者未主动提及相关症状
- 异常程度与症状不匹配，需要了解详细情况
- 提示潜在并发症，需要筛查相关表现
- 需要明确病因、诱因或病程进展
- 与主诉相关，有助于诊断或评估严重程度

🛑 **无需追问**（满足以下情况）：
- 异常轻微，临床意义不大
- 与当前主诉无关的偶然发现
- 患者已充分描述过相关症状
- 该信息对当前诊疗决策无实质影响
- 需要进一步检查而非问诊来明确

【输出要求】
生成一个问题，满足：
1. **目的明确**：直接针对检查异常，询问相关症状或病史
2. **简洁易懂**：用患者能理解的语言，避免专业术语
3. **临床价值**：对诊断、治疗或预后评估有实际帮助
4. **不重复**：与之前问题不重复

如果不需要追问，返回空字符串 ""

【优秀提问示例】

场景1：血红蛋白低（贫血）
✅ "最近有没有感觉特别容易累？走路或爬楼梯会气喘吗？"
✅ "大便颜色正常吗？有没有发黑的情况？"

场景2：肝功能异常
✅ "最近有没有发现眼睛或皮肤发黄？"
✅ "肚子有没有胀，能不能吃得下饭？"

场景3：炎症指标高
✅ "有没有发烧？体温最高多少度？"
✅ "身体哪个部位特别不舒服或者肿了？"

❌ 避免泛泛的问题："检查结果有些异常，您最近身体还有其他不适吗？"
"""
        
        user_prompt = '请生成一个问题，输出JSON格式：{"question": "问题内容"} 或 {"question": ""} 表示无需继续提问'
        
        try:
            obj, _, _ = self._llm.generate_json(
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
    
    def generate_clarification_question(
        self,
        diagnosis_info: dict[str, Any],
        collected_info: dict[str, Any]
    ) -> str:
        """基于诊断不确定性生成澄清问题
        
        Args:
            diagnosis_info: 诊断信息，包括current_diagnosis, uncertainty_reason, test_results, rule_out等
            collected_info: 已收集的信息
            
        Returns:
            单个问题字符串，如果不需要继续问则返回空字符串
        """
        
        # 构建已问问题列表
        asked_summary = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(self.questions_asked)])
        
        current_diagnosis = diagnosis_info.get("current_diagnosis", "未明确")
        uncertainty_reason = diagnosis_info.get("uncertainty_reason", "")
        test_results_summary = diagnosis_info.get("test_results", [])
        rule_out_list = diagnosis_info.get("rule_out", [])
        
        system_prompt = f"""你是一名经验丰富的{self._dept_name()}医生，当前面临诊断不确定的情况，需要通过补充问诊来明确诊断。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前诊断】
{current_diagnosis}

【不确定原因】
{uncertainty_reason}

【检查结果摘要】
{json.dumps(test_results_summary, ensure_ascii=False, indent=2)}

【需要鉴别的诊断】
{json.dumps(rule_out_list, ensure_ascii=False, indent=2)}

【已收集的信息】
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

【之前的提问】
{asked_summary if asked_summary else "（尚未提问）"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【临床思维流程】

1️⃣ **诊断不确定性分析**
   - 为什么当前诊断不确定？缺少哪些关键信息？
   - 有哪些鉴别诊断需要排除？
   - 哪些症状、体征或病史能够帮助鉴别？

2️⃣ **鉴别诊断的关键点**
   - A诊断和B诊断的主要区别是什么？
   - 有哪些特征性症状或病史可以帮助鉴别？
   - 患者是否有未提及但对诊断有价值的信息？

3️⃣ **病程和伴随症状**
   - 症状的时间顺序和演变过程
   - 是否有伴随症状或诱发因素
   - 既往类似发作的情况

4️⃣ **治疗和用药史**
   - 之前是否接受过治疗？效果如何？
   - 是否服用过相关药物？
   - 有无过敏史或禁忌症？

【决策指南】

✅ **需要追问**（满足以下任一条件）：
- 有助于在多个鉴别诊断中明确主要诊断
- 能够排除重要的鉴别诊断
- 补充关键的病史、症状细节或伴随症状
- 了解治疗史或用药史，有助于评估病情
- 与当前诊断的不确定性直接相关

🛑 **无需追问**（满足以下情况）：
- 已有足够信息支持当前诊断
- 鉴别诊断已基本排除
- 该信息对诊断决策无实质影响
- 需要进一步检查而非问诊来明确
- 与之前问题重复

【输出要求】
生成一个问题，满足：
1. **针对性强**：直接针对诊断不确定性或鉴别诊断
2. **简洁易懂**：用患者能理解的语言
3. **临床价值**：对明确诊断有直接帮助
4. **不重复**：与之前问题不重复

如果不需要追问，返回空字符串 ""

【优秀提问示例】

场景1：鉴别紧张性头痛 vs 偏头痛
✅ "头痛的时候，是整个头都疼，还是一侧疼得更明显？"
✅ "头疼的时候，有没有恶心想吐的感觉？"
✅ "光线或声音会让头疼加重吗？"

场景2：鉴别病毒性感染 vs 细菌性感染
✅ "咳出来的痰是什么颜色的？是清的还是黄绿色的？"
✅ "鼻涕是清水样的还是浓稠的？"

场景3：鉴别功能性 vs 器质性疾病
✅ "这个症状是最近才出现的，还是很多年了？"
✅ "之前做过什么检查吗？结果怎么样？"
✅ "有没有治疗过？用了什么药？效果如何？"

❌ 避免泛泛的问题："您还有其他症状吗？"
"""
        
        user_prompt = '请生成一个问题，输出JSON格式：{"question": "问题内容"} 或 {"question": ""} 表示无需继续提问'
        
        try:
            obj, _, _ = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {"question": ""},
                temperature=0.3
            )
            question = str(obj.get("question", ""))
            
            # 检查是否与之前的问题重复
            if question and self._is_duplicate_question(question):
                return ""
            
            return question
        except Exception:
            return ""
    
    def _is_duplicate_question(self, new_question: str) -> bool:
        """检测新问题是否与已问问题重复
        
        Args:
            new_question: 新生成的问题
            
        Returns:
            True表示重复，False表示不重复
        """
        if not new_question or not self.questions_asked:
            return False
        
        new_q = new_question.lower().strip("？?。. ")
        
        for asked_q in self.questions_asked:
            asked_q_normalized = asked_q.lower().strip("？?。. ")
            
            # 1. 完全相同
            if new_q == asked_q_normalized:
                return True
            
            # 2. 高度相似（关键词重叠度检测）
            # 提取核心关键词（去除常见助词和标点）
            stop_words = {"的", "了", "吗", "呢", "啊", "您", "你", "有没有", "是不是", "还", "也", "都", "和", "或者", "以及"}
            
            def extract_keywords(text: str) -> set:
                """提取关键词"""
                import re
                # 移除标点和空格
                text = re.sub(r'[^\w]', ' ', text)
                words = text.split()
                # 过滤停用词和单字词（除了一些重要的单字）
                important_single = {"痛", "疼", "麻", "红", "肿", "热", "冷", "晕", "吐", "泻"}
                keywords = {w for w in words if w not in stop_words and (len(w) > 1 or w in important_single)}
                return keywords
            
            new_keywords = extract_keywords(new_q)
            asked_keywords = extract_keywords(asked_q_normalized)
            
            if not new_keywords or not asked_keywords:
                continue
            
            # 计算关键词重叠度
            overlap = new_keywords & asked_keywords
            overlap_ratio = len(overlap) / min(len(new_keywords), len(asked_keywords))
            
            # 如果重叠度超过55%，认为是重复问题（阈值从0.7降至0.55，减少漏网）
            if overlap_ratio > 0.55:
                return True
        
        return False
    
    def process_patient_answer(self, question: str, answer: str) -> None:
        """处理患者的回答，更新收集的信息"""
        self.questions_asked.append(question)
        
        # 如果是第一个问题且询问主诉，提取主诉
        if len(self.questions_asked) == 1 and any(keyword in question for keyword in ["哪里不舒服", "什么症状", "怎么了", "主诉"]):
            # 简单提取：将第一个回答作为初步主诉
            self.collected_info["chief_complaint"] = answer[:100]  # 限制长度
        
        # 如果有LLM，使用LLM提取结构化信息
        if self._llm is not None:
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
                
                obj, _, _ = self._llm.generate_json(
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
                pass
    
    def assess_interview_quality(self) -> dict[str, Any]:
        """评估当前问诊质量，提供改进建议
        
        Returns:
            dict: 包含质量评分和建议
        """
        quality_report = {
            "completeness_score": 0,  # 0-100
            "depth_score": 0,  # 0-100
            "efficiency_score": 0,  # 0-100
            "overall_score": 0,  # 0-100
            "missing_areas": [],  # 缺失的关键信息
            "suggestions": [],  # 改进建议
            "warning": None  # 警告信息
        }
        
        # 1. 完整性评估（40分）
        completeness = 0
        
        # 检查主诉（必须有，10分）
        if self.collected_info.get("chief_complaint"):
            completeness += 10
        
        # 检查持续时间（必须有，15分）
        history = self.collected_info.get("history", {})
        if "duration" in history and history["duration"]:
            completeness += 15
        else:
            quality_report["missing_areas"].append("症状持续时间")
        
        # 检查严重程度（必须有，15分）
        if "severity" in history and history["severity"]:
            completeness += 15
        else:
            quality_report["missing_areas"].append("症状严重程度")
        
        # 检查伴随症状（10分）
        if self.collected_info.get("symptoms") or history.get("associated_symptoms"):
            completeness += 10
        
        quality_report["completeness_score"] = min(100, completeness * 2)
        
        # 2. 深度评估（30分）
        depth = 0
        
        # 检查症状量化
        quantified_keywords = ["几分", "1-10", "评分", "多少", "频率", "次数"]
        quantified_questions = sum(1 for q in self.questions_asked 
                                  if any(kw in q for kw in quantified_keywords))
        if quantified_questions > 0:
            depth += 10
        
        # 检查鉴别诊断相关问题
        differential_keywords = ["加重", "缓解", "吃饭", "运动", "姿势", "时间", "扩散"]
        differential_questions = sum(1 for q in self.questions_asked 
                                    if any(kw in q for kw in differential_keywords))
        if differential_questions >= 2:
            depth += 10
        
        # 检查危险征象筛查
        red_flag_keywords = ["发烧", "发热", "体重", "意识", "呼吸", "胸痛", "血"]
        red_flag_questions = sum(1 for q in self.questions_asked 
                                if any(kw in q for kw in red_flag_keywords))
        if red_flag_questions > 0:
            depth += 10
        
        quality_report["depth_score"] = min(100, depth * 3.3)
        
        # 3. 效率评估（30分）
        if len(self.questions_asked) == 0:
            efficiency = 0
        else:
            # 避免重复问题
            unique_ratio = len(set(self.questions_asked)) / len(self.questions_asked)
            efficiency = int(unique_ratio * 30)
            
            # 惩罚过于泛泛的问题
            vague_keywords = ["还有什么", "其他症状", "不舒服", "怎么样"]
            vague_count = sum(1 for q in self.questions_asked 
                             if any(kw in q for kw in vague_keywords))
            if vague_count > 0:
                efficiency -= vague_count * 5
        
        quality_report["efficiency_score"] = max(0, min(100, efficiency * 3.3))
        
        # 综合评分
        quality_report["overall_score"] = int(
            quality_report["completeness_score"] * 0.4 +
            quality_report["depth_score"] * 0.3 +
            quality_report["efficiency_score"] * 0.3
        )
        
        # 识别缺失的关键信息（避免重复添加已在completeness检查中添加的）
        # 持续时间和严重程度在completeness评估时已经添加到missing_areas
        
        # 检查危险征象筛查
        if not red_flag_questions:
            if "危险征象筛查" not in quality_report["missing_areas"]:
                quality_report["missing_areas"].append("危险征象筛查")
            quality_report["suggestions"].append("建议筛查红旗症状：'最近有没有发烧、体重下降、夜间盗汗？'")
        
        if not differential_questions:
            quality_report["missing_areas"].append("鉴别诊断相关信息")
            quality_report["suggestions"].append("建议询问加重/缓解因素：'什么情况下症状会加重或缓解？'")
        
        # 检查暴露史（新增）
        exposure_keywords = ["吸烟", "抽烟", "烟", "酒", "喝酒", "饮酒", "职业", "工作", "接触", 
                            "化学", "粉尘", "辐射", "装修", "宠物", "药物", "保健品", "电子烟"]
        has_exposure = any(q for q in self.questions_asked 
                          if any(kw in q for kw in exposure_keywords))
        if not has_exposure:
            quality_report["missing_areas"].append("暴露史（吸烟/饮酒/职业/环境/药物等,常是漏诊线索）")
            quality_report["suggestions"].append("建议询问暴露史：'您平时吸烟或使用电子烟吗？有没有特殊的职业接触？'")
        
        # 警告（不直接显示，由调用方记录到detail_logger）
        if quality_report["overall_score"] < 50:
            quality_report["warning"] = f"问诊质量偏低（{quality_report['overall_score']}/100），建议补充关键信息"
        elif quality_report["overall_score"] < 70:
            quality_report["warning"] = f"问诊质量中等（{quality_report['overall_score']}/100），可进一步完善"
        else:
            quality_report["warning"] = f"问诊质量良好（{quality_report['overall_score']}/100）"
        
        return quality_report
    
    def decide_tests(self) -> list[dict[str, Any]]:
        """根据收集的信息决定需要做哪些检查"""
        # 检索检查指南
        chunks = self._retriever.retrieve(
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
            obj, _, _ = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2
            )
            return obj.get("ordered_tests", [])
        except Exception as e:
            self._logger.error(f"❌ 检查开单失败: {e}")
            return []
    
    def make_diagnosis(self, test_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """综合分析后做出诊断"""
        # 检索诊疗指南
        chunks = self._retriever.retrieve(
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

【诊断要求 - 临床推理框架】

⚠️ **强制要求**：必须完成以下临床推理步骤，否则诊断质量不合格

🔍 **步骤1：证据收集与整理**
- 列出所有支持性证据（症状、体征、检查结果）
- 列出所有矛盾性证据（不支持当前诊断的信息）
- 评估证据的可靠性和权重

🧠 **步骤2：鉴别诊断思维**（必须完成！）
- **至少列出3个鉴别诊断**，包括：
  * 最可能的诊断（主诊断）
  * 需要排除的严重疾病（如肿瘤、急症）
  * 最常见的相似疾病
- ⚠️ **避免思维定势 - 多系统考虑**：
  * 胸痛：不仅考虑心脏，还需考虑肺、食管、胸壁、心理
  * 腹痛：不仅考虑胃肠，还需考虑泌尿、妇科、血管、代谢
  * 头痛：不仅考虑神经，还需考虑眼科、耳鼻喉、血管、中毒
  * **检查暴露史**：职业病、中毒、药物不良反应常被漏诊
- 对每个鉴别诊断，说明：
  * 支持该诊断的证据
  * 不支持该诊断的证据
  * 如何通过进一步检查鉴别

⚖️ **步骤3：诊断推理**
- 解释为什么选择某个诊断作为主诊断
- 说明排除其他诊断的理由
- 评估诊断的确定性（高/中/低）
- 指出诊断中的不确定因素

📋 **步骤4：输出诊断**
1. **主诊断**：一个最确定的诊断（简洁明确，≤20字）
2. **鉴别诊断**：2-3个需要排除的疾病（明确列出）
3. **诊断依据**：列举关键证据（症状+检查）
4. **不确定因素**：列出诊断中的疑点或需要进一步明确的问题
5. **进一步检查建议**：如果诊断不确定，建议哪些检查可以明确

🚫 **禁止**：
- 避免使用"疑似"、"可能"、"待查"等模糊表述（除非确实无法确定）
- 不要只给一个诊断不考虑鉴别
- 不要忽略矛盾的证据
- 不要给出过度具体的诊断（如果证据不足）

【任务】
综合分析并给出诊断和治疗建议。
"""
        
        user_prompt = '''输出JSON格式（必须包含完整的鉴别诊断推理）：
{
  "diagnosis": {
    "name": "一个明确的主要诊断名称",
    "confidence": "high/medium/low",
    "evidence": ["支持证据1", "支持证据2", "支持证据3"],
    "differential": [
      {"disease": "鉴别诊断1", "support": "支持依据", "against": "排除理由"},
      {"disease": "鉴别诊断2", "support": "支持依据", "against": "排除理由"},
      {"disease": "鉴别诊断3", "support": "支持依据", "against": "排除理由"}
    ],
    "reasoning": "为什么选择主诊断的推理过程（200-300字）",
    "uncertainty": "诊断中的不确定因素",
    "further_tests": ["建议的进一步检查"]
  },
  "treatment_plan": {
    "medications": ["用药1", "用药2"],
    "lifestyle": ["生活建议1"],
    "followup": "随访计划"
  }
}'''
        
        try:
            obj, _, _ = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2
            )
            return obj
        except Exception as e:
            self._logger.error(f"❌ 诊断失败: {e}")
            return {
                "diagnosis": {"name": "诊断失败", "confidence": "low", "differential": []},
                "treatment_plan": {"medications": [], "lifestyle": [], "followup": ""}
            }
    
    def _dept_name(self) -> str:
        """根据科室代码返回科室中文名"""
        dept_names = {
            "neurology": "神经医学",
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
        conversation_history = self.collected_info.get("conversation_history", [])
        return {
            "questions_count": len(self.questions_asked),
            "qa_pairs": [
                {"question": conv.get("question", ""), "answer": conv.get("answer", "")}
                for conv in conversation_history
            ],
            "collected_info": self.collected_info
        }
    
    def summarize_chief_complaint(self) -> str:
        """从问诊中总结患者主诉
        
        医生通过问诊和患者描述，总结出简洁的主诉
        
        Returns:
            总结的主诉字符串
        """
        conversation_history = self.collected_info.get("conversation_history", [])
        if not self.questions_asked or not conversation_history:
            return "患者主诉不明"
        
        # LLM模式：智能总结（只使用前5轮）
        qa_text = "\n".join([
            f"医生：{conv.get('question', '')}\n患者：{conv.get('answer', '')}"
            for conv in conversation_history[:5]
        ])
        
        system_prompt = f"""你是一名{self._dept_name()}医生，需要根据问诊记录总结患者的主诉。

【问诊记录】
{qa_text}

【任务】
总结患者的核心主诉（chief complaint），要求：
1. 简洁明确，一般10-30字
2. 包含主要症状、部位、时间
3. 使用医学术语，但保持可读性
4. 格式示例："反复上腹痛3天" "头痛伴恶心呕吐2周" "发热咳嗽5天"

不要包含检查结果、诊断推测，只描述患者的主观症状。
"""
        
        user_prompt = '请总结主诉，输出JSON格式：{"chief_complaint": "总结的主诉"}'
        
        try:
            obj, _, _ = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1
            )
            return obj.get("chief_complaint", "患者主诉不明")
        except Exception:
            # LLM失败时使用第一个回答
            first_answer = conversation_history[0].get("answer", "") if conversation_history else ""
            return first_answer[:50] if first_answer else "患者主诉不明"
