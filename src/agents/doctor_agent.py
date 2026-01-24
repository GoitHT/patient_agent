"""医生智能体：基于RAG知识库进行问诊、开单、诊断"""
from __future__ import annotations

import json
from typing import Any

from rag import ChromaRetriever
from services.llm_client import LLMClient
from utils import get_logger


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
        self._retriever = retriever
        self._llm = llm
        self._max_questions = max_questions
        self._logger = get_logger("hospital_agent.doctor")
        self.collected_info: dict[str, Any] = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "history": {},
            "exam_findings": {}
        }
        self.questions_asked: list[str] = []
        self._patient_answers: list[str] = []
    
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
        self._patient_answers = []
    
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
        
        if self._llm is None:
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
            chunks = self._retriever.retrieve(
                f"{self.dept} {context} 问诊要点",
                filters={"dept": self.dept},
                k=3
            )
            kb_context = self._format_chunks(chunks)
        
        # 构建已问问题列表 - 完整展示，用于避免重复
        if self.questions_asked:
            asked_list = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(self.questions_asked)])
            asked_summary = f"已问过的问题（共{len(self.questions_asked)}个）：\n{asked_list}"
        else:
            asked_summary = "（尚未开始问诊）"
        
        answers_summary = "\n".join([f"A{i+1}: {a[:50]}..." for i, a in enumerate(self._patient_answers)])
        
        system_prompt = f"""你是一名经验丰富的{self._dept_name()}医生，正在进行{context}。你需要通过系统的问诊收集关键信息，做出准确诊断。

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

【患者回答摘要】
{answers_summary if answers_summary else "（尚无回答）"}

【临床知识参考】
{kb_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 **重复问题检测 - 强制要求**
在生成新问题前，你必须：
1. 仔细阅读上方【问诊历史】中的所有已问问题
2. 确保新问题与已问问题在内容和意图上都不重复
3. 如果某个方面已经问过，应该基于已有答案深入追问，而不是换个说法再问一遍

❌ 重复问题示例：
- 已问"是否扩散到其他部位？" → 不要再问"有没有扩散现象？"
- 已问"接触什么会加重？" → 不要再问"有什么因素会诱发或加重吗？"
- 已问"疼痛程度如何？" → 不要再问"能给疼痛打个分吗？"

✅ 正确做法：
- 已问"是否扩散？"患者答"有" → 可以问"具体扩散到哪些部位？"
- 已问"接触水加重"患者确认 → 可以问"除了水，其他液体如肥皂水会怎样？"

【问诊策略 - 按优先级顺序】

1️⃣ **首要信息**（必问，诊断基础）
   - 症状特点：性质、部位、程度（要求量化：1-10分评分）
   - 时间信息：何时开始、持续时间、发作频率（具体到小时/天/周）
   - 伴随症状：是否有其他不适（系统性询问）

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
⚠️ **重要原则**：问诊的目标是获取足够做出临床决策的信息，而不是问满固定轮数。信息足够时应主动停止，避免过度问诊。

分析当前问诊进度，决定下一步行动：

🛑 **应该停止提问**（优先判断，满足以下任一条件即可停止）：
1. **诊断清晰度**：已收集足够信息可以做出初步诊断或制定检查计划
2. **核心信息完整**（问诊完整性检查）：
   - ✅ 主要症状已量化（严重度用1-10评分、持续时间具体到天/周）
   - ✅ 危险征象已排查（发热、体重下降、意识改变、急性恶化等）
   - ✅ 鉴别诊断关键信息已获取（至少2-3个鉴别点）
   - ✅ 时间轴清晰（起病时间、进展趋势）
   - ✅ 加重/缓解因素明确
3. **检查优先**：进一步明确需要依赖检查而非问诊（如需要影像、化验）
4. **患者状态**：患者表现痛苦、疲劳或回答质量下降，应尽快转入诊疗
5. **边际效益递减**：继续提问获得的新信息价值不大
6. **已达上限**：问诊次数接近或达到上限（{len(self.questions_asked)}/{self._max_questions}）

✅ **继续提问**（仅当以下关键信息缺失时）：
- 🔴 **必问项缺失**（诊断基础）：
  * 症状性质/部位/程度未明确
  * 持续时间不详（回答"最近"等模糊描述）
  * 症状未量化（无1-10评分）
- 🔴 **危险征象未排查**：存在可疑的红旗症状需要确认
- 🔴 **鉴别诊断关键点缺失**：需要特定信息来区分不同疾病
- 💡 **重要线索待挖掘**：患者提到但未详述的异常表现

⚖️ **判断标准**：
- 如果核心信息已获取，即使未达到最大轮数，也应停止
- 如果患者回答已经足够支持临床决策，不要为了问而问
- 优先考虑"这个问题对诊断/治疗有实质帮助吗？"而非"还能问什么"

【输出要求】
1. **首先判断是否应该停止**：仔细评估上述"停止提问"的条件，如果满足任一条件，直接返回空字符串 ""
2. **如果确实需要继续**：
   - 生成一个简洁、口语化的问题（避免医学术语，患者容易理解）
   - 问题要有明确的临床目的，能填补关键信息缺口
   - 避免开放式泛问（如"还有什么不舒服"）
   - 一次只问一个问题，不要多个问题组合
   - **必须确保新问题与已问问题完全不重复**
3. **质量优先于数量**：宁可问5个高质量问题得到足够信息，也不要机械地问满10个问题

【良好问题示例】

✅ **量化追问**：
  "如果用1-10分给疼痛打分，10分是最痛，现在是几分？"
  "一天发作几次？每次持续多长时间？"
  
✅ **深度挖掘**：
  "疼痛是钝痛、刺痛、还是绞痛？能描述一下具体的感觉吗？"
  "疼痛会扩散到其他地方吗？比如后背、肩膀？"
  
✅ **鉴别关键**：
  "吃饭后疼痛会加重还是缓解？空腹时怎么样？"
  "躺下时疼痛会加重吗？弯腰或咳嗽时呢？"
  
✅ **危险筛查**：
  "最近有没有发烧？体重有没有下降？"
  "有没有觉得心慌、气短、或者晕倒过？"
  
✅ **病史关联**：
  "以前有过类似的情况吗？当时怎么处理的？"
  "有没有高血压、糖尿病这些慢性病？"

❌ 避免："您还有什么其他症状吗？"（太泛，无针对性）
❌ 避免："请描述一下您的疼痛性质、部位和持续时间。"（一次问太多）
❌ 避免：与已问问题意思相同或高度相似的问题
❌ 避免："最近怎么样？"（过于开放，缺乏目的性）
"""
        
        user_prompt = """请分析当前问诊情况并决定下一步行动。

**决策流程**（必须按顺序执行）：
1. **评估信息完整性**：当前已收集的信息是否足够做出临床决策？
2. **检查停止条件**：是否满足任何一个"应该停止提问"的条件？
3. **如果满足停止条件**：返回空字符串，理由说明为什么信息已足够
4. **如果需要继续**：生成一个针对关键信息缺口的高质量问题

**重要提醒**：
- 不要为了达到问诊轮数而机械提问
- 信息足够时应主动停止，体现专业的临床判断能力
- 先检查【问诊历史】，确保不重复已问过的问题
- 每个问题都应该有明确的诊断价值

输出JSON格式：
{{
  "question": "问题内容（如果信息足够或应停止则为空字符串）",
  "reason": "决策理由（为什么继续问/为什么停止）"
}}"""
        
        # 尝试最多2次，避免重复问题
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                obj, _, _ = self._llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=lambda: {"question": "", "reason": ""},
                    temperature=0.3 + (attempt * 0.1)  # 第二次尝试时稍微提高温度
                )
                question = str(obj.get("question", "")).strip()
                reason = str(obj.get("reason", "")).strip()
                
                # 如果为空或者不重复，直接返回
                if not question:
                    return ""
                
                # 检查是否重复（作为最后的安全网）
                if not self._is_duplicate_question(question):
                    if reason and attempt == 0:
                        self._logger.debug(f"  💡 问题目的: {reason}")
                    return question
                
                # 如果是重复的，且还有尝试次数，继续下一轮
                if attempt < max_attempts - 1:
                    self._logger.warning(f"  ⚠️  LLM生成了重复问题（尝试{attempt+1}/{max_attempts}），重新生成...")
                    # 在system_prompt中添加额外提醒
                    system_prompt += f"\n\n⚠️ 注意：刚才生成的问题「{question}」与已问问题重复，请生成完全不同的问题！"
                else:
                    # 最后一次尝试仍然重复，返回空
                    self._logger.warning(f"  ❌ 多次尝试后仍生成重复问题，跳过本轮提问")
                    return ""
                    
            except Exception as e:
                self._logger.error(f"  ❌ 生成问题时出错: {e}")
                # 发生异常时返回规则问题
                all_questions = self._rule_based_questions(chief_complaint)
                remaining = [q for q in all_questions if q not in self.questions_asked]
                return remaining[0] if remaining else ""
        
        return ""  # 安全返回
    
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
        
        if self._llm is None:
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
            
            # 如果重叠度超过70%，认为是重复问题
            if overlap_ratio > 0.7:
                return True
        
        return False
    
    def process_patient_answer(self, question: str, answer: str) -> None:
        """处理患者的回答，更新收集的信息"""
        self.questions_asked.append(question)
        self._patient_answers.append(answer)
        
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
        required_keys = ["chief_complaint", "duration", "severity"]
        for key in required_keys:
            if key == "chief_complaint":
                if self.collected_info.get(key):
                    completeness += 10
            elif key in self.collected_info.get("history", {}):
                completeness += 15
        
        # 检查是否有伴随症状
        if self.collected_info.get("symptoms") or self.collected_info.get("history", {}).get("associated_symptoms"):
            completeness += 10
        
        quality_report["completeness_score"] = min(100, completeness * 2.5)
        
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
        
        # 识别缺失的关键信息
        if "duration" not in self.collected_info.get("history", {}):
            quality_report["missing_areas"].append("症状持续时间（具体到天/周/月）")
            quality_report["suggestions"].append("建议询问：'这个症状有多久了？具体从什么时候开始的？'")
        
        if "severity" not in self.collected_info.get("history", {}):
            quality_report["missing_areas"].append("症状严重程度量化")
            quality_report["suggestions"].append("建议量化询问：'如果用1-10分评价，10分最严重，现在是几分？'")
        
        if not red_flag_questions:
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
        
        # 警告
        if quality_report["overall_score"] < 50:
            quality_report["warning"] = f"⚠️  问诊质量偏低（{quality_report['overall_score']}/100），建议补充关键信息"
        elif quality_report["overall_score"] < 70:
            quality_report["warning"] = f"💡 问诊质量中等（{quality_report['overall_score']}/100），可进一步完善"
        else:
            quality_report["warning"] = f"✅ 问诊质量良好（{quality_report['overall_score']}/100）"
        
        return quality_report
    
    def decide_tests(self) -> list[dict[str, Any]]:
        """根据收集的信息决定需要做哪些检查"""
        if self._llm is None:
            return self._rule_based_tests()
        
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
        if self._llm is None:
            return self._rule_based_diagnosis()
        
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
                for q, a in zip(self.questions_asked, self._patient_answers)
            ],
            "collected_info": self.collected_info
        }
    
    def summarize_chief_complaint(self) -> str:
        """从问诊中总结患者主诉
        
        医生通过问诊和患者描述，总结出简洁的主诉
        
        Returns:
            总结的主诉字符串
        """
        if not self.questions_asked or not self._patient_answers:
            return "患者主诉不明"
        
        if self._llm is None:
            # 规则模式：使用第一个回答作为主诉
            first_answer = self._patient_answers[0] if self._patient_answers else "不适"
            return first_answer[:50]  # 限制长度
        
        # LLM模式：智能总结
        qa_text = "\n".join([
            f"医生：{q}\n患者：{a}"
            for q, a in zip(self.questions_asked[:5], self._patient_answers[:5])  # 只使用前5轮
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
                fallback=lambda: {"chief_complaint": self._patient_answers[0][:50] if self._patient_answers else "不适"},
                temperature=0.1
            )
            return obj.get("chief_complaint", "患者主诉不明")
        except Exception:
            # LLM失败时使用第一个回答
            return self._patient_answers[0][:50] if self._patient_answers else "患者主诉不明"
