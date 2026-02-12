"""RAG 检索关键词生成器

严格遵循规则：
1. 只生成关键词，不进行医学推理
2. 根据节点功能和患者上下文生成针对性关键词
3. 区分不同知识库的检索关键词
4. 避免跨节点检索和泛化搜索

知识库查询标准：

【医学指南库】
根据患者主诉、关键症状及当前流程节点目标，
检索诊断标准、治疗方案、决策依据或是否需要辅助检查的权威指南内容。

【高质量对话库】
根据患者主诉及当前问诊或解释阶段，
检索结构化医患问诊、检查结果解读及沟通示例对话。

【临床案例库】
根据患者主诉、关键症状及已完成检查，
检索典型病例、结果解读案例及诊疗决策路径相关内容。

【规则流程库】
根据当前流程节点类型，
检索标准操作规程、检查准备事项、医疗文书模板及宣教与随访流程。
"""
from __future__ import annotations
from typing import Any, Literal
from dataclasses import dataclass


@dataclass
class NodeContext:
    """节点上下文信息"""
    node_id: str  # 节点编号（如 C5, S4）
    node_name: str  # 节点名称（如 "准备问诊"）
    dept: str | None = None  # 科室代码
    dept_name: str | None = None  # 科室名称
    chief_complaint: str | None = None  # 主诉
    patient_age: int | None = None  # 年龄
    patient_gender: str | None = None  # 性别
    ordered_tests: list[dict] | None = None  # 已开检查
    test_results: list[dict] | None = None  # 检查结果
    preliminary_diagnosis: str | None = None  # 初步诊断
    specialty_summary: dict | None = None  # 专科小结


class RAGKeywordGenerator:
    """RAG 检索关键词生成器
    
    核心原则：
    - 基于节点功能生成针对性关键词
    - 充分利用患者上下文信息
    - 区分不同知识库的检索意图
    - 避免医学推理和诊断结论
    
    知识库查询标准：
    1. 医学指南库：患者主诉+关键症状+节点目标 → 诊断标准、治疗方案、决策依据、辅助检查指导
    2. 高质量对话库：患者主诉+问诊/解释阶段 → 结构化问诊、检查结果解读、沟通示例
    3. 临床案例库：患者主诉+关键症状+已完成检查 → 典型病例、结果解读案例、诊疗决策路径
    4. 规则流程库：流程节点类型 → 标准操作规程、检查准备事项、文书模板、宣教随访流程
    """
    
    # 节点检索目标映射
    NODE_RETRIEVAL_GOALS = {
        # 通用流程节点
        "C5": {
            "goal": "获取通用就诊流程标准操作规程",
            "dbs": ["HospitalProcess_db"],
        },
        "C8": {
            "goal": "获取检查准备事项、缴费流程、历史检查记录",
            "dbs": ["HospitalProcess_db", "UserHistory_db"],
        },
        "C11": {
            "goal": "获取问诊对话参考、相似病例的检查结果解读",
            "dbs": ["HighQualityQA_db", "ClinicalCase_db"],
        },
        "C12": {
            "goal": "获取诊断指南、治疗方案、相似临床案例",
            "dbs": ["MedicalGuide_db", "ClinicalCase_db"],
        },
        "C14": {
            "goal": "获取文书模板、患者历史病历信息",
            "dbs": ["HospitalProcess_db", "UserHistory_db"],
        },
        "C15": {
            "goal": "获取健康宣教材料、随访计划模板",
            "dbs": ["HospitalProcess_db"],
        },
        # 专科子图节点
        "S4": {
            "goal": "获取专科知识、问诊参考、相似症状案例",
            "dbs": ["MedicalGuide_db", "HighQualityQA_db", "ClinicalCase_db"],
        },
        "S6": {
            "goal": "获取诊断标准、鉴别诊断依据、辅助检查建议",
            "dbs": ["MedicalGuide_db"],
        },
    }
    
    def generate_keywords(
        self, 
        ctx: NodeContext,
        db_name: Literal[
            "HospitalProcess_db",  # 规则流程库
            "MedicalGuide_db",     # 医学指南库
            "ClinicalCase_db",      # 临床案例库
            "HighQualityQA_db",     # 高质量对话库
            "UserHistory_db",       # 患者对话历史库
        ]
    ) -> str:
        """生成指定知识库的检索关键词
        
        Args:
            ctx: 节点上下文信息
            db_name: 目标知识库名称
            
        Returns:
            检索关键词字符串（可直接用于检索）
        """
        # 根据节点和知识库生成关键词
        method_name = f"_generate_{ctx.node_id.lower()}_{self._simplify_db_name(db_name)}"
        
        # 尝试调用专门的生成方法
        if hasattr(self, method_name):
            return getattr(self, method_name)(ctx)
        
        # 备用：通用生成逻辑
        return self._generate_generic(ctx, db_name)
    
    def _simplify_db_name(self, db_name: str) -> str:
        """简化知识库名称用于方法命名"""
        mapping = {
            "HospitalProcess_db": "process",
            "MedicalGuide_db": "guide",
            "ClinicalCase_db": "case",
            "HighQualityQA_db": "qa",
            "UserHistory_db": "history",
        }
        return mapping.get(db_name, "generic")

    def _extract_symptom_keywords(self, text: str, max_terms: int = 6) -> list[str]:
        """从主诉文本中提取症状关键词（轻量规则，避免推理）"""
        if not text:
            return []

        # 常见无意义词
        stop_words = {
            "患者", "病人", "家属", "妈妈", "母亲", "父亲", "孩子", "本人",
            "不舒服", "不适", "感觉", "觉得", "出现", "发现", "最近", "今天", "昨天",
            "一直", "一个", "有点", "有些", "没有", "不", "可以", "需要", "请问",
            "怎么办", "怎么", "如何", "情况", "症状", "问题", "能否", "是否",
        }

        # 按常见标点切分
        for ch in [";", "；", "，", ",", "。", ".", "、", "\n"]:
            text = text.replace(ch, " ")

        parts = [p.strip() for p in text.split(" ") if p.strip()]
        keywords: list[str] = []
        for part in parts:
            if part in stop_words:
                continue
            if len(part) < 2:
                continue
            keywords.append(part)
            if len(keywords) >= max_terms:
                break
        return keywords
    
    # ==================== C5: 准备问诊 ====================
    
    def _generate_c5_process(self, ctx: NodeContext) -> str:
        """C5 - 规则流程库：标准操作规程
        
        遵循标准：根据当前流程节点类型（准备问诊），
        检索标准操作规程、就诊流程指引。
        
        输入：流程节点类型 + 科室
        输出：门诊就诊流程、问诊接诊SOP
        """
        keywords = []
        
        # 1. 科室（如果有）
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 2. 节点类型关键词：标准操作规程、就诊流程
        keywords.extend(["门诊", "就诊流程", "问诊", "接诊", "标准操作规程", "SOP"])
        
        return " ".join(keywords)
    
    # ==================== C8: 开单与准备说明 ====================
    
    def _generate_c8_process(self, ctx: NodeContext) -> str:
        """C8 - 规则流程库：检查准备事项
        
        遵循标准：根据当前流程节点类型（开单与准备说明），
        检索检查准备事项、缴费预约流程。
        
        输入：流程节点类型 + 已开检查项目
        输出：检查准备注意事项、缴费预约流程
        """
        keywords = []
        
        # 1. 基于已开检查项目（必须）
        if ctx.ordered_tests:
            for test in ctx.ordered_tests[:3]:
                test_name = test.get("name", "")
                test_type = test.get("type", "")
                if test_name:
                    keywords.append(test_name)
                    if test_type in ["endoscopy", "imaging"]:
                        keywords.extend(["禁食", "准备事项"])
        
        # 2. 节点类型关键词：检查准备事项
        keywords.extend(["检查准备事项", "注意事项", "缴费流程", "预约流程"])
        
        return " ".join(keywords)
    
    def _generate_c8_history(self, ctx: NodeContext) -> str:
        """C8 - 患者历史库：历史检查记录"""
        keywords = []
        
        # 必须基于已开检查项目生成关键词
        if ctx.ordered_tests:
            for test in ctx.ordered_tests[:3]:
                test_name = test.get("name", "")
                if test_name:
                    keywords.append(test_name)
            keywords.append("历史检查")
        
        return " ".join(keywords)
    
    # ==================== C11: 报告回诊 ====================
    
    def _generate_c11_qa(self, ctx: NodeContext) -> str:
        """C11 - 高质量对话库：检查结果解读及沟通示例
        
        遵循标准：根据患者主诉及当前解释阶段（报告回诊），
        检索检查结果解读及沟通示例对话。
        
        输入：患者主诉 + 检查结果（特别是异常结果）
        输出：检查结果解读对话、医患沟通示例
        """
        keywords = []
        
        # 1. 患者主诉
        if ctx.chief_complaint:
            keywords.append(ctx.chief_complaint[:20])
        
        # 2. 检查项目和异常结果
        if ctx.test_results:
            for result in ctx.test_results[:3]:
                test_name = result.get("test_name", "")
                if test_name:
                    keywords.append(test_name)
                # 异常结果作为解读重点
                if result.get("abnormal"):
                    summary = result.get("summary", "")[:20]
                    if summary:
                        keywords.append(summary)
        
        # 3. 阶段目标关键词：检查结果解读、沟通示例
        keywords.extend(["检查结果解读", "医患沟通示例", "结果说明对话"])
        return " ".join(keywords)
    
    def _generate_c11_case(self, ctx: NodeContext) -> str:
        """C11 - 临床案例库：典型病例、结果解读案例
        
        遵循标准：根据患者主诉、关键症状及已完成检查，
        检索典型病例、结果解读案例。
        
        输入：患者主诉 + 科室 + 已完成检查 + 关键症状（异常结果）
        输出：典型病例、检查结果解读案例
        """
        keywords = []
        
        # 1. 患者主诉
        if ctx.chief_complaint:
            keywords.append(ctx.chief_complaint[:20])
        
        # 2. 科室
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 3. 已完成检查（必须体现）
        if ctx.test_results:
            for result in ctx.test_results[:3]:
                test_name = result.get("test_name", "")
                if test_name:
                    keywords.append(test_name)
                # 关键症状（异常结果）
                if result.get("abnormal"):
                    summary = result.get("summary", "")[:15]
                    if summary:
                        keywords.append(summary)
        
        # 4. 目标关键词：典型病例、结果解读案例
        keywords.extend(["典型病例", "检查结果解读案例"])
        return " ".join(keywords)
    
    # ==================== C12: 综合分析与诊断 ====================
    
    def _generate_c12_guide(self, ctx: NodeContext) -> str:
        """C12 - 医学指南库：诊断标准、治疗方案、决策依据
        
        遵循标准：根据患者主诉、关键症状及当前流程节点目标，
        检索诊断标准、治疗方案、决策依据的权威指南内容。
        
        输入：患者主诉 + 科室 + 关键症状（检查结果异常） + 节点目标（综合诊断）
        输出：诊断标准、治疗方案、临床决策依据、权威指南
        """
        keywords = []
        
        # 1. 患者主诉（必须）
        if ctx.chief_complaint:
            keywords.append(ctx.chief_complaint[:30])
        
        # 2. 科室（提供专科背景）
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 3. 关键症状（从检查结果提取）
        if ctx.test_results:
            for result in ctx.test_results[:2]:
                test_name = result.get("test_name", "")
                if test_name:
                    keywords.append(test_name)
                # 提取异常结果作为关键症状
                if result.get("abnormal"):
                    summary = result.get("summary", "")[:20]
                    if summary:
                        keywords.append(summary)
        
        # 4. 节点目标关键词：诊断标准、治疗方案、决策依据
        keywords.extend(["诊断标准", "治疗方案", "临床决策依据", "权威指南"])
        return " ".join(keywords)
    
    def _generate_c12_case(self, ctx: NodeContext) -> str:
        """C12 - 临床案例库：典型病例、诊疗决策路径
        
        遵循标准：根据患者主诉、关键症状及已完成检查，
        检索典型病例、诊疗决策路径相关内容。
        
        输入：患者主诉 + 科室 + 已完成检查 + 关键症状（异常结果）
        输出：典型病例、诊疗决策路径、临床案例分析
        """
        keywords = []
        
        # 1. 患者主诉
        if ctx.chief_complaint:
            keywords.append(ctx.chief_complaint[:30])
        
        # 2. 科室
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 3. 已完成检查及关键症状
        if ctx.test_results:
            for result in ctx.test_results[:3]:
                test_name = result.get("test_name", "")
                if test_name:
                    keywords.append(test_name)
                # 关键症状（异常结果）
                if result.get("abnormal"):
                    summary = result.get("summary", "")[:15]
                    if summary:
                        keywords.append(summary)
        
        # 4. 目标关键词：典型病例、诊疗决策路径
        keywords.extend(["典型病例", "诊疗决策路径", "临床案例分析"])
        return " ".join(keywords)
    
    # ==================== C14: 生成文书 ====================
    
    def _generate_c14_process(self, ctx: NodeContext) -> str:
        """C14 - 规则流程库：医疗文书模板
        
        遵循标准：根据当前流程节点类型（生成文书），
        检索医疗文书模板。
        
        输入：流程节点类型 + 科室 + 初步诊断
        输出：医疗文书模板（门诊病历、诊断证明、病假条、宣教单）
        """
        keywords = []
        
        # 1. 科室
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 2. 初步诊断
        if ctx.preliminary_diagnosis:
            keywords.append(ctx.preliminary_diagnosis[:20])
        
        # 3. 节点类型关键词：医疗文书模板
        keywords.extend(["医疗文书模板", "门诊病历", "诊断证明", "病假条", "宣教单"])
        
        return " ".join(keywords)
    
    def _generate_c14_history(self, ctx: NodeContext) -> str:
        """C14 - 患者历史库：历史病历信息"""
        keywords = ["历史病历", "既往就诊"]
        
        # 使用主诉
        if ctx.chief_complaint:
            keywords.insert(0, ctx.chief_complaint[:20])
        
        return " ".join(keywords)
    
    # ==================== C15: 宣教与随访 ====================
    
    def _generate_c15_process(self, ctx: NodeContext) -> str:
        """C15 - 规则流程库：宣教与随访流程
        
        遵循标准：根据当前流程节点类型（宣教与随访），
        检索宣教与随访流程。
        
        输入：流程节点类型 + 科室 + 初步诊断
        输出：健康宣教流程、随访计划、注意事项、红旗症状
        """
        keywords = []
        
        # 1. 科室
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 2. 初步诊断
        if ctx.preliminary_diagnosis:
            keywords.append(ctx.preliminary_diagnosis[:20])
        
        # 3. 节点类型关键词：宣教与随访流程
        keywords.extend(["健康宣教流程", "随访计划", "注意事项", "红旗症状", "患者教育"])
        return " ".join(keywords)
    
    # ==================== S4: 专科问诊 ====================
    
    def _generate_s4_guide(self, ctx: NodeContext) -> str:
        """S4 - 医学指南库：专科诊断标准、问诊要点
        
        遵循标准：根据患者主诉、当前流程节点目标（专科问诊），
        检索专科诊断标准、问诊决策依据的权威指南内容。
        
        输入：科室 + 患者主诉 + 节点目标（专科问诊）
        输出：专科诊断标准、问诊要点、Red Flags、鉴别诊断要点、临床决策
        """
        keywords = []
        
        # 1. 科室（必须，提供专科背景）
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 2. 患者主诉（提取症状关键词）
        if ctx.chief_complaint:
            keywords.extend(self._extract_symptom_keywords(ctx.chief_complaint))

        # 3. 年龄/性别线索（如有）
        if ctx.patient_age:
            keywords.append(f"{ctx.patient_age}岁")
        if ctx.patient_gender:
            keywords.append(ctx.patient_gender)
        
        # 4. 节点目标关键词：专科诊断标准、问诊要点、决策依据
        keywords.extend(["专科诊断标准", "问诊要点", "Red Flags", "鉴别诊断要点", "临床决策"])
        return " ".join(keywords)
    
    def _generate_s4_qa(self, ctx: NodeContext) -> str:
        """S4 - 高质量对话库：结构化医患问诊示例
        
        遵循标准：根据患者主诉及当前问诊阶段（专科问诊），
        检索结构化医患问诊示例对话。
        
        输入：科室 + 患者主诉 + 阶段（专科问诊）
        输出：结构化问诊对话、医患问诊示例、专科问诊技巧
        """
        keywords = []
        
        # 1. 科室（提供专科背景）
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 2. 患者主诉（提取症状关键词）
        if ctx.chief_complaint:
            keywords.extend(self._extract_symptom_keywords(ctx.chief_complaint))

        # 3. 年龄/性别线索（如有）
        if ctx.patient_age:
            keywords.append(f"{ctx.patient_age}岁")
        if ctx.patient_gender:
            keywords.append(ctx.patient_gender)
        
        # 4. 阶段目标关键词：结构化问诊、医患对话示例
        keywords.extend(["结构化问诊", "医患问诊示例", "专科问诊对话", "沟通技巧"])
        return " ".join(keywords)
    
    def _generate_s4_case(self, ctx: NodeContext) -> str:
        """S4 - 临床案例库：典型病例、诊疗决策参考
        
        遵循标准：根据患者主诉、关键症状，
        检索典型病例、诊疗决策路径相关内容。
        
        输入：科室 + 患者主诉
        输出：典型病例、诊疗决策参考、临床案例
        """
        keywords = []
        
        # 1. 科室
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 2. 患者主诉（提取症状关键词）
        if ctx.chief_complaint:
            keywords.extend(self._extract_symptom_keywords(ctx.chief_complaint))
        else:
            # 无主诉时，使用科室常见症状作为备选
            keywords.append("常见症状")

        # 3. 年龄/性别线索（如有）
        if ctx.patient_age:
            keywords.append(f"{ctx.patient_age}岁")
        if ctx.patient_gender:
            keywords.append(ctx.patient_gender)
        
        # 4. 目标关键词：典型病例、诊疗决策
        keywords.extend(["典型病例", "诊疗决策参考", "临床案例"])
        return " ".join(keywords)
    
    # ==================== S6: 初步判断 ====================
    
    def _generate_s6_guide(self, ctx: NodeContext) -> str:
        """S6 - 医学指南库：诊断标准、辅助检查决策依据
        
        遵循标准：根据患者主诉、关键症状及当前流程节点目标（初步判断），
        检索诊断标准、鉴别诊断、是否需要辅助检查的决策依据。
        
        输入：患者主诉 + 科室 + 关键症状（从专科小结提取） + 节点目标（初步判断）
        输出：诊断标准、鉴别诊断、辅助检查决策依据、临床指南
        """
        keywords = []
        
        # 1. 患者主诉（必须）
        if ctx.chief_complaint:
            keywords.append(ctx.chief_complaint[:30])
        
        # 2. 科室
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        # 3. 关键症状（从专科小结提取）
        if ctx.specialty_summary:
            interview = ctx.specialty_summary.get("interview", {})
            if isinstance(interview, dict):
                for key in ["onset_time", "severity", "frequency", "character"]:
                    value = interview.get(key)
                    if value and isinstance(value, str) and value not in ["不详", "未知", ""]:
                        keywords.append(value[:15])
        
        # 4. 节点目标关键词：诊断标准、辅助检查决策
        keywords.extend(["诊断标准", "鉴别诊断", "是否需要辅助检查", "检查决策依据", "临床指南"])
        return " ".join(keywords)
    
    # ==================== 通用备用方法 ====================
    
    def _generate_generic(self, ctx: NodeContext, db_name: str) -> str:
        """通用关键词生成（备用方案）
        
        严格遵循标准为各知识库生成关键词：
        
        【医学指南库】
        输入：患者主诉 + 关键症状 + 节点目标
        输出：诊断标准、治疗方案、决策依据、辅助检查指导
        
        【高质量对话库】
        输入：患者主诉 + 问诊/解释阶段
        输出：结构化问诊、检查结果解读、沟通示例
        
        【临床案例库】
        输入：患者主诉 + 关键症状 + 已完成检查
        输出：典型病例、结果解读案例、诊疗决策路径
        
        【规则流程库】
        输入：流程节点类型
        输出：标准操作规程、检查准备事项、文书模板、宣教随访流程
        """
        keywords = []
        
        # 1. 基础信息
        if ctx.dept_name:
            keywords.append(ctx.dept_name)
        
        if ctx.chief_complaint:
            keywords.append(ctx.chief_complaint[:30])
        
        # 2. 关键症状/已完成检查（适用于指南库和案例库）
        if db_name in ["MedicalGuide_db", "ClinicalCase_db"]:
            if ctx.test_results:
                for result in ctx.test_results[:2]:
                    test_name = result.get("test_name", "")
                    if test_name:
                        keywords.append(test_name)
        
        # 3. 根据知识库类型添加目标关键词
        if "Process" in db_name:
            # 规则流程库：标准操作规程
            keywords.extend(["标准操作规程", "流程", "SOP"])
        elif "Guide" in db_name:
            # 医学指南库：诊断标准、治疗方案、决策依据
            keywords.extend(["诊断标准", "治疗方案", "临床决策依据", "权威指南"])
        elif "Case" in db_name:
            # 临床案例库：典型病例、诊疗决策
            keywords.extend(["典型病例", "诊疗决策路径", "临床案例"])
        elif "QA" in db_name:
            # 高质量对话库：医患问诊、沟通示例
            keywords.extend(["医患问诊", "沟通示例对话", "结构化问诊"])
        
        return " ".join(keywords) if keywords else ctx.node_name
