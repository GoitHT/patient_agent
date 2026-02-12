"""智能RAG查询优化器 - 根据上下文动态生成精准查询

本模块提供智能查询策略,根据患者信息、问诊进展、检查结果等动态生成最优查询。

主要功能:
1. 上下文感知查询生成 - 利用患者历史、当前症状等信息
2. 多策略查询优化 - 针对不同场景使用不同查询策略
3. 同义词扩展 - 自动扩展医学同义词提高召回率
4. 查询重写 - 将通用查询重写为更精准的医学查询
"""
from __future__ import annotations

import re
from typing import Any
from dataclasses import dataclass


@dataclass
class QueryContext:
    """查询上下文信息"""
    # 患者信息
    patient_id: str | None = None
    age: int | None = None
    gender: str | None = None
    
    # 症状信息
    chief_complaint: str | None = None
    symptom_duration: str | None = None
    symptom_severity: str | None = None
    
    # 就诊信息
    dept: str | None = None
    dept_name: str | None = None
    
    # 问诊进展
    qa_history: list[dict[str, str]] | None = None
    specialty_summary: dict[str, Any] | None = None
    
    # 检查信息
    ordered_tests: list[dict[str, Any]] | None = None
    test_results: list[dict[str, Any]] | None = None
    abnormal_results: list[dict[str, Any]] | None = None
    
    # 诊断信息
    preliminary_diagnosis: str | None = None
    diagnosis_uncertainty: str | None = None
    
    # 其他上下文
    additional_context: dict[str, Any] | None = None


class RAGQueryOptimizer:
    """RAG查询优化器 - 根据上下文动态生成精准查询"""
    
    def __init__(self):
        # 医学同义词映射
        self.synonyms = {
            # 症状同义词
            "头痛": ["头疼", "头部疼痛", "头部不适", "颅痛"],
            "眩晕": ["头晕", "眩晕感", "晕眩", "头昏"],
            "恶心": ["恶心感", "想吐", "欲呕", "反胃"],
            "呕吐": ["呕吐物", "呕", "吐"],
            "发热": ["发烧", "体温升高", "热度"],
            "乏力": ["疲劳", "无力", "倦怠", "疲乏"],
            "胸痛": ["胸部疼痛", "胸闷痛", "胸部不适"],
            "腹痛": ["腹部疼痛", "肚子痛", "腹部不适"],
            "咳嗽": ["咳", "咳痰"],
            
            # 检查同义词
            "CT": ["计算机断层扫描", "X线计算机断层成像"],
            "MRI": ["核磁共振", "磁共振成像", "核磁"],
            "脑电图": ["EEG", "脑电"],
            "心电图": ["ECG", "心电"],
            "血常规": ["血细胞分析", "血液常规检查", "CBC"],
            "生化": ["生化检查", "生化全套", "肝肾功能"],
            
            # 诊疗同义词
            "诊断": ["诊疗", "确诊", "判断", "鉴别"],
            "治疗": ["处理", "处置", "干预", "疗法"],
            "用药": ["药物治疗", "服药", "给药"],
            "随访": ["复查", "回诊", "定期复诊"],
        }
        
        # 科室关键词映射
        self.dept_keywords = {
            "neurology": {
                "symptoms": ["头痛", "眩晕", "肢体麻木", "肢体无力", "抽搐", "意识障碍", "言语不清"],
                "diseases": ["脑梗死", "脑出血", "癫痫", "偏头痛", "帕金森", "神经炎"],
                "tests": ["头颅CT", "头颅MRI", "脑电图", "肌电图", "TCD"],
            },
            "cardiology": {
                "symptoms": ["胸痛", "心悸", "气促", "呼吸困难"],
                "diseases": ["冠心病", "心律失常", "心力衰竭", "高血压"],
                "tests": ["心电图", "心脏超声", "冠脉造影", "动态心电图"],
            },
            "respiratory": {
                "symptoms": ["咳嗽", "咳痰", "气促", "胸痛"],
                "diseases": ["肺炎", "哮喘", "慢阻肺", "肺结核"],
                "tests": ["胸部CT", "肺功能", "痰培养"],
            },
        }
        
        # 场景特定关键词
        self.scenario_keywords = {
            "sop": ["流程", "规范", "标准", "操作", "指南"],
            "diagnosis": ["诊断", "鉴别", "判断", "确诊", "排除"],
            "treatment": ["治疗", "处理", "方案", "用药", "干预"],
            "education": ["宣教", "教育", "注意事项", "健康指导"],
            "preparation": ["准备", "禁忌", "注意事项", "要求"],
            "red_flags": ["红旗", "警示", "危险", "紧急", "严重"],
            "history": ["病史", "既往史", "过敏史", "家族史"],
            "case": ["案例", "病例", "实例", "转归"],
        }
    
    def optimize_query(
        self,
        base_query: str,
        context: QueryContext,
        scenario: str = "general",
        expand_synonyms: bool = True,
        max_keywords: int = 10
    ) -> str:
        """优化查询字符串
        
        Args:
            base_query: 基础查询（可以是简单关键词或句子）
            context: 查询上下文信息
            scenario: 查询场景（sop/diagnosis/treatment/education等）
            expand_synonyms: 是否扩展同义词
            max_keywords: 最大关键词数量（避免查询过长）
        
        Returns:
            优化后的查询字符串
        """
        keywords = []
        
        # 1. 提取基础查询中的关键词
        base_keywords = self._extract_keywords(base_query)
        keywords.extend(base_keywords[:3])  # 保留最重要的3个基础关键词
        
        # 2. 添加患者特征关键词（年龄、性别）
        if context.age is not None:
            age_group = self._get_age_group(context.age)
            keywords.append(age_group)
        
        if context.gender:
            keywords.append(context.gender)
        
        # 3. 添加主诉症状及其变体
        if context.chief_complaint:
            complaint_keywords = self._extract_keywords(context.chief_complaint)
            keywords.extend(complaint_keywords[:2])
            
            # 添加症状持续时间（急性vs慢性）
            if context.symptom_duration:
                duration_type = self._classify_duration(context.symptom_duration)
                if duration_type:
                    keywords.append(duration_type)
            
            # 添加症状严重程度
            if context.symptom_severity:
                keywords.append(context.symptom_severity)
        
        # 4. 添加科室特定关键词
        if context.dept and context.dept in self.dept_keywords:
            dept_info = self.dept_keywords[context.dept]
            
            # 根据场景选择合适的科室关键词
            if scenario in ["diagnosis", "treatment"]:
                keywords.extend(dept_info["diseases"][:2])
            elif scenario == "preparation":
                keywords.extend(dept_info["tests"][:2])
        
        # 5. 添加问诊历史中的重要信息
        if context.qa_history:
            qa_keywords = self._extract_qa_keywords(context.qa_history)
            keywords.extend(qa_keywords[:2])
        
        # 6. 添加检查结果关键词
        if context.test_results:
            test_keywords = self._extract_test_keywords(context.test_results)
            keywords.extend(test_keywords[:2])
        
        # 特别关注异常结果
        if context.abnormal_results:
            abnormal_keywords = self._extract_abnormal_keywords(context.abnormal_results)
            keywords.extend(abnormal_keywords[:2])
        
        # 7. 添加场景特定关键词
        if scenario in self.scenario_keywords:
            scenario_kw = self.scenario_keywords[scenario][:2]
            keywords.extend(scenario_kw)
        
        # 8. 添加诊断信息（如果有初步诊断）
        if context.preliminary_diagnosis:
            keywords.append(context.preliminary_diagnosis)
        
        # 9. 去重并限制数量
        unique_keywords = []
        seen = set()
        for kw in keywords:
            kw_clean = kw.strip()
            if kw_clean and kw_clean not in seen and len(kw_clean) > 1:
                unique_keywords.append(kw_clean)
                seen.add(kw_clean)
                if len(unique_keywords) >= max_keywords:
                    break
        
        # 10. 扩展同义词（可选）
        if expand_synonyms:
            expanded = self._expand_synonyms(unique_keywords, max_expand=2)
            unique_keywords.extend(expanded)
        
        # 11. 构建最终查询（按重要性排序）
        optimized_query = " ".join(unique_keywords[:max_keywords])
        
        return optimized_query
    
    def generate_contextual_query(
        self,
        purpose: str,
        context: QueryContext,
        **kwargs
    ) -> str:
        """根据目的和上下文生成查询（高层接口）
        
        Args:
            purpose: 查询目的（interview_sop/diagnosis_guide/treatment_plan等）
            context: 查询上下文
            **kwargs: 其他参数（传递给optimize_query）
        
        Returns:
            生成的查询字符串
        """
        # 根据不同目的生成不同的基础查询
        purpose_queries = {
            # C5: 问诊准备
            "interview_sop": self._generate_interview_sop_query(context),
            "patient_history": self._generate_patient_history_query(context),
            
            # S4: 专科问诊
            "specialty_knowledge": self._generate_specialty_knowledge_query(context),
            "quality_qa": self._generate_quality_qa_query(context),
            "clinical_case": self._generate_clinical_case_query(context),
            
            # C8: 开单准备
            "test_preparation": self._generate_test_preparation_query(context),
            "test_history": self._generate_test_history_query(context),
            "hospital_flow": self._generate_hospital_flow_query(context),
            
            # C11: 回诊分析
            "treatment_guide": self._generate_treatment_guide_query(context),
            "similar_cases": self._generate_similar_cases_query(context),
            
            # C12: 诊断分析
            "diagnosis_support": self._generate_diagnosis_support_query(context),
            "differential": self._generate_differential_query(context),
            
            # C14: 文书生成
            "document_template": self._generate_document_template_query(context),
            "medical_record_history": self._generate_medical_record_history_query(context),
            
            # C15: 宣教随访
            "education_material": self._generate_education_material_query(context),
            "followup_guide": self._generate_followup_guide_query(context),
        }
        
        base_query = purpose_queries.get(purpose, "")
        if not base_query:
            # 如果没有预定义的查询，返回简单的上下文查询
            base_query = self._generate_fallback_query(context)
        
        # 确定场景类型
        scenario_map = {
            "interview_sop": "sop",
            "hospital_flow": "sop",
            "specialty_knowledge": "diagnosis",
            "diagnosis_support": "diagnosis",
            "differential": "diagnosis",
            "treatment_guide": "treatment",
            "test_preparation": "preparation",
            "education_material": "education",
            "clinical_case": "case",
            "similar_cases": "case",
        }
        scenario = scenario_map.get(purpose, "general")
        
        # 调用优化器
        return self.optimize_query(base_query, context, scenario=scenario, **kwargs)
    
    # ========== 私有辅助方法 ==========
    
    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词（优先提取症状词汇）"""
        if not text:
            return []
        
        # 医学症状关键词（高优先级）
        symptom_keywords = [
            "疼痛", "头痛", "胸痛", "腹痛", "背痛", "关节痛", "肌肉痛",
            "发热", "发烧", "咳嗽", "气短", "呼吸困难", "胸闷", "心悸",
            "恶心", "呕吐", "腹泻", "便秘", "腹胀", "食欲",
            "头晕", "眩晕", "乏力", "疲劳", "失眠", "嗜睡",
            "麻木", "无力", "抽搐", "震颤", "瘫痪",
            "出血", "肿胀", "红肿", "瘙痒", "皮疹",
            "弹响", "僵硬", "活动受限", "跛行",
            "不适", "酸胀", "压迫感", "紧缩感", "勒住"
        ]
        
        # 提取文本中的所有中文词汇
        stopwords = {"的", "是", "了", "和", "与", "或", "等", "及", "、", "，", "。", "在", "有", "我", "会", "就", "没", "很", "比较", "感觉", "觉得"}
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        
        # 优先收集症状关键词
        priority_keywords = []
        regular_keywords = []
        
        for word in words:
            if word in stopwords or len(word) < 2:
                continue
            # 检查是否包含症状词汇
            is_symptom = any(symptom in word for symptom in symptom_keywords)
            if is_symptom:
                if word not in priority_keywords:
                    priority_keywords.append(word)
            else:
                if word not in regular_keywords:
                    regular_keywords.append(word)
        
        # 症状词优先，然后补充其他关键词
        result = priority_keywords[:4] + regular_keywords[:3]
        return result[:5]  # 最多返回5个关键词
    
    def _get_age_group(self, age: int) -> str:
        """将年龄转换为年龄组标签"""
        if age < 18:
            return "儿童青少年"
        elif age < 40:
            return "青年"
        elif age < 60:
            return "中年"
        else:
            return "老年"
    
    def _classify_duration(self, duration: str) -> str | None:
        """将症状持续时间分类为急性/慢性"""
        if not duration:
            return None
        
        duration_lower = duration.lower()
        
        # 急性：小时、天、突然
        if any(kw in duration_lower for kw in ["小时", "hour", "天", "day", "突然", "急性", "急"]):
            return "急性"
        
        # 慢性：周、月、年、长期
        if any(kw in duration_lower for kw in ["周", "week", "月", "month", "年", "year", "长期", "慢性", "反复"]):
            return "慢性"
        
        return None
    
    def _extract_qa_keywords(self, qa_history: list[dict[str, str]]) -> list[str]:
        """从问诊历史中提取关键词"""
        keywords = []
        
        # 从最近的几轮问答中提取（最多3轮）
        recent_qa = qa_history[-3:] if len(qa_history) > 3 else qa_history
        
        for qa in recent_qa:
            answer = qa.get("answer", "")
            if answer:
                kws = self._extract_keywords(answer)
                keywords.extend(kws[:2])  # 每轮最多2个关键词
        
        return keywords
    
    def _extract_test_keywords(self, test_results: list[dict[str, Any]]) -> list[str]:
        """从检查结果中提取关键词"""
        keywords = []
        
        for result in test_results[:3]:  # 最多处理3个检查
            test_name = result.get("test_name", "")
            if test_name:
                keywords.append(test_name)
        
        return keywords
    
    def _extract_abnormal_keywords(self, abnormal_results: list[dict[str, Any]]) -> list[str]:
        """从异常结果中提取关键词"""
        keywords = []
        
        for result in abnormal_results[:2]:  # 最多处理2个异常
            summary = result.get("summary", "")
            if summary:
                kws = self._extract_keywords(summary)
                keywords.extend(kws[:2])
        
        return keywords
    
    def _expand_synonyms(self, keywords: list[str], max_expand: int = 2) -> list[str]:
        """扩展同义词"""
        expanded = []
        
        for kw in keywords:
            if kw in self.synonyms:
                synonyms = self.synonyms[kw][:max_expand]
                expanded.extend(synonyms)
        
        return expanded
    
    # ========== 场景特定查询生成器 ==========
    
    def _generate_interview_sop_query(self, ctx: QueryContext) -> str:
        """生成问诊SOP查询（优化用于C5节点）"""
        parts = ["门诊", "问诊", "流程", "SOP", "操作规范"]
        
        if ctx.dept_name:
            parts.insert(0, ctx.dept_name)  # 科室放在最前面，提高权重
        
        if ctx.chief_complaint:
            # 提取主诉关键词，增强查询精度
            complaint_keywords = ctx.chief_complaint[:30]
            parts.extend([complaint_keywords, "问诊要点", "注意事项"])
        
        # 添加患者特征辅助检索
        if ctx.age:
            if ctx.age < 18:
                parts.append("儿童")
            elif ctx.age >= 65:
                parts.append("老年")
        
        if ctx.gender:
            parts.append(ctx.gender)
        
        return " ".join(parts)
    
    def _generate_patient_history_query(self, ctx: QueryContext) -> str:
        """生成患者历史查询"""
        parts = ["患者", "历史", "记录"]
        
        if ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        return " ".join(parts)
    
    def _generate_specialty_knowledge_query(self, ctx: QueryContext) -> str:
        """生成专科知识查询（优化：提取症状关键词）"""
        parts = []
        
        # 1. 科室名称（必需）
        if ctx.dept_name:
            parts.append(ctx.dept_name)
        
        # 2. 从主诉中提取症状关键词（而非简单截断）
        if ctx.chief_complaint:
            keywords = self._extract_keywords(ctx.chief_complaint)
            parts.extend(keywords[:3])  # 取前3个关键词
        
        # 3. 年龄特征（某些疾病有年龄倾向）
        if ctx.age is not None:
            age_group = self._get_age_group(ctx.age)
            parts.append(age_group)
        
        # 4. 专科诊疗关键词
        parts.extend(["鉴别诊断", "诊疗指南"])
        
        return " ".join(parts)
    
    def _generate_quality_qa_query(self, ctx: QueryContext) -> str:
        """生成高质量问诊查询（优化：根据症状生成问诊方向）"""
        parts = []
        
        # 1. 科室专科
        if ctx.dept_name:
            parts.append(ctx.dept_name)
        
        # 2. 提取症状关键词
        symptom_keywords = []
        if ctx.chief_complaint:
            symptom_keywords = self._extract_keywords(ctx.chief_complaint)
            parts.extend(symptom_keywords[:2])  # 取前2个症状
        
        # 3. 问诊方向关键词
        parts.extend(["问诊", "评估"])
        
        # 4. 患者特征（性别可能影响问诊重点）
        if ctx.gender:
            parts.append(ctx.gender)
        
        # 5. 如果有症状关键词，添加相关问诊词
        if symptom_keywords:
            # 根据症状类型添加问诊重点
            if any(kw in symptom_keywords for kw in ["疼痛", "痛"]):
                parts.append("疼痛特征")
            if any(kw in symptom_keywords for kw in ["发热", "发烧"]):
                parts.append("伴随症状")
        
        return " ".join(parts)
    
    def _generate_clinical_case_query(self, ctx: QueryContext) -> str:
        """生成临床案例查询（优化：症状组合+患者特征）"""
        parts = []
        
        # 1. 年龄组和性别（案例检索的重要特征）
        if ctx.age is not None:
            age_group = self._get_age_group(ctx.age)
            parts.append(age_group)
        
        if ctx.gender:
            parts.append(ctx.gender)
        
        # 2. 提取主要症状关键词（症状组合）
        if ctx.chief_complaint:
            keywords = self._extract_keywords(ctx.chief_complaint)
            parts.extend(keywords[:3])  # 取前3个关键词形成症状组合
        
        # 3. 如果有科室，添加科室相关
        if ctx.dept_name:
            parts.append(ctx.dept_name)
        
        # 4. 案例相关词
        parts.extend(["病例", "诊疗"])
        
        return " ".join(parts)
    
    def _generate_test_preparation_query(self, ctx: QueryContext) -> str:
        """生成检查准备查询"""
        parts = []
        
        if ctx.ordered_tests:
            for test in ctx.ordered_tests[:2]:
                test_name = test.get("name", "")
                if test_name:
                    parts.append(test_name)
        
        parts.extend(["准备", "禁忌", "注意事项"])
        
        return " ".join(parts)
    
    def _generate_test_history_query(self, ctx: QueryContext) -> str:
        """生成检查历史查询"""
        parts = ["患者", "检查", "历史"]
        
        if ctx.ordered_tests:
            for test in ctx.ordered_tests[:2]:
                test_name = test.get("name", "")
                if test_name:
                    parts.append(test_name)
        
        return " ".join(parts)
    
    def _generate_hospital_flow_query(self, ctx: QueryContext) -> str:
        """生成医院流程查询"""
        return "缴费 预约 报告领取 流程"
    
    def _generate_treatment_guide_query(self, ctx: QueryContext) -> str:
        """生成治疗指南查询"""
        parts = []
        
        if ctx.preliminary_diagnosis:
            parts.append(ctx.preliminary_diagnosis)
        elif ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        # 添加检查结果关键词
        if ctx.test_results:
            for result in ctx.test_results[:2]:
                test_name = result.get("test_name", "")
                if test_name:
                    parts.append(test_name)
        
        # 添加异常关键词
        if ctx.abnormal_results:
            for result in ctx.abnormal_results[:1]:
                summary = result.get("summary", "")
                if summary:
                    parts.append(summary[:30])
        
        parts.extend(["诊疗方案", "处理建议"])
        
        return " ".join(parts)
    
    def _generate_similar_cases_query(self, ctx: QueryContext) -> str:
        """生成相似案例查询"""
        parts = []
        
        if ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        # 添加检查名称
        if ctx.test_results:
            for result in ctx.test_results[:3]:
                test_name = result.get("test_name", "")
                if test_name:
                    parts.append(test_name)
        
        # 添加异常关键词
        if ctx.abnormal_results:
            for result in ctx.abnormal_results[:2]:
                summary = result.get("summary", "")
                if summary:
                    kws = self._extract_keywords(summary)
                    parts.extend(kws[:1])
        
        parts.extend(["病例", "处理", "转归"])
        
        return " ".join(parts)
    
    def _generate_diagnosis_support_query(self, ctx: QueryContext) -> str:
        """生成诊断支持查询"""
        parts = []
        
        if ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        if ctx.dept_name:
            parts.append(ctx.dept_name)
        
        # 添加关键症状（从问诊历史）
        if ctx.specialty_summary:
            key_findings = ctx.specialty_summary.get("key_findings", [])
            parts.extend(key_findings[:2])
        
        parts.extend(["诊断", "鉴别", "依据"])
        
        return " ".join(parts)
    
    def _generate_differential_query(self, ctx: QueryContext) -> str:
        """生成鉴别诊断查询"""
        parts = []
        
        if ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        if ctx.preliminary_diagnosis:
            parts.append(ctx.preliminary_diagnosis)
        
        parts.extend(["鉴别诊断", "类似疾病", "排除"])
        
        return " ".join(parts)
    
    def _generate_document_template_query(self, ctx: QueryContext) -> str:
        """生成文书模板查询"""
        return "门诊病历 诊断证明 病假条 宣教单 模板"
    
    def _generate_medical_record_history_query(self, ctx: QueryContext) -> str:
        """生成病历历史查询"""
        parts = ["病历", "诊断", "既往史"]
        
        if ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        return " ".join(parts)
    
    def _generate_education_material_query(self, ctx: QueryContext) -> str:
        """生成宣教材料查询"""
        parts = []
        
        if ctx.preliminary_diagnosis:
            parts.append(ctx.preliminary_diagnosis)
        elif ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:20])
        
        if ctx.dept_name:
            parts.append(ctx.dept_name)
        
        parts.extend(["宣教", "健康教育", "注意事项"])
        
        return " ".join(parts)
    
    def _generate_followup_guide_query(self, ctx: QueryContext) -> str:
        """生成随访指南查询"""
        parts = []
        
        if ctx.preliminary_diagnosis:
            parts.append(ctx.preliminary_diagnosis)
        
        parts.extend(["随访", "复查", "回诊安排"])
        
        return " ".join(parts)
    
    def _generate_fallback_query(self, ctx: QueryContext) -> str:
        """生成默认查询（当没有匹配的目的时）"""
        parts = []
        
        if ctx.chief_complaint:
            parts.append(ctx.chief_complaint[:30])
        
        if ctx.dept:
            parts.append(ctx.dept)
        
        return " ".join(parts) if parts else "医学知识"


# 全局单例
_query_optimizer = None


def get_query_optimizer() -> RAGQueryOptimizer:
    """获取查询优化器全局单例"""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = RAGQueryOptimizer()
    return _query_optimizer


__all__ = ["RAGQueryOptimizer", "QueryContext", "get_query_optimizer"]
