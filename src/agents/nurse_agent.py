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
    
    def __init__(self, llm: LLMClient):
        """
        Args:
            llm: 语言模型客户端（必需，用于智能分诊）
        """
        self.llm = llm
        self.triage_history: list[dict[str, str]] = []
    
    def reset(self) -> None:
        """重置分诊历史（用于处理新的就诊流程）"""
        self.triage_history = []
    
    def triage(self, chief_complaint: str) -> str:
        """
        根据主诉进行分诊到15个标准科室之一
        
        Args:
            chief_complaint: 患者主诉
            
        Returns:
            科室代码（internal_medicine, surgery, orthopedics等）
        """
        # 参数验证
        if not chief_complaint or not chief_complaint.strip():
            raise ValueError("主诉不能为空")
        
        chief_complaint = chief_complaint.strip()
        
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
        
        user_prompt = f"""患者主诉：{chief_complaint}

请判断应该挂哪个科室，输出JSON格式：
{{
  "dept": "科室代码（如internal_medicine）",
  "reason": "分诊理由"
}}"""
        
        try:
            obj, _, _ = self.llm.generate_json(
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
            self.triage_history.append({
                "chief_complaint": chief_complaint,
                "dept": dept,
                "reason": reason
            })
            
            return dept
            
        except Exception as e:
            print(f"⚠️  分诊异常: {str(e)}，默认分诊至内科")
            dept = "internal_medicine"
            self.triage_history.append({
                "chief_complaint": chief_complaint,
                "dept": dept,
                "reason": f"异常回退：{str(e)}，默认内科"
            })
            return dept
    
    def get_triage_summary(self) -> dict[str, int | list[dict[str, str]]]:
        """获取分诊摘要"""
        return {
            "total_triages": len(self.triage_history),
            "history": self.triage_history
        }
