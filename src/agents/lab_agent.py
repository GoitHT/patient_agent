"""检验科智能体：处理医生开具的检查，生成或返回检查结果"""
from __future__ import annotations

import json
from typing import Any

from services.llm_client import LLMClient


class LabAgent:
    """检验科智能体：负责处理检验检查，生成结果"""
    
    def __init__(self, llm: LLMClient | None = None):
        """
        Args:
            llm: 语言模型客户端（可选，用于生成缺失的检查结果）
        """
        self._llm = llm
        self._processed_tests: list[dict[str, Any]] = []
    
    def reset(self) -> None:
        """重置检验科状态（用于处理新患者）"""
        self._processed_tests = []
    
    def generate_test_results(
        self,
        lab_context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        根据完整上下文智能生成检查结果
        
        这是一个统一的接口方法，接受包含所有必要信息的上下文字典，
        检验科Agent根据医生开具的检查项目、患者主诉、病历信息等智能生成检查结果。
        
        Args:
            lab_context: 包含以下键的字典:
                - ordered_tests: list[dict] - 医生开具的检查列表
                - chief_complaint: str - 患者主诉
                - case_info: str - 病例信息文本
                - real_tests_reference: str|None - 真实检查结果参考（从数据集）
                - dept: str - 科室代码
                - patient_id: str - 患者ID
                
        Returns:
            检查结果列表 [{"test_name": "血常规", "result": {...}, "abnormal": True, ...}, ...]
        """
        ordered_tests = lab_context.get("ordered_tests", [])
        chief_complaint = lab_context.get("chief_complaint", "")
        case_info_text = lab_context.get("case_info", "")
        real_tests_reference = lab_context.get("real_tests_reference")
        dept = lab_context.get("dept", "")
        patient_id = lab_context.get("patient_id", "")
        
        if not ordered_tests:
            return []
        
        # 构造伪病例数据结构以复用现有逻辑
        case_data = {
            "Case Information": case_info_text,
            "dept": dept,
            "patient_id": patient_id
        }
        
        # 如果有真实检查结果参考，将其附加到Case Information中
        if real_tests_reference:
            case_data["Case Information"] = case_info_text + "\n\n真实检查结果参考:\n" + real_tests_reference
        
        # 调用现有的process_test_orders方法
        return self.process_test_orders(
            ordered_tests=ordered_tests,
            case_data=case_data,
            chief_complaint=chief_complaint,
            physical_state=None,
            existing_results=None
        )
    
    def process_test_orders(
        self,
        ordered_tests: list[dict[str, Any]],
        case_data: dict[str, Any],
        chief_complaint: str,
        physical_state: dict[str, Any] | None = None,
        existing_results: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        处理医生开具的检查单，返回检查结果
        
        Args:
            ordered_tests: 医生开具的检查列表 [{"name": "血常规", "type": "lab", ...}, ...]
            case_data: 病例数据（包含Case Information等）
            chief_complaint: 患者主诉
            physical_state: 患者物理状态快照
            existing_results: 已有的检查结果列表
            
        Returns:
            检查结果列表 [{"test_name": "血常规", "result": {...}, "abnormal": True, ...}, ...]
        """
        if not ordered_tests:
            return []
        
        results = []
        
        for test_order in ordered_tests:
            test_name = test_order.get("name", "")
            test_type = test_order.get("type", "lab")
            
            # 1. 优先从病例数据中查找现有结果
            existing_result = self._find_existing_result(test_name, case_data)
            
            if existing_result:
                # 使用病例中的真实结果
                result = self._format_existing_result(test_name, test_type, existing_result)
                results.append(result)
            else:
                # 2. 如果病例中没有，使用LLM生成结果
                if self._llm:
                    generated_result = self._generate_result_with_llm(
                        test_name=test_name,
                        test_type=test_type,
                        chief_complaint=chief_complaint,
                        case_data=case_data,
                        physical_state=physical_state,
                        existing_results=existing_results or []
                    )
                    results.append(generated_result)
                else:
                    # 3. 规则模式：返回正常结果
                    fallback_result = self._generate_fallback_result(test_name, test_type)
                    results.append(fallback_result)
        
        # 记录处理的检查
        self._processed_tests.extend(results)
        
        return results
    
    def _find_existing_result(self, test_name: str, case_data: dict[str, Any]) -> str | None:
        """
        从病例数据中查找现有的检查结果
        
        Args:
            test_name: 检查名称
            case_data: 病例数据
            
        Returns:
            检查结果文本，如果没有则返回None
        """
        if not case_data:
            return None
        
        # 从 Case Information 中查找
        case_info = case_data.get("Case Information", "")
        if not case_info:
            return None
        
        # 定义检查名称的多种可能写法（映射表）
        test_name_variants = {
            "血常规": ["血常规", "血细胞分析", "全血细胞计数", "CBC"],
            "尿常规": ["尿常规", "尿液分析", "尿检"],
            "肝功能": ["肝功能", "肝功", "肝酶", "转氨酶"],
            "肾功能": ["肾功能", "肾功", "肌酐", "尿素氮"],
            "血糖": ["血糖", "空腹血糖", "餐后血糖", "GLU"],
            "CT": ["CT", "计算机断层扫描", "电子计算机断层扫描"],
            "MRI": ["MRI", "核磁共振", "磁共振成像"],
            "X光": ["X光", "X线", "胸片", "X-ray"],
            "心电图": ["心电图", "ECG", "EKG"],
            "B超": ["B超", "超声", "超声检查", "彩超"],
            "胃镜": ["胃镜", "上消化道内镜", "食管胃十二指肠镜"],
            "肠镜": ["肠镜", "结肠镜", "纤维结肠镜"],
        }
        
        # 获取该检查的所有变体名称
        variants = test_name_variants.get(test_name, [test_name])
        
        # 搜索匹配的检查结果
        for variant in variants:
            # 在病例信息中查找该检查名称
            if variant in case_info:
                # 提取该检查的结果段落
                start_idx = case_info.find(variant)
                
                # 寻找结果的结束位置（下一个检查项目或段落）
                end_markers = [
                    "\n\n", "。\n", "。 ", 
                    "诊断：", "治疗：", "讨论：",
                    "血常规", "尿常规", "肝功能", "肾功能", 
                    "CT", "MRI", "X光", "心电图", "B超"
                ]
                
                end_idx = start_idx + 500  # 默认最多提取500字符
                remaining = case_info[start_idx:]
                
                for marker in end_markers:
                    pos = remaining.find(marker, len(variant) + 1)  # 从检查名称后开始找
                    if pos != -1 and start_idx + pos < end_idx:
                        end_idx = start_idx + pos
                
                result_text = case_info[start_idx:end_idx].strip()
                
                # 确保提取的文本包含实际结果（不只是检查名称）
                if len(result_text) > len(variant) + 5:  # 至少比名称多5个字符
                    return result_text
        
        return None
    
    def _format_existing_result(
        self, 
        test_name: str, 
        test_type: str, 
        result_text: str
    ) -> dict[str, Any]:
        """
        格式化病例中的现有结果
        
        Args:
            test_name: 检查名称
            test_type: 检查类型
            result_text: 检查结果原文
            
        Returns:
            格式化的检查结果字典
        """
        # 判断是否异常（简单规则）
        abnormal_keywords = [
            "异常", "升高", "降低", "增高", "减少", "偏高", "偏低",
            "阳性", "+++", "++", "异常回声", "占位", "肿块", "扩大",
            "狭窄", "堵塞", "梗阻", "积液", "出血", "破裂"
        ]
        
        is_abnormal = any(keyword in result_text for keyword in abnormal_keywords)
        
        # 提取关键信息作为摘要（取前100字符）
        summary = result_text[:100].strip()
        if len(result_text) > 100:
            summary += "..."
        
        return {
            "test_name": test_name,
            "test": test_name,
            "type": test_type,
            "result": result_text,
            "abnormal": is_abnormal,
            "summary": summary,
            "source": "case_data",  # 标记来源
            "timestamp": None,  # 可以从病例中提取时间
        }
    
    def _generate_result_with_llm(
        self,
        test_name: str,
        test_type: str,
        chief_complaint: str,
        case_data: dict[str, Any],
        physical_state: dict[str, Any] | None,
        existing_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        使用LLM生成检查结果
        
        Args:
            test_name: 检查名称
            test_type: 检查类型
            chief_complaint: 患者主诉
            case_data: 病例数据
            physical_state: 患者物理状态
            existing_results: 已有的检查结果
            
        Returns:
            生成的检查结果字典
        """
        # 构建上下文信息
        case_summary = self._extract_case_summary(case_data)
        
        # 构建已有结果摘要
        existing_summary = ""
        if existing_results:
            existing_summary = "\n已有检查结果：\n"
            for res in existing_results:
                name = res.get("test_name", "")
                abnormal = res.get("abnormal", False)
                summary = res.get("summary", "")
                status = "异常" if abnormal else "正常"
                existing_summary += f"- {name} ({status}): {summary}\n"
        
        # 构建物理状态信息
        physical_summary = ""
        if physical_state:
            symptoms = physical_state.get("symptoms", {})
            vital_signs = physical_state.get("vital_signs", {})
            
            if symptoms:
                physical_summary += "\n患者症状：\n"
                for symptom, severity in symptoms.items():
                    physical_summary += f"- {symptom} (严重度: {severity:.1f}/10)\n"
            
            if vital_signs:
                physical_summary += "\n生命体征：\n"
                for sign, value in vital_signs.items():
                    physical_summary += f"- {sign}: {value:.1f}\n"
        
        # 根据检查类型和名称构建专业的检验报告格式指导
        report_templates = self._get_report_template(test_name, test_type)
        
        system_prompt = f"""你是一名资深的医学检验科主任，拥有20年临床检验经验。你的任务是根据患者的临床信息，生成专业、逼真、符合医学规范的检查结果。

【检查类型与报告规范】
{report_templates}

【核心原则】
1. ✅ 临床一致性：结果必须能解释患者的主诉和症状（如发热→WBC升高、CRP升高）
2. ✅ 数据真实性：数值必须在合理范围内（如WBC: 4-10×10⁹/L，不能出现40或0.5）
3. ✅ 逻辑关联性：多个指标之间要有内在联系（如贫血时RBC↓、Hb↓、HCT↓应同步）
4. ✅ 异常分级：轻度异常（偏离10-30%）、中度（30-50%）、重度（>50%或危急值）
5. ✅ 诊断价值：异常结果应指向具体疾病方向，避免"未见异常"的无效报告
6. ⚠️ 禁止编造：不要生成明显违背医学常识的结果（如糖尿病患者血糖正常）

【输出质量标准】
- 实验室检查：每项指标需包含【检测值 + 单位 + 参考范围 + 异常标记】
- 影像学检查：需包含【检查部位 + 影像表现 + 测量数据 + 印象诊断】
- 内镜检查：需包含【观察部位 + 黏膜描述 + 病变特征 + 病理提示】
- 异常结果：必须在summary中用1-2句话突出关键异常点及临床意义
"""
        
        user_prompt = f"""【患者临床信息】
检查项目：{test_name}
检查类型：{test_type}
患者主诉：{chief_complaint}

【病例背景】
{case_summary}
{physical_summary}
{existing_summary}

【任务要求】
请严格按照上述【检查类型与报告规范】中的格式模板，生成该检查的详细结果报告。输出必须为严格的JSON格式：

{{
  "test_name": "{test_name}",
  "result": "完整的检查结果描述（必须严格遵循上述报告格式模板，包含所有必需项目、具体数值、单位、参考范围）",
  "abnormal": true/false,
  "summary": "关键发现摘要（1-2句话，用通俗语言说明主要异常及临床意义）",
  "key_findings": ["关键异常1", "关键异常2", "..."],
  "clinical_significance": "该结果对诊断的提示意义"
}}

【关键要求】
1. result字段必须完全按照上述报告格式模板生成，不得简化或省略
2. 实验室检查(lab)：每个指标必须包含【检测值 单位 箭头标记 (参考范围)】
3. 影像学检查(imaging)：必须包含【检查所见 + 病灶描述 + 印象/诊断】
4. 功能检查(functional)：必须包含【波形/节律描述 + 数值指标 + 诊断结论】
5. 内镜检查(endoscopy)：必须包含【各部位观察 + 病灶详细描述 + 内镜诊断】

【异常判断标准】
- 若患者有明显症状（如发热、疼痛、出血），相关检查应显示异常
- 异常程度应与症状严重度匹配（轻症→轻度异常，重症→显著异常）
- 多个相关指标应协同变化（不要只有一个指标异常）

【正常结果要求】
- 即使正常也要给出具体数值，不能只写"正常"或"未见异常"
- 数值应在参考范围中段（如WBC写6.5而非4.1或9.9）

请严格按照报告格式模板生成专业、真实的检查报告。"""
        
        try:
            obj, raw_text, finish_reason = self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=lambda: {
                    "test_name": test_name,
                    "result": f"{test_name}检查完成，未见明显异常。",
                    "abnormal": False,
                    "summary": "未见异常",
                    "key_findings": [],
                    "clinical_significance": "结果正常"
                },
                temperature=0.3  # 适中温度，保证结果合理性
            )
            
            # 确保包含必需字段
            result = {
                "test_name": obj.get("test_name", test_name),
                "test": test_name,
                "type": test_type,
                "result": obj.get("result", ""),
                "abnormal": obj.get("abnormal", False),
                "summary": obj.get("summary", ""),
                "key_findings": obj.get("key_findings", []),
                "clinical_significance": obj.get("clinical_significance", ""),
                "source": "llm_generated",  # 标记来源
                "timestamp": None,
            }
            
            return result
            
        except Exception as e:
            # LLM失败时使用fallback
            return self._generate_fallback_result(test_name, test_type)
    
    def _generate_fallback_result(self, test_name: str, test_type: str) -> dict[str, Any]:
        """
        生成fallback检查结果（规则模式）- 使用真实的医学报告格式
        
        Args:
            test_name: 检查名称
            test_type: 检查类型
            
        Returns:
            默认的检查结果字典
        """
        # 更真实的正常结果（按照医学报告格式）
        normal_results = {
            "血常规": """白细胞计数(WBC) 6.5×10⁹/L (参考范围: 4.0-10.0)
红细胞计数(RBC) 4.8×10¹²/L (参考范围: 4.0-5.5)
血红蛋白(Hb) 145 g/L (参考范围: 120-160)
血小板计数(PLT) 220×10⁹/L (参考范围: 100-300)
中性粒细胞百分比(NEUT%) 60% (参考范围: 50-70)
淋巴细胞百分比(LYMPH%) 30% (参考范围: 20-40)""",
            
            "肝功能": """丙氨酸氨基转移酶(ALT) 25 U/L (参考范围: 0-40)
天门冬氨酸氨基转移酶(AST) 28 U/L (参考范围: 0-40)
总蛋白(TP) 70 g/L (参考范围: 60-80)
白蛋白(ALB) 45 g/L (参考范围: 35-55)
总胆红素(TBIL) 15 μmol/L (参考范围: 5-21)
直接胆红素(DBIL) 5 μmol/L (参考范围: 0-7)""",
            
            "肾功能": """尿素氮(BUN) 5.5 mmol/L (参考范围: 2.9-8.2)
肌酐(Cr) 85 μmol/L (参考范围: 53-106)
尿酸(UA) 320 μmol/L (参考范围: 208-428)
肾小球滤过率(eGFR) 95 mL/min/1.73m² (参考范围: >90)""",
            
            "凝血功能": """凝血酶原时间(PT) 12.0秒 (参考范围: 11-13)
国际标准化比值(INR) 1.0 (参考范围: 0.8-1.2)
活化部分凝血活酶时间(APTT) 35秒 (参考范围: 28-43)
纤维蛋白原(FIB) 3.0 g/L (参考范围: 2.0-4.0)
D-二聚体(D-Dimer) 0.3 mg/L (参考范围: <0.5)""",
            
            "心电图": """窦性心律，心率 72次/分 (正常: 60-100)
PR间期 0.16秒，QRS时限 0.09秒
QT/QTc 0.38/0.41秒
各导联ST-T未见异常
诊断：窦性心律，心电图正常""",
            
            "胸部CT": """检查技术：平扫
检查所见：双肺纹理清晰，未见实质性病灶。双肺门影不大，纵隔内未见肿大淋巴结。气管及主支气管通畅。心影大小正常。双侧胸腔未见积液。
印象：双肺未见明显异常。""",
            
            "腹部超声": """肝脏：大小正常，包膜光整，实质回声均匀，肝内血管走行正常。
胆囊：大小形态正常，壁不厚，腔内未见异常回声。
胰腺：显示清晰，回声均匀，胰管不扩张。
脾脏：大小形态正常，实质回声均匀。
双肾：大小形态正常，皮髓质分界清，集合系统未见分离。
超声提示：肝、胆、胰、脾、双肾未见异常。""",
        }
        
        # 尝试匹配检查名称
        for key, value in normal_results.items():
            if key in test_name or test_name in key:
                return {
                    "test_name": test_name,
                    "test": test_name,
                    "type": test_type,
                    "result": value,
                    "abnormal": False,
                    "summary": "检查结果正常，未见异常",
                    "source": "fallback",
                    "timestamp": None,
                }
        
        # 默认通用结果
        result_text = f"{test_name}检查完成，各项指标均在正常范围内，未见明显异常。"
        
        return {
            "test_name": test_name,
            "test": test_name,
            "type": test_type,
            "result": result_text,
            "abnormal": False,
            "summary": "未见异常",
            "source": "fallback",
            "timestamp": None,
        }
    
    def _extract_case_summary(self, case_data: dict[str, Any]) -> str:
        """
        从病例数据中提取摘要信息
        
        Args:
            case_data: 病例数据
            
        Returns:
            病例摘要文本
        """
        if not case_data:
            return "无病例信息"
        
        case_info = case_data.get("Case Information", "")
        
        # 提取关键部分（前300字符）
        summary = case_info[:300].strip()
        if len(case_info) > 300:
            summary += "..."
        
        return summary
    
    def _get_report_template(self, test_name: str, test_type: str) -> str:
        """
        根据检查项目类型返回专业的报告格式模板
        
        Args:
            test_name: 检查名称
            test_type: 检查类型
            
        Returns:
            报告格式指导文本
        """
        # 实验室检查模板
        lab_templates = {
            "血常规": """【血常规检查报告格式】
报告单位：××医院检验科
检查项目必须包含：
- 白细胞计数(WBC) ×10⁹/L  参考范围：4.0-10.0
- 红细胞计数(RBC) ×10¹²/L  参考范围：男4.0-5.5，女3.5-5.0
- 血红蛋白(Hb) g/L  参考范围：男120-160，女110-150
- 血小板计数(PLT) ×10⁹/L  参考范围：100-300
- 中性粒细胞百分比(NEUT%) %  参考范围：50-70
- 淋巴细胞百分比(LYMPH%) %  参考范围：20-40
- 单核细胞百分比(MONO%) %  参考范围：3-8
- 嗜酸性粒细胞百分比(EOS%) %  参考范围：0.5-5
- 红细胞压积(HCT) %  参考范围：35-50
- 平均红细胞体积(MCV) fL  参考范围：80-100
异常项目用 ↑ 或 ↓ 标注""",
            
            "肝功能": """【肝功能检查报告格式】
报告单位：××医院检验科
检查项目必须包含：
- 丙氨酸氨基转移酶(ALT) U/L  参考范围：0-40
- 天门冬氨酸氨基转移酶(AST) U/L  参考范围：0-40
- 总蛋白(TP) g/L  参考范围：60-80
- 白蛋白(ALB) g/L  参考范围：35-55
- 球蛋白(GLO) g/L  参考范围：20-35
- 白球比(A/G)  参考范围：1.2-2.4
- 总胆红素(TBIL) μmol/L  参考范围：5-21
- 直接胆红素(DBIL) μmol/L  参考范围：0-7
- 间接胆红素(IBIL) μmol/L  参考范围：5-14
- γ-谷氨酰转肽酶(GGT) U/L  参考范围：0-50
- 碱性磷酸酶(ALP) U/L  参考范围：40-150""",
            
            "肾功能": """【肾功能检查报告格式】
报告单位：××医院检验科
检查项目必须包含：
- 尿素氮(BUN) mmol/L  参考范围：2.9-8.2
- 肌酐(Cr) μmol/L  参考范围：男53-106，女44-97
- 尿酸(UA) μmol/L  参考范围：男208-428，女155-357
- 肾小球滤过率(eGFR) mL/min/1.73m²  参考范围：>90""",
            
            "血脂": """【血脂检查报告格式】
报告单位：××医院检验科
检查项目必须包含：
- 总胆固醇(TC) mmol/L  参考范围：<5.2
- 甘油三酯(TG) mmol/L  参考范围：<1.7
- 高密度脂蛋白胆固醇(HDL-C) mmol/L  参考范围：>1.0
- 低密度脂蛋白胆固醇(LDL-C) mmol/L  参考范围：<3.4
- 载脂蛋白A1(ApoA1) g/L  参考范围：1.0-1.6
- 载脂蛋白B(ApoB) g/L  参考范围：0.6-1.1""",
            
            "凝血功能": """【凝血功能检查报告格式】
报告单位：××医院检验科
检查项目必须包含：
- 凝血酶原时间(PT) 秒  参考范围：11-13
- 国际标准化比值(INR)  参考范围：0.8-1.2
- 活化部分凝血活酶时间(APTT) 秒  参考范围：28-43
- 纤维蛋白原(FIB) g/L  参考范围：2.0-4.0
- D-二聚体(D-Dimer) mg/L  参考范围：<0.5""",
        }
        
        # 影像学检查模板
        imaging_templates = {
            "CT": f"""【{test_name} CT检查报告格式】
报告单位：××医院影像科
报告结构：
1. 检查技术：平扫/增强扫描
2. 检查所见：
   - 扫描范围各层面描述
   - 病灶位置、大小（cm×cm×cm）、密度（HU值）
   - 周围结构关系
   - 淋巴结/血管情况
3. 印象/诊断：
   - 主要病变诊断
   - 鉴别诊断建议
示例："肺窗示右下肺见片状高密度影，大小约4.2cm×3.5cm，CT值约+35HU，边界模糊，周围见渗出。纵隔窗示右肺门淋巴结肿大，短径约1.2cm。双侧胸腔未见积液。印象：右下肺炎症改变，建议抗感染治疗后复查。\"""",
            
            "MRI": f"""【{test_name} MRI检查报告格式】
报告单位：××医院影像科
报告结构：
1. 检查序列：T1WI、T2WI、FLAIR、DWI等
2. 检查所见：
   - T1WI信号特征（等/高/低信号）
   - T2WI信号特征
   - 病灶位置、大小、形态
   - 增强扫描特点（如有）
3. 印象/诊断
示例："T1WI示左侧额叶见类圆形稍低信号影，大小约2.5cm×2.3cm，边界清楚；T2WI及FLAIR呈高信号，DWI呈高信号，ADC图呈低信号；增强扫描病灶环形强化。周围见指状水肿带。中线结构无移位。印象：左额叶占位性病变，考虑胶质瘤可能，建议活检明确。\"""",
            
            "X光": f"""【{test_name} X线检查报告格式】
报告单位：××医院放射科
报告结构：
1. 检查体位：正位/侧位/斜位
2. 检查所见：
   - 骨质密度、连续性
   - 关节间隙
   - 软组织情况
3. 印象/诊断
示例："正侧位片示：右侧第4-5肋骨见线性致密影，局部骨皮质连续性中断，周围软组织肿胀。肺野清晰，未见气胸征象。印象：右侧第4-5肋骨骨折。\"""",
            
            "超声": f"""【{test_name}超声检查报告格式】
报告单位：××医院超声科
报告结构：
1. 检查所见：
   - 脏器大小、形态、边界
   - 实质回声（均匀/不均匀）
   - 病灶位置、大小（cm×cm）、回声特征
   - 血流信号（CDFI）
2. 超声提示/诊断
示例："肝脏大小正常，包膜光整，实质回声均匀。于右叶见一低回声结节，大小约2.1cm×1.8cm，边界尚清，内部回声不均，CDFI示周边及内部可见点状血流信号。胆囊、脾脏、双肾未见异常。超声提示：肝右叶占位性病变，建议进一步检查明确性质。\"""",
        }
        
        # 功能检查模板
        functional_templates = {
            "心电图": """【心电图检查报告格式】
报告单位：××医院心电图室
报告结构：
1. 心律、心率
2. P波、PR间期、QRS波群、QT间期
3. ST段、T波
4. 诊断结论
示例：
"窦性心律，心率 68次/分
PR间期 0.16秒，QRS时限 0.09秒，QT/QTc 0.38/0.42秒
V1-V6导联R波递增良好
II、III、aVF导联ST段压低0.1mV，T波低平
诊断：1. 窦性心律  2. 下壁导联ST-T改变，提示心肌缺血可能，建议结合临床\"""",
            
            "脑电图": """【脑电图检查报告格式】
报告单位：××医院神经电生理室
报告结构：
1. 背景节律：α节律频率、波幅
2. 睡眠相关波形（如有）
3. 异常放电（棘波、尖波、慢波）
4. 诱发试验结果
5. 结论
示例："清醒期背景为8-9Hz α节律，波幅30-50μV，睁眼抑制良好。双侧颞区可见散在θ慢波，波幅40-60μV。过度换气后双侧半球同步出现3Hz棘慢波综合，持续5秒。闪光刺激未见异常。结论：异常脑电图，双侧同步棘慢波发放，符合全面性癫痫放电。\"""",
        }
        
        # 内镜检查模板
        endoscopy_templates = {
            "胃镜": """【胃镜检查报告格式】
报告单位：××医院内镜中心
报告结构：
1. 食管：黏膜、蠕动、贲门
2. 胃：黏膜颜色、皱襞、蠕动、胃液
3. 十二指肠：球部、降部黏膜
4. 病灶描述：位置、大小、形态、颜色
5. 内镜诊断
示例："食管黏膜光滑，未见异常。胃底、胃体黏膜充血水肿，皱襞增粗，胃窦部见一不规则溃疡，大小约1.5cm×1.2cm，边缘隆起，底部覆白苔，周围黏膜充血明显，取活检2块。幽门开放好，十二指肠球部及降部黏膜未见异常。内镜诊断：1. 慢性萎缩性胃炎  2. 胃窦溃疡（活检待病理）\"""",
            
            "肠镜": """【结肠镜检查报告格式】
报告单位：××医院内镜中心
报告结构：
1. 进镜深度
2. 各段肠道描述（直肠-回盲部）
3. 病灶描述：位置、大小、形态、Yamada分型
4. 处理措施
5. 内镜诊断
示例："进镜至回盲部，退镜观察。回盲部、升结肠、横结肠、降结肠、乙状结肠黏膜光滑，血管纹理清晰。直肠距肛缘约10cm处见一有蒂息肉，大小约0.8cm×0.7cm，Yamada II型，表面光滑，予圈套器切除，创面喷洒止血粉。病理送检。内镜诊断：直肠息肉（已电切，送病理）\"""",
        }
        
        # 病理检查模板
        pathology_templates = {
            "活检": """【组织病理学检查报告格式】
报告单位：××医院病理科
报告结构：
1. 送检标本：部位、数量、大小
2. 大体描述：颜色、质地、切面
3. 镜下所见：组织结构、细胞形态、特殊染色
4. 免疫组化结果（如有）
5. 病理诊断
示例："送检标本：胃窦黏膜组织2块，大小0.2cm×0.2cm。镜下：胃黏膜组织，腺体排列紊乱，固有层见大量淋巴细胞、浆细胞浸润。腺上皮细胞增生，部分呈肠化生改变。间质纤维组织增生。免疫组化：Ki-67(+)约15%，P53(-)，Her-2(-)。病理诊断：（胃窦）慢性萎缩性胃炎伴肠化生。\"""",
        }
        
        # 根据检查类型选择模板
        if test_type == "lab":
            # 尝试精确匹配
            for key in lab_templates:
                if key in test_name or test_name in key:
                    return lab_templates[key]
            # 通用实验室检查格式
            return """【实验室检查报告通用格式】
必须包含：
- 项目名称
- 检测值 + 单位
- 参考范围
- 异常标记(↑↓)
每个指标单独一行，格式：项目名称  检测值 单位 [↑/↓]  (参考范围: X-X)"""
        
        elif test_type == "imaging":
            for key in imaging_templates:
                if key in test_name or test_name in key:
                    return imaging_templates[key]
            return imaging_templates.get("CT", "影像学检查报告")
        
        elif test_type == "functional":
            for key in functional_templates:
                if key in test_name or test_name in key:
                    return functional_templates[key]
            return functional_templates.get("心电图", "功能检查报告")
        
        elif test_type == "endoscopy":
            for key in endoscopy_templates:
                if key in test_name or test_name in key:
                    return endoscopy_templates[key]
            return endoscopy_templates.get("胃镜", "内镜检查报告")
        
        elif "病理" in test_name or "活检" in test_name:
            return pathology_templates["活检"]
        
        else:
            return """【通用医学检查报告格式】
- lab（实验室检查）：必须提供精确数值、单位、参考范围，标注↑↓箭头
- imaging（影像学检查）：描述解剖结构、密度/信号、病变特征（位置、大小、形态）
- functional（功能检查）：描述波形特征、间期、频率、异常模式
- endoscopy（内镜检查）：描述黏膜外观、病变性质、活检部位"""
    
    def get_processing_summary(self) -> dict[str, Any]:
        """
        获取检验科处理摘要
        
        Returns:
            包含处理统计的字典
        """
        abnormal_count = sum(1 for test in self._processed_tests if test.get("abnormal"))
        
        return {
            "total_tests": len(self._processed_tests),
            "abnormal_count": abnormal_count,
            "normal_count": len(self._processed_tests) - abnormal_count,
            "tests": self._processed_tests,
        }
