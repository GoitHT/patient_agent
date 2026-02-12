# 对话质量评估与高质量对话库实施文档

## 📋 概述

本次实施在S4专科问诊节点中集成了**对话质量评估系统**，实现了对医患问答的实时评估和高质量对话的自动存储，用于建设高质量对话知识库。

## 🎯 实施目标

1. **实时评估**：在S4节点的每轮问答后，自动评估对话质量
2. **多维度评分**：从医生和患者两个维度进行综合评估
3. **自动存储**：将高质量对话自动存入向量库，供后续检索使用
4. **质量统计**：问诊结束后展示整体质量统计报告

---

## 📊 评估指标体系

### 患者回答评估指标

| 指标 | 符号 | 范围 | 说明 | 评估方法 |
|------|------|------|------|---------|
| **相关性** | α | [0,1] | 回答是否直接且充分地回应问题 | 问答语义相似度（余弦距离） |
| **忠实性** | β | [0,1] | 回答是否基于病历信息且符合SP要求 | LLM评估或默认评分 |
| **鲁棒性** | γ | [0,1] | 是否泄露不应泄露的信息 | 关键词检测+长度惩罚 |
| **能力** | - | [0,1] | 综合指标 | (α + β + γ) / 3 |

#### 相关性评估细节

- **基于语义嵌入**：使用 `BAAI/bge-large-zh-v1.5` 模型计算问答语义相似度
- **余弦相似度**：将[-1,1]范围映射到[0,1]
- **长度惩罚**：回答过短（<5字符）会降低相关性分数

#### 忠实性评估细节

- **有LLM**：使用LLM评估回答是否符合病历信息
- **无LLM**：默认分数0.75（假设SP遵循病历）

#### 鲁棒性评估细节

检测以下泄露信息：
- 疾病诊断名称（"癫痫"、"脑梗"、"肿瘤"等）
- 详细病历描述（"病历"、"CT显示"、"MRI显示"等）
- 医学专业术语（"症状学"、"体征"、"鉴别诊断"等）
- 过长的详细描述（>200字符）

---

### 医生提问评估指标

| 指标 | 符号 | 范围 | 说明 | 评估方法 |
|------|------|------|------|---------|
| **具体性** | δ | [0,1] | 问题是否精确和明确 | 关键词检测（时间/地点/程度描述） |
| **针对性** | ε | [0,1] | 问题是否有助于诊断 | 诊断要素关键词匹配 |
| **专业性** | ζ | [0,1] | 是否体现医学专业性 | 医学术语使用频率 |
| **质量** | - | [0,1] | 综合指标 | (δ + ε + ζ) / 3 |

#### 具体性评估细节

**高分关键词**（具体问题）：
- 时间："什么时候"、"多久"、"多长时间"、"几次"
- 地点/部位："哪里"、"什么部位"
- 性质/程度："什么性质"、"什么程度"
- 变化："伴随"、"诱因"、"缓解"、"加重"

**低分关键词**（开放问题）：
- "怎么样"、"如何"、"情况"、"有没有"、"是否"

#### 针对性评估细节

**诊断要素关键词**：
- 症状特征：症状、疼痛、不适、感觉、表现
- 时间特征：开始、持续、频率、发作
- 诱因和缓解：诱因、缓解、加重、因素
- 伴随症状：伴随、同时、还有
- 既往史：以前、曾经、病史、治疗
- 系统回顾：头痛、眩晕、恶心、呕吐、发热

#### 专业性评估细节

**专业术语评分**：
- 症状描述：性质、部位、持续时间、程度、频率
- 医学术语：伴随症状、诱发因素、缓解因素、加重因素
- 病史采集：既往史、家族史、过敏史、用药史
- 专科术语（神经内科）：意识、肢体、感觉、运动、反射

---

### 综合评分

```
综合得分 = (医生质量 + 患者能力) / 2
高质量对话阈值 = 0.7
```

当综合得分 ≥ 0.7 时，对话被标记为高质量并存储到向量库。

---

## 🏗️ 技术实现

### 1. 核心模块：`src/rag/qa_evaluator.py`

#### 主要类

```python
class DialogueQualityEvaluator:
    """对话质量评估器"""
    
    def evaluate_patient_answer(...) -> PatientAnswerMetrics
    def evaluate_doctor_question(...) -> DoctorQuestionMetrics
    def evaluate_dialogue(...) -> DialogueQualityScore
    def store_high_quality_dialogue(...) -> bool
```

#### 数据结构

```python
@dataclass
class PatientAnswerMetrics:
    relevance: float      # α: 相关性
    faithfulness: float   # β: 忠实性
    robustness: float     # γ: 鲁棒性
    
    @property
    def ability(self) -> float:
        return (self.relevance + self.faithfulness + self.robustness) / 3.0

@dataclass
class DoctorQuestionMetrics:
    specificity: float     # δ: 具体性
    targetedness: float    # ε: 针对性
    professionalism: float # ζ: 专业性
    
    @property
    def quality(self) -> float:
        return (self.specificity + self.targetedness + self.professionalism) / 3.0

@dataclass
class DialogueQualityScore:
    question: str
    answer: str
    patient_metrics: PatientAnswerMetrics
    doctor_metrics: DoctorQuestionMetrics
    
    @property
    def overall_score(self) -> float:
        return (self.doctor_metrics.quality + self.patient_metrics.ability) / 2.0
```

---

### 2. S4节点集成

#### 集成位置
文件：`src/graphs/dept_subgraphs/common_specialty_subgraph.py`
节点：`s4_specialty_interview`

#### 集成流程

```
问诊开始
    ↓
初始化评估器 → DialogueQualityEvaluator
    ↓
问诊循环（每轮）
    ├─ 医生提问
    ├─ 患者回答
    ├─ 【新增】评估对话质量
    ├─ 【新增】存储高质量对话
    └─ 记录到state
    ↓
问诊结束
    ↓
【新增】展示质量统计
    ↓
医生总结主诉
```

#### 代码示例

```python
# 1. 初始化评估器
qa_evaluator = DialogueQualityEvaluator(
    llm=llm,
    spllm_root=spllm_root,
    high_quality_threshold=0.7
)

# 2. 每轮问答后评估
for i in range(remaining_questions):
    question = doctor_agent.generate_one_question(...)
    answer = patient_agent.respond_to_doctor(question, ...)
    
    # 评估对话质量
    dialogue_score = qa_evaluator.evaluate_dialogue(
        question=question,
        answer=answer,
        patient_info=patient_info,
        context=context
    )
    
    qa_scores.append(dialogue_score)
    
    # 存储高质量对话
    if dialogue_score.is_high_quality():
        success = qa_evaluator.store_high_quality_dialogue(
            dialogue_score=dialogue_score,
            patient_id=state.patient_id,
            metadata={...}
        )
        if success:
            high_quality_count += 1

# 3. 展示质量统计
if qa_scores:
    avg_doctor_quality = sum(s.doctor_metrics.quality for s in qa_scores) / len(qa_scores)
    avg_patient_ability = sum(s.patient_metrics.ability for s in qa_scores) / len(qa_scores)
    avg_overall = sum(s.overall_score for s in qa_scores) / len(qa_scores)
    
    logger.info(f"医生平均质量: {avg_doctor_quality:.2f}/1.0")
    logger.info(f"患者平均能力: {avg_patient_ability:.2f}/1.0")
    logger.info(f"综合得分: {avg_overall:.2f}/1.0")
    logger.info(f"高质量对话: {high_quality_count}/{len(qa_scores)} 轮")
```

---

### 3. 高质量对话存储

#### 存储位置
向量库：`SPLLM-RAG1/chroma/HighQualityQA_db/`

#### 存储内容

```python
doc_content = f"【问】{question}\n【答】{answer}"

doc_metadata = {
    "patient_id": patient_id,
    "question": question,
    "answer": answer,
    "overall_score": overall_score,
    "doctor_quality": doctor_quality,
    "patient_ability": patient_ability,
    "dept": dept,
    "stage": "specialty_interview",
    ...
}
```

#### 检索使用

存储的高质量对话可在后续问诊中检索：

```python
# S4节点中已集成
qa_chunks = retriever.retrieve(
    f"{dept_name} 专科问诊 问题",
    filters={"scenario": "quality_qa"},
    k=3
)
```

---

## 📈 输出示例

### 问诊过程中的评估

```
💬 问诊第 1 轮
  医生: 请详细描述您头痛的性质，是搏动性、压迫性还是针刺样的疼痛？
  患者: 主要是右侧太阳穴部位的搏动性疼痛。
  📊 Q1 质量评分: 医生=0.78, 患者=0.85, 综合=0.82
  ✅ 高质量对话已存储

💬 问诊第 2 轮
  医生: 头痛发作时是否伴有恶心、呕吐或畏光？
  患者: 有的，疼的厉害的时候会恶心。
  📊 Q2 质量评分: 医生=0.82, 患者=0.80, 综合=0.81
  ✅ 高质量对话已存储
```

### 问诊结束后的统计

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💎 对话质量统计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📈 对话轮数: 3 轮
  👨‍⚕️ 医生平均质量: 0.78/1.0
     • 具体性: 0.75
     • 针对性: 0.80
     • 专业性: 0.80
  👤 患者平均能力: 0.82/1.0
     • 相关性: 0.85
     • 忠实性: 0.80
     • 鲁棒性: 0.82
  🎯 综合得分: 0.80/1.0
  ✨ 高质量对话: 2/3 轮已存入知识库
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🧪 测试验证

### 测试脚本

```bash
cd C:\Users\32765\Desktop\langgraph\patient_agent
conda activate langgraph
python test_qa_quality.py
```

### 测试案例

测试脚本包含4个典型案例：

1. **高质量对话** - 具体且专业的提问
2. **中等质量对话** - 开放式提问
3. **低质量对话** - 患者泄露信息
4. **高质量对话** - 针对性强的提问

### 测试输出

```
🧪 测试对话质量评估功能
════════════════════════════════════════════════════════════════════════════════

【案例 1】高质量对话 - 具体且专业的提问
问：请详细描述您头痛的性质，是搏动性、压迫性还是针刺样的疼痛？
答：主要是右侧太阳穴部位的搏动性疼痛，跟心跳一样一跳一跳的。

📊 质量评分：
  👨‍⚕️ 医生提问质量: 0.780
     • 具体性 (δ): 0.750
     • 针对性 (ε): 0.800
     • 专业性 (ζ): 0.790
  👤 患者回答能力: 0.850
     • 相关性 (α): 0.850
     • 忠实性 (β): 0.750
     • 鲁棒性 (γ): 0.950
  🎯 综合得分: 0.815
  ✨ 高质量对话 (≥ 0.7)
```

---

## 🔄 数据流

```
S4 问诊节点
    ↓
医生提问 + 患者回答
    ↓
DialogueQualityEvaluator.evaluate_dialogue()
    ├─ 评估患者回答
    │   ├─ 相关性（语义相似度）
    │   ├─ 忠实性（病历一致性）
    │   └─ 鲁棒性（信息泄露检测）
    ├─ 评估医生提问
    │   ├─ 具体性（问题明确性）
    │   ├─ 针对性（诊断相关性）
    │   └─ 专业性（医学术语）
    └─ 综合评分
    ↓
DialogueQualityScore
    ├─ overall_score >= 0.7 ?
    │   ├─ Yes → 存储到 HighQualityQA_db
    │   └─ No  → 仅记录评分
    └─ 追加到 qa_scores 列表
    ↓
问诊结束后统计展示
    ├─ 平均医生质量
    ├─ 平均患者能力
    ├─ 综合得分
    └─ 高质量对话数量
```

---

## 📊 效果评估

### 预期效果

1. **质量提升**
   - 自动识别高质量对话
   - 为后续问诊提供优质范例
   - 持续积累最佳实践

2. **知识积累**
   - 高质量对话库自动增长
   - S4节点可检索优质问诊问题
   - 提升问诊专业性和效率

3. **评估反馈**
   - 实时了解对话质量
   - 识别潜在问题（信息泄露、问题不明确等）
   - 支持系统持续优化

---

## ⚙️ 配置选项

### 评估器配置

```python
evaluator = DialogueQualityEvaluator(
    llm=llm,                         # LLM客户端（可选）
    spllm_root=spllm_root,           # SPLLM-RAG1根目录
    embed_model="BAAI/bge-large-zh-v1.5",  # 嵌入模型
    high_quality_threshold=0.7       # 高质量阈值
)
```

### 调整建议

- **提高门槛**：设置 `high_quality_threshold=0.8` 只存储顶级对话
- **降低门槛**：设置 `high_quality_threshold=0. 6` 扩大知识库
- **禁用存储**：不初始化评估器或设置 `threshold=1.0`

---

## 📝 注意事项

### 1. 依赖项

- **嵌入模型**：需要 `BAAI/bge-large-zh-v1.5` 已缓存
- **向量库**：`SPLLM-RAG1/chroma/HighQualityQA_db/` 需存在
- **LLM**：忠实性评估可选使用LLM（否则使用默认分数）

### 2. 性能考虑

- **评估开销**：每轮问答增加约0.1-0.2秒
- **存储开销**：每个高质量对话约10-20KB
- **向量库大小**：建议定期清理或归档

### 3. 质量控制

- **阈值调整**：根据实际效果调整 `high_quality_threshold`
- **定期审查**：人工抽查存储的高质量对话
- **持续优化**：基于反馈优化评估算法

---

## 🚀 后续优化方向

1. **评估算法优化**
   - 引入更复杂的语义分析
   - 基于医学知识图谱的评估
   - 机器学习模型替代规则评分

2. **存储策略优化**
   - 去重机制（避免存储相似对话）
   - 分类存储（按科室、症状分类）
   - 版本管理（跟踪对话演变）

3. **应用场景扩展**
   - 医生培训：展示最佳问诊范例
   - 患者教育：展示标准回答方式
   - 质量监控：识别异常对话模式

---

## 📞 技术支持

如遇问题，请检查：

1. **评估器未启用**：检查日志中是否有 "对话质量评估器已启用"
2. **存储失败**：检查 `SPLLM-RAG1/chroma/HighQualityQA_db/` 是否存在
3. **评分异常**：检查嵌入模型是否正确加载
4. **无高质量对话**：尝试降低 `high_quality_threshold` 阈值

---

**实施完成日期**: 2026年2月11日
**版本**: v1.0
**状态**: ✅ 已完成并测试
