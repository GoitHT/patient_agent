# RAG 功能增强实施文档

## 📋 概述

本次实施完成了对系统中 RAG（检索增强生成）功能的全面增强，涵盖 7 个关键节点，实现了多维度知识库检索、场景化检索策略和患者历史记忆功能。

## 🎯 实施内容

### 1. 增强 AdaptiveRAGRetriever 核心能力

**文件**: `src/rag/adaptive_rag_retriever.py`

#### 新增功能：

1. **场景化检索策略** - 根据不同使用场景优化检索
   - `scenario="patient_history"`: 专注患者历史记录（C5/C8/C14）
   - `scenario="clinical_case"`: 专注临床案例（C11/C12）
   - `scenario="quality_qa"`: 专注高质量问诊（S4）
   - 默认策略：均衡检索所有库

2. **患者历史检查记录检索** - 避免重复开单
   - 新增方法: `retrieve_patient_test_history()`
   - 支持按检查关键词过滤
   - 用于 C8 节点检查重复开单

3. **启用临床案例库** - 提供相似病例参考
   - 激活 `ClinicalCase_db` 检索
   - 支持基于症状和检查结果的案例匹配

#### 代码示例：

```python
# 场景化检索
chunks = retriever.retrieve(
    "患者主诉 症状描述",
    filters={"patient_id": "P001", "scenario": "patient_history"},
    k=3
)

# 历史检查记录检索
test_history = retriever.retrieve_patient_test_history(
    patient_id="P001",
    test_keywords=["CT", "MRI"],
    k=5
)
```

---

### 2. C5 节点 - 问诊准备

**文件**: `src/graphs/common_opd_graph.py`

#### 增强内容：

1. **检索医院通用 SOP** ✅ 原有功能
2. **新增：检索患者历史对话记录** 🆕
   - 查询：患者既往主诉、问答记录、关注点
   - 用途：帮助医生快速掌握患者背景
   - 效果：生成个性化问诊思路与沟通策略

#### 检索流程：

```
🔍 检索医院通用SOP → 获取标准流程
            ↓
🔍 检索患者历史对话 → 了解患者背景（如有patient_id）
            ↓
📝 初始化问诊记录 → 准备进入专科问诊
```

---

### 3. C8 节点 - 开单与准备说明

**文件**: `src/graphs/common_opd_graph.py`

#### 增强内容：

1. **检索医院通用流程 SOP** ✅ 原有功能
2. **新增：检索患者历史检查开单记录** 🆕
   - 查询：该患者历史检查/检验记录
   - 检查：是否近期已做过相同检查（如"3天前已做CT"）
   - 用途：避免重复开单，减少患者负担

#### 检索流程：

```
🔍 检索通用流程SOP → 获取缴费/预约流程
            ↓
🔍 检索患者历史检查记录 → 检查重复开单
            ↓
⚠️  发现重复 → 提示医生（可选择不重复）
✅ 无重复 → 正常开单
```

#### 输出示例：

```
🔍 检索患者历史检查记录（检查重复开单）...
  ⚠️  发现 2 条历史检查记录
     • 2024-01-15: 头颅CT平扫，未见异常...
     • 2024-01-20: 血常规检查，结果正常...
```

---

### 4. C11 节点 - 报告回诊

**文件**: `src/graphs/common_opd_graph.py`

#### 增强内容：

1. **新增：检索相似临床案例** 🆕
   - 查询：基于异常检查结果的相似病例
   - 内容：具有相似异常指标的真实病例（含处理与转归）
   - 用途：结合真实世界证据，准确解读报告

#### 检索流程：

```
📊 获取检查结果 → 识别异常项目
            ↓
🔍 检索相似临床案例 → 查找相似异常指标的病例
            ↓
📖 案例参考 → 辅助制定针对性后续方案
```

#### 输出示例：

```
🔍 检索相似临床案例...
  ✅ 找到 3 个相似案例
     [1] 患者A，头痛+MRI异常，最终诊断为...
     [2] 患者B，相似症状，治疗方案为...
```

---

### 5. C12 节点 - 综合分析诊断

**文件**: `src/graphs/common_opd_graph.py`

#### 增强内容：

1. **检索医学指南** ✅ 原有功能
2. **新增：检索相似临床案例** 🆕
   - 查询：主诉 + 科室 + 检查结果关键词
   - 内容：诊断标准、相似病例、循证治疗方案
   - 用途：综合理论与实践，辅助准确诊断

#### 检索流程：

```
🔍 检索文书模板 → 获取标准格式
            ↓
🔍 检索诊后处置SOP → 获取流程规范
            ↓
🔍 检索专科诊疗方案 → 获取治疗指南
            ↓
🔍 检索相似临床案例 → 获取真实病例参考
            ↓
🤖 LLM综合分析 → 生成最终诊断和方案
```

#### 输出示例：

```
🔍 检索诊断相关知识...
  ✅ 专科知识: 4 条
🔍 检索相似临床案例（用于诊断参考）...
  ✅ 找到 5 个相似案例
  ✅ 共检索到 17 个知识片段
```

---

### 6. C14 节点 - 生成文书

**文件**: `src/graphs/common_opd_graph.py`

#### 增强内容：

1. **新增：检索患者历史病历** 🆕
   - 查询：该患者既往病历、诊断、过敏史、既往史
   - 用途：确保病史一致性，避免信息冲突
   - 效果：自动生成连贯、完整的标准化医疗文书

#### 检索流程：

```
🔍 检索患者历史病历 → 获取既往诊断、过敏史等
            ↓
📝 构建历史上下文 → 整合关键信息
            ↓
🤖 LLM生成文书 → 门诊病历、诊断证明、病假条、宣教单
            ↓
✅ 确保病史一致性 → 历史信息自动关联
```

#### 输出示例：

```
🔍 检索患者历史病历信息...
  ✅ 找到 3 条历史病历记录
     • 已整合历史病历信息用于生成文书
     
【患者历史病历摘要】
2024-01-10: 诊断-偏头痛，药物-XXX
2023-12-05: 过敏史-青霉素过敏
```

---

### 7. S4 节点 - 专科问诊

**文件**: `src/graphs/dept_subgraphs/common_specialty_subgraph.py`

#### 增强内容：

1. **检索专科知识库** ✅ 原有功能
2. **新增：检索高质量问诊库** 🆕
   - 查询：结构化问诊问题链、专科问诊要点
   - 用途：在动态问诊中推荐高质量问题
3. **新增：检索相似临床案例** 🆕
   - 查询：相似症状患者的完整问诊流程
   - 用途：参考典型病例的问诊思路

#### 检索流程：

```
🔍 检索专科知识库 → 红旗症状、检查建议
            ↓
🔍 检索高质量问诊库 → 推荐高质量问题
            ↓
🔍 检索临床案例 → 相似症状的问诊流程
            ↓
👨‍⚕️ Agent动态问诊 → 利用所有检索结果生成问题
```

#### 医生提问增强：

```python
# 原来：只使用专科知识
question = doctor_agent.generate_one_question(
    context=context_desc,
    rag_chunks=chunks  # 仅专科知识
)

# 现在：整合多个知识库
question = doctor_agent.generate_one_question(
    context=context_desc,
    rag_chunks=chunks + qa_chunks + case_chunks  # 专科+问诊库+案例
)
```

#### 输出示例：

```
🔍 检索神经内科专科知识库...
  ✅ 专科知识: 4 条
  ✅ 高质量问诊参考: 3 条
  ✅ 临床案例参考: 2 条
```

---

## 🗄️ 数据库架构

### 现有向量库：

| 数据库名称 | 路径 | 用途 | 使用节点 |
|-----------|------|------|---------|
| **MedicalGuide_db** | `SPLLM-RAG1/chroma/MedicalGuide_db/` | 医学指南、诊疗规范 | C5, C8, C12, C15, S4, S6 |
| **ClinicalCase_db** | `SPLLM-RAG1/chroma/ClinicalCase_db/` | 临床案例库 | C11, C12, S4 |
| **UserHistory_db** | `SPLLM-RAG1/chroma/UserHistory_db/` | 患者历史记录 | C5, C8, C14 |

### 检索策略对比：

| 场景（scenario） | 优先检索库 | 辅助检索库 | 适用节点 |
|----------------|-----------|-----------|---------|
| `patient_history` | UserHistory | MedicalGuide | C5, C8, C14 |
| `clinical_case` | ClinicalCase | MedicalGuide | C11, C12 |
| `quality_qa` | HighQualityQA | ClinicalCase + UserHistory | S4 |
| 默认（无场景） | MedicalGuide + HighQualityQA | ClinicalCase + UserHistory | 其他 |

---

## 🚀 使用示例

### 示例 1: C5 问诊准备使用患者历史

```python
# 原来
chunks = self.retriever.retrieve(
    f"门诊 问诊要点 {state.chief_complaint}",
    filters={"dept": "hospital", "type": "sop"},
    k=4
)

# 现在
# 1. 检索通用SOP
chunks = self.retriever.retrieve(
    f"门诊 问诊要点 {state.chief_complaint}",
    filters={"dept": "hospital", "type": "sop"},
    k=4
)

# 2. 检索患者历史（如果有patient_id）
if state.patient_id:
    history_chunks = self.retriever.retrieve(
        f"患者主诉 症状描述 {state.chief_complaint}",
        filters={"patient_id": state.patient_id, "scenario": "patient_history"},
        k=3
    )
    # 医生可以了解患者既往就诊历史
```

### 示例 2: C8 避免重复开单

```python
# 检查患者是否近期做过相同检查
if state.patient_id and state.ordered_tests:
    test_keywords = [t.get('name') for t in state.ordered_tests]
    
    test_history = self.retriever.retrieve_patient_test_history(
        patient_id=state.patient_id,
        test_keywords=test_keywords,
        k=5
    )
    
    if test_history:
        # 提示医生：患者3天前刚做过CT检查
        # 医生可以选择：查看旧报告 or 重新开单
```

### 示例 3: S4 高质量问诊

```python
# 医生提问时参考高质量问诊库
qa_chunks = retriever.retrieve(
    f"{dept_name} 专科问诊 问题",
    filters={"scenario": "quality_qa"},
    k=3
)

# 例如检索到：
# Q1: "请详细描述头痛的性质：是搏动性、压迫性还是针刺样？"
# Q2: "头痛发作时是否伴有恶心、呕吐或畏光？"
# Q3: "头痛是否有固定的诱发因素，如劳累、情绪紧张等？"

# 医生Agent生成问题时会参考这些高质量示例
```

---

## 📊 测试验证

### 测试脚本

运行测试脚本验证所有增强功能：

```bash
cd C:\Users\32765\Desktop\langgraph\patient_agent
conda activate langgraph
python test_enhanced_rag.py
```

### 测试内容

1. ✅ 默认均衡检索 - 同时查询多个知识库
2. ✅ 场景化检索 - 根据 scenario 参数优化检索策略
3. ✅ 患者历史检索 - 支持 patient_id 过滤
4. ✅ 临床案例检索 - 提供相似病例参考
5. ✅ 高质量问诊检索 - 推荐优质问诊问题
6. ✅ 历史检查记录检索 - 避免重复开单

---

## 🎯 预期效果

### 1. 提升问诊质量
- S4 节点问诊更加专业、全面
- 医生能参考高质量问诊模板
- 问诊深度和效率显著提升

### 2. 避免医疗资源浪费
- C8 节点自动检查重复开单
- 减少不必要的重复检查
- 降低患者经济负担

### 3. 增强诊断准确性
- C11/C12 节点参考相似病例
- 结合真实世界证据
- 诊断更加准确、可解释

### 4. 确保病历一致性
- C14 节点自动关联历史病历
- 避免信息冲突和遗漏
- 生成连贯完整的医疗文书

### 5. 个性化医疗服务
- C5 节点了解患者背景
- 制定针对性问诊策略
- 提供个性化沟通方式

---

## 📝 注意事项

### 1. 数据库依赖

所有增强功能依赖于以下向量库：

- `SPLLM-RAG1/chroma/MedicalGuide_db/` - 必需
- `SPLLM-RAG1/chroma/ClinicalCase_db/` - 必需（已启用）
- `SPLLM-RAG1/chroma/UserHistory_db/` - 可选（需患者历史数据）

### 2. patient_id 要求

以下功能需要提供有效的 `patient_id`：
- C5: 检索患者历史对话
- C8: 检索历史检查记录
- C14: 检索历史病历
- S4: 检索患者历史（可选）

### 3. 性能考虑

- 每个节点的检索次数增加（2-4倍）
- 建议监控检索延迟
- 可通过 `k` 参数调整检索数量

### 4. 数据质量

RAG 效果取决于向量库数据质量：
- 确保医学知识库内容准确
- 定期更新临床案例库
- 及时录入患者历史记录

---

## 🔄 后续优化建议

1. **缓存机制** - 缓存常用检索结果，减少重复查询
2. **检索排序** - 基于相关性分数动态调整检索策略
3. **自适应k值** - 根据查询复杂度自动调整检索数量
4. **检索日志** - 记录检索效果，用于持续优化
5. **用户反馈** - 收集医生反馈，优化检索策略

---

## 📞 技术支持

如遇问题，请检查：

1. 向量库路径是否正确：`SPLLM-RAG1/chroma/`
2. 嵌入模型是否已缓存：`SPLLM-RAG1/model_cache/`
3. patient_id 是否有效
4. 检索日志输出（查看 `hospital_agent.adaptive_rag` 日志）

---

**实施完成日期**: 2026年2月11日
**版本**: v1.0
**状态**: ✅ 已完成并测试
