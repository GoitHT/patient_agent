# 增强版 RAG 系统文档

## 系统架构

### 五大知识库

1. **医学指南库（MedicalGuide_db）**
   - 用途：为医生诊断提供医学专业知识
   - 内容：诊疗指南、临床路径、疾病诊断标准、检查指征、治疗方案
   - 分块策略：层次分块（保留文档结构）
   - 典型场景：疾病诊断、治疗方案制定、检查项目准备

2. **医院流程库（HospitalProcess_db）** ⭐ 新增
   - 用途：医院通用流程和表单模板
   - 内容：医院SOP（挂号/缴费/预约）、文书模板（病历/证明/病假条）、患者宣教材料
   - 分块策略：固定分块（适合模板和流程文档）
   - 典型场景：C8开单流程、C14文书生成、C12诊后处置
   - **与医学指南库区分**：医院流程重流程和模板，医学指南重专业知识

3. **临床案例库（ClinicalCase_db）**
   - 用途：提供标准案例参考
   - 内容：典型病例、治疗方案
   - 分块策略：语义分块（按病历段落）

4. **高质量问答库（HighQualityQA_db）**
   - 用途：医患对话自进化机制
   - 内容：高质量问答对
   - 分块策略：最小分块（保持完整性）
   - 特性：双向进化闭环

5. **医患对话摘要库（UserHistory_db）**
   - 用途：帮助医生构建问答链
   - 内容：结构化对话摘要
   - 存储：仅摘要，不存储全部对话

## 核心特性

### 1. 融合检索（Hybrid Retrieval）

**实现方式：BM25 + 向量检索**

```python
from src.rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    spllm_root="./SPLLM-RAG1",
    bm25_weight=0.4,      # BM25 权重
    vector_weight=0.6,    # 向量检索权重
)

# 融合检索
results = retriever.retrieve(
    query="头痛患者如何诊断？",
    k=5
)
```

**优势：**
- BM25：精确匹配专业术语、疾病名称
- 向量检索：理解语义相似性
- RRF 融合：综合两种检索优势

### 2. 动态 Chunk 策略

**根据文档类型自适应选择分块策略：**

```python
from src.rag.dynamic_chunker import DynamicChunker, ChunkStrategy

chunker = DynamicChunker()

# 自动识别文档类型并选择策略
documents = [
    {"text": "诊疗指南内容...", "meta": {}},
    {"text": "临床病例...", "meta": {}},
]

chunked = chunker.chunk_documents(documents)
```

**分块策略：**

| 文档类型 | 策略 | 说明 |
|---------|------|------|
| 医学指南 | 层次分块 | 保留章节结构 |
| 临床病例 | 语义分块 | 按病历段落 |
| 问答对 | 固定分块 | 保持完整性 |
| 对话记录 | 语义分块 | 按对话回合 |

### 3. 分层检索策略

**根据查询类型动态调整检索策略：**

```python
from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever

rag = EnhancedRAGRetriever(
    spllm_root="./SPLLM-RAG1",
    enable_hybrid=True,
    enable_rerank=False,
)

# 自动识别查询类型并分层检索
results = rag.retrieve(
    query="上次患者的诊断结果是什么？",  # 历史查询
    filters={"patient_id": "P001"},
    k=5,
    enable_hierarchical=True
)
```

**查询类型与检索策略：**

| 查询类型 | 主要检索库 | 权重分配 |
|---------|-----------|---------|
| 事实查询 | 指南库 + 问答库 | 0.6 + 0.4 |
| 流程查询 | 指南库 + 案例库 + 问答库 | 0.5 + 0.3 + 0.2 |
| 案例查询 | 案例库 + 问答库 + 指南库 | 0.5 + 0.3 + 0.2 |
| 历史查询 | 历史库 + 问答库 | 0.7 + 0.3 |

### 4. 自进化机制

**高质量问答库持续更新：**

```python
# 在医患对话后，评估并存储高质量问答
rag.update_high_quality_qa(
    question="患者主诉头痛一周，该做哪些检查？",
    answer="建议进行神经系统体格检查、头颅 CT 或 MRI...",
    quality_score=0.85  # 只有高分问答才会存储
)

# 更新患者历史摘要
rag.update_history(
    patient_id="P001",
    dialogue_summary="患者主诉头痛，伴随恶心呕吐...",
    diagnosis="偏头痛",
    treatment="布洛芬 400mg，每日三次"
)
```

## 使用指南

### 完整使用示例

```python
from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever

# 1. 初始化 RAG 系统
rag = EnhancedRAGRetriever(
    spllm_root="./SPLLM-RAG1",
    enable_hybrid=True,        # 启用混合检索
    enable_rerank=False,       # 可选：启用重排序
    cosine_threshold=0.3,
)

# 2. 执行检索
results = rag.retrieve(
    query="头痛患者如何诊断？",
    filters={
        "patient_id": "P001",  # 可选：检索患者历史
        "dept": "神经内科"
    },
    k=5,
    enable_hierarchical=True   # 启用分层检索
)

# 3. 使用检索结果
for result in results:
    print(f"来源: {result['meta']['source']}")
    print(f"分数: {result['score']:.3f}")
    print(f"内容: {result['text'][:100]}...")
    print("-" * 60)

# 4. 对话后更新知识库
rag.update_high_quality_qa(
    question="用户的问题",
    answer="医生的回答",
    quality_score=0.85
)

rag.update_history(
    patient_id="P001",
    dialogue_summary="对话摘要",
    diagnosis="诊断结果",
    treatment="治疗方案"
)
```

### 重建向量库（使用动态分块）

```bash
# 在 SPLLM-RAG1 目录下运行
cd SPLLM-RAG1
python rebuild_database_with_dynamic_chunk.py
```

这将：
1. 使用动态分块策略重新处理所有文档
2. 为不同类型的文档应用最优分块策略
3. 重建所有四个向量库

## 性能优化

### 1. 混合检索权重调优

```python
retriever = HybridRetriever(
    bm25_weight=0.4,     # 增加此值以强化关键词匹配
    vector_weight=0.6,   # 增加此值以强化语义理解
)
```

**建议：**
- 专业术语多：增加 BM25 权重（0.5-0.6）
- 口语化查询：增加向量权重（0.6-0.7）

### 2. 分块大小优化

```python
from src.rag.dynamic_chunker import ChunkConfig

config = ChunkConfig(
    chunk_size=600,        # 基础块大小
    chunk_overlap=100,     # 重叠大小
    min_chunk_size=100,    # 最小块
    max_chunk_size=1500,   # 最大块
)
```

**建议：**
- 医学指南：较大块（800-1000）
- 问答对：保持完整（不分块或大块）
- 对话记录：中等块（500-700）

### 3. 检索阈值调整

```python
rag = EnhancedRAGRetriever(
    cosine_threshold=0.3,  # 余弦距离阈值
)
```

**建议：**
- 严格匹配：0.2-0.25
- 平衡：0.3-0.35
- 宽松：0.4-0.5

## 依赖安装

```bash
pip install rank-bm25 jieba langchain-chroma langchain-huggingface sentence-transformers
```

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    增强版 RAG 系统                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 查询分析     │→ │ 分层检索策略  │→ │ 融合检索     │     │
│  │ Query Type   │  │ Hierarchical  │  │ BM25+Vector │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────┐       │
│  │            四大知识库                            │       │
│  ├─────────────────────────────────────────────────┤       │
│  │ 医学指南库  │ 临床案例库 │ 问答库 │ 历史库     │       │
│  │ [层次分块]  │ [语义分块] │ [固定] │ [动态更新] │       │
│  └─────────────────────────────────────────────────┘       │
│                            ↓                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 结果融合     │→ │ 重排序(可选)  │→ │ 返回结果     │     │
│  │ RRF Fusion   │  │ Cross-Encoder│  │ Formatted    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│                     ↓ 自进化机制 ↓                          │
│  ┌─────────────────────────────────────────────────┐       │
│  │        高质量问答 & 历史记忆持续更新              │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 进一步优化方向

1. **重排序模型**：集成 BGE-reranker 提升精度
2. **动态权重**：根据查询自动调整 BM25/向量权重
3. **多模态检索**：支持医学图像检索
4. **实时更新**：流式更新知识库
5. **质量评估**：自动评估问答质量

## 常见问题

### Q1: 如何选择是否启用混合检索？

**A:** 
- 启用混合检索（推荐）：查询包含专业术语、疾病名称
- 仅向量检索：口语化查询、语义理解为主

### Q2: 分块策略如何影响检索效果？

**A:** 
- 块太小：上下文不足，语义不完整
- 块太大：噪音多，检索不精确
- 动态分块：根据文档类型自适应，效果最佳

### Q3: 如何评估检索质量？

**A:** 可以使用以下指标：
- 准确率：检索结果的相关性
- 召回率：是否检索到关键信息
- 响应时间：检索速度

建议定期对比测试不同配置的效果。
