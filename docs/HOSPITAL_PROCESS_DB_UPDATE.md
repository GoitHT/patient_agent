# 医院规则流程库 (HospitalProcess_db) 更新说明

## 📝 更新概述

本次更新在RAG系统中新增了独立的**医院规则流程库 (HospitalProcess_db)**，用于存储医院通用流程、表单模板等，与医学指南库 (MedicalGuide_db) 明确区分。

## ✨ 主要改动

### 1. 数据库架构

**新增知识库：**
- `HospitalProcess_db` - 医院规则流程库
  - 医院通用流程SOP（挂号、缴费、预约等）
  - 文书模板（病历、诊断证明、病假条等）
  - 患者宣教材料
  - 医院内部规范流程

**与医学指南库的区别：**
| 知识库 | 用途 | 内容类型 | 检索场景 |
|--------|------|----------|----------|
| MedicalGuide_db | 医学专业知识 | 诊疗指南、检查指征、治疗方案 | 疾病诊断、治疗决策 |
| HospitalProcess_db | 医院流程规范 | SOP流程、文书模板、宣教材料 | 开单流程、文书生成、患者教育 |

### 2. 代码修改

#### 2.1 hybrid_retriever.py
- ✅ 添加 `HospitalProcess_db` 检索支持
- ✅ 更新知识库说明文档
- ✅ 实现关键词触发机制（仅在查询包含流程相关词时检索）

**关键词触发列表：**
```python
["流程", "模板", "证明", "病假", "病历", "表单", "SOP", "缴费", "预约", "挂号", "诊断书", "宣教"]
```

#### 2.2 enhanced_rag_retriever.py
- ✅ 新增查询类型 `HOSPITAL_PROCESS`
- ✅ 更新查询分类逻辑（优先识别医院流程查询）
- ✅ 添加分层检索策略（医院流程查询时优先检索 HospitalProcess_db）

**分层检索策略：**
```python
QueryType.HOSPITAL_PROCESS: {
    "libraries": ["HospitalProcess_db", "HighQualityQA_db"],
    "weights": [0.8, 0.2],  # 80%权重给流程库
    "k_per_lib": [k, k//2]
}
```

#### 2.3 common_opd_graph.py
- ✅ **C8节点**：明确从 HospitalProcess_db 检索医院通用流程
- ✅ **C12节点**：更新文书模板和医院SOP的检索来源标注
- ✅ **C14节点**：明确从 HospitalProcess_db 检索文书模板

**节点修改示例：**
```python
# C8: 开单与准备说明
# 使用：医院流程库(HospitalProcess_db) - 检索医院通用流程、缴费预约SOP
hospital_chunks = self.retriever.retrieve(
    query="缴费 预约 检查流程",
    filters={"dept": "hospital", "type": "sop"},
    k=4,
)

# C14: 文书生成
# 使用：医院流程库(HospitalProcess_db) - 检索病历/证明/病假条模板
template_chunks = self.retriever.retrieve(
    query="门诊病历 诊断证明 病假条 模板",
    filters={"dept": "forms"},
    k=4,
)
```

### 3. 文档更新

- ✅ 新增：`docs/hospital_process_db_guide.md` - 医院规则流程库完整使用指南
- ✅ 更新：`docs/enhanced_rag_system.md` - 添加第五大知识库说明

## 🚀 使用方法

### 创建/更新数据库

```bash
# 进入项目目录
cd SPLLM-RAG1

# 重建医院流程库（使用动态分块）
python create_database_general.py --mode rebuild --db process

# 增量更新
python create_database_general.py --mode update --db process
```

### 添加新模板

1. 在 `SPLLM-RAG1/data/HospitalProcess_data/` 目录下添加新的 `.md` 文件
2. 运行更新命令同步到数据库

### 在代码中使用

```python
from src.rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(spllm_root="./SPLLM-RAG1")

# 检索医院流程
results = retriever.retrieve(
    query="如何开具诊断证明",
    filters={"dept": "forms"},
    k=4
)
```

## 📊 验证测试

### 快速验证

```python
# 验证数据库是否正确创建
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
db = Chroma(
    persist_directory="./SPLLM-RAG1/chroma/HospitalProcess_db",
    embedding_function=embeddings,
    collection_name="HospitalProcess"
)

# 检查文档数量
print(f"文档总数: {db._collection.count()}")

# 测试检索
results = db.similarity_search("病历模板", k=3)
for doc in results:
    print(f"- {doc.page_content[:100]}...")
```

### 运行完整测试

```bash
# 测试RAG检索
python test_adaptive_rag.py
```

## 📁 文件清单

### 修改的文件
- `src/rag/hybrid_retriever.py`
- `src/rag/enhanced_rag_retriever.py`
- `src/graphs/common_opd_graph.py`
- `docs/enhanced_rag_system.md`

### 新增的文件
- `docs/hospital_process_db_guide.md`
- `docs/HOSPITAL_PROCESS_DB_UPDATE.md` (本文件)

### 已存在的数据库相关文件
- `SPLLM-RAG1/create_database_general.py` (已包含 HospitalProcess_db 逻辑)
- `SPLLM-RAG1/data/HospitalProcess_data/` (数据源目录)
- `SPLLM-RAG1/scripts/sync_hospital_process_data.py` (同步脚本)

## 🔍 检索日志示例

更新后的系统会在日志中明确显示知识库来源：

```
🔍 RAG检索 [C8 - 医院通用流程[医院流程库]]:
   Query: 缴费 预约 检查流程
   Filters: {'dept': 'hospital', 'type': 'sop'}
   结果数: 4条
   • 医院流程库 (HospitalProcess_db): 4条

🔍 RAG检索 [C14 - 文书模板[医院流程库]]:
   Query: 门诊病历 诊断证明 病假条 宣教单 模板
   Filters: {'dept': 'forms'}
   结果数: 4条
   • 医院流程库 (HospitalProcess_db): 4条
```

## ⚠️ 注意事项

1. **数据库初始化**：首次使用前需运行 `create_database_general.py` 创建数据库
2. **关键词匹配**：自动检索依赖关键词触发，确保查询包含相关词汇
3. **元数据规范**：新增文档需正确标注 `type: hospital_process` 和 `dept` 字段
4. **与医学指南库的区分**：
   - 医学专业知识 → MedicalGuide_db
   - 医院流程模板 → HospitalProcess_db

## 📚 参考文档

- [医院规则流程库详细指南](./hospital_process_db_guide.md)
- [增强版RAG系统文档](./enhanced_rag_system.md)
- [RAG整合指南](./rag_integration_guide.md)

## 🔄 后续优化建议

1. 根据实际使用情况优化关键词触发列表
2. 调整 BM25 和向量检索的权重以提高准确率
3. 定期更新医院流程文档以保持时效性
4. 收集检索统计数据以改进分层策略

---

**更新日期**: 2026年2月11日  
**影响范围**: RAG检索系统、图节点 C8/C12/C14  
**向后兼容**: ✅ 完全兼容现有代码
