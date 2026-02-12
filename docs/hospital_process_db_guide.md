# 医院规则流程库 (HospitalProcess_db) 使用指南

## 📚 概述

`HospitalProcess_db` 是专门用于存储医院通用流程、表单模板和SOP文档的向量数据库，与医学专业知识库 `MedicalGuide_db` 相互独立，确保检索的精准性和效率。

## 🎯 设计目标

### 知识库区分

- **MedicalGuide_db**（医学指南库）：医学专业知识
  - 诊疗指南、临床路径
  - 疾病诊断标准、鉴别诊断
  - 检查指征、治疗方案
  - 专科诊疗规范

- **HospitalProcess_db**（规则流程库）：医院通用流程
  - 医院通用流程SOP（挂号、缴费、预约等）
  - 文书模板（病历、诊断证明、病假条等）
  - 患者宣教材料
  - 医院内部规范流程

## 📁 数据结构

### 数据目录

```
SPLLM-RAG1/
├── data/
│   ├── HospitalProcess_data/        # 医院流程数据源
│   │   ├── forms_template_emr.md    # 病历模板
│   │   ├── forms_template_diagnosis_cert.md  # 诊断证明模板
│   │   ├── forms_template_sick_leave.md      # 病假条模板
│   │   ├── forms_template_education_sheet.md # 宣教单模板
│   │   ├── hospital_sop_intake.md   # 挂号流程SOP
│   │   ├── hospital_sop_billing_reports.md   # 缴费流程SOP
│   │   ├── hospital_sop_followup.md # 随访流程SOP
│   │   └── hospital_education_common.md      # 通用宣教内容
│   └── MedicalGuide_data/          # 医学指南数据（对比）
└── chroma/
    ├── HospitalProcess_db/         # 医院流程向量库
    └── MedicalGuide_db/           # 医学指南向量库
```

### 元数据规范

每个文档应包含以下元数据：

```python
{
    "type": "hospital_process",      # 固定类型标识
    "dept": "hospital" | "forms",    # hospital: 流程SOP, forms: 表单模板
    "source": "文件名.md",            # 来源文件
    "category": "sop" | "template",  # 类别
}
```

## 🔧 创建与维护

### 1. 初始化数据库

```bash
# 完整重建（使用动态分块）
cd SPLLM-RAG1
python create_database_general.py --mode rebuild --db process

# 增量更新（添加新文档）
python create_database_general.py --mode update --db process
```

### 2. 数据同步脚本

使用 `scripts/sync_hospital_process_data.py` 自动同步文档：

```python
# 同步 kb/forms 和 kb/hospital 下的模板文件
python scripts/sync_hospital_process_data.py
```

### 3. 添加新模板

在 `SPLLM-RAG1/data/HospitalProcess_data/` 目录下添加新的 `.md` 文件：

```markdown
<!-- forms_template_prescription.md -->
# 门诊处方模板

## 基本信息
- 患者姓名：
- 年龄：
- 性别：
- 就诊日期：

## 诊断
主要诊断：

## 处方
1. 药品名称：
   - 剂量：
   - 用法：
   - 数量：

## 医嘱
遵医嘱服药，注意观察。

---
医生签名：
日期：
```

然后运行更新命令：

```bash
python create_database_general.py --mode update --db process
```

## 🔍 检索使用

### 在代码中使用

#### 方法1：直接检索（推荐）

```python
from src.rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(spllm_root="./SPLLM-RAG1")

# 检索医院流程
results = retriever.retrieve(
    query="如何开具诊断证明",
    filters={"dept": "forms"},  # 表单模板
    k=4
)

# 检索缴费流程
results = retriever.retrieve(
    query="门诊缴费流程",
    filters={"dept": "hospital", "type": "sop"},  # 医院SOP
    k=4
)
```

#### 方法2：智能分类检索

```python
from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever

retriever = EnhancedRAGRetriever(
    spllm_root="./SPLLM-RAG1",
    enable_hybrid=True
)

# 系统会自动识别查询类型并选择合适的库
results = retriever.retrieve(
    query="病历模板怎么写",  # 自动识别为 HOSPITAL_PROCESS 类型
    k=5
)
```

### 在图节点中使用

#### C8节点：开单与准备说明

```python
# 检索医院通用流程SOP
hospital_chunks = self.retriever.retrieve(
    query="缴费 预约 检查流程",
    filters={"dept": "hospital", "type": "sop"},
    k=4,
)
```

#### C14节点：文书生成

```python
# 检索文书模板
template_chunks = self.retriever.retrieve(
    query="门诊病历 诊断证明 病假条 模板",
    filters={"dept": "forms"},
    k=4,
)
```

#### C12节点：综合分析

```python
# 检索医院通用SOP
chunks_hospital = self.retriever.retrieve(
    query="诊后处置 随访 SOP",
    filters={"dept": "hospital", "type": "sop"},
    k=4,
)

# 检索文书模板
chunks_forms = self.retriever.retrieve(
    query="门诊病历 诊断证明 病假条 宣教单 模板",
    filters={"dept": "forms"},
    k=4,
)
```

## 🎨 查询优化

### 使用查询优化器

```python
from src.rag.query_optimizer import get_query_optimizer, QueryContext

optimizer = get_query_optimizer()

# 构建查询上下文
query_ctx = QueryContext(
    patient_id=state.patient_id,
    chief_complaint=state.chief_complaint,
    dept=state.dept,
)

# 生成优化的查询
query = optimizer.generate_contextual_query("document_template", query_ctx)

# 检索
results = retriever.retrieve(query, filters={"dept": "forms"}, k=4)
```

## 📊 关键词触发

系统会根据查询中的关键词自动选择是否检索 `HospitalProcess_db`：

### 触发关键词

```python
HOSPITAL_PROCESS_KEYWORDS = [
    "流程", "模板", "证明", "病假", "病历", 
    "表单", "SOP", "缴费", "预约", "挂号",
    "诊断书", "宣教"
]
```

### 自动检索示例

在 `hybrid_retriever.py` 中：

```python
# 仅在查询包含流程相关关键词时检索
if any(kw in query for kw in ["流程", "模板", "证明", ...]):
    process_results = self.hybrid_retrieve(query, "HospitalProcess_db", k=k)
    for r in process_results:
        r["meta"]["source"] = "HospitalProcess"
    all_results.extend(process_results)
```

## 🔐 最佳实践

### 1. 文档组织

- **命名规范**：
  - 表单模板：`forms_template_*.md`
  - 医院流程：`hospital_sop_*.md`
  - 宣教材料：`hospital_education_*.md`

- **内容结构**：保持一致的Markdown格式
  ```markdown
  # 标题
  
  ## 适用范围
  
  ## 流程/模板内容
  
  ## 注意事项
  ```

### 2. 元数据标注

确保每个文档在向量化时添加正确的元数据：

```python
doc["meta"]["type"] = "hospital_process"
doc["meta"]["dept"] = "hospital" or "forms"
doc["meta"]["source"] = file.name
```

### 3. 定期维护

- 每月检查过时的流程文档
- 更新医院政策变动相关的SOP
- 根据实际使用情况优化模板

### 4. 检索优化

- 使用明确的查询关键词
- 合理设置 `k` 值（推荐 3-5）
- 利用 `filters` 精确定位

## 📈 监控与日志

### 检索日志

系统会自动记录每次检索：

```
🔍 RAG检索 [C14 - 文书模板[医院流程库]]:
   Query: 门诊病历 诊断证明 病假条 宣教单 模板
   Filters: {'dept': 'forms'}
   结果数: 4条
   • 医院流程库 (HospitalProcess_db): 4条
```

### 统计分析

```python
# 查看检索统计
from src.graphs.log_helpers import RAG_RETRIEVAL_STATS

print(RAG_RETRIEVAL_STATS)
# 输出：
# {
#   'C14': {
#     'total_retrievals': 50,
#     'avg_results': 4.2,
#     'libraries': {
#       'HospitalProcess_db': 50,
#       'UserHistory_db': 10
#     }
#   }
# }
```

## 🛠️ 故障排查

### 常见问题

1. **检索不到结果**
   - 检查 filters 是否正确
   - 确认数据库已正确创建
   - 验证查询关键词是否匹配

2. **检索结果不准确**
   - 调整 BM25 和向量检索的权重
   - 使用更明确的查询语句
   - 检查文档质量和元数据

3. **数据库损坏**
   ```bash
   # 重建数据库
   cd SPLLM-RAG1
   python create_database_general.py --mode rebuild --db process
   ```

### 验证数据库

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 加载数据库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
db = Chroma(
    persist_directory="./SPLLM-RAG1/chroma/HospitalProcess_db",
    embedding_function=embeddings,
    collection_name="HospitalProcess"
)

# 检查文档数量
collection = db._collection
print(f"文档总数: {collection.count()}")

# 查看样本
results = collection.get(limit=5, include=["documents", "metadatas"])
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"\n{meta}")
    print(f"{doc[:100]}...")
```

## 📚 参考资料

- [增强版 RAG 系统文档](./enhanced_rag_system.md)
- [RAG 整合指南](./rag_integration_guide.md)
- [查询优化器实现](./qa_quality_implementation.md)

## 🔄 更新记录

- **2026-02-11**: 创建医院规则流程库独立文档
- **2026-02-11**: 更新hybrid_retriever支持HospitalProcess_db
- **2026-02-11**: 更新enhanced_rag_retriever分层检索策略
- **2026-02-11**: 更新common_opd_graph中C8/C12/C14节点使用流程库

---

如有问题或建议，请参考项目文档或联系开发团队。
