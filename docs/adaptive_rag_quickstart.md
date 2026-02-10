# Adaptive RAG 快速启动指南

## 🚀 5分钟快速启动

### 步骤 1: 检查目录结构

确保你的目录结构如下：
```
项目/
├── patient_agent/           # 当前项目
│   ├── src/
│   ├── requirements.txt
│   └── ...
└── SPLLM-RAG1/             # SPLLM-RAG1 项目（需要在同级目录）
    ├── chroma/
    ├── model_cache/
    └── ...
```

### 步骤 2: 安装依赖

```bash
cd patient_agent
pip install -r requirements.txt
```

**预计时间**: 2-3分钟

### 步骤 3: 验证配置

检查 `src/config.yaml`：

```yaml
rag:
  skip_rag: false               # ✅ 确保为 false
  spllm_root: ../SPLLM-RAG1     # ✅ 检查路径是否正确
```

### 步骤 4: 启动系统

```bash
cd src
python main.py
```

### 步骤 5: 验证输出

看到以下日志表示成功：
```
🚀 初始化 Adaptive RAG（SPLLM-RAG1）
   → SPLLM-RAG1: /path/to/SPLLM-RAG1
   → 阈值: 0.3
✅ 嵌入模型加载成功（维度=768）
📦 AdaptiveRAG 初始化: spllm_root=/path/to/SPLLM-RAG1
```

## ⚠️ 常见启动问题

### 问题 1: 找不到 SPLLM-RAG1

**症状**：
```
⚠️  SPLLM-RAG1 路径不存在: ...
🔄 回退到基础 RAG 系统
```

**解决**：
1. 确认 SPLLM-RAG1 与 patient_agent 在同级目录
2. 或修改 `spllm_root` 为绝对路径：
   ```yaml
   spllm_root: C:/Users/xxx/Desktop/项目/patient_agent/SPLLM-RAG1
   ```

### 问题 2: 向量库不存在

**症状**：
```
⚠️  向量库路径不存在: .../chroma/MedicalGuide_db
```

**解决**：
```bash
cd SPLLM-RAG1
python create_database_general.py
```

### 问题 3: 缺少依赖

**症状**：
```
ModuleNotFoundError: No module named 'langchain_chroma'
```

**解决**：
```bash
pip install langchain_chroma langchain_huggingface sentence_transformers
```

## 🧪 快速测试

### 测试 1: 基础检索

```python
from rag.adaptive_rag_retriever import AdaptiveRAGRetriever
from pathlib import Path

# 初始化
retriever = AdaptiveRAGRetriever(
    spllm_root="../SPLLM-RAG1"
)

# 检索
results = retriever.retrieve("高血压的治疗方案", k=3)
print(f"检索到 {len(results)} 条结果")
for r in results:
    print(f"- [{r['meta']['source']}] 分数:{r['score']:.2f}")
```

### 测试 2: 患者历史检索

```python
results = retriever.retrieve(
    "上次就诊情况",
    filters={"patient_id": "test_001"},
    k=3
)
```

## 📊 性能基准

首次启动：
- ⏱️ 模型加载: 3-5秒
- ⏱️ 向量库加载: 1-2秒
- 💾 内存占用: ~800MB

后续检索：
- ⏱️ 单次检索: 100-300ms
- 📈 精度: 比基础 RAG 提升 40-60%

## 🔄 如果遇到问题

1. **跳过 RAG（测试模式）**：
   ```yaml
   rag:
     skip_rag: true
   ```

2. **查看详细日志**：
   ```bash
   export LOGLEVEL=DEBUG
   python main.py
   ```

3. **查看完整文档**：
   - [完整集成文档](./adaptive_rag_integration.md)
   - [配置说明](./dependency_management.md)

## ✅ 启动成功标志

系统正常运行时，你会看到：
- ✅ Adaptive RAG 初始化成功
- ✅ 4个向量库加载完成
- ✅ 嵌入模型运行正常
- ✅ 患者问诊流程顺利进行

---

祝使用愉快！如有问题，请参考完整文档或联系开发团队。
