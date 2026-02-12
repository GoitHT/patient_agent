# 将增强版 RAG 整合到 patient_agent 项目

## 📋 整合步骤

### 1. 更新依赖

在项目根目录的 `requirements.txt` 中添加：

```txt
# RAG 增强功能
rank-bm25>=0.2.2
jieba>=0.42.1
```

然后安装：
```bash
pip install rank-bm25 jieba
```

### 2. 替换现有 RAG 模块

#### 方式 A：完全替换（推荐）

在 `src/config.yaml` 中配置：

```yaml
rag:
  retriever_type: "enhanced"  # 使用增强版
  spllm_root: "./SPLLM-RAG1"
  enable_hybrid: true
  enable_hierarchical: true
  cosine_threshold: 0.3
```

在 `src/rag.py` 中修改：

```python
# 原来的导入
# from src.rag.adaptive_rag_retriever import AdaptiveRAGRetriever

# 改为
from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever

# 初始化时
def init_rag_retriever(config):
    """初始化 RAG 检索器"""
    return EnhancedRAGRetriever(
        spllm_root=config.get("spllm_root", "./SPLLM-RAG1"),
        enable_hybrid=config.get("enable_hybrid", True),
        enable_rerank=config.get("enable_rerank", False),
        cosine_threshold=config.get("cosine_threshold", 0.3),
    )
```

#### 方式 B：兼容模式（保留原有功能）

添加配置开关，支持两种模式：

```python
def init_rag_retriever(config):
    """初始化 RAG 检索器（兼容模式）"""
    retriever_type = config.get("retriever_type", "adaptive")
    
    if retriever_type == "enhanced":
        from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever
        return EnhancedRAGRetriever(
            spllm_root=config["spllm_root"],
            enable_hybrid=config.get("enable_hybrid", True),
        )
    else:
        from src.rag.adaptive_rag_retriever import AdaptiveRAGRetriever
        return AdaptiveRAGRetriever(
            spllm_root=config["spllm_root"],
        )
```

### 3. 整合到医生智能体

在 `src/agents/doctor_agent.py` 中使用：

```python
class DoctorAgent:
    def __init__(self, rag_retriever, llm_client):
        self.rag = rag_retriever
        self.llm = llm_client
    
    def diagnose(self, patient_symptoms, patient_id=None):
        """诊断患者"""
        # 1. 检索相关医学知识
        query = f"患者症状：{patient_symptoms}"
        
        results = self.rag.retrieve(
            query=query,
            filters={"patient_id": patient_id} if patient_id else None,
            k=5,
            enable_hierarchical=True  # 启用分层检索
        )
        
        # 2. 构建上下文
        context = "\n\n".join([
            f"参考资料 {i+1} (来源: {r['meta']['source']}):\n{r['text']}"
            for i, r in enumerate(results)
        ])
        
        # 3. 生成诊断
        prompt = f"""
基于以下医学知识，诊断患者病情：

{context}

患者症状：
{patient_symptoms}

请给出诊断建议：
"""
        
        diagnosis = self.llm.generate(prompt)
        
        return {
            "diagnosis": diagnosis,
            "reference_sources": [r['meta']['source'] for r in results],
            "confidence": self._calculate_confidence(results)
        }
    
    def _calculate_confidence(self, results):
        """根据检索结果计算诊断置信度"""
        if not results:
            return 0.0
        
        # 基于检索分数计算平均置信度
        avg_score = sum(r['score'] for r in results) / len(results)
        return min(1.0, avg_score)
```

### 4. 实现对话后的知识库更新

在对话流程结束后，更新知识库：

```python
class DialogueManager:
    def __init__(self, rag_retriever):
        self.rag = rag_retriever
    
    def end_dialogue(self, patient_id, dialogue_history, diagnosis, treatment):
        """对话结束时的处理"""
        
        # 1. 生成对话摘要
        summary = self._generate_summary(dialogue_history)
        
        # 2. 更新患者历史
        self.rag.update_history(
            patient_id=patient_id,
            dialogue_summary=summary,
            diagnosis=diagnosis,
            treatment=treatment
        )
        
        # 3. 提取高质量问答对
        qa_pairs = self._extract_qa_pairs(dialogue_history)
        
        for qa in qa_pairs:
            quality_score = self._evaluate_quality(qa)
            
            if quality_score > 0.7:
                self.rag.update_high_quality_qa(
                    question=qa['question'],
                    answer=qa['answer'],
                    quality_score=quality_score
                )
    
    def _generate_summary(self, dialogue_history):
        """生成对话摘要"""
        # 使用 LLM 生成摘要
        dialogue_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in dialogue_history
        ])
        
        summary_prompt = f"""
请总结以下医患对话的关键信息：

{dialogue_text}

要求：
1. 提取主要症状
2. 诊断结果
3. 治疗方案
4. 患者关注点
"""
        
        return self.llm.generate(summary_prompt)
    
    def _extract_qa_pairs(self, dialogue_history):
        """从对话中提取问答对"""
        qa_pairs = []
        
        for i in range(len(dialogue_history) - 1):
            if dialogue_history[i]['role'] == 'patient':
                question = dialogue_history[i]['content']
                
                # 找到医生的回答
                for j in range(i + 1, len(dialogue_history)):
                    if dialogue_history[j]['role'] == 'doctor':
                        answer = dialogue_history[j]['content']
                        qa_pairs.append({
                            'question': question,
                            'answer': answer
                        })
                        break
        
        return qa_pairs
    
    def _evaluate_quality(self, qa_pair):
        """评估问答质量"""
        # 简单评估规则（可以用 LLM 实现更复杂的评估）
        answer = qa_pair['answer']
        
        # 评分标准
        score = 0.5  # 基础分
        
        # 答案长度合理
        if 50 <= len(answer) <= 500:
            score += 0.2
        
        # 包含专业术语
        medical_terms = ['诊断', '治疗', '检查', '药物', '症状', '疾病']
        if any(term in answer for term in medical_terms):
            score += 0.2
        
        # 结构清晰
        if any(marker in answer for marker in ['1.', '2.', '首先', '其次', '建议']):
            score += 0.1
        
        return min(1.0, score)
```

### 5. 在 LangGraph 中集成

在 `src/graphs/common_opd_graph.py` 中：

```python
from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever

def create_hospital_graph(config):
    """创建医院流程图"""
    
    # 初始化 RAG
    rag = EnhancedRAGRetriever(
        spllm_root=config["rag"]["spllm_root"],
        enable_hybrid=config["rag"].get("enable_hybrid", True),
    )
    
    # 定义节点
    def doctor_consult_node(state):
        """医生问诊节点"""
        patient_id = state.get("patient_id")
        symptoms = state.get("symptoms")
        
        # 使用 RAG 检索
        results = rag.retrieve(
            query=f"患者症状：{symptoms}",
            filters={"patient_id": patient_id},
            k=5,
            enable_hierarchical=True
        )
        
        # 更新状态
        state["rag_context"] = results
        state["diagnosis"] = generate_diagnosis(symptoms, results)
        
        return state
    
    def finalize_node(state):
        """结束节点 - 更新知识库"""
        # 更新患者历史
        rag.update_history(
            patient_id=state["patient_id"],
            dialogue_summary=state["dialogue_summary"],
            diagnosis=state.get("diagnosis"),
            treatment=state.get("treatment")
        )
        
        # 更新高质量问答（如果有）
        if state.get("qa_pairs"):
            for qa in state["qa_pairs"]:
                rag.update_high_quality_qa(
                    question=qa["question"],
                    answer=qa["answer"],
                    quality_score=qa.get("quality", 0.8)
                )
        
        return state
    
    # 构建图
    graph = StateGraph(HospitalState)
    graph.add_node("consult", doctor_consult_node)
    graph.add_node("finalize", finalize_node)
    # ... 其他节点
    
    return graph.compile()
```

### 6. 配置文件示例

更新 `src/config.yaml`：

```yaml
# RAG 配置
rag:
  retriever_type: "enhanced"
  spllm_root: "./SPLLM-RAG1"
  
  # 混合检索配置
  enable_hybrid: true
  bm25_weight: 0.4
  vector_weight: 0.6
  
  # 分层检索配置
  enable_hierarchical: true
  
  # 检索参数
  cosine_threshold: 0.3
  default_k: 5
  
  # 重排序（可选）
  enable_rerank: false
  
  # 自进化配置
  auto_update_qa: true
  qa_quality_threshold: 0.7
  
  # 缓存配置
  cache_folder: "./SPLLM-RAG1/model_cache"
  embed_model: "BAAI/bge-large-zh-v1.5"
```

### 7. 测试集成

创建测试文件 `tests/test_rag_integration.py`：

```python
import pytest
from src.rag.enhanced_rag_retriever import EnhancedRAGRetriever

def test_basic_retrieval():
    """测试基础检索"""
    rag = EnhancedRAGRetriever(
        spllm_root="./SPLLM-RAG1",
        enable_hybrid=True,
    )
    
    results = rag.retrieve(
        query="头痛患者如何诊断？",
        k=3
    )
    
    assert len(results) > 0
    assert all('text' in r for r in results)
    assert all('score' in r for r in results)

def test_patient_history():
    """测试患者历史更新"""
    rag = EnhancedRAGRetriever(spllm_root="./SPLLM-RAG1")
    
    # 更新历史
    rag.update_history(
        patient_id="TEST_001",
        dialogue_summary="测试患者就诊记录",
        diagnosis="测试诊断"
    )
    
    # 检索历史
    results = rag.retrieve(
        query="最近就诊记录",
        filters={"patient_id": "TEST_001"},
        k=2
    )
    
    assert any(r['meta'].get('patient_id') == 'TEST_001' for r in results)

def test_qa_update():
    """测试问答库更新"""
    rag = EnhancedRAGRetriever(spllm_root="./SPLLM-RAG1")
    
    rag.update_high_quality_qa(
        question="测试问题",
        answer="测试答案",
        quality_score=0.9
    )
    
    # 验证可以检索到
    results = rag.retrieve(query="测试问题", k=1)
    assert len(results) > 0
```

运行测试：
```bash
pytest tests/test_rag_integration.py -v
```

## 📊 性能监控

添加监控代码以跟踪 RAG 性能：

```python
import time
import logging

class RAGMonitor:
    """RAG 性能监控"""
    
    def __init__(self):
        self.logger = logging.getLogger("rag_monitor")
        self.metrics = {
            "total_queries": 0,
            "total_time": 0,
            "avg_results": 0,
        }
    
    def log_query(self, query, results, elapsed_time):
        """记录查询"""
        self.metrics["total_queries"] += 1
        self.metrics["total_time"] += elapsed_time
        self.metrics["avg_results"] = (
            (self.metrics["avg_results"] * (self.metrics["total_queries"] - 1)
             + len(results)) / self.metrics["total_queries"]
        )
        
        self.logger.info(
            f"查询: {query[:50]}... | "
            f"结果数: {len(results)} | "
            f"耗时: {elapsed_time:.2f}s"
        )
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "total_queries": self.metrics["total_queries"],
            "avg_time": self.metrics["total_time"] / max(1, self.metrics["total_queries"]),
            "avg_results": self.metrics["avg_results"],
        }

# 使用示例
monitor = RAGMonitor()

def retrieve_with_monitoring(rag, query, **kwargs):
    """带监控的检索"""
    start_time = time.time()
    results = rag.retrieve(query, **kwargs)
    elapsed = time.time() - start_time
    
    monitor.log_query(query, results, elapsed)
    return results
```

## ✅ 验证清单

完成整合后，检查以下项目：

- [ ] 依赖已安装（`rank-bm25`, `jieba`）
- [ ] 向量库已重建（使用动态分块）
- [ ] RAG 初始化成功
- [ ] 基础检索正常
- [ ] 患者历史记忆正常
- [ ] 问答库更新正常
- [ ] 分层检索工作正常
- [ ] 性能可接受（响应时间 < 2秒）
- [ ] 测试通过

## 🔍 故障排查

如遇问题，按以下步骤排查：

1. **检查依赖**
   ```bash
   pip list | grep -E "rank-bm25|jieba"
   ```

2. **检查向量库**
   ```bash
   ls -la SPLLM-RAG1/chroma/
   ```

3. **查看日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **测试基础功能**
   ```bash
   python example_enhanced_rag.py
   ```

## 📞 获取帮助

- 📖 查看完整文档：`docs/enhanced_rag_system.md`
- 💻 查看示例代码：`example_enhanced_rag.py`
- 🚀 查看快速开始：`QUICKSTART_RAG.md`
