# 🏥 患者门诊管理多智能体系统

<div align="center">

**基于 LangGraph 的医院门诊诊疗流程多智能体模拟平台**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.7-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[功能特性](#-核心特性) • [快速开始](#-快速开始) • [项目结构](#-项目结构) • [运行指南](#-运行指南) • [核心模块](#-核心模块详解)

</div>

---

## 📋 项目简介

患者门诊管理多智能体系统是一个基于 **LangGraph** 编排的医院门诊诊疗流程模拟平台。系统采用多智能体协作模式（医生、护士、患者、检验科），支持 **神经医学科、消化科**等多个科室，通过本地 **RAG 知识库**检索和可选的 **DeepSeek LLM** 增强，实现了高度可追踪、可复现的医疗流程仿真。

### 🎯 核心特性

- 🤖 **多智能体协作**：医生、护士、患者、检验科四方智能体紧密协作
- 🏗️ **LangGraph 编排**：完整门诊流程（16个节点）+ 专科子图（3个节点），支持多科室扩展
- 📚 **本地 RAG 系统**：基于 ChromaDB 的向量检索，关键节点强制检索知识库并记录引用溯源
- 🔒 **完全确定性**：Mock 外部系统基于 seed 保证可复现，便于测试和调试
- 📊 **完整审计追踪**：每步操作记录 audit_trail、citations，支持流程回放
- 👥 **多患者并发**：支持多医生多患者并发场景，自动负载均衡和队列管理
- 🌍 **物理环境模拟**：模拟真实医院空间、时间流逝、设备排队等约束
- 💾 **数据库支持**：支持 MySQL 持久化，完整的患者就诊记录管理

### 🏥 支持的科室

**当前已实现**：
- 神经医学科 (Neurology)
- 消化科 (Gastroenterology)

**架构特性**：通用专科子图设计，新增科室仅需配置知识库和科室参数即可

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- pip 包管理器

### 安装依赖

```bash
# 克隆或进入项目目录
cd patient_agent

# 安装依赖
pip install -r requirements.txt
```

### 初始化知识库

```bash
# 构建向量索引（ChromaDB）
python scripts/build_index.py
```

> 💡 **提示**：首次运行会自动加载 `kb/` 目录下的所有知识库文件并构建向量索引。

### 运行系统

#### 基本运行（单患者）

```bash
# 使用默认配置
python src/main.py

# 指定数据集ID
python src/main.py --dataset-id 15 --max-questions 5
```

#### 多患者并发模式

```bash
# 运行3个患者，间隔60秒进入
python src/main.py --num-patients 3 --patient-interval 60
```

#### 配置 DeepSeek API

```bash
# 设置环境变量（PowerShell）
$env:DEEPSEEK_API_KEY="sk-your-key-here"

# 或创建 .env 文件
# DEEPSEEK_API_KEY=sk-your-key-here

# 启用LLM增强
python src/main.py --enable-reports
```

### 配置管理

项目支持通过 `config.yaml` 进行配置，优先级从高到低：

1. 命令行参数
2. 环境变量
3. `config.yaml` 配置文件
4. 代码默认值

```yaml
# src/config.yaml - 关键配置项

llm:
  backend: deepseek        # mock 或 deepseek
  enable_reports: false    # 使用LLM增强检查报告

agent:
  max_questions: 3         # 医生最多问题数
  max_triage_questions: 3  # 护士分诊最多问题数

mode:
  multi_patient: true      # 启用多患者模式
  num_patients: 1          # 患者数量
  patient_interval: 60     # 患者进入间隔（秒）

physical:
  enable_simulation: true  # 启用物理环境模拟
  interactive: false       # 交互式命令模式

database:
  enabled: true
  connection_string: "mysql+pymysql://root:123456@localhost:3306/agent"
```

---

## 📁 项目结构

```
patient_agent/
├── src/
│   ├── agents/                      # 智能体实现
│   │   ├── doctor_agent.py          # 医生智能体（问诊、开单、诊断）
│   │   ├── nurse_agent.py           # 护士智能体（分诊、宣教）
│   │   ├── patient_agent.py         # 患者智能体（模拟患者回答）
│   │   └── lab_agent.py             # 检验智能体（解读结果）
│   ├── coordination/
│   │   └── coordinator.py           # HospitalCoordinator（多患者调度）
│   ├── environment/
│   │   ├── hospital_world.py        # 物理环境模拟
│   │   ├── command_system.py        # 交互命令系统
│   │   └── staff_tracker.py         # 人员跟踪
│   ├── graphs/                      # LangGraph 流程编排
│   │   ├── common_opd_graph.py      # 通用门诊流程（C1-C16）
│   │   ├── router.py                # 图构建器
│   │   └── dept_subgraphs/
│   │       └── common_specialty_subgraph.py  # 通用专科子图（S4-S6）
│   ├── services/                    # 外部系统 Mock
│   │   ├── appointment.py           # 预约服务
│   │   ├── billing.py               # 缴费服务
│   │   ├── lab.py                   # 实验室检查
│   │   ├── imaging.py               # 影像检查
│   │   ├── endoscopy.py             # 内镜检查
│   │   ├── neurophysiology.py       # 神经生理检查
│   │   ├── llm_client.py            # LLM 客户端
│   │   ├── medical_record.py        # 病例管理（文件存储）
│   │   └── medical_record_db_service.py  # 病例管理（数据库存储）
│   ├── state/
│   │   └── schema.py                # BaseState 定义
│   ├── rag.py                       # RAG 检索器（ChromaDB）
│   ├── loaders.py                   # 数据加载器（诊断数据集）
│   ├── utils.py                     # 工具函数（JSON解析、日志等）
│   ├── config.py                    # 配置管理
│   ├── main.py                      # CLI 主程序
│   └── prompts/                     # LLM 提示词模板
├── kb/                              # 知识库
│   ├── hospital/                    # 医院通用知识
│   │   ├── sop_*.md                 # 标准操作流程
│   │   ├── education_common.md      # 通用健康教育
│   │   └── sop_*.md
│   ├── forms/                       # 文书模板
│   │   ├── template_emr.md
│   │   ├── template_diagnosis_cert.md
│   │   └── template_sick_leave.md
│   ├── neuro/                       # 神经医学科知识
│   │   ├── education_neuro.md
│   │   ├── guide_redflags.md        # 红旗症状指南
│   │   ├── plan_neuro.md            # 诊疗方案
│   │   └── prep_*.md                # 检查前准备
│   └── gastro/                      # 消化科知识
│       ├── education_gastro.md
│       ├── guide_redflags.md
│       ├── plan_gastro.md
│       └── prep_*.md
├── medical_records/                 # 患者病例数据
├── logs/
│   └── patients/                    # 每个患者详细日志
├── scripts/
│   ├── build_index.py               # 构建向量索引
│   └── seed_kb_examples.py          # 初始化示例数据
├── tests/                           # 测试用例
├── config.yaml                      # 全局配置文件
├── requirements.txt                 # 依赖清单
└── README.md
```

---

## 🏗️ 流程设计

### 门诊流程图（C1-C16）

```
患者挂号登记 (C1-C4)
    ↓
初诊问诊准备 (C5) [RAG: 通用SOP]
    ↓
专科问诊 (C6 → S4-S6) [RAG: 专科知识库]
    ↓
判断是否需要辅助检查 (C7)
    ├─→ 是 → 开单准备说明 (C8) [RAG: 检查准备]
    │         ↓
    │     缴费和预约 (C9)
    │         ↓
    │     获取检查结果 (C10a) [Mock/LLM生成]
    │         ↓
    │     增强报告叙述 (C10b) [LLM可选]
    │         ↓
    │     报告回诊 (C11) [RAG: 诊疗方案]
    │         ↓
    └─────→ 综合分析诊断 (C12) [RAG: 诊疗方案/文书]
            ↓
        处置决策 (C13) [检查升级建议]
            ↓
        生成诊疗文书 (C14)
            ↓
        健康宣教与随访 (C15) [RAG: 健康教育]
            ↓
        完成流程 (C16)
```

### 专科子图（S4-S6）

**S4: Specialty Interview** - 一问一答模式
- 医生根据问诊要点逐步提问
- 患者基于病例数据回答
- RAG 检索科室知识库指导提问

**S5: Physical Exam** - 体格检查
- 模拟真实检查流程
- 基于主诉和病历生成检查发现

**S6: Preliminary Judgment** - 初步判断
- 综合分析决定是否需要辅助检查
- RAG 检索诊疗指南确定检查建议

---

## 🧪 核心模块详解

### 1. 医生智能体 (`doctor_agent.py`)

**职责**：问诊、检查建议、诊断制定

**关键方法**：

```python
class DoctorAgent:
    def reset(self) -> None:
        """重置医生状态（处理新患者前必须调用）"""
    
    def generate_one_question(self, chief_complaint: str, context: str) -> str:
        """生成单个问题（一问一答模式）"""
    
    def ask_patient(self, patient_agent, chief_complaint: str, context: str) -> dict:
        """完整问诊流程"""
    
    def suggest_tests(self, collected_info: dict) -> list[dict]:
        """建议检查项目"""
    
    def analyze_and_diagnose(self, collected_info: dict, test_results: list) -> dict:
        """综合分析给出诊断"""
```

### 2. 护士智能体 (`nurse_agent.py`)

**职责**：分诊、生命体征测量、宣教

```python
class NurseAgent:
    def triage(self, patient_description: str) -> str:
        """科室分诊"""
    
    def explain_test_prep(self, test_name: str, prep_info: dict) -> str:
        """解释检查前准备"""
```

### 3. 患者智能体 (`patient_agent.py`)

**职责**：模拟真实患者症状和回答

```python
class PatientAgent:
    def describe_to_nurse(self) -> str:
        """向护士描述症状"""
    
    def answer_doctor_question(self, question: str) -> str:
        """回答医生问题（基于病例数据）"""
```

### 4. 医院协调器 (`coordination/coordinator.py`)

**职责**：多患者并发管理、医生资源调度

```python
class HospitalCoordinator:
    def register_doctor(self, doctor_id: str, name: str, dept: str) -> None:
        """注册医生"""
    
    def register_patient(self, patient_id: str, patient_data: dict, dept: str) -> str:
        """患者挂号"""
    
    def get_available_doctors(self, dept: str) -> list:
        """获取空闲医生"""
    
    def assign_doctor_manually(self, patient_id: str, doctor_id: str) -> bool:
        """手动指定医生"""
```

### 5. 物理环境模拟 (`environment/hospital_world.py`)

**职责**：模拟医院物理空间、时间、资源

```python
class HospitalWorld:
    def add_agent(self, agent_id: str, agent_type: str, initial_location: str) -> bool:
        """添加agent到环境"""
    
    def move_agent(self, agent_id: str, target_location: str) -> (bool, str):
        """移动agent（自动寻路）"""
    
    def advance_time(self, minutes: int) -> None:
        """推进时间"""
    
    def use_device(self, agent_id: str, device_name: str) -> (bool, str):
        """使用医疗设备（自动排队）"""
    
    def perform_exam(self, patient_id: str, exam_type: str, priority: int) -> (bool, str):
        """执行检查"""
```

### 6. RAG 检索系统 (`rag.py`)

**向量数据库**：ChromaDB
**嵌入模型**：HashEmbeddingFunction（完全本地、确定性）
**支持过滤**：按 dept 和 type 过滤

```python
class ChromaRetriever:
    def retrieve(self, query: str, filters: dict = None, k: int = 3) -> list[dict]:
        """检索知识片段"""
        # 自动包含 doc_id, chunk_id, source, score 等元数据
```

**知识库结构**：

```
kb/
├── hospital/dept=hospital
│   ├── sop_intake.md (type=sop)
│   └── education_common.md (type=education)
├── forms/dept=forms
│   ├── template_emr.md (type=template)
│   └── template_diagnosis_cert.md
├── neuro/dept=neuro
│   ├── education_neuro.md (type=education)
│   ├── guide_redflags.md (type=guide)
│   ├── plan_neuro.md (type=plan)
│   └── prep_mri.md (type=prep)
└── gastro/dept=gastro
    ├── education_gastro.md
    ├── guide_redflags.md
    ├── plan_gastro.md
    └── prep_*.md
```

### 7. 状态管理 (`state/schema.py`)

**BaseState** 包含完整的就诊状态：

```python
class BaseState(BaseModel):
    run_id: str                    # 运行ID
    dept: str                      # 科室
    patient_id: str                # 患者ID
    chief_complaint: str           # 主诉
    history_present_illness: dict  # 现病史
    ordered_tests: List[dict]      # 检查/检验单
    test_results: List[dict]       # 检查报告
    diagnosis: dict                # 诊断
    treatment_plan: dict           # 治疗方案
    escalations: List[str]         # 升级建议
    audit_trail: List[dict]        # 审计追踪
    retrieved_chunks: List[dict]   # RAG检索结果
```

**审计追踪格式**：

```json
{
  "ts": "2026-02-02T10:30:00Z",
  "node_name": "C5_common_intake",
  "inputs_summary": {...},
  "outputs_summary": {...},
  "decision": "proceed_to_specialty",
  "citations": [
    {
      "doc_id": "hospital_sop_001",
      "chunk_id": "ch_003",
      "score": 0.89
    }
  ],
  "flags": ["LLM_USED", "RAG_RETRIEVED"]
}
```

---

## 🔧 高级用法

### 自定义配置

```bash
# 使用自定义配置文件
python src/main.py --config my_config.yaml
```

### 审计追踪分析

```python
import json

# 加载保存的追踪
with open("trace.json") as f:
    trace = json.load(f)

# 分析RAG引用
for entry in trace.get("audit_trail", []):
    if entry.get("citations"):
        print(f"{entry['node_name']}: {len(entry['citations'])} citations")

# 检查LLM调用
llm_calls = [e for e in trace.get("audit_trail", []) if "LLM_USED" in e.get("flags", [])]
print(f"Total LLM calls: {len(llm_calls)}")
```

### 批量处理

```bash
# 处理多个数据集ID
for i in 1..10 {
  python src/main.py --dataset-id $i --save-trace "trace_$i.json"
}
```

---

## 📊 外部系统 Mock

所有 Mock 服务基于 `seed` 保证确定性输出：

| 服务 | 模块 | 功能 | 确定性 |
|------|------|------|--------|
| 预约服务 | `appointment.py` | 挂号、签到、叫号 | ✅ Seed-based |
| 缴费服务 | `billing.py` | 费用计算、记录 | ✅ Seed-based |
| 实验室 | `lab.py` | 血常规、肝功能等 | ✅ Mock数据 |
| 影像检查 | `imaging.py` | CT、MRI、超声 | ✅ Mock数据 |
| 内镜检查 | `endoscopy.py` | 胃镜、肠镜 | ✅ Mock数据 |
| 神经生理 | `neurophysiology.py` | EEG、EMG、NCV | ✅ Mock数据 |

---

## 🛡️ 安全机制

### 红旗症状识别

系统自动检测危重症状并触发升级：

- 🚨 **急诊**：生命体征异常、急性胸痛等
- 🏥 **住院**：严重并发症、需要住院治疗
- 👥 **会诊**：疑难病例、多学科协作
- ➡️ **转诊**：超出本科室诊疗范围

---

## 💾 数据持久化

### 文件存储模式

病例数据存储在 `medical_records/` 目录：

```
medical_records/
├── patient_001.json
├── patient_002.json
└── ...
```

### 数据库存储模式

支持 MySQL 持久化，表结构：

**Patient 表**：患者基本信息
**MedicalCase 表**：就诊病例（支持多次就诊）
**Examination 表**：检查检验结果

配置数据库：

```yaml
# config.yaml
database:
  enabled: true
  connection_string: "mysql+pymysql://user:password@host:port/dbname"
  backup_to_file: true  # 同时备份到文件
```

---

## 🧪 运行示例

### 示例 1：基础运行

```bash
python src/main.py
# 输出完整的诊疗流程日志
```

### 示例 2：多患者并发

```bash
python src/main.py --num-patients 3 --patient-interval 60
# 3个患者，间隔60秒依次进入，系统自动分配医生
```

### 示例 3：启用LLM增强

```bash
# 设置API Key
$env:DEEPSEEK_API_KEY="sk-xxx"

# 运行
python src/main.py --enable-reports --backend deepseek
# 使用LLM增强检查报告叙述
```

### 示例 4：交互模式

```bash
python src/main.py --interactive
# 进入交互式命令模式，可实时观察和控制流程
```

---

## 📖 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | - |
| `DEEPSEEK_MODEL` | 模型名称 | `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | API 端点 | `https://api.deepseek.com` |
| `AGENT_MAX_QUESTIONS` | 医生最多问题数 | `3` |
| `ENABLE_RAG` | 启用RAG系统 | `true` |

---

## 📝 日志系统

系统为每个患者生成详细的日志文件：

```
logs/patients/
├── patient_001_20260202_103000.log
├── patient_002_20260202_103100.log
└── ...
```

**日志包含**：
- ✅ 完整的诊疗流程记录
- ✅ 医生问诊对话
- ✅ RAG 检索结果与引用
- ✅ 检查报告和诊断结果
- ✅ 审计追踪和决策理由

---

## 🔗 依赖组件

| 组件 | 版本 | 用途 |
|------|------|------|
| LangGraph | 1.0.7 | 流程编排 |
| ChromaDB | 1.4.1 | 向量数据库 |
| Pydantic | 2.12.5 | 数据验证 |
| SQLAlchemy | 2.0.36 | ORM |
| PyMySQL | 1.1.1 | MySQL 驱动 |
| Typer | 0.21.1 | CLI 框架 |
| Rich | 14.3.0 | 彩色输出 |

---

## ✅ 系统能力矩阵

| 能力维度 | 实现状态 | 说明 |
|---------|---------|------|
| **多智能体协作** | ✅ 完整 | 医生、护士、患者、检验科协作 |
| **多患者并发** | ✅ 完整 | 自动负载均衡和队列管理 |
| **物理环境模拟** | ✅ 完整 | 时间、空间、设备约束 |
| **RAG知识检索** | ✅ 完整 | 关键节点强制检索 |
| **审计追踪** | ✅ 完整 | 完整的决策记录和溯源 |
| **数据库持久化** | ✅ 完整 | MySQL支持 |
| **LLM增强** | ⚡ 可选 | DeepSeek集成 |
| **红旗症状识别** | ✅ 完整 | 自动升级触发 |
| **多科室支持** | ✅ 可扩展 | 神经科、消化科，框架支持新增 |

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## ⚠️ 免责声明

本项目仅用于技术演示和教学目的，不构成任何医疗建议。如有健康问题，请咨询专业医疗机构。

---

<div align="center">

**Made with ❤️ by Patient Agent Team**

</div>
