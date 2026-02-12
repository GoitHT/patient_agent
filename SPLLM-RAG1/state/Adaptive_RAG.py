import os
import sys
import operator
import re
import logging
from typing import List, TypedDict, Annotated

# --- 智能模型下载：检查本地是否有模型，没有则临时允许下载 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CACHE_FOLDER = os.path.join(ROOT_DIR, "model_cache")
EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# 检查模型是否存在
model_cache_path = os.path.join(CACHE_FOLDER, "models--BAAI--bge-large-zh-v1.5")
model_exists = os.path.exists(model_cache_path) and os.path.isdir(model_cache_path)

if not model_exists:
    print(f"⚠️  未检测到本地模型缓存: {model_cache_path}")
    print("📥 首次运行，将在线下载模型...")
    # 临时允许下载
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
else:
    print(f"✅ 检测到本地模型缓存，使用离线模式")
    # 强制使用离线模式
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

os.environ['HF_HOME'] = CACHE_FOLDER  # 指定HuggingFace缓存目录

# 修复：导入正确的Chroma版本
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# --- 1. 全局配置（不变） ---
DEEPSEEK_KEY = 'sk-16ecbb2a436c410e870b3ec10c87a84b'
DEEPSEEK_BASE = 'https://api.deepseek.com'
COSINE_DISTANCE_THRESHOLD = 0.3  # 临时放宽，确保能检索到
# EMBED_MODEL_NAME 已在上面定义


# --- 2. 嵌入模型初始化（支持自动下载） ---
def init_embeddings():
    """从本地缓存初始化嵌入模型，首次运行自动下载"""

    print(f"📂 嵌入模型缓存路径: {CACHE_FOLDER}")
    print(f"📂 缓存路径是否存在: {os.path.exists(CACHE_FOLDER)}")

    # 检查缓存中是否有模型文件
    if os.path.exists(CACHE_FOLDER):
        import glob
        model_files = glob.glob(os.path.join(CACHE_FOLDER, "**/*.bin"), recursive=True) + \
                      glob.glob(os.path.join(CACHE_FOLDER, "**/*.safetensors"), recursive=True)
        print(f"📂 找到 {len(model_files)} 个模型文件")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            },
            cache_folder=CACHE_FOLDER
        )
        if model_exists:
            print("✅ 嵌入模型初始化成功（使用本地缓存）")
        else:
            print("✅ 嵌入模型下载并初始化成功")
            # 下载完成后，重新设置为离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # 测试嵌入模型是否正常工作
        test_emb = embeddings.embed_query("测试文本")
        print(f"🔬 嵌入向量测试: 维度={len(test_emb)}, 范数={sum(x * x for x in test_emb) ** 0.5:.4f}")

        return embeddings
    except Exception as e:
        print(f"❌ 嵌入模型初始化失败: {e}")

        # 备选方案：尝试直接加载sentence-transformers
        try:
            print("🔄 尝试备选加载方案...")
            from sentence_transformers import SentenceTransformer

            # 直接使用本地模型路径
            model_path = os.path.join(CACHE_FOLDER, "models--BAAI--bge-large-zh-v1.5")
            if not os.path.exists(model_path):
                model_path = EMBED_MODEL_NAME

            model = SentenceTransformer(
                model_path,
                cache_folder=CACHE_FOLDER,
                device='cpu'
            )

            # 包装成LangChain兼容的嵌入模型
            class LocalEmbeddings(HuggingFaceEmbeddings):
                def __init__(self, model):
                    self.model = model
                    super().__init__()

                def embed_documents(self, texts):
                    return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

                def embed_query(self, text):
                    return self.model.encode([text], normalize_embeddings=True, batch_size=32)[0]

            embeddings = LocalEmbeddings(model)
            print("✅ 备选嵌入模型加载成功")
            return embeddings

        except Exception as e2:
            print(f"❌ 备选方案也失败: {e2}")

            # 返回一个简单的占位符嵌入模型（仅用于测试）
            class DummyEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 768 for _ in texts]

                def embed_query(self, text):
                    return [0.1] * 768

            print("⚠️ 使用虚拟嵌入模型（功能受限）")
            return DummyEmbeddings()


embeddings = init_embeddings()


# --- 3. 核心修复：动态加载向量库（关键！每次检索都重新读取最新数据） ---
def get_high_quality_qa_db():
    """
    动态加载高质量问答向量库
    解决：模块启动时初始化、更新后无法读取新数据的问题
    """
    # 精准计算路径（适配你的文件结构：SPLLM-RAG1y/chroma/HighQualityQA_db）
    # Adaptive_RAG.py 在 state/ 下，所以根目录是 state/ 的上一级
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)  # 正确指向 SPLLM-RAG1y/
    DB_PATH = os.path.join(ROOT_DIR, "chroma", "HighQualityQA_db")

    # 调试：打印实际路径，确认是否正确
    print(f"[动态加载] HighQualityQA_db 路径: {DB_PATH}")
    print(f"[动态加载] 路径是否存在: {os.path.exists(DB_PATH)}")

    # 动态创建/加载向量库（强制余弦距离）
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="HighQualityQA",
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 调试：打印向量库内实际文档数
    doc_count = db._collection.count()
    print(f"[动态加载] 向量库内文档总数: {doc_count}")
    return db


# --- 工具函数（不变） ---
@tool
def retrieve_docs(query: str):
    """当用户询问医学专业问题，或需要联系患者之前的沟通历史时，调用此工具。"""
    return "已触发检索流程"


# --- 状态定义（不变） ---
class GraphState(TypedDict):
    patient_id: str
    question: str
    messages: Annotated[List[BaseMessage], operator.add]
    generation: str
    documents: List[str]
    history_context: str
    score: float
    evaluation_result: dict


# --- 4. 核心修复：retrieve_high_quality_qa 函数（彻底重构） ---
def retrieve_high_quality_qa(query: str, top_k: int = 3) -> str:
    """
    检索高质量问答向量库（修复版）
    核心改进：动态加载向量库 + 先调试后判断 + 作用域修复
    """
    # 第一步：先校验查询（排除空查询）
    if not query or query.strip() == "":
        print("[QA匹配调试] 无查询内容")
        return ""

    # 第二步：动态加载最新的向量库（关键！）
    try:
        high_quality_qa_db = get_high_quality_qa_db()
    except Exception as e:
        print(f"[QA匹配调试] 向量库加载失败: {str(e)}")
        return ""

    # 第三步：校验向量库是否有数据
    doc_count = high_quality_qa_db._collection.count()
    if doc_count == 0:
        print(f"[QA匹配调试] 向量库内无数据（总数：{doc_count}）")
        return ""

    # 第四步：执行检索（余弦距离）
    try:
        results = high_quality_qa_db.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        print(f"[QA匹配调试] 检索失败: {str(e)}")
        return ""

    if not results:
        print(f"[QA匹配调试] 无检索结果")
        return ""

    # 第五步：过滤并格式化结果
    relevant_qa = []
    for doc, distance in results:
        print(f"[QA匹配调试] 余弦距离={distance:.4f}, 文档内容: {doc.page_content[:100]}...")

        if distance < COSINE_DISTANCE_THRESHOLD:
            similarity = max(0, (1 - distance / 2) * 100)
            question = doc.metadata.get("question", "")
            answer = doc.metadata.get("answer", "")
            relevant_qa.append(
                f"问题：「{question}」\n"
                f"答案：「{answer[:300]}...」\n"
                f"(相似度：{similarity:.1f}%)"
            )

    if relevant_qa:
        print(f"--- [高质量问答匹配] 找到 {len(relevant_qa)} 条高相关优质问答 ---")
        return "\n\n".join(relevant_qa)
    else:
        min_distance = min(d for _, d in results)
        print(f"--- [高质量问答匹配] 最小距离 {min_distance:.4f}，高于阈值 {COSINE_DISTANCE_THRESHOLD} ---")
        return ""


# --- 5. 节点函数（仅修改 agent_node 中的检索调用，其余不变） ---
def agent_node(state: GraphState):
    print("--- [记忆检索] 正在调取历史记忆... ---")
    p_id = state.get("patient_id", "default")
    query = state.get("question", "").strip()  # 去除首尾空格

    # 加载历史记忆（不变）
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    history_db = Chroma(
        persist_directory=os.path.join(ROOT_DIR, "chroma", "UserHistory_db"),
        embedding_function=embeddings
    )
    related_memories = history_db.similarity_search(query, k=2, filter={"patient_id": p_id})
    history_text = "\n".join([d.page_content for d in related_memories]) if related_memories else "尚无相关历史记录"
    print(f"--- [检索结果] 已调取患者 {p_id} 的专属历史 ---")

    # 核心修改：调用修复后的检索函数（动态加载向量库）
    high_quality_qa_text = retrieve_high_quality_qa(query)

    # 后续 Prompt 逻辑（不变）
    prompt_path = os.path.join(CURRENT_DIR, "prompt", "Node1_Triage.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()

    prompt_parts = [SYSTEM_PROMPT, f"【当前患者历史记录】：\n{history_text}"]
    if high_quality_qa_text:
        prompt_parts.append(f"【全量高相关优质问答参考】：\n{high_quality_qa_text}")
    full_system_prompt = "\n\n".join(prompt_parts)

    llm = ChatOpenAI(model='deepseek-chat', openai_api_key=DEEPSEEK_KEY, openai_api_base=DEEPSEEK_BASE, temperature=0)
    llm_with_tools = llm.bind_tools([retrieve_docs])
    prompt = ChatPromptTemplate.from_messages([("system", full_system_prompt), ("user", "{input}")])
    chain = prompt | llm_with_tools
    response = chain.invoke({"input": query})
    return {"messages": [response], "history_context": history_text, "question": query}


# --- 其余节点函数（retrieve_node/generate_node/direct_answer_node/evaluation_node/record_memory_node）保持不变 ---
def retrieve_node(state: GraphState):
    import concurrent.futures
    from functools import lru_cache
    print("--- [节点 2] 结合历史记忆执行检索与多维提炼（优化版） ---")
    question = f"背景：{state['history_context']} 问题：{state['question']}"
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)

    # ========== 新增：定义分隔符变量，避免f-string内直接使用反斜杠 ==========
    DOC_SEPARATOR = "\n---\n"  # 文档片段分隔符，提前定义

    # ========== 优化1：缓存向量库实例，避免重复加载 ==========
    @lru_cache(maxsize=2)
    def get_vector_db(db_name):
        """缓存向量库实例，重复调用时直接复用"""
        db_path = os.path.join(ROOT_DIR, "chroma", db_name)
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # 确保余弦距离
        )

    # ========== 优化2：并行检索两个向量库（替代串行） ==========
    def retrieve_single_db(db_name, query, k=3):  # 优化3：降低k值（从5→3），减少处理量
        db = get_vector_db(db_name)
        docs = db.similarity_search(query, k=k)
        # 优化4：提前过滤空文档/无效文档
        valid_docs = [d for d in docs if d.page_content.strip() and len(d.page_content) > 20]
        return valid_docs

    # 并行执行指南库+案例库检索
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        guide_future = executor.submit(retrieve_single_db, "MedicalGuide_db", question)
        case_future = executor.submit(retrieve_single_db, "ClinicalCase_db", question)
        guide_docs = guide_future.result()
        case_docs = case_future.result()

    # 无有效文档时快速返回
    if not guide_docs and not case_docs:
        return {
            "documents": ["未找到相关资料"],
            "messages": [ToolMessage(
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                content="未找到相关资料"
            )]
        }

    # ========== 优化5：批量处理文档，减少LLM调用次数 ==========
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=DEEPSEEK_KEY,
        openai_api_base=DEEPSEEK_BASE,
        temperature=0,
        request_timeout=15  # 优化6：缩短超时时间，避免卡顿
    )
    refined_context = {"guides": [], "cases": [], "conflicts": []}

    # 批量处理指南文档（✅ 修复：使用预定义分隔符变量）
    if guide_docs:
        # 优化7：精简文档内容（截取前1000字符），减少LLM输入长度
        guide_texts = [f"指南片段{i + 1}：{d.page_content[:1000]}" for i, d in enumerate(guide_docs)]
        # 替换直接的"\n---\n"为预定义变量，消除f-string内的反斜杠
        guide_prompt = f"""批量提取以下医学指南的核心信息（每条仅保留：核心推荐、针对症状、来源）：
{DOC_SEPARATOR.join(guide_texts)}"""
        guide_response = llm.invoke(guide_prompt).content
        refined_context["guides"] = [guide_response]  # 批量输出，替代逐个调用

    # 批量处理案例文档（✅ 修复：使用预定义分隔符变量）
    if case_docs:
        case_texts = [f"案例片段{i + 1}：{d.page_content[:800]}" for i, d in enumerate(case_docs)]  # 进一步精简
        # 修正：使用预定义分隔符变量，彻底避免f-string内的反斜杠
        case_prompt = f"""批量分析以下临床案例对当前问题的参考价值（问题：{question}），每条回答包含：
1. 相似症状 2. 治疗方案 3. 医生经验/警示
{DOC_SEPARATOR.join(case_texts)}"""
        case_response = llm.invoke(case_prompt).content
        refined_context["cases"] = [case_response]  # 批量输出，替代逐个调用

    # ========== 优化8：精简冲突分析Prompt，提升效率 ==========
    if refined_context["guides"] and refined_context["cases"]:
        conflict_prompt = f"""对比以下指南与案例，仅指出案例中未被指南覆盖的临床实操点：
指南核心：{refined_context['guides'][0][:500]}
案例核心：{refined_context['cases'][0][:500]}"""
        conflict_res = llm.invoke(conflict_prompt).content
        refined_context["conflicts"] = conflict_res
    else:
        refined_context["conflicts"] = "无足够数据进行对比"

    return {
        "documents": [str(refined_context)],
        "messages": [ToolMessage(
            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            content="双库联合检索完成（优化版）"
        )]
    }

def generate_node(state: GraphState):
    print("--- [节点 3] 基于检索结果生成回答 ---")
    refined_info = state.get("documents", ["无可用参考资料"])[0]
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(CURRENT_DIR, "prompt", "Node3_Generate.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read()

    llm = ChatOpenAI(model='deepseek-chat', openai_api_key=DEEPSEEK_KEY, openai_api_base=DEEPSEEK_BASE, temperature=0)
    response = llm.invoke([
        ("system", f"{SYSTEM_PROMPT}\n历史记忆：{state['history_context']}"),
        ("user", f"问题：{state['question']}\n资料：{refined_info}")
    ])
    return {"generation": response.content}


def direct_answer_node(state: GraphState):
    print("--- [节点 4] 综合回答控制中心 ---")
    llm = ChatOpenAI(model='deepseek-chat', openai_api_key=DEEPSEEK_KEY, openai_api_base=DEEPSEEK_BASE, temperature=0.7)
    docs = state.get("documents", [])
    is_fallback = len(docs) > 0
    h_context = state.get("history_context", "无相关历史记录")

    if is_fallback:
        system_msg = (
            f"你是一位专业的医生。已知患者历史背景：{h_context}。\n"
            "虽然在医学库中未匹配到对标条目，但请基于通用医学知识给出建议。"
            "注意：1. 语气专业谨慎；2. 提示建议仅供参考。"
        )
    else:
        system_msg = (
            f"你是一个专业的医疗助手。参考患者之前的沟通记录：{h_context}。\n"
            "请友好地回答用户。打招呼请礼貌回应；常识问题请简洁明了。"
        )

    response = llm.invoke([("system", system_msg), ("user", state["question"])])
    return {"generation": response.content}


def evaluation_node(state: GraphState):
    import json
    print("--- [节点 5] 质量评估中心 ---")
    question = state.get("question", "")
    generation = state.get("generation", "")
    history_context = state.get("history_context", "无记录")
    answer_text = generation.content if (hasattr(generation, 'content') and generation.content) else str(generation)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(CURRENT_DIR, "prompt", "Node5_Evaluation.txt")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            full_system_prompt = f.read()
    except FileNotFoundError:
        print("❌ 评估配置文件Node5_Evaluation.txt未找到")
        return {"score": 0.0, "evaluation_result": {"total_score": 0.0, "is_high_quality": False}}

    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        ("user", "【患者提问】：{question}\n【历史背景】：{history_context}\n【医生回答】：{answer_text}")
    ])

    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=DEEPSEEK_KEY,
        openai_api_base=DEEPSEEK_BASE,
        temperature=0,
        request_timeout=30
    )
    chain = prompt | llm

    try:
        response = chain.invoke({"question": question, "history_context": history_context, "answer_text": answer_text})
        eval_content = response.content.strip()
        evaluation_result = json.loads(eval_content)

        if "total_score" not in evaluation_result:
            base_score = sum(evaluation_result.get("dimension_scores", {}).values())
            bonus = evaluation_result.get("bonus_points", 0.0)
            penalty = evaluation_result.get("penalty_points", 0.0)
            total_score = round(base_score + bonus - penalty, 1)
            evaluation_result["total_score"] = total_score

        dim_scores = evaluation_result.get("dimension_scores", {})
        total_score = evaluation_result.get("total_score", 0.0)
        has_medical_guide = any(
            key in answer_text for key in ["剂量", "检查", "用药", "就医时机", "方案", "疗程", "减量"])
        has_fabricated_info = evaluation_result.get("penalty_points", 0.0) <= -1.4
        is_high_quality = (
                total_score >= 8.0 and
                dim_scores.get("医学准确性", 0.0) >= 1.2 and
                dim_scores.get("安全合规性", 0.0) >= 1.0 and
                has_medical_guide and
                not has_fabricated_info
        )

        evaluation_result["is_high_quality"] = is_high_quality
        evaluation_result["answer_text"] = answer_text
        print(f"⭐ 评估结果: 总分 {evaluation_result['total_score']}, 高质量对话: {is_high_quality}")
        print(f"   7维度分数: {evaluation_result['dimension_scores']}")
        print(f"   加分项: {evaluation_result['bonus_points']} | 扣分项: {evaluation_result['penalty_points']}")
        return {"score": total_score, "evaluation_result": evaluation_result}
    except json.JSONDecodeError:
        print(f"❌ 评估结果JSON解析失败，原始响应：{eval_content}")
        return {"score": 0.0, "evaluation_result": {"total_score": 0.0, "is_high_quality": False}}
    except Exception as e:
        print(f"❌ 评估节点调用失败: {str(e)}")
        return {"score": 3.0, "evaluation_result": {"total_score": 3.0, "is_high_quality": False}}


def record_memory_node(state: GraphState):
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    sys.path.append(ROOT_DIR)

    try:
        from create_database_general import store_chat_history_rag, store_doctor_qa_evolution, init_high_quality_qa_db
    except ImportError as e:
        print(f"❌ 导入记忆存储模块失败: {str(e)}")
        return state

    print("--- [节点 6] 记忆持久化与进化中心 ---")
    p_id = state.get("patient_id", "default")
    question = state.get("question", "")
    generation = state.get("generation", "")
    useful_info = state.get("history_context", "")
    evaluation_result = state.get("evaluation_result", {})
    is_high_quality = evaluation_result.get("is_high_quality", False)
    total_score = evaluation_result.get("total_score", 0.0)
    answer_text = evaluation_result.get("answer_text",
                                        generation.content if hasattr(generation, 'content') else str(generation))

    # 存储对话历史 + 高质量问答
    store_chat_history_rag(question, answer_text, p_id)
    store_doctor_qa_evolution(
        question=question,
        answer=answer_text,
        rag_info=useful_info,
        patient_id=p_id,
        score=total_score,
        is_high_quality=is_high_quality
    )

    # 核心新增：更新向量库后，**立即重新初始化**（确保下一次检索能读到最新数据）
    init_high_quality_qa_db()
    print("✅ 高质量问答向量库已同步最新数据")
    return state


# --- 决策函数（不变） ---
def should_retrieve(state: GraphState):
    last_message = state["messages"][-1]
    return "retrieve" if last_message.tool_calls else "direct_response"


def decide_to_generate(state: GraphState):
    print("--- [决策闸门] 评估检索相关性 ---")
    docs = state.get("documents", [])
    if not docs or "未找到相关资料" in str(docs):
        print(">>> 决策：知识库完全无关，切换至大模型直接回答")
        return "fallback"

    llm = ChatOpenAI(model='deepseek-chat', openai_api_key=DEEPSEEK_KEY, openai_api_base=DEEPSEEK_BASE, temperature=0)
    check_prompt = f"判定以下资料与问题是否相关。只需回复 [合格] 或 [不相关]。\n问题：{state['question']}\n资料：{str(docs)[:500]}"
    verification = llm.invoke(check_prompt).content

    if "[不相关]" in verification:
        print(">>> 决策：资料质量不佳，切换至大模型直接回答")
        return "fallback"
    return "generate"


# --- 图构建与编译（不变） ---
workflow = StateGraph(GraphState)
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("direct_answer_node", direct_answer_node)
workflow.add_node("record_memory", record_memory_node)
workflow.add_node("evaluate", evaluation_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_retrieve,
                               {"retrieve": "retrieve_node", "direct_response": "direct_answer_node"})
workflow.add_conditional_edges("retrieve_node", decide_to_generate,
                               {"generate": "generate_node", "fallback": "direct_answer_node"})
workflow.add_edge("generate_node", "evaluate")
workflow.add_edge("direct_answer_node", "evaluate")
workflow.add_edge("evaluate", "record_memory")
workflow.add_edge("record_memory", END)
app = workflow.compile()

# --- 主函数（测试用，不变） ---
if __name__ == "__main__":
    test_input = {
        "patient_id": "test_001",
        "question": "高血压患者突发头痛、一侧肢体无力和言语含糊该怎么办？",
        "messages": [],
        "generation": "",
        "documents": [],
        "history_context": "",
        "score": 0.0,
        "evaluation_result": {}
    }
    print("\n=== 🚀 运行Adaptive RAG测试 ===")
    result = app.invoke(test_input)
    print(f"\n📝 最终回答：\n{result['generation']}")
    print(
        f"\n📊 评估结果：\n总分: {result['score']}, 高质量对话: {result['evaluation_result'].get('is_high_quality', False)}")