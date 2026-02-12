import os
import sys
import chardet
import json
import re
import csv
import numpy as np
import shutil
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# --- 关键修复：在导入任何HuggingFace库之前设置环境变量和路径 ---
# 获取当前脚本所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, ROOT_DIR)  # 添加项目根目录到路径
CACHE_FOLDER = os.path.join(CURRENT_DIR, "model_cache")
MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# 检查模型是否存在（使用绝对路径）
model_cache_path = os.path.join(CACHE_FOLDER, "models--BAAI--bge-large-zh-v1.5")
model_exists = os.path.exists(model_cache_path) and os.path.isdir(model_cache_path)

if not model_exists:
    print(f"⚠️  未检测到本地模型: {model_cache_path}")
    print("📥 将在线下载模型（约1.3GB）...")
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
else:
    print(f"✅ 使用本地缓存模型: {model_cache_path}")
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 设置缓存路径
os.environ['HF_HOME'] = CACHE_FOLDER

# 现在才导入HuggingFace相关库
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 导入进度条（用于 rebuild 模式）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm 未安装，rebuild 模式将无进度条显示")

# 导入动态分块器（用于 rebuild 模式）
try:
    from src.rag.dynamic_chunker import DynamicChunker, ChunkConfig, ChunkStrategy
    DYNAMIC_CHUNKER_AVAILABLE = True
except ImportError:
    DYNAMIC_CHUNKER_AVAILABLE = False
    print("⚠️  DynamicChunker 未找到，rebuild 模式将使用固定分块")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_normalized_embeddings():
    """创建归一化的嵌入模型（支持自动下载）"""
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,  # 强制归一化
            'batch_size': 32
        },
        cache_folder=CACHE_FOLDER
    )
    
    # 下载完成后恢复离线模式
    if not model_exists:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print("✅ 模型下载完成，已切换回离线模式")
    
    # 验证归一化效果
    test_emb = embeddings.embed_query("测试文本")
    norm = np.linalg.norm(test_emb)
    print(f"嵌入向量归一化验证：范数 = {norm:.4f}（理想值=1.0）")
    return embeddings


# 全局嵌入模型（确保所有向量库使用同一归一化模型）
EMBEDDINGS = get_normalized_embeddings()


# =============================================================================
# 动态分块相关函数（用于 rebuild 模式）
# =============================================================================

def load_documents_from_json_rebuild(file_path: Path) -> List[Dict[str, Any]]:
    """从 JSON 文件加载文档（rebuild 模式专用）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"📄 从 {file_path.name} 加载了 {len(data) if isinstance(data, list) else 1} 个文档")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        logger.error(f"❌ 加载文件失败 {file_path}: {e}")
        return []


def load_documents_from_txt_rebuild(file_path: Path) -> List[Dict[str, Any]]:
    """从 TXT 文件加载文档（rebuild 模式专用）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单分割（按双换行符）
        sections = content.split('\n\n')
        docs = []
        
        for i, section in enumerate(sections):
            if section.strip():
                docs.append({
                    "text": section.strip(),
                    "meta": {
                        "source": file_path.name,
                        "section_id": i
                    }
                })
        
        logger.info(f"📄 从 {file_path.name} 加载了 {len(docs)} 个文档段")
        return docs
    except Exception as e:
        logger.error(f"❌ 加载文件失败 {file_path}: {e}")
        return []


def create_vector_db_with_progress(
    documents: List[Document],
    embeddings,
    db_path: str,
    collection_name: str,
    batch_size: int = 100
):
    """创建向量库（带进度显示和批处理）"""
    # 删除旧数据库
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        logger.info(f"   → 已删除旧向量库")
    
    total_docs = len(documents)
    logger.info(f"   → 开始创建向量库: {total_docs} 个文档")
    logger.info(f"   → 批处理大小: {batch_size}")
    
    # 估算时间
    estimated_time = (total_docs / batch_size) * 2
    logger.info(f"   → 预计耗时: {estimated_time/60:.1f} 分钟")
    
    db = None
    start_time = time.time()
    
    # 分批处理
    if TQDM_AVAILABLE:
        iterator = tqdm(range(0, total_docs, batch_size), desc=f"   创建 {collection_name}", unit="batch")
    else:
        iterator = range(0, total_docs, batch_size)
    
    for i in iterator:
        batch = documents[i:i+batch_size]
        
        if db is None:
            # 第一批：创建数据库
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=db_path,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # 后续批次：添加到现有数据库
            db.add_documents(batch)
    
    elapsed = time.time() - start_time
    logger.info(f"   → 完成！实际耗时: {elapsed/60:.1f} 分钟")
    
    return db


# --- 修复2：统一向量库创建逻辑（指定余弦距离） ---
def create_chroma_db_with_cosine(docs, db_path, collection_name):
    """
    创建指定余弦距离的Chroma向量库
    :param docs: 文档列表（不能为空）
    :param db_path: 持久化路径
    :param collection_name: 集合名称
    :return: Chroma向量库实例
    """
    # 如果路径存在，先删除（确保重新创建时使用指定的距离函数）
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"已删除旧向量库：{db_path}")

    # 检查文档列表是否为空
    if not docs:
        raise ValueError(f"无法创建空的向量库 {db_path}，文档列表不能为空")

    # 创建向量库（显式指定余弦距离）
    db = Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDINGS,
        persist_directory=db_path,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}  # 强制使用余弦距离
    )
    db.persist()
    print(f"✅ 已创建余弦距离向量库：{db_path}，包含 {len(docs)} 个文档")
    return db


# --- 2. 核心加载器：修复 JSON 解析 ---
def load_single_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    def get_encoding(path):
        with open(path, 'rb') as f:
            return chardet.detect(f.read())['encoding'] or 'utf-8'

    try:
        if ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding=get_encoding(file_path)).load()
        elif ext == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                docs = []
                rows = []
                # 兼容性修复：适配单条记录 JSON 或 列表 JSON
                if isinstance(data, list):
                    rows = data
                elif isinstance(data, dict):
                    # 如果字典本身就有这些医学键，说明它本身就是一行记录
                    if "medicalRecordId" in data or "主诉" in data:
                        rows = [data]
                    else:
                        # 否则按 Sheet 结构处理
                        for val in data.values():
                            if isinstance(val, list): rows.extend(val)
                for row in rows:
                    content = " | ".join([f"{k}: {str(v).strip()}" for k, v in row.items() if v])
                    content = re.sub(r'\s+', ' ', content)
                    if len(content) > 10:
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": file_path}
                        ))
                return docs
        return []
    except Exception as e:
        print(f"\n❌ 读取文件 {file_path} 失败: {e}")
        return []


# --- 3. 增量同步逻辑：适配根目录chroma + 余弦距离 ---
def update_vector_db(db_name, data_folder, use_dynamic_chunker=True):
    """
    增量更新向量库
    :param db_name: 数据库名称
    :param data_folder: 数据文件夹
    :param use_dynamic_chunker: 是否使用动态分块器（默认True）
    """
    # 向量库路径：根目录/chroma/{db_name}
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma", db_name)
    data_dir = os.path.join("data", data_folder)
    print(f"\n>>> 🚀 开始同步数据库: {db_name}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return

    # 获取文件列表
    allowed_extensions = {".txt", ".md", ".json"}
    files_to_process = [f for f in os.listdir(data_dir) if os.path.splitext(f)[-1].lower() in allowed_extensions]

    if not files_to_process:
        print(f"⚠️ 数据目录 {data_dir} 中没有可处理的文件")
        return

    # 根据参数选择分块器
    if use_dynamic_chunker and DYNAMIC_CHUNKER_AVAILABLE:
        print(f"📊 使用动态分块策略")
        chunker = DynamicChunker()
        # 根据数据库类型选择分块配置
        if "Guide" in db_name:
            chunk_config = ChunkConfig(
                strategy=ChunkStrategy.HIERARCHICAL,
                chunk_size=800,
                chunk_overlap=100,
                min_chunk_size=100,
                max_chunk_size=2000
            )
        elif "Case" in db_name:
            chunk_config = ChunkConfig(
                strategy=ChunkStrategy.SEMANTIC,
                chunk_size=600,
                chunk_overlap=80,
                min_chunk_size=100,
                max_chunk_size=1500
            )
        else:
            chunk_config = ChunkConfig(
                strategy=ChunkStrategy.FIXED,
                chunk_size=600,
                chunk_overlap=60
            )
        use_dynamic = True
    else:
        if use_dynamic_chunker and not DYNAMIC_CHUNKER_AVAILABLE:
            print(f"⚠️ 动态分块器不可用，使用固定分块策略")
        else:
            print(f"📊 使用固定分块策略")
        # 固定切分器：强制用于所有文件，防止 Token 溢出
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]
        )
        use_dynamic = False

    # 检查向量库是否存在
    if os.path.exists(db_path):
        # 加载已有向量库
        db = Chroma(
            persist_directory=db_path,
            embedding_function=EMBEDDINGS,
            collection_name=db_name,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # 获取已存在文件清单
        results = db.get()
        processed_files = set()
        if results and results['metadatas']:
            processed_files = {os.path.basename(m['source']) for m in results['metadatas'] if 'source' in m}
    else:
        # 首次创建向量库
        db = None
        processed_files = set()
        print(f"🆕 首次创建向量库: {db_name}")

    # 处理所有文件
    for i, filename in enumerate(files_to_process):
        # 如果文件已在库中，直接跳过
        if filename in processed_files:
            print(f"⏩ 文件已存在，跳过: {filename}")
            continue

        file_path = os.path.join(data_dir, filename)
        print(f"📄 正在处理 ({i + 1}/{len(files_to_process)}): {filename} ", end="", flush=True)

        raw_docs = load_single_document(file_path)
        if raw_docs:
            # 根据分块策略进行切分
            if use_dynamic:
                # 转换为动态分块器需要的格式
                docs_for_chunking = []
                for doc in raw_docs:
                    docs_for_chunking.append({
                        "text": doc.page_content,
                        "meta": doc.metadata
                    })
                chunked = chunker.chunk_documents(docs_for_chunking, chunk_config)
                # 转换回 LangChain Document
                current_splits = [
                    Document(page_content=d["text"], metadata=d["meta"])
                    for d in chunked
                ]
            else:
                # 使用固定分块
                current_splits = text_splitter.split_documents(raw_docs)
            
            if current_splits:
                if db is None:
                    # 首次创建向量库
                    db = create_chroma_db_with_cosine(current_splits, db_path, db_name)
                else:
                    # 批量入库到已有向量库
                    batch_size = 50
                    for j in range(0, len(current_splits), batch_size):
                        batch = current_splits[j: j + batch_size]
                        db.add_documents(batch)
                        print(".", end="", flush=True)
                    db.persist()
                print(f" ✅ 完成 (新增 {len(current_splits)} 个片段)")
            else:
                print("⚠️ 内容无效")
        else:
            print("❌ 加载失败")

    print(f"✨ {db_name} 同步完成！")


# --- 4. 实时对话存储函数：修复版（确保余弦距离） ---
def store_chat_history_rag(question: str, answer: str, patient_id: str, db_name="UserHistory_db"):
    """
    增加了 patient_id 参数，实现多患者隔离存储，向量库指向根目录chroma
    修复：强制使用余弦距离存储
    """
    # 构建文档
    doc_content = f"患者问: {question} | 医生答: {answer}"
    doc = Document(
        page_content=doc_content,
        metadata={"patient_id": patient_id, "source": "conversation_history"}
    )

    # 向量库路径：根目录/chroma/{db_name}
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma", db_name)

    # 切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["|", "。", "；", "！", "？", "，", " "]
    )
    splits = text_splitter.split_documents([doc])

    if not splits:
        print(f"⚠️ 无法存储空对话")
        return

    # 检查向量库是否存在
    if os.path.exists(db_path):
        # 加载已有向量库
        db = Chroma(
            persist_directory=db_path,
            embedding_function=EMBEDDINGS,
            collection_name="UserHistory",
            collection_metadata={"hnsw:space": "cosine"}
        )
        db.add_documents(splits)
        db.persist()
    else:
        # 首次创建向量库
        create_chroma_db_with_cosine(
            docs=splits,
            db_path=db_path,
            collection_name="UserHistory"
        )

    print(f"✅ 患者 {patient_id} 的对话已完成切片索引")


# --- 5. 医生进化存储函数：双存储（用户专属+全量汇总），适配state/dataset ---
def store_doctor_qa_evolution(question, answer, rag_info, patient_id, score, is_high_quality):
    """
    仅高质量对话才会被存入CSV，用于Few-shot学习
    新增：同步写入【state/dataset】下的用户专属CSV + 全量高分对话汇总CSV，标记patient_id便于溯源
    """
    if not is_high_quality:
        print(f"⚠️ 对话非高质量（总分{score}），未达进化标准，不执行 CSV 存储")
        return

    # CSV存储路径：根目录/state/dataset/
    root_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(root_dir, "state", "dataset")
    os.makedirs(csv_dir, exist_ok=True)

    # ========== 1. 写入用户专属CSV ==========
    user_csv_path = os.path.join(csv_dir, f"doctor_evolve_{patient_id}.csv")
    file_exists = os.path.isfile(user_csv_path)
    # 新增patient_id字段，用于溯源和向量库去重
    headers = [
        "patient_id",
        "question1", "qus_embedding", "rag_info1", "answer1",
        "qus2_embedding", "question2", "answer2", "rag_info2",
        "total_score", "is_high_quality"
    ]
    # 构造数据行
    row = [
        patient_id,
        "N/A", "vector_placeholder", "N/A", "N/A",
        "vector_placeholder", question, answer, rag_info,
        score, is_high_quality
    ]
    # 写入用户专属库
    with open(user_csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

    # ========== 2. 新增：写入全量高分对话汇总CSV ==========
    summary_csv_path = os.path.join(csv_dir, "doctor_evolve_summary.csv")
    summary_file_exists = os.path.isfile(summary_csv_path)
    with open(summary_csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not summary_file_exists:
            writer.writerow(headers)  # 字段与用户CSV一致，含patient_id
        writer.writerow(row)  # 复用同一行数据，确保数据一致
    print(f"🚀 高质量问答已存入：用户专属库({user_csv_path}) + 全量汇总库({summary_csv_path}) (Score: {score})")


# --- 6. 修复3：高质量问答向量库初始化/更新函数（余弦距离版） ---
def init_high_quality_qa_db():
    """
    初始化/更新高质量问答向量库（HighQualityQA_db）
    【修复版】：使用余弦距离+归一化嵌入，将问题和答案分开存储
    基于【state/dataset/doctor_evolve_summary.csv】构建，向量库指向根目录chroma
    【增强】：即使 CSV 不存在也会创建空向量库
    """
    db_name = "HighQualityQA_db"
    # 向量库路径：根目录/chroma/HighQualityQA_db
    root_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(root_dir, "chroma", db_name)
    # CSV汇总文件路径：根目录/state/dataset/doctor_evolve_summary.csv
    csv_dir = os.path.join(root_dir, "state", "dataset")
    summary_csv_path = os.path.join(csv_dir, "doctor_evolve_summary.csv")

    # 1. 检查汇总CSV是否存在
    if not os.path.exists(summary_csv_path):
        logger.warning(f"⚠️ 全量高分对话汇总CSV不存在({summary_csv_path})，创建空向量库")
        # 创建空向量库
        if not os.path.exists(db_path):
            placeholder_doc = Document(
                page_content="高质量问答库初始化占位符",
                metadata={
                    "type": "placeholder",
                    "patient_id": "system",
                    "question": "初始化问题",
                    "answer": "初始化答案",
                    "source": "high_quality_qa_init"
                }
            )
            db = Chroma.from_documents(
                documents=[placeholder_doc],
                embedding=EMBEDDINGS,
                persist_directory=db_path,
                collection_name="HighQualityQA",
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ 高质量问答库创建成功（空库）")
        else:
            logger.info(f"ℹ️  高质量问答库已存在，无需更新")
        return

    # 2. 读取CSV并构造文档
    high_quality_docs = []
    with open(summary_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question2"]
            answer = row["answer2"]
            patient_id = row["patient_id"]

            # ⚡️ 关键改进：将问题单独存储为一个文档（便于问题匹配）
            question_doc = Document(
                page_content=question,  # 只存储问题文本，便于匹配
                metadata={
                    "patient_id": patient_id,
                    "question": question,
                    "answer": answer,  # 答案放在metadata中
                    "full_answer": answer,
                    "score": row["total_score"],
                    "source": "high_quality_qa_summary",
                    "doc_type": "question"  # 标记文档类型
                }
            )
            high_quality_docs.append(question_doc)

            # 可选：也可以存储答案文档，用于答案检索
            answer_doc = Document(
                page_content=answer,  # 存储答案文本
                metadata={
                    "patient_id": patient_id,
                    "question": question,
                    "answer": answer,
                    "score": row["total_score"],
                    "source": "high_quality_qa_summary",
                    "doc_type": "answer"
                }
            )
            high_quality_docs.append(answer_doc)

    if not high_quality_docs:
        logger.warning("ℹ️ 无高质量问答数据，创建空向量库")
        # 创建空向量库
        if not os.path.exists(db_path):
            placeholder_doc = Document(
                page_content="高质量问答库初始化占位符",
                metadata={
                    "type": "placeholder",
                    "patient_id": "system",
                    "question": "初始化问题",
                    "answer": "初始化答案",
                    "source": "high_quality_qa_init"
                }
            )
            db = Chroma.from_documents(
                documents=[placeholder_doc],
                embedding=EMBEDDINGS,
                persist_directory=db_path,
                collection_name="HighQualityQA",
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ 高质量问答库创建成功（空库）")
        return

    # 3. 批量入库（切片后入库，防止Token溢出）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["|", "。", "；", "！", "？", "，", " "]
    )
    splits = text_splitter.split_documents(high_quality_docs)

    if not splits:
        print("⚠️ 切片后无有效文档")
        return

    # 4. 初始化/加载向量库（指定余弦距离）
    if os.path.exists(db_path):
        # 加载已有向量库
        db = Chroma(
            persist_directory=db_path,
            embedding_function=EMBEDDINGS,
            collection_name="HighQualityQA",
            collection_metadata={"hnsw:space": "cosine"}
        )

        # 读取已存在的记录（通过question+patient_id做唯一标识）
        processed_qa = set()
        results = db.get()
        if results and results['metadatas']:
            processed_qa = {(m.get("question"), m.get("patient_id")) for m in results['metadatas'] if
                            m.get("question") and m.get("patient_id")}

        # 过滤掉已存在的文档
        new_splits = []
        for split in splits:
            question = split.metadata.get("question")
            patient_id = split.metadata.get("patient_id")
            if (question, patient_id) not in processed_qa:
                new_splits.append(split)

        splits = new_splits
    else:
        # 首次创建向量库
        db = create_chroma_db_with_cosine(splits, db_path, "HighQualityQA")
        return

    # 5. 批量入库
    if splits:
        batch_size = 50
        for j in range(0, len(splits), batch_size):
            batch = splits[j:j + batch_size]
            db.add_documents(batch)
        db.persist()
        print(f"✅ 高质量问答向量库更新完成，新增 {len(splits)} 个片段（问题+答案分别存储）")
    else:
        print("ℹ️ 无新增高质量问答，向量库无需更新")


# =============================================================================
# Rebuild 模式：使用动态分块完全重建向量库
# =============================================================================

def rebuild_medical_guide_db_dynamic():
    """重建医学指南库（使用动态分块 - 层次分块）"""
    if not DYNAMIC_CHUNKER_AVAILABLE:
        logger.warning("⚠️  DynamicChunker 不可用，使用固定分块")
        update_vector_db("MedicalGuide_db", "MedicalGuide_data")
        return
    
    logger.info("🏗️  开始重建：医学指南库（动态分块）")
    
    data_dir = Path(CURRENT_DIR) / "data" / "MedicalGuide_data"
    output_dir = Path(CURRENT_DIR) / "chroma"
    
    if not data_dir.exists():
        logger.warning(f"⚠️  目录不存在: {data_dir}")
        return
    
    # 加载数据
    guide_files = list(data_dir.glob("*.txt"))
    all_docs = []
    
    for file in guide_files:
        docs = load_documents_from_txt_rebuild(file)
        for doc in docs:
            doc["meta"]["type"] = "guideline"
        all_docs.extend(docs)
    
    if not all_docs:
        logger.warning("⚠️  没有找到医学指南文档")
        return
    
    # 使用层次分块策略
    chunker = DynamicChunker()
    config = ChunkConfig(
        strategy=ChunkStrategy.HIERARCHICAL,
        chunk_size=800,
        chunk_overlap=100,
        min_chunk_size=100,
        max_chunk_size=2000
    )
    
    chunked_docs = chunker.chunk_documents(all_docs, config)
    
    # 转换为 LangChain Document 格式
    lc_docs = [
        Document(page_content=doc["text"], metadata=doc["meta"])
        for doc in chunked_docs
    ]
    
    # 创建向量库
    db_path = str(output_dir / "MedicalGuide_db")
    create_vector_db_with_progress(
        documents=lc_docs,
        embeddings=EMBEDDINGS,
        db_path=db_path,
        collection_name="MedicalGuide",
        batch_size=100
    )
    
    logger.info(f"✅ 医学指南库创建成功: {len(lc_docs)} 个块")


def rebuild_hospital_process_db_dynamic():
    """重建医院流程库（使用动态分块 - 固定分块适用于模板类文档）"""
    if not DYNAMIC_CHUNKER_AVAILABLE:
        logger.warning("⚠️  DynamicChunker 不可用，使用固定分块")
        update_vector_db("HospitalProcess_db", "HospitalProcess_data")
        return
    
    logger.info("🏗️  开始重建：医院流程库（动态分块）")
    
    data_dir = Path(CURRENT_DIR) / "data" / "HospitalProcess_data"
    output_dir = Path(CURRENT_DIR) / "chroma"
    
    if not data_dir.exists():
        logger.warning(f"⚠️  目录不存在: {data_dir}")
        return
    
    # 加载数据（支持 txt 和 json 格式）
    process_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md")) + list(data_dir.glob("*.json"))
    all_docs = []
    
    for file in process_files:
        if file.suffix in [".txt", ".md"]:
            docs = load_documents_from_txt_rebuild(file)
            for doc in docs:
                doc["meta"]["type"] = "hospital_process"
            all_docs.extend(docs)
        elif file.suffix == ".json":
            docs = load_documents_from_json_rebuild(file)
            for item in docs:
                doc = None
                if isinstance(item, dict):
                    if "text" in item or "content" in item:
                        doc = {
                            "text": item.get("text") or item.get("content", ""),
                            "meta": item.get("meta", {})
                        }
                    else:
                        doc = {"text": str(item), "meta": {}}
                elif isinstance(item, str):
                    doc = {"text": item, "meta": {}}
                
                if doc and doc["text"].strip():
                    doc["meta"]["type"] = "hospital_process"
                    doc["meta"]["source"] = file.name
                    all_docs.append(doc)
    
    if not all_docs:
        logger.warning("⚠️  没有找到医院流程文档")
        return
    
    # 使用固定分块策略（适合模板和流程文档）
    chunker = DynamicChunker()
    config = ChunkConfig(
        strategy=ChunkStrategy.FIXED,
        chunk_size=600,
        chunk_overlap=60,
        min_chunk_size=100,
        max_chunk_size=1500
    )
    
    chunked_docs = chunker.chunk_documents(all_docs, config)
    
    # 转换为 LangChain Document 格式
    lc_docs = [
        Document(page_content=doc["text"], metadata=doc["meta"])
        for doc in chunked_docs
    ]
    
    # 创建向量库
    db_path = str(output_dir / "HospitalProcess_db")
    create_vector_db_with_progress(
        documents=lc_docs,
        embeddings=EMBEDDINGS,
        db_path=db_path,
        collection_name="HospitalProcess",
        batch_size=100
    )
    
    logger.info(f"✅ 医院流程库创建成功: {len(lc_docs)} 个块")


def rebuild_clinical_case_db_dynamic():
    """重建临床案例库（使用动态分块 - 语义分块）"""
    if not DYNAMIC_CHUNKER_AVAILABLE:
        logger.warning("⚠️  DynamicChunker 不可用，使用固定分块")
        update_vector_db("ClinicalCase_db", "ClinicalCase_data")
        return
    
    logger.info("🏗️  开始重建：临床案例库（动态分块）")
    
    data_dir = Path(CURRENT_DIR) / "data" / "ClinicalCase_data"
    output_dir = Path(CURRENT_DIR) / "chroma"
    
    if not data_dir.exists():
        logger.warning(f"⚠️  目录不存在: {data_dir}")
        return
    
    # 加载数据
    case_files = list(data_dir.glob("*.json"))
    all_docs = []
    
    for file in case_files:
        raw_docs = load_documents_from_json_rebuild(file)
        
        for item in raw_docs:
            doc = None
            
            if isinstance(item, dict):
                if "text" in item or "content" in item:
                    doc = {
                        "text": item.get("text") or item.get("content", ""),
                        "meta": item.get("meta", {})
                    }
                else:
                    doc = {"text": str(item), "meta": {}}
            elif isinstance(item, str):
                doc = {"text": item, "meta": {}}
            else:
                logger.warning(f"   ⚠️  跳过不支持的数据类型: {type(item)}")
                continue
            
            if doc and doc["text"].strip():
                doc["meta"]["type"] = "case"
                doc["meta"]["source"] = file.name
                all_docs.append(doc)
    
    if not all_docs:
        logger.warning("⚠️  没有找到临床案例文档")
        return
    
    # 使用语义分块策略
    chunker = DynamicChunker()
    config = ChunkConfig(
        strategy=ChunkStrategy.SEMANTIC,
        chunk_size=600,
        chunk_overlap=80,
        min_chunk_size=100,
        max_chunk_size=1500
    )
    
    chunked_docs = chunker.chunk_documents(all_docs, config)
    
    # 转换为 LangChain Document 格式
    lc_docs = [
        Document(page_content=doc["text"], metadata=doc["meta"])
        for doc in chunked_docs
    ]
    
    # 创建向量库
    db_path = str(output_dir / "ClinicalCase_db")
    create_vector_db_with_progress(
        documents=lc_docs,
        embeddings=EMBEDDINGS,
        db_path=db_path,
        collection_name="ClinicalCase",
        batch_size=100
    )
    
    logger.info(f"✅ 临床案例库创建成功: {len(lc_docs)} 个块")


def rebuild_all_databases():
    """使用动态分块重建所有向量库"""
    print("=" * 70)
    logger.info("=" * 60)
    logger.info("🚀 开始使用动态分块策略重建向量库")
    logger.info("=" * 60)
    print()
    
    total_start = time.time()
    
    # 重建各个库
    rebuild_medical_guide_db_dynamic()
    print()
    rebuild_hospital_process_db_dynamic()
    print()
    rebuild_clinical_case_db_dynamic()
    print()
    
    # 高质量问答库使用原有逻辑
    logger.info("🏗️  开始重建：高质量问答库")
    init_high_quality_qa_db()
    print()
    
    # 用户历史库
    logger.info("🏗️  开始重建：用户历史库")
    db_path = os.path.join(CURRENT_DIR, "chroma", "UserHistory_db")
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    placeholder_doc = Document(
        page_content="患者历史记忆库初始化",
        metadata={"type": "placeholder", "patient_id": "system"}
    )
    db = Chroma.from_documents(
        documents=[placeholder_doc],
        embedding=EMBEDDINGS,
        persist_directory=db_path,
        collection_name="UserHistory",
        collection_metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"✅ 用户历史库创建成功（空库）")
    print()
    
    total_elapsed = time.time() - total_start
    
    print("=" * 70)
    logger.info("=" * 60)
    logger.info("✅ 所有向量库重建完成！")
    logger.info(f"⏱️  总耗时: {total_elapsed/60:.1f} 分钟")
    logger.info("=" * 60)
    print("=" * 70)


# =============================================================================
# 主函数：支持命令行参数
# =============================================================================

def main():
    """主函数：支持 rebuild 和update 两种模式，支持动态/固定分块"""
    parser = argparse.ArgumentParser(
        description="向量库管理工具：支持增量更新和完全重建，支持动态/固定分块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 增量更新模式（默认，使用动态分块）
  python create_database_general.py
  python create_database_general.py --mode update --chunker dynamic
  
  # 增量更新模式（使用固定分块）
  python create_database_general.py --chunker fixed
  
  # 完全重建模式（使用动态分块）
  python create_database_general.py --mode rebuild
  
  # 完全重建模式（使用固定分块）
  python create_database_general.py --mode rebuild --chunker fixed
  
  # 只更新/重建特定数据库
  python create_database_general.py --mode update --db guide
  python create_database_general.py --mode rebuild --db case
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['update', 'rebuild'],
        default='update',
        help='运行模式：update=增量更新（默认），rebuild=完全重建'
    )
    
    parser.add_argument(
        '--chunker',
        choices=['dynamic', 'fixed'],
        default='dynamic',
        help='分块策略：dynamic=动态自适应分块（默认），fixed=固定大小分块'
    )
    
    parser.add_argument(
        '--db',
        choices=['all', 'guide', 'process', 'case', 'qa', 'history'],
        default='all',
        help='指定数据库：all=全部（默认），guide=医学指南，process=医院流程，case=临床案例，qa=问答库，history=历史库'
    )
    
    args = parser.parse_args()
    
    # 确定是否使用动态分块
    use_dynamic = (args.chunker == 'dynamic')
    chunker_name = "动态自适应分块" if use_dynamic else "固定大小分块"
    
    if args.mode == 'rebuild':
        print(f"\n📦 运行模式: REBUILD（完全重建 + {chunker_name}）\n")
        
        if use_dynamic:
            # 使用动态分块的重建模式
            if args.db == 'all':
                rebuild_all_databases()
            elif args.db == 'guide':
                rebuild_medical_guide_db_dynamic()
            elif args.db == 'process':
                rebuild_hospital_process_db_dynamic()
            elif args.db == 'case':
                rebuild_clinical_case_db_dynamic()
            elif args.db == 'qa':
                init_high_quality_qa_db()
            elif args.db == 'history':
                db_path = os.path.join(CURRENT_DIR, "chroma", "UserHistory_db")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                placeholder_doc = Document(
                    page_content="患者历史记忆库初始化",
                    metadata={"type": "placeholder", "patient_id": "system"}
                )
                db = Chroma.from_documents(
                    documents=[placeholder_doc],
                    embedding=EMBEDDINGS,
                    persist_directory=db_path,
                    collection_name="UserHistory",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"✅ 用户历史库创建成功（空库）")
        else:
            # 使用固定分块的重建模式（先删除再增量更新）
            print("🔄 使用固定分块进行重建...")
            if args.db in ['all', 'guide']:
                db_path = os.path.join(CURRENT_DIR, "chroma", "MedicalGuide_db")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    print(f"✅ 已删除旧向量库: MedicalGuide_db")
                update_vector_db("MedicalGuide_db", "MedicalGuide_data", use_dynamic_chunker=False)
            
            if args.db in ['all', 'process']:
                db_path = os.path.join(CURRENT_DIR, "chroma", "HospitalProcess_db")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    print(f"✅ 已删除旧向量库: HospitalProcess_db")
                update_vector_db("HospitalProcess_db", "HospitalProcess_data", use_dynamic_chunker=False)
            
            if args.db in ['all', 'case']:
                db_path = os.path.join(CURRENT_DIR, "chroma", "ClinicalCase_db")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    print(f"✅ 已删除旧向量库: ClinicalCase_db")
                update_vector_db("ClinicalCase_db", "ClinicalCase_data", use_dynamic_chunker=False)
            
            if args.db in ['all', 'qa']:
                db_path = os.path.join(CURRENT_DIR, "chroma", "HighQualityQA_db")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                    print(f"✅ 已删除旧向量库: HighQualityQA_db")
                init_high_quality_qa_db()
            
            if args.db in ['all', 'history']:
                db_path = os.path.join(CURRENT_DIR, "chroma", "UserHistory_db")
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                placeholder_doc = Document(
                    page_content="患者历史记忆库初始化",
                    metadata={"type": "placeholder", "patient_id": "system"}
                )
                db = Chroma.from_documents(
                    documents=[placeholder_doc],
                    embedding=EMBEDDINGS,
                    persist_directory=db_path,
                    collection_name="UserHistory",
                    collection_metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"✅ 用户历史库创建成功（空库）")
    
    else:  # update 模式
        print(f"\n📦 运行模式: UPDATE（增量更新 + {chunker_name}）\n")
        
        if args.db in ['all', 'guide']:
            update_vector_db("MedicalGuide_db", "MedicalGuide_data", use_dynamic_chunker=use_dynamic)
        
        if args.db in ['all', 'process']:
            update_vector_db("HospitalProcess_db", "HospitalProcess_data", use_dynamic_chunker=use_dynamic)
        
        if args.db in ['all', 'case']:
            update_vector_db("ClinicalCase_db", "ClinicalCase_data", use_dynamic_chunker=use_dynamic)
        
        if args.db in ['all', 'qa']:
            init_high_quality_qa_db()
        
        if args.db == 'history':
            logger.info("⚠️ 用户历史库由运行时动态更新，无需手动初始化")


if __name__ == "__main__":
    main()