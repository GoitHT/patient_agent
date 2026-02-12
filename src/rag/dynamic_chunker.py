"""动态 Chunk 策略 - 根据内容类型自适应分块
实现智能文档分块，提升检索质量
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ChunkStrategy(Enum):
    """分块策略类型"""
    FIXED = "fixed"  # 固定大小
    SEMANTIC = "semantic"  # 语义分块
    HIERARCHICAL = "hierarchical"  # 层次分块
    ADAPTIVE = "adaptive"  # 自适应分块


@dataclass
class ChunkConfig:
    """分块配置"""
    strategy: ChunkStrategy
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    
    # 语义分块参数
    semantic_threshold: float = 0.7  # 语义相似度阈值
    
    # 层次分块参数
    hierarchy_levels: List[str] = None  # 层次标记（如：章节、段落）
    
    # 自适应参数
    content_density_threshold: float = 0.3  # 内容密度阈值（中文字符/总字符）


class DynamicChunker:
    """动态分块器 - 根据内容自适应选择分块策略
    
    特性：
        - 医学文本识别（指南、病例、问答）
        - 自适应块大小（根据内容密度）
        - 保留文档结构（标题、列表、段落）
        - 智能重叠（避免截断关键信息）
    """
    
    def __init__(
        self,
        default_config: ChunkConfig = None,
        logger: logging.Logger = None
    ):
        """
        Args:
            default_config: 默认分块配置
            logger: 日志记录器
        """
        self.default_config = default_config or ChunkConfig(
            strategy=ChunkStrategy.ADAPTIVE,
            chunk_size=500,
            chunk_overlap=50
        )
        self._logger = logger or logging.getLogger("hospital_agent.chunker")
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        config: ChunkConfig = None
    ) -> List[Dict[str, Any]]:
        """批量分块文档
        
        Args:
            documents: 文档列表，格式: [{"text": "...", "meta": {...}}]
            config: 分块配置（可选）
            
        Returns:
            分块后的文档列表
        """
        config = config or self.default_config
        chunked_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            meta = doc.get("meta", {})
            
            # 识别文档类型
            doc_type = self._identify_document_type(text, meta)
            
            # 选择分块策略
            strategy = self._select_strategy(text, doc_type, config)
            
            # 执行分块
            chunks = self._chunk_text(text, strategy, config)
            
            # 添加元数据
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "meta": {
                        **meta,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "doc_type": doc_type,
                        "chunk_strategy": strategy.value,
                        "chunk_size": len(chunk)
                    }
                })
        
        self._logger.info(
            f"📝 分块完成: {len(documents)} 文档 → {len(chunked_docs)} 块"
        )
        return chunked_docs
    
    def _identify_document_type(self, text: str, meta: Dict[str, Any]) -> str:
        """识别文档类型
        
        Returns:
            文档类型: "guideline", "case", "qa", "dialogue", "general"
        """
        # 从元数据识别
        if "type" in meta:
            return meta["type"]
        
        # 从内容特征识别
        text_lower = text.lower()
        
        # 医学指南特征
        guideline_keywords = ["诊疗指南", "临床路径", "共识", "标准操作程序", "SOP"]
        if any(kw in text for kw in guideline_keywords):
            return "guideline"
        
        # 临床病例特征
        case_keywords = ["主诉", "现病史", "既往史", "体格检查", "诊断", "治疗方案"]
        case_count = sum(1 for kw in case_keywords if kw in text)
        if case_count >= 3:
            return "case"
        
        # 问答对特征
        if "问：" in text or "答：" in text or "Q:" in text or "A:" in text:
            return "qa"
        
        # 对话记录特征
        if "医生：" in text or "患者：" in text:
            return "dialogue"
        
        return "general"
    
    def _select_strategy(
        self,
        text: str,
        doc_type: str,
        config: ChunkConfig
    ) -> ChunkStrategy:
        """根据文档类型和内容选择分块策略"""
        # 如果配置指定了固定策略，直接使用
        if config.strategy != ChunkStrategy.ADAPTIVE:
            return config.strategy
        
        # 自适应选择策略
        if doc_type == "guideline":
            # 医学指南：层次分块（保留章节结构）
            return ChunkStrategy.HIERARCHICAL
        
        elif doc_type == "case":
            # 临床病例：语义分块（按病历段落）
            return ChunkStrategy.SEMANTIC
        
        elif doc_type == "qa":
            # 问答对：固定分块（保持完整性）
            return ChunkStrategy.FIXED
        
        elif doc_type == "dialogue":
            # 对话记录：语义分块（按对话回合）
            return ChunkStrategy.SEMANTIC
        
        else:
            # 一般文本：根据内容密度决定
            density = self._calculate_content_density(text)
            if density > config.content_density_threshold:
                return ChunkStrategy.SEMANTIC
            else:
                return ChunkStrategy.FIXED
    
    def _calculate_content_density(self, text: str) -> float:
        """计算内容密度（中文字符占比）"""
        if not text:
            return 0.0
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        return chinese_chars / total_chars if total_chars > 0 else 0.0
    
    def _chunk_text(
        self,
        text: str,
        strategy: ChunkStrategy,
        config: ChunkConfig
    ) -> List[str]:
        """执行分块"""
        if strategy == ChunkStrategy.FIXED:
            return self._fixed_chunk(text, config)
        
        elif strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_chunk(text, config)
        
        elif strategy == ChunkStrategy.HIERARCHICAL:
            return self._hierarchical_chunk(text, config)
        
        else:
            return self._fixed_chunk(text, config)
    
    def _fixed_chunk(self, text: str, config: ChunkConfig) -> List[str]:
        """固定大小分块（带智能重叠）"""
        chunks = []
        chunk_size = config.chunk_size
        overlap = config.chunk_overlap
        
        # 按句子分割（避免截断句子）
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 添加重叠（取当前块的最后部分）
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _semantic_chunk(self, text: str, config: ChunkConfig) -> List[str]:
        """语义分块（按段落和语义边界）"""
        chunks = []
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前段落加上新段落不超过最大块大小
            if len(current_chunk) + len(para) <= config.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果单个段落超过最大块大小，按句子分割
                if len(para) > config.max_chunk_size:
                    sub_chunks = self._fixed_chunk(para, config)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
        
        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _hierarchical_chunk(self, text: str, config: ChunkConfig) -> List[str]:
        """层次分块（保留标题和章节结构）"""
        chunks = []
        
        # 识别标题（支持多种格式）
        # 格式1: # 标题, ## 标题
        # 格式2: 一、标题, 1. 标题, （一）标题
        # 格式3: 【标题】
        
        lines = text.split('\n')
        current_section = ""
        current_header = ""
        
        for line in lines:
            line = line.strip()
            
            # 检测是否为标题
            is_header = self._is_header(line)
            
            if is_header:
                # 保存之前的章节
                if current_section:
                    chunks.append(f"{current_header}\n{current_section}".strip())
                
                # 开始新章节
                current_header = line
                current_section = ""
            else:
                current_section += line + "\n"
                
                # 如果当前章节过长，分块
                if len(current_section) > config.chunk_size:
                    chunk_text = f"{current_header}\n{current_section}".strip()
                    sub_chunks = self._semantic_chunk(chunk_text, config)
                    chunks.extend(sub_chunks)
                    current_section = ""
        
        # 添加最后一个章节
        if current_section:
            chunks.append(f"{current_header}\n{current_section}".strip())
        
        return chunks if chunks else [text]
    
    def _is_header(self, line: str) -> bool:
        """判断是否为标题行"""
        if not line:
            return False
        
        # Markdown 标题
        if re.match(r'^#{1,6}\s+', line):
            return True
        
        # 中文编号标题
        if re.match(r'^[一二三四五六七八九十]+[、.]', line):
            return True
        
        if re.match(r'^\d+[、.]', line):
            return True
        
        if re.match(r'^[（\(][一二三四五六七八九十]+[）\)]', line):
            return True
        
        # 【标题】格式
        if re.match(r'^【.+】$', line):
            return True
        
        # 全大写或加粗标记
        if line.isupper() or line.startswith('**'):
            return True
        
        return False
    
    def _split_sentences(self, text: str) -> List[str]:
        """智能句子分割（保留标点）"""
        # 中英文句子结束符
        sentence_endings = r'[。！？；.!?;]'
        
        # 分割句子（保留分隔符）
        sentences = re.split(f'({sentence_endings})', text)
        
        # 重组句子（将分隔符附加到句子后）
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        
        return [s for s in result if s.strip()]


# 工具函数
def create_chunker_for_medical_documents() -> DynamicChunker:
    """创建适用于医学文档的分块器"""
    config = ChunkConfig(
        strategy=ChunkStrategy.ADAPTIVE,
        chunk_size=600,  # 医学文本适中大小
        chunk_overlap=100,  # 较大重叠保证上下文
        min_chunk_size=100,
        max_chunk_size=1500,
        semantic_threshold=0.7,
        content_density_threshold=0.5  # 中文内容为主
    )
    
    return DynamicChunker(default_config=config)


__all__ = ["DynamicChunker", "ChunkStrategy", "ChunkConfig", "create_chunker_for_medical_documents"]
