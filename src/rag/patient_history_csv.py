"""患者对话历史CSV存储模块
负责患者对话的CSV存储和检索，每个患者一个CSV文件
"""
import os
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)


class PatientHistoryCSV:
    """患者对话历史CSV存储管理器
    
    特性：
        - 每个患者一个独立CSV文件
        - 线程安全的读写操作
        - 自动创建目录结构
        - 支持按关键词和时间范围检索
    """
    
    def __init__(self, storage_root: Path | str):
        """初始化CSV存储管理器
        
        Args:
            storage_root: CSV文件存储根目录
        """
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        logger.info(f"✅ 患者对话CSV存储初始化: {self.storage_root}")
    
    def _get_patient_csv_path(self, file_id: str) -> Path:
        """获取患者CSV文件路径
        
        Args:
            file_id: 文件标识符（可以是patient_id或case_id）
            
        Returns:
            患者CSV文件路径
        """
        filename = f"patient_{file_id}.csv"
        return self.storage_root / filename
    
    def store_conversation(
        self,
        patient_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_id: Optional[str] = None
    ) -> bool:
        """存储患者对话记录
        
        Args:
            patient_id: 患者ID（用于记录内容）
            question: 患者问题
            answer: 医生回答
            metadata: 额外元数据（可选）
            file_id: 文件标识符（可选，默认使用patient_id。如果提供则使用它作为文件名）
            
        Returns:
            是否存储成功
        """
        try:
            # 使用file_id命名文件（如果提供），否则使用patient_id
            csv_path = self._get_patient_csv_path(file_id if file_id else patient_id)
            
            # 准备记录
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                "timestamp": timestamp,
                "patient_id": patient_id,
                "question": question,
                "answer": answer,
                "metadata": str(metadata) if metadata else ""
            }
            
            # 线程安全写入
            with self._lock:
                file_exists = csv_path.exists()
                
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    fieldnames = ["timestamp", "patient_id", "question", "answer", "metadata"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    # 如果文件不存在，写入表头
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerow(row)
            
            logger.debug(f"✅ 患者 {patient_id} 对话已存储到CSV")
            return True
            
        except Exception as e:
            logger.error(f"❌ 存储患者 {patient_id} 对话失败: {e}")
            return False
    
    def retrieve_history(
        self,
        patient_id: str,
        query: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_records: int = 10,
        since: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """检索患者对话历史
        
        Args:
            patient_id: 患者ID
            query: 查询文本（可选，用于模糊匹配）
            keywords: 关键词列表（可选，用于精确匹配）
            max_records: 最多返回记录数
            since: 起始时间（可选，格式：YYYY-MM-DD）
            
        Returns:
            对话历史记录列表
        """
        try:
            csv_path = self._get_patient_csv_path(patient_id)
            
            if not csv_path.exists():
                logger.debug(f"患者 {patient_id} 尚无对话历史")
                return []
            
            results = []
            
            with self._lock:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # 时间过滤
                        if since and row["timestamp"] < since:
                            continue
                        
                        # 关键词过滤
                        if keywords:
                            combined_text = f"{row['question']} {row['answer']}".lower()
                            if not any(kw.lower() in combined_text for kw in keywords):
                                continue
                        
                        # 查询文本模糊匹配
                        if query:
                            combined_text = f"{row['question']} {row['answer']}".lower()
                            if query.lower() not in combined_text:
                                continue
                        
                        # 构建结果
                        results.append({
                            "timestamp": row["timestamp"],
                            "patient_id": row["patient_id"],
                            "question": row["question"],
                            "answer": row["answer"],
                            "metadata": row.get("metadata", ""),
                            "text": f"患者问: {row['question']} | 医生答: {row['answer']}"
                        })
            
            # 按时间倒序（最新的在前）并限制数量
            results = sorted(results, key=lambda x: x["timestamp"], reverse=True)
            results = results[:max_records]
            
            logger.debug(f"📜 检索患者 {patient_id} 历史: 返回 {len(results)} 条记录")
            return results
            
        except Exception as e:
            logger.error(f"❌ 检索患者 {patient_id} 历史失败: {e}")
            return []
    
    def retrieve_test_history(
        self,
        patient_id: str,
        test_keywords: List[str],
        max_records: int = 5
    ) -> List[Dict[str, Any]]:
        """检索患者历史检查记录（用于避免重复开单）
        
        Args:
            patient_id: 患者ID
            test_keywords: 检查关键词列表（如 ["CT", "MRI", "血常规"]）
            max_records: 最多返回记录数
            
        Returns:
            包含检查关键词的历史记录
        """
        return self.retrieve_history(
            patient_id=patient_id,
            keywords=test_keywords,
            max_records=max_records
        )
    
    def get_all_patient_ids(self) -> List[str]:
        """获取所有患者ID列表
        
        Returns:
            患者ID列表
        """
        patient_ids = []
        
        try:
            for csv_file in self.storage_root.glob("patient_*.csv"):
                # 从文件名提取患者ID: patient_{id}.csv
                patient_id = csv_file.stem.replace("patient_", "")
                patient_ids.append(patient_id)
            
            return patient_ids
            
        except Exception as e:
            logger.error(f"❌ 获取患者ID列表失败: {e}")
            return []
    
    def get_patient_record_count(self, patient_id: str) -> int:
        """获取患者记录数量
        
        Args:
            patient_id: 患者ID
            
        Returns:
            记录数量
        """
        try:
            csv_path = self._get_patient_csv_path(patient_id)
            
            if not csv_path.exists():
                return 0
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return sum(1 for _ in reader)
                
        except Exception as e:
            logger.error(f"❌ 获取患者 {patient_id} 记录数失败: {e}")
            return 0
    
    def clear_patient_history(self, patient_id: str) -> bool:
        """清除患者历史记录（删除CSV文件）
        
        Args:
            patient_id: 患者ID
            
        Returns:
            是否删除成功
        """
        try:
            csv_path = self._get_patient_csv_path(patient_id)
            
            if csv_path.exists():
                with self._lock:
                    csv_path.unlink()
                logger.info(f"✅ 已删除患者 {patient_id} 的历史记录")
                return True
            else:
                logger.debug(f"患者 {patient_id} 无历史记录")
                return False
                
        except Exception as e:
            logger.error(f"❌ 删除患者 {patient_id} 历史记录失败: {e}")
            return False


# 全局单例实例（延迟初始化）
_global_csv_manager: Optional[PatientHistoryCSV] = None


def get_patient_history_csv(storage_root: Path | str = None) -> PatientHistoryCSV:
    """获取全局患者历史CSV管理器实例
    
    Args:
        storage_root: CSV存储根目录（首次调用时需要提供）
        
    Returns:
        PatientHistoryCSV实例
    """
    global _global_csv_manager
    
    if _global_csv_manager is None:
        if storage_root is None:
            # 默认路径：项目根目录/patient_history_csv
            from pathlib import Path
            default_root = Path(__file__).parent.parent.parent / "patient_history_csv"
            storage_root = default_root
        
        _global_csv_manager = PatientHistoryCSV(storage_root)
    
    return _global_csv_manager
