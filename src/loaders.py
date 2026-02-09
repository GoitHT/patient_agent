"""数据加载器模块 - 从本地 Excel 文件加载患者数据"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from utils import get_logger

# 初始化logger
logger = get_logger("hospital_agent.dataset_loader")

# Excel文件路径（与loaders.py在同一目录，即src目录）
DEFAULT_EXCEL_PATH = Path(__file__).parent / "patient_text.xlsx"

# 全局数据集缓存（避免重复加载）
_DATASET_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_ENABLED = True  # 是否启用内存缓存
_CACHE_LOCK = threading.RLock()  # 缓存锁，防止并发加载


def _load_excel_data(excel_path: str | Path = DEFAULT_EXCEL_PATH) -> pd.DataFrame:
    """
    从Excel文件加载患者数据
    
    Args:
        excel_path: Excel文件路径
    
    Returns:
        包含患者数据的DataFrame
    """
    # 转换为绝对路径
    excel_path = Path(excel_path).resolve()
    cache_key = str(excel_path)
    
    with _CACHE_LOCK:
        # 检查内存缓存
        if _CACHE_ENABLED and cache_key in _DATASET_CACHE:
            logger.debug(f"📂 使用缓存的Excel数据")
            return _DATASET_CACHE[cache_key]
        
        # 从文件加载
        if not excel_path.exists():
            raise FileNotFoundError(f"患者数据文件不存在: {excel_path}")
        
        logger.info(f"📂 从Excel文件加载患者数据: {excel_path.name}")
        df = pd.read_excel(excel_path)
        
        # 验证必需的列
        required_columns = ['Patient-SN', 'case_character']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Excel文件缺少必需的列: {missing_columns}")
        
        logger.info(f"✅ 成功加载 {len(df)} 条患者数据")
        
        # 存入缓存
        if _CACHE_ENABLED:
            _DATASET_CACHE[cache_key] = df
        
        return df


def load_diagnosis_arena_case(case_id: int | None = None, excel_path: str | Path = DEFAULT_EXCEL_PATH) -> dict[str, Any]:
    """
    从本地Excel文件加载患者数据
    
    Args:
        case_id: 患者ID（对应Excel中的行索引），None表示随机选择
        excel_path: Excel文件路径
    
    数据格式：
    {
        "id": 0,  # 行索引
        "Patient-SN": "8868522351",  # 患者编号
        "case_character": "患者信息（主诉、病史、家族史等）",
        "treatment_plan": "治疗方案"  # 可选
    }
    
    Returns:
        {
            "full_case": dict,  # 完整病例数据
            "known_case": dict,  # 患者可见部分（仅基本信息）
            "ground_truth": dict  # 标准答案（治疗方案等）
        }
    """
    try:
        # 加载Excel数据
        df = _load_excel_data(excel_path)
        
        # 确定使用的病例索引
        if case_id is not None:
            if case_id < 0 or case_id >= len(df):
                raise ValueError(f"case_id {case_id} 超出范围 [0, {len(df)-1}]")
            actual_case_id = case_id
            logger.debug(f"📚 加载患者数据 - 索引: {case_id}")
        else:
            # 随机选择
            import random
            actual_case_id = random.randint(0, len(df) - 1)
            logger.info(f"🎲 随机选择患者 - 索引: {actual_case_id}")
        
        # 获取该行数据
        row = df.iloc[actual_case_id]
        
        # 构建完整病例数据
        full_case = {
            "id": actual_case_id,
            "Patient-SN": str(row['Patient-SN']),
            "Case Information": str(row['case_character']),  # 患者信息（主诉、病史等）
            "treatment_plan": str(row.get('treatment_plan', '')) if 'treatment_plan' in row else '',
        }
        
        # 不显示加载提示，避免重复
        
        # 患者可见部分（模拟真实场景：患者只知道自己的症状）
        known_case = {
            "id": full_case["id"],
            "Patient-SN": full_case["Patient-SN"],
            "Case Information": full_case["Case Information"],
        }
        
        # 标准答案（用于最终评估，如果有的话）
        ground_truth = {
            "treatment_plan": full_case.get("treatment_plan", ""),
        }
        
        return {
            "full_case": full_case,
            "known_case": known_case,
            "ground_truth": ground_truth,
        }
        
    except FileNotFoundError as e:
        error_msg = f"❌ 错误：找不到患者数据文件 {excel_path}"
        logger.error(error_msg)
        print(f"\n{'='*80}")
        print(error_msg)
        print(f"{'='*80}\n")
        raise
    except Exception as e:
        error_msg = f"❌ 错误：加载患者数据失败 - {e}"
        logger.error(error_msg)
        print(f"\n{'='*80}")
        print(error_msg)
        print(f"{'='*80}\n")
        raise RuntimeError(f"数据加载失败: {e}") from e




def clear_dataset_cache():
    """清除内存中的数据集缓存"""
    global _DATASET_CACHE
    with _CACHE_LOCK:
        _DATASET_CACHE.clear()
    logger.info("🗑️ 数据集内存缓存已清除")


def get_cache_info() -> dict[str, Any]:
    """获取缓存信息"""
    return {
        "enabled": _CACHE_ENABLED,
        "cached_datasets": list(_DATASET_CACHE.keys()),
        "cache_size": len(_DATASET_CACHE),
    }


def _get_dataset_size(excel_path: str | Path | None = None) -> int:
    """
    获取数据集大小（患者数量）
    
    Args:
        excel_path: Excel文件路径，None表示使用默认路径
    
    Returns:
        数据集中的患者数量
    """
    try:
        # 如果传入None，使用默认路径
        if excel_path is None:
            excel_path = DEFAULT_EXCEL_PATH
        df = _load_excel_data(excel_path)
        return len(df)
    except Exception as e:
        logger.warning(f"获取数据集大小失败: {e}")
        return 100  # 默认值


__all__ = ["load_diagnosis_arena_case", "clear_dataset_cache", "get_cache_info", "_get_dataset_size"]
