"""数据加载器模块 - 从 HuggingFace 加载 DiagnosisArena 数据集"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

from utils import get_logger

# 初始化logger
logger = get_logger("hospital_agent.dataset_loader")

# 是否启用自动翻译（可通过环境变量控制）
ENABLE_TRANSLATION = os.getenv("ENABLE_DATASET_TRANSLATION", "true").lower() in ("true", "1", "yes")

# 全局数据集缓存（避免重复加载）
_DATASET_CACHE: dict[str, Any] = {}
_CACHE_ENABLED = True  # 是否启用内存缓存
_CACHE_LOCK = threading.RLock()  # 缓存锁，防止并发加载


def _translate_to_chinese(text: str, field_name: str = "") -> str:
    """
    使用LLM将英文医疗文本翻译为中文
    
    Args:
        text: 待翻译文本
        field_name: 字段名称（用于提示）
    
    Returns:
        翻译后的中文文本
    """
    if not text or not text.strip():
        return text
    
    # 快速检测：如果已经主要是中文，跳过翻译
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > len(text) * 0.3:  # 30%以上是中文
        logger.debug(f"  ✓ {field_name} 已为中文，跳过翻译")
        return text
    
    try:
        from services.llm_client import build_llm_client
        
        llm = build_llm_client("deepseek")
        
        system_prompt = "你是一个专业的医疗翻译专家，擅长将英文医疗文本准确翻译为中文。"
        
        user_prompt = (
            f"请将以下医疗文本翻译为中文。要求：\n"
            f"1. 保持医学术语的准确性\n"
            f"2. 保留所有数值、单位、时间等关键信息\n"
            f"3. 语句通顺自然，符合中文医学表达习惯\n"
            f"4. 不要添加任何解释或额外内容\n\n"
            f"【待翻译文本】\n{text}\n\n"
            f"【翻译结果】（仅输出翻译后的中文，不要包含其他内容）"
        )
        
        translated = llm.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # 低温度保证翻译准确性
            max_tokens=1500
        )
        
        # 清理可能的前缀
        translated = translated.strip()
        for prefix in ["翻译结果：", "翻译：", "中文：", "【翻译结果】", "翻译后的中文："]:
            if translated.startswith(prefix):
                translated = translated[len(prefix):].strip()
        
        logger.debug(f"  ✓ {field_name} 翻译完成 ({len(text)} → {len(translated)} 字符)")
        return translated
        
    except Exception as e:
        logger.warning(f"  ⚠️ {field_name} 翻译失败: {e}，使用原文")
        return text


def _translate_case_data(case_data: dict[str, Any]) -> dict[str, Any]:
    """
    将病例数据翻译为中文
    
    Args:
        case_data: 原始病例数据
    
    Returns:
        翻译后的病例数据
    """
    # 检查是否启用翻译
    if not ENABLE_TRANSLATION:
        logger.info("  ℹ️  自动翻译已禁用，使用原始数据")
        return case_data
    
    logger.info("  🌏 开始翻译病例数据为中文...")
    
    translated = {}
    
    # 需要翻译的字段
    text_fields = [
        "Case Information",
        "Physical Examination", 
        "Diagnostic Tests",
        "Final Diagnosis"
    ]
    
    for field in text_fields:
        if field in case_data:
            original_text = case_data[field]
            translated_text = _translate_to_chinese(original_text, field)
            translated[field] = translated_text
        else:
            translated[field] = ""
    
    # Options 需要特殊处理（翻译每个选项）
    if "Options" in case_data and isinstance(case_data["Options"], dict):
        logger.debug("  🔄 翻译诊断选项...")
        translated_options = {}
        for key, value in case_data["Options"].items():
            translated_options[key] = _translate_to_chinese(value, f"Option {key}")
        translated["Options"] = translated_options
    else:
        translated["Options"] = {}
    
    # 其他字段直接复制
    translated["id"] = case_data.get("id", 0)
    translated["Right Option"] = case_data.get("Right Option", "")
    
    logger.info("  ✅ 病例数据翻译完成")
    return translated


def load_diagnosis_arena_case(case_id: int | None = None, use_mock: bool = False, local_cache_dir: str = "./diagnosis_dataset") -> dict[str, Any]:
    """
    从 HuggingFace 加载诊断数据集（支持本地缓存）
    
    Args:
        case_id: 病例ID，None表示随机
        use_mock: 是否直接使用Mock数据（跳过HuggingFace加载）
        local_cache_dir: 本地缓存目录（首次从HF下载后保存到此目录）
    
    数据格式：
    {
        "id": 1,
        "Case Information": "患者基本信息+主诉",
        "Physical Examination": "体格检查结果",
        "Diagnostic Tests": "实验室/影像检查结果",
        "Final Diagnosis": "最终诊断（标准答案）",
        "Options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "Right Option": "A"
    }
    
    Returns:
        {
            "full_case": dict,  # 完整病例（含标准答案）
            "known_case": dict,  # 患者可见部分（仅 Case Information）
            "ground_truth": dict  # 标准答案（Final Diagnosis, Right Option）
        }
    """
    # 如果指定使用Mock数据，直接返回
    if use_mock:
        return _get_mock_case(case_id)
    
    try:
        from datasets import load_dataset
        from pathlib import Path
        
        # 检查本地缓存是否存在
        cache_path = Path(local_cache_dir)
        local_json = cache_path / "dataset.json"
        
        # 构建缓存键（基于文件路径）
        cache_key = str(local_json.absolute())
        
        # 使用锁保护缓存检查和加载过程
        with _CACHE_LOCK:
            # 检查内存缓存
            if _CACHE_ENABLED and cache_key in _DATASET_CACHE:
                dataset = _DATASET_CACHE[cache_key]
                # 静默使用缓存，不输出日志（避免重复日志）
            else:
                # 优先从本地加载
                if local_json.exists():
                    # 二次检查：可能其他线程已经加载了
                    if _CACHE_ENABLED and cache_key in _DATASET_CACHE:
                        dataset = _DATASET_CACHE[cache_key]
                    else:
                        logger.info(f"📂 从本地缓存加载数据集: {local_json}")
                        try:
                            dataset = load_dataset("json", data_files=str(local_json), split="train")
                            logger.info(f"✅ 本地数据集加载成功 (共 {len(dataset)} 条)")
                            
                            # 存入内存缓存
                            if _CACHE_ENABLED:
                                _DATASET_CACHE[cache_key] = dataset
                        except Exception as e:
                            logger.warning(f"⚠️ 本地缓存加载失败: {e}，尝试从 HuggingFace 重新下载")
                            dataset = None
                else:
                    dataset = None
            
            # 如果本地没有，从 HuggingFace 下载
            if dataset is None:
                # 三次检查：可能其他线程刚刚下载完成
                if _CACHE_ENABLED and cache_key in _DATASET_CACHE:
                    dataset = _DATASET_CACHE[cache_key]
                else:
                    logger.info("🌐 从 HuggingFace 下载数据集...")
                    try:
                        dataset = load_dataset("SII-SPIRAL-MED/DiagnosisArena", split="train")
                    except (ValueError, KeyError):
                        # 如果没有train split，尝试加载整个数据集
                        dataset = load_dataset("SII-SPIRAL-MED/DiagnosisArena")
                        # 取第一个split
                        if isinstance(dataset, dict):
                            split_name = list(dataset.keys())[0]
                            dataset = dataset[split_name]
                    
                    # 保存到本地
                    logger.info(f"💾 保存数据集到本地: {local_json}")
                    cache_path.mkdir(parents=True, exist_ok=True)
                    # 使用 orient='records' 格式保存，避免 Arrow 格式问题
                    dataset.to_json(str(local_json), force_ascii=False, orient='records', lines=True)
                    logger.info(f"✅ 数据集已保存 (共 {len(dataset)} 条)")
                    
                    # 存入内存缓存
                    if _CACHE_ENABLED:
                        _DATASET_CACHE[cache_key] = dataset
        
        # 如果指定 case_id，获取特定病例
        if case_id is not None:
            if case_id < 0 or case_id >= len(dataset):
                raise ValueError(f"case_id {case_id} 超出范围 [0, {len(dataset)-1}]")
            case_data = dataset[case_id]
        else:
            # 随机选择一个病例
            import random
            case_data = dataset[random.randint(0, len(dataset) - 1)]
        
        # 解析数据
        full_case = {
            "id": case_data.get("id", 0),
            "Case Information": case_data.get("Case Information", ""),
            "Physical Examination": case_data.get("Physical Examination", ""),
            "Diagnostic Tests": case_data.get("Diagnostic Tests", ""),
            "Final Diagnosis": case_data.get("Final Diagnosis", ""),
            "Options": case_data.get("Options", {}),
            "Right Option": case_data.get("Right Option", ""),
        }
        
        # 翻译为中文
        logger.info(f"📚 加载病例ID: {full_case['id']}")
        full_case = _translate_case_data(full_case)
        
        # 患者可见部分（模拟真实患者只知道自己的症状）
        known_case = {
            "id": full_case["id"],
            "Case Information": full_case["Case Information"],
            # 患者不知道检查结果和诊断
        }
        
        # 标准答案（用于最终评估）
        ground_truth = {
            "Final Diagnosis": full_case["Final Diagnosis"],
            "Options": full_case["Options"],
            "Right Option": full_case["Right Option"],
            "Physical Examination": full_case["Physical Examination"],
            "Diagnostic Tests": full_case["Diagnostic Tests"],
        }
        
        return {
            "full_case": full_case,
            "known_case": known_case,
            "ground_truth": ground_truth,
        }
        
    except ImportError:
        # 如果没有安装 datasets 库，返回示例数据
        warning_msg = "⚠️ 警告：未安装 datasets 库，使用Mock示例数据！真实数据请运行: pip install datasets"
        print(f"\n{'='*80}")
        print(f"❌ {warning_msg}")
        print(f"{'='*80}\n")
        logger.warning(warning_msg)
        return _get_mock_case(case_id)
    except Exception as e:
        # 如果加载失败（网络问题、数据集不存在等），返回示例数据
        error_msg = f"⚠️ 警告：无法从 HuggingFace 加载数据 ({e})，使用Mock示例数据！"
        print(f"\n{'='*80}")
        print(f"❌ {error_msg}")
        print("💡 可能原因:")
        print("   1. 网络连接问题，无法访问 HuggingFace")
        print("   2. 数据集不存在或已移除")
        print("   3. HuggingFace token 配置问题")
        print(f"{'='*80}\n")
        logger.error(f"数据加载失败: {e}")
        return _get_mock_case(case_id)


def _get_mock_case(case_id: int | None = None) -> dict[str, Any]:
    """返回模拟病例数据（当无法访问 HuggingFace 时）"""
    mock_cases = [
        {
            "id": 1,
            "Case Information": "患者，男，45岁，主诉：上腹痛3天，伴反酸、烧心。既往有吸烟史10年。",
            "Physical Examination": "上腹部轻压痛，无反跳痛，墨菲氏征阴性。",
            "Diagnostic Tests": "胃镜：胃窦部糜烂性胃炎，Hp阳性。血常规正常。",
            "Final Diagnosis": "幽门螺杆菌相关性胃炎",
            "Options": {
                "A": "幽门螺杆菌相关性胃炎",
                "B": "胃食管反流病",
                "C": "消化性溃疡",
                "D": "急性胰腺炎"
            },
            "Right Option": "A"
        },
        {
            "id": 2,
            "Case Information": "患者，女，62岁，主诉：突发右侧肢体无力伴言语不清1小时。有高血压病史5年。",
            "Physical Examination": "神志清楚，右侧肢体肌力3级，巴宾斯基征阳性。",
            "Diagnostic Tests": "头颅CT：左侧基底节区低密度影。血压180/110mmHg。",
            "Final Diagnosis": "急性脑梗死",
            "Options": {
                "A": "脑出血",
                "B": "急性脑梗死",
                "C": "短暂性脑缺血发作",
                "D": "脑肿瘤"
            },
            "Right Option": "B"
        },
    ]
    
    idx = case_id if case_id is not None and 0 <= case_id < len(mock_cases) else 0
    case_data = mock_cases[idx]
    
    known_case = {
        "id": case_data["id"],
        "Case Information": case_data["Case Information"],
    }
    
    ground_truth = {
        "Final Diagnosis": case_data["Final Diagnosis"],
        "Options": case_data["Options"],
        "Right Option": case_data["Right Option"],
        "Physical Examination": case_data["Physical Examination"],
        "Diagnostic Tests": case_data["Diagnostic Tests"],
    }
    
    return {
        "full_case": case_data,
        "known_case": known_case,
        "ground_truth": ground_truth,
    }


def clear_dataset_cache():
    """清除内存中的数据集缓存"""
    global _DATASET_CACHE
    _DATASET_CACHE.clear()
    logger.info("🗑️ 数据集内存缓存已清除")


def get_cache_info() -> dict[str, Any]:
    """获取缓存信息"""
    return {
        "enabled": _CACHE_ENABLED,
        "cached_datasets": list(_DATASET_CACHE.keys()),
        "cache_size": len(_DATASET_CACHE),
    }


def _get_dataset_size(local_cache_dir: str = "./diagnosis_dataset") -> int:
    """
    获取数据集大小（病例数量）
    
    Args:
        local_cache_dir: 本地缓存目录
    
    Returns:
        数据集中的病例数量
    """
    try:
        from datasets import load_dataset
        from pathlib import Path
        
        # 检查本地缓存
        cache_path = Path(local_cache_dir)
        local_json = cache_path / "dataset.json"
        cache_key = str(local_json.absolute())
        
        with _CACHE_LOCK:
            # 先检查内存缓存
            if _CACHE_ENABLED and cache_key in _DATASET_CACHE:
                return len(_DATASET_CACHE[cache_key])
            
            if local_json.exists():
                # 从本地加载（使用 jsonlines 格式）
                dataset = load_dataset("json", data_files=str(local_json), split="train")
                # 存入缓存
                if _CACHE_ENABLED:
                    _DATASET_CACHE[cache_key] = dataset
                return len(dataset)
            else:
                # 从 HuggingFace 加载
                try:
                    dataset = load_dataset("SII-SPIRAL-MED/DiagnosisArena", split="train")
                except (ValueError, KeyError):
                    dataset = load_dataset("SII-SPIRAL-MED/DiagnosisArena")
                    if isinstance(dataset, dict):
                        split_name = list(dataset.keys())[0]
                        dataset = dataset[split_name]
                
                # 存入缓存
                if _CACHE_ENABLED:
                    _DATASET_CACHE[cache_key] = dataset
                return len(dataset)
    except Exception as e:
        logger.warning(f"获取数据集大小失败: {e}")
        return 100  # 默认值


__all__ = ["load_diagnosis_arena_case", "clear_dataset_cache", "get_cache_info", "_get_dataset_size"]
