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
        
        # 验证必需的列（仅支持新版结构化字段）
        required_columns = [
            '姓名', '性别', '年龄', '民族', '职业', '病史陈述者',
            '主诉',
            '现病史_详细描述', '现病史_起病情况', '现病史_病程', '现病史_病情发展',
            '既往史_疾病史', '既往史_手术史', '既往史_过敏史', '既往史_预防接种史', '既往史_外伤史',
            '个人史', '个人史_饮酒史', '个人史_抽烟史', '个人史_月经史',
            '婚育史',
            '家族史_父亲', '家族史_母亲', '家族史_兄弟姐妹', '家族史_疾病',
            '体格检查_生命体征', '体格检查_皮肤黏膜', '体格检查_浅表淋巴结', '体格检查_头颈部',
            '体格检查_心肺血管', '体格检查_腹部', '体格检查_脊柱四肢', '体格检查_神经系统',
            '辅助检查', '初步诊断', '诊断依据', '治疗原则',
            '医患问答参考_问', '医患问答参考_答',
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                "Excel文件缺少新版结构化字段，请使用新数据模板。\n"
                f"缺少字段: {missing_columns}"
            )
        
        logger.info(f"✅ 成功加载 {len(df)} 条患者数据")
        
        # 存入缓存
        if _CACHE_ENABLED:
            _DATASET_CACHE[cache_key] = df
        
        return df


# ---------------------------------------------------------------------------
# 字段分组常量
# ---------------------------------------------------------------------------

# 所有主要数据字段（必须在 Excel 中存在）
_CORE_PATIENT_FIELDS: list[str] = [
    # 基本信息
    "姓名", "性别", "年龄", "民族", "职业", "病史陈述者",
    # 主诉
    "主诉",
    # 现病史
    "现病史_详细描述", "现病史_起病情况", "现病史_病程", "现病史_病情发展",
    # 既往史
    "既往史_疾病史", "既往史_手术史", "既往史_过敏史", "既往史_预防接种史", "既往史_外伤史",
    # 个人史
    "个人史", "个人史_饮酒史", "个人史_抽烟史", "个人史_月经史",
    # 婚育史
    "婚育史",
    # 家族史
    "家族史_父亲", "家族史_母亲", "家族史_兄弟姐妹", "家族史_疾病",
    # 体格检查（医生/护士操作，患者不掌握）
    "体格检查_生命体征", "体格检查_皮肤黏膜", "体格检查_浅表淋巴结", "体格检查_头颈部",
    "体格检查_心肺血管", "体格检查_腹部", "体格检查_脊柱四肢", "体格检查_神经系统",
    # 辅助检查
    "辅助检查",
    # 诊断与治疗
    "初步诊断", "诊断依据", "治疗原则",
    # 医患问答参考
    "医患问答参考_问", "医患问答参考_答",
]

# 新版 Excel 中已有的扩展字段（从 Excel 读取，患者侧不可见）
_FUTURE_USE_FIELDS: list[str] = [
    "标准病历参考_主诉", "标准病历参考_现病史", "标准病历参考_既往史",
    "标准病历参考_体格检查", "标准病历参考_辅助检查", "标准病历参考_诊断结果",
    "高级问题题目", "高级问题答案",
    "考试题目", "考试题目答案",
]

# 患者不可见的医疗数据字段（体格检查 + 辅助检查），供医生/系统参考
_MEDICAL_DATA_FIELDS: list[str] = [
    # 体格检查（所有子项）
    "体格检查_生命体征", "体格检查_皮肤黏膜", "体格检查_浅表淋巴结", "体格检查_头颈部",
    "体格检查_心肺血管", "体格检查_腹部", "体格检查_脊柱四肢", "体格检查_神经系统",
    # 辅助检查
    "辅助检查",
]

# 属于「患者可见」的字段（known_case 包含的字段子集，不包含 体格检查/辅助检查/诊断/治疗）
_KNOWN_CASE_FIELDS: list[str] = [
    "姓名", "性别", "年龄", "民族", "职业", "病史陈述者",
    "主诉",
    "现病史_详细描述", "现病史_起病情况", "现病史_病程", "现病史_病情发展",
    "既往史_疾病史", "既往史_手术史", "既往史_过敏史", "既往史_预防接种史", "既往史_外伤史",
    "个人史", "个人史_饮酒史", "个人史_抽烟史", "个人史_月经史",
    "婚育史",
    "家族史_父亲", "家族史_母亲", "家族史_兄弟姐妹", "家族史_疾病",
]


def _build_case_info_text(case: dict[str, Any]) -> str:
    """
    将结构化病例字段拼合为纯文本摘要，供日志与提示词使用。
    """
    lines: list[str] = []

    # 基本信息行
    basic_parts = []
    for label, key in [("姓名", "姓名"), ("性别", "性别"), ("年龄", "年龄"),
                        ("民族", "民族"), ("职业", "职业")]:
        val = case.get(key, "")
        if val:
            basic_parts.append(f"{label}：{val}")
    if basic_parts:
        lines.append("，".join(basic_parts))

    narrator = case.get("病史陈述者", "")
    if narrator:
        lines.append(f"病史陈述者：{narrator}")

    # 主诉
    chief = case.get("主诉", "")
    if chief:
        lines.append(f"主诉：{chief}")

    # 现病史
    pih_parts: list[str] = []
    for sub_label, sub_key in [
        ("详细描述", "现病史_详细描述"), ("起病情况", "现病史_起病情况"),
        ("病程", "现病史_病程"), ("病情发展", "现病史_病情发展"),
    ]:
        val = case.get(sub_key, "")
        if val:
            pih_parts.append(f"{sub_label}：{val}")
    if pih_parts:
        lines.append("现病史：" + "；".join(pih_parts))

    # 既往史
    ph_parts: list[str] = []
    for sub_label, sub_key in [
        ("疾病史", "既往史_疾病史"), ("手术史", "既往史_手术史"),
        ("过敏史", "既往史_过敏史"), ("预防接种史", "既往史_预防接种史"),
        ("外伤史", "既往史_外伤史"),
    ]:
        val = case.get(sub_key, "")
        if val:
            ph_parts.append(f"{sub_label}：{val}")
    if ph_parts:
        lines.append("既往史：" + "；".join(ph_parts))

    # 个人史
    ph2_parts: list[str] = []
    for sub_label, sub_key in [
        ("个人史", "个人史"), ("饮酒史", "个人史_饮酒史"),
        ("抽烟史", "个人史_抽烟史"), ("月经史", "个人史_月经史"),
    ]:
        val = case.get(sub_key, "")
        if val:
            ph2_parts.append(f"{sub_label}：{val}")
    if ph2_parts:
        lines.append("个人史：" + "；".join(ph2_parts))

    # 婚育史
    marriage = case.get("婚育史", "")
    if marriage:
        lines.append(f"婚育史：{marriage}")

    # 家族史
    fh_parts: list[str] = []
    for sub_label, sub_key in [
        ("父亲", "家族史_父亲"), ("母亲", "家族史_母亲"),
        ("兄弟姐妹", "家族史_兄弟姐妹"), ("疾病", "家族史_疾病"),
    ]:
        val = case.get(sub_key, "")
        if val:
            fh_parts.append(f"{sub_label}：{val}")
    if fh_parts:
        lines.append("家族史：" + "；".join(fh_parts))

    return "\n".join(lines)


def load_diagnosis_arena_case(case_id: int | None = None, excel_path: str | Path = DEFAULT_EXCEL_PATH) -> dict[str, Any]:
    """
    从本地Excel文件加载患者数据（支持新版结构化字段格式）
    
    Args:
        case_id: 患者ID（对应Excel中的行索引），None表示随机选择
        excel_path: Excel文件路径
    
    数据字段（Excel列名）：
        基本信息: 姓名 性别 年龄 民族 职业 病史陈述者
        主诉/现病史: 主诉 现病史_详细描述 现病史_起病情况 现病史_病程 现病史_病情发展
        既往史: 既往史_疾病史 既往史_手术史 既往史_过敏史 既往史_预防接种史 既往史_外伤史
        个人史: 个人史 个人史_饮酒史 个人史_抽烟史 个人史_月经史
        婚育史/家族史: 婚育史 家族史_父亲 家族史_母亲 家族史_兄弟姐妹 家族史_疾病
        体格检查: 体格检查_生命体征 ~ 体格检查_神经系统（共8项，患者不可见，放入 medical_data）
        辅助检查: 辅助检查（患者不可见，放入 medical_data）
        诊断: 初步诊断（放入 ground_truth 用于评估）
        LLM生成字段: 诊断依据 治疗原则 随访计划 治疗方案 治疗药物 医嘱（由系统运行后LLM生成写入数据库）
        医患问答: 医患问答参考_问 医患问答参考_答
        留待以后使用（初始化为空）:
            标准病历参考_* 高级问题题目 高级问题答案 考试题目 考试题目答案
    
    Returns:
        {
            "full_case": dict,     # 完整病例数据（所有字段）
            "known_case": dict,    # 患者可见部分（基本信息+主诉+现病史+既往史+个人史+家族史）
            "medical_data": dict,  # 患者不可见的医疗数据（所有体格检查+辅助检查），供医生/系统参考
            "ground_truth": dict   # 仅含初步诊断，用于后期评估
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
        
        # 安全获取字段值的辅助函数
        def _get(field: str) -> str:
            """从行数据中安全获取字符串值，若列不存在或值为 NaN 则返回空字符串"""
            if field not in df.columns:
                return ""
            val = row.get(field, "")
            if val is None or (isinstance(val, float) and __import__("math").isnan(val)):
                return ""
            return str(val).strip()
        
        # 构建完整病例数据（仅新版结构化字段）
        full_case: dict[str, Any] = {
                "id": actual_case_id,
                # 基本信息
                "姓名": _get("姓名"),
                "性别": _get("性别"),
                "年龄": _get("年龄"),
                "民族": _get("民族"),
                "职业": _get("职业"),
                "病史陈述者": _get("病史陈述者"),
                # 主诉
                "主诉": _get("主诉"),
                # 现病史
                "现病史_详细描述": _get("现病史_详细描述"),
                "现病史_起病情况": _get("现病史_起病情况"),
                "现病史_病程": _get("现病史_病程"),
                "现病史_病情发展": _get("现病史_病情发展"),
                # 既往史
                "既往史_疾病史": _get("既往史_疾病史"),
                "既往史_手术史": _get("既往史_手术史"),
                "既往史_过敏史": _get("既往史_过敏史"),
                "既往史_预防接种史": _get("既往史_预防接种史"),
                "既往史_外伤史": _get("既往史_外伤史"),
                # 个人史
                "个人史": _get("个人史"),
                "个人史_饮酒史": _get("个人史_饮酒史"),
                "个人史_抽烟史": _get("个人史_抽烟史"),
                "个人史_月经史": _get("个人史_月经史"),
                # 婚育史
                "婚育史": _get("婚育史"),
                # 家族史
                "家族史_父亲": _get("家族史_父亲"),
                "家族史_母亲": _get("家族史_母亲"),
                "家族史_兄弟姐妹": _get("家族史_兄弟姐妹"),
                "家族史_疾病": _get("家族史_疾病"),
                # 体格检查
                "体格检查_生命体征": _get("体格检查_生命体征"),
                "体格检查_皮肤黏膜": _get("体格检查_皮肤黏膜"),
                "体格检查_浅表淋巴结": _get("体格检查_浅表淋巴结"),
                "体格检查_头颈部": _get("体格检查_头颈部"),
                "体格检查_心肺血管": _get("体格检查_心肺血管"),
                "体格检查_腹部": _get("体格检查_腹部"),
                "体格检查_脊柱四肢": _get("体格检查_脊柱四肢"),
                "体格检查_神经系统": _get("体格检查_神经系统"),
                # 辅助检查
                "辅助检查": _get("辅助检查"),
                # 诊断与治疗
                "初步诊断": _get("初步诊断"),
                "诊断依据": _get("诊断依据"),
                "治疗原则": _get("治疗原则"),
                # 医患问答参考
                "医患问答参考_问": _get("医患问答参考_问"),
                "医患问答参考_答": _get("医患问答参考_答"),
                # 标准病历参考及教学扩展字段（从 Excel 读取）
                "标准病历参考_主诉": _get("标准病历参考_主诉"),
                "标准病历参考_现病史": _get("标准病历参考_现病史"),
                "标准病历参考_既往史": _get("标准病历参考_既往史"),
                "标准病历参考_体格检查": _get("标准病历参考_体格检查"),
                "标准病历参考_辅助检查": _get("标准病历参考_辅助检查"),
                "标准病历参考_诊断结果": _get("标准病历参考_诊断结果"),
                "高级问题题目": _get("高级问题题目"),
                "高级问题答案": _get("高级问题答案"),
                "考试题目": _get("考试题目"),
                "考试题目答案": _get("考试题目答案"),
        }

        # 患者可见部分（基本信息 + 主诉 + 现病史 + 既往史 + 个人史 + 婚育史 + 家族史）
        known_case: dict[str, Any] = {
                "id": full_case["id"],
                # 患者可见的结构化字段
                **{k: full_case[k] for k in _KNOWN_CASE_FIELDS},
                # 标准病历参考及教学字段属于医生侧评估材料，患者不可见，保持为空
                "标准病历参考_主诉": "",
                "标准病历参考_现病史": "",
                "标准病历参考_既往史": "",
                "标准病历参考_体格检查": "",
                "标准病历参考_辅助检查": "",
                "标准病历参考_诊断结果": "",
                "高级问题题目": "",
                "高级问题答案": "",
                "考试题目": "",
                "考试题目答案": "",
        }

        # 患者不可见的医疗数据：所有体格检查 + 辅助检查（供医生/系统参考，患者智能体不可见）
        medical_data: dict[str, Any] = {
                # 体格检查（全部8项）
                "体格检查_生命体征": full_case["体格检查_生命体征"],
                "体格检查_皮肤黏膜": full_case["体格检查_皮肤黏膜"],
                "体格检查_浅表淋巴结": full_case["体格检查_浅表淋巴结"],
                "体格检查_头颈部": full_case["体格检查_头颈部"],
                "体格检查_心肺血管": full_case["体格检查_心肺血管"],
                "体格检查_腹部": full_case["体格检查_腹部"],
                "体格检查_脊柱四肢": full_case["体格检查_脊柱四肢"],
                "体格检查_神经系统": full_case["体格检查_神经系统"],
                # 辅助检查
                "辅助检查": full_case["辅助检查"],
        }

        # 标准答案（仅含初步诊断，用于后期评估）
        # 诊断依据/治疗原则/随访计划/治疗方案/治疗药物/医嘱由系统运行后LLM生成并写入数据库
        ground_truth: dict[str, Any] = {
                "初步诊断": full_case["初步诊断"],
        }
        
        return {
            "full_case": full_case,
            "known_case": known_case,
            "medical_data": medical_data,
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


__all__ = [
    "load_diagnosis_arena_case",
    "clear_dataset_cache",
    "get_cache_info",
    "_get_dataset_size",
    "_build_case_info_text",
    "_CORE_PATIENT_FIELDS",
    "_FUTURE_USE_FIELDS",
    "_KNOWN_CASE_FIELDS",
    "_MEDICAL_DATA_FIELDS",
]
