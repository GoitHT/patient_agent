"""工具函数模块 - 合并了原 utils/ 目录下的所有小文件"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# ============================================================================
# JSON 解析工具 (原 json_parser.py)
# ============================================================================

def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json_with_retry(
    text: str,
    *,
    fallback: Callable[[], dict[str, Any]],
    max_attempts: int = 2,
) -> tuple[dict[str, Any], bool]:
    """Parse JSON with 1 retry; fallback to a rule-based template.

    Returns: (parsed_dict, used_fallback)
    """
    
    logger = logging.getLogger("hospital_agent.utils")
    last_err: Exception | None = None
    candidate = (text or "").strip()
    
    for attempt in range(max_attempts):
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("Top-level JSON is not an object")
            return obj, False
        except Exception as e:  # noqa: BLE001 - fallback is required by spec
            last_err = e
            if attempt == 0:
                # 第一次失败，尝试提取JSON对象
                extracted = _extract_json_object(candidate)
                if extracted and extracted != candidate:
                    candidate = extracted.strip()
                    logger.debug(f"JSON解析失败（尝试{attempt+1}/{max_attempts}），提取JSON对象后重试")
                else:
                    logger.debug(f"JSON解析失败（尝试{attempt+1}/{max_attempts}），无法提取有效JSON")
            else:
                # 最后一次失败，记录详细错误
                logger.warning(f"JSON解析失败（所有{max_attempts}次尝试均失败）: {e}")
                
                # 尝试定位错误位置并显示上下文
                if hasattr(e, 'pos') and e.pos:
                    error_pos = e.pos
                    # 显示错误位置前后50个字符
                    start = max(0, error_pos - 50)
                    end = min(len(candidate), error_pos + 50)
                    context = candidate[start:end]
                    logger.warning(f"错误位置附近: ...{context}...")
                    logger.warning(f"            {' ' * (error_pos - start)}^ 错误在这里")
                else:
                    # 没有位置信息，显示前200字符
                    logger.warning(f"失败的文本预览: {candidate[:200]}...")

    return fallback(), True


# ============================================================================
# 文本处理工具 (原 text.py)
# ============================================================================

DEFAULT_NEGATIONS = ["无", "否认", "未见", "没有", "不伴", "未", "不"]


def contains_positive(text: str, keyword: str, *, negations: list[str] | None = None) -> bool:
    t = text or ""
    kw = keyword or ""
    if not t or not kw:
        return False
    negs = negations or DEFAULT_NEGATIONS

    for m in re.finditer(re.escape(kw), t, flags=re.IGNORECASE):
        start = m.start()
        negated = False
        for neg in negs:
            neg_start = start - len(neg)
            if neg_start >= 0 and t[neg_start:start] == neg:
                negated = True
                break
        if not negated:
            return True
    return False


def contains_any_positive(text: str, keywords: list[str], *, negations: list[str] | None = None) -> bool:
    return any(contains_positive(text, k, negations=negations) for k in keywords)


# ============================================================================
# 安全规则工具 (原 safety.py)
# ============================================================================

ESCALATION_OPTIONS = ["急诊", "住院", "会诊", "转诊"]


def disclaimer_text() -> str:
    return "免责声明：本系统仅用于流程仿真与工程演示，不构成医疗建议；如出现红旗症状请立即线下就医/急诊。"


def apply_safety_rules(state: Any) -> None:
    """Rule-based triage and escalation triggers.

    - Updates state.escalations in-place.
    - Writes reasons into state.preliminary_assessment["safety_reasons"].
    """

    reasons: list[str] = []
    escalations: list[str] = list(getattr(state, "escalations", []) or [])
    cc = getattr(state, "chief_complaint", "") or ""

    if contains_any_positive(cc, ["黑便", "呕血", "大量便血"]):
        escalations.append("急诊")
        reasons.append("提示消化道出血红旗")
    if contains_any_positive(cc, ["意识障碍", "昏迷", "偏瘫", "肢体无力", "言语不清"]):
        escalations.append("急诊")
        reasons.append("提示神经系统红旗/卒中待排")

    test_results = getattr(state, "test_results", []) or []
    if any(r.get("type") == "imaging" and r.get("abnormal") for r in test_results):
        escalations.append("会诊")
        reasons.append("影像异常需进一步评估")
    if any(r.get("type") in {"lab", "endoscopy"} and r.get("abnormal") for r in test_results):
        escalations.append("会诊")
        reasons.append("检查异常需结合临床进一步处理")

    # de-dup while preserving order
    seen: set[str] = set()
    cleaned: list[str] = []
    for e in escalations:
        if e in seen:
            continue
        if e not in ESCALATION_OPTIONS:
            continue
        cleaned.append(e)
        seen.add(e)

    state.escalations = cleaned
    state.preliminary_assessment.setdefault("safety_reasons", [])
    state.preliminary_assessment["safety_reasons"] = reasons


# ============================================================================
# 时间工具 (原 time.py)
# ============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================
# Prompt 加载工具 (原 prompts.py)
# ============================================================================

def prompt_dir() -> Path:
    # src/utils.py -> src/prompts
    return Path(__file__).resolve().parent / "prompts"


def load_prompt(filename: str) -> str:
    path = prompt_dir() / filename
    return path.read_text(encoding="utf-8")


# ============================================================================
# 日志工具 (原 logging.py)
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅在支持ANSI的终端生效）"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # 添加颜色
        if sys.stdout.isatty():  # 仅在终端环境中使用颜色
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        return super().format(record)


def setup_dual_logging(log_file: str = "hospital_agent.log", console_level: int = logging.WARNING) -> None:
    """设置双通道日志系统：详细日志到文件，简洁日志到终端"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    
    # 文件处理器 - 记录所有详细信息
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # 终端处理器 - 只显示重要信息
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = ColoredFormatter(
        "%(message)s",  # 简洁格式，不显示时间和日志级别
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str = "hospital_agent") -> logging.Logger:
    """获取配置好的logger实例"""
    logger = logging.getLogger(name)
    logger.propagate = True  # 使用根logger的配置
    return logger


# ============================================================================
# ID 生成工具
# ============================================================================

def make_run_id(dept: str) -> str:
    """生成运行ID"""
    import time
    raw = f"{dept}:{time.time()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]
