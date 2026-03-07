"""工具函数模块 - 合并了原 utils/ 目录下的所有小文件"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sys
import threading
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


def _clean_json_string(text: str) -> str:
    """清理JSON字符串中的控制字符（换行、制表符），以标准JSON规范处理引号边界。
    
    注意：不尝试修复字符串内部的裸引号，避免因启发式判断失误导致后续JSON全部解析错误。
    结构性问题（缺失逗号、截断）由 _repair_json 负责。
    """
    if not text:
        return text
    
    # 移除BOM标记
    text = text.replace('\ufeff', '')
    
    result = []
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        
        if in_string:
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                if i + 1 < len(text) and text[i + 1] == '\n':
                    continue  # 跳过 \r\n 中的 \r
                result.append('\\n')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)


def _repair_json(text: str) -> str:
    """修复两类结构性JSON错误：
    1. 属性间缺失逗号（LLM常见遗漏）
    2. 字符串/括号未闭合（LLM输出被截断）
    """
    if not text:
        return text
    
    # 修复缺失的逗号：值结束后直接跟着下一个属性键（没有逗号）
    # 匹配：字符串/数字/真假值/null/}] 后跟换行空白再跟 "key":
    text = re.sub(
        r'("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null|[}\]])'  # 值
        r'(\s*\n\s*)'                                                    # 换行
        r'("(?:[^"\\]|\\.)*"\s*:)',                                   # 下一个键
        r'\1,\2\3',
        text
    )
    
    # 修复截断：找出未闭合的字符串和括号，在末尾补全
    stack: list[str] = []
    in_string = False
    escape_next = False
    
    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char in ('{', '['):
                stack.append(char)
            elif char == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif char == ']' and stack and stack[-1] == '[':
                stack.pop()
    
    suffix: list[str] = []
    if in_string:
        suffix.append('"')
    for bracket in reversed(stack):
        suffix.append('}' if bracket == '{' else ']')
    
    return text + ''.join(suffix)


def parse_json_with_retry(
    text: str,
    *,
    fallback: Callable[[], dict[str, Any]],
    max_attempts: int = 3,
) -> tuple[dict[str, Any], bool]:
    """JSON解析，含多次重试策略：提取 → 结构修复 → 兜底。

    重试策略：
      attempt 0: 清理控制字符后直接解析
      attempt 1: 从原始文本提取 {...} 块后重解析
      attempt 2: 对提取块执行结构修复（补逗号、闭合截断）后重解析

    Returns: (parsed_dict, used_fallback)
    """
    logger = logging.getLogger("hospital_agent.utils")
    last_err: Exception | None = None
    raw = (text or "").strip()
    candidate = _clean_json_string(raw)
    
    for attempt in range(max_attempts):
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise ValueError("Top-level JSON is not an object")
            return obj, False
        except Exception as e:  # noqa: BLE001 - fallback is required by spec
            last_err = e
            if attempt < max_attempts - 1:
                # 还有后续重试机会，准备下一个候选
                if attempt == 0:
                    # 第一次失败：提取 {...} 块，从原始文本重新清理
                    extracted = _extract_json_object(raw)
                    if extracted and extracted.strip() != raw:
                        candidate = _clean_json_string(extracted.strip())
                        logger.debug(f"JSON解析失败（尝试{attempt+1}/{max_attempts}），提取JSON对象后重试")
                    else:
                        logger.debug(f"JSON解析失败（尝试{attempt+1}/{max_attempts}），无法提取有效JSON")
                elif attempt == 1:
                    # 第二次失败：结构修复（补缺失逗号 + 闭合截断）
                    base = _extract_json_object(raw) or raw
                    repaired = _repair_json(base.strip())
                    candidate = _clean_json_string(repaired)
                    logger.debug(f"JSON解析失败（尝试{attempt+1}/{max_attempts}），尝试结构修复后重试")
            else:
                # 最后一次失败，记录详细错误
                logger.warning(f"JSON解析失败（所有{max_attempts}次尝试均失败）: {e}")
                if hasattr(e, 'pos') and e.pos:
                    error_pos = e.pos
                    start = max(0, error_pos - 50)
                    end = min(len(candidate), error_pos + 50)
                    context = candidate[start:end]
                    logger.warning(f"错误位置附近: ...{context}...")
                    logger.warning(f"            {' ' * (error_pos - start)}^ 错误在这里")
                else:
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
    # src/utils/common.py -> src/utils -> src -> src/prompts
    return Path(__file__).resolve().parent.parent / "prompts"


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


_setup_logging_lock = threading.Lock()

def setup_console_logging(console_level: int = logging.INFO) -> None:
    """设置终端日志系统：仅输出到控制台"""
    with _setup_logging_lock:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        # 仅移除控制台处理器，保留文件处理器，避免重复输出
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
        
        # 终端处理器 - 显示重要信息
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = ColoredFormatter(
            "%(message)s",  # 简洁格式，不显示时间和日志级别
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        console_handler._hospital_console = True
        
        # 防御性检查：避免意外添加重复 handler（正常情况不会触发）
        for existing in root_logger.handlers:
            if (
                isinstance(existing, logging.StreamHandler)
                and not isinstance(existing, logging.FileHandler)
                and getattr(existing, '_hospital_console', False)
            ):
                return
        root_logger.addHandler(console_handler)


# 保留旧函数名作为别名，以保持向后兼容性
def setup_dual_logging(log_file: str = "hospital_agent.log", console_level: int = logging.WARNING) -> None:
    """已弃用：使用setup_console_logging代替"""
    setup_console_logging(console_level=console_level)


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
