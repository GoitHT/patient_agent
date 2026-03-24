"""按运行批次初始化与写入指标日志（中文输出）。"""

from __future__ import annotations

import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


METRICS_ROOT_DIR = Path("logs/metrics")
METRICS_RUNS_DIR = METRICS_ROOT_DIR / "runs"
_CURRENT_METRICS_LOG_PATHS: dict[str, str] = {}
_WRITE_LOCK = threading.Lock()
_EMBED_LOCK = threading.Lock()
_EMBEDDINGS = None

_RAG_STATS = {
    "latencies_ms": [],
    "recall_queries": 0,
    "recall_hits": 0,
    "grounded_scores": [],
}

_CONSULT_STATS = {
    "total_rounds": [],
    "effective_rounds": [],
    "diag_total": 0,
    "diag_correct": 0,
}


def _reset_runtime_stats() -> None:
    _RAG_STATS["latencies_ms"] = []
    _RAG_STATS["recall_queries"] = 0
    _RAG_STATS["recall_hits"] = 0
    _RAG_STATS["grounded_scores"] = []

    _CONSULT_STATS["total_rounds"] = []
    _CONSULT_STATS["effective_rounds"] = []
    _CONSULT_STATS["diag_total"] = 0
    _CONSULT_STATS["diag_correct"] = 0


def _now_iso() -> str:
    return datetime.now().isoformat()


def _append_lines(file_path: str, lines: list[str]) -> None:
    with _WRITE_LOCK:
        with Path(file_path).open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\n", " ").strip()


def get_current_metrics_log_paths() -> dict[str, str]:
    return dict(_CURRENT_METRICS_LOG_PATHS)


def create_run_metrics_logs() -> dict[str, str]:
    """为当前运行创建三类指标日志文件。"""
    METRICS_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = METRICS_RUNS_DIR / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "rag": run_dir / "rag_knowledge_retrieval.log",
        "performance": run_dir / "system_performance_process_efficiency.log",
        "consultation": run_dir / "consultation_diagnosis_effectiveness.log",
    }

    header = (
        "# 自动生成：本次运行指标日志\n"
        f"# 运行批次: {run_stamp}\n"
        f"# 创建时间: {datetime.now().isoformat()}\n\n"
    )

    sections = {
        "rag": (
            "[指标范围]\n"
            "检索召回率 Recall@k\n"
            "引用一致性 Groundedness\n"
            "检索时延 Retrieval Latency\n\n"
        ),
        "performance": (
            "[指标范围]\n"
            "并发吞吐量 Throughput\n"
            "平均诊疗时长 Average Treatment Duration\n\n"
        ),
        "consultation": (
            "[指标范围]\n"
            "问诊质量 Consultation Quality\n"
            "平均有效问诊轮次 Average Effective Rounds\n"
            "诊断准确率 Diagnosis Accuracy\n\n"
        ),
    }

    for key, path in files.items():
        path.write_text(header + sections[key], encoding="utf-8")

    global _CURRENT_METRICS_LOG_PATHS

    _CURRENT_METRICS_LOG_PATHS = {
        "run_dir": str(run_dir),
        "rag": str(files["rag"]),
        "performance": str(files["performance"]),
        "consultation": str(files["consultation"]),
    }
    _reset_runtime_stats()
    return dict(_CURRENT_METRICS_LOG_PATHS)


def log_retrieval_latency(
    *,
    query: str,
    latency_ms: float,
    result_count: int,
    k: int,
    db_name: str = "",
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
    node_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    rag_log = paths.get("rag")
    if not rag_log:
        return

    _RAG_STATS["latencies_ms"].append(float(latency_ms))

    _append_lines(
        rag_log,
        [
            "[检索时延]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"节点ID={_safe_text(node_id)}",
            f"目标知识库={_safe_text(db_name)}",
            f"k={int(k)}",
            f"返回条数={int(result_count)}",
            f"耗时毫秒={float(latency_ms):.3f}",
            f"查询文本={_safe_text(query)}",
            "---",
        ],
    )


def log_recall_at_k(
    *,
    query: str,
    k: int,
    retrieved_doc_ids: list[str],
    gold_doc_ids: list[str],
    recall_at_k: float,
    hit: bool,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
    node_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    rag_log = paths.get("rag")
    if not rag_log:
        return

    _RAG_STATS["recall_queries"] += 1
    if bool(hit):
        _RAG_STATS["recall_hits"] += 1

    _append_lines(
        rag_log,
        [
            "[检索召回率]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"节点ID={_safe_text(node_id)}",
            f"k={int(k)}",
            f"是否命中={str(bool(hit)).lower()}",
            f"Recall@k={float(recall_at_k):.6f}",
            f"检索文档ID={retrieved_doc_ids}",
            f"标准文档ID={gold_doc_ids}",
            f"查询文本={_safe_text(query)}",
            "---",
        ],
    )


def _load_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    with _EMBED_LOCK:
        if _EMBEDDINGS is not None:
            return _EMBEDDINGS
        model_name = os.getenv("METRICS_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
            )
        except Exception:
            _EMBEDDINGS = None
        return _EMBEDDINGS


def _compute_groundedness_similarity_chargram(answer_text: str, citation_texts: list[str]) -> float:
    """Fallback groundedness score via character 3-gram cosine similarity."""
    answer = _safe_text(answer_text)
    evidence = " ".join(_safe_text(t) for t in citation_texts if _safe_text(t))
    if not answer or not evidence:
        return 0.0

    def to_ngrams(text: str, n: int = 3) -> dict[str, int]:
        compact = "".join(text.split())
        if len(compact) < n:
            return {compact: 1} if compact else {}
        counts: dict[str, int] = {}
        for i in range(len(compact) - n + 1):
            gram = compact[i : i + n]
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    a = to_ngrams(answer)
    b = to_ngrams(evidence)
    if not a or not b:
        return 0.0

    dot = 0.0
    for key, av in a.items():
        dot += av * b.get(key, 0)

    a_norm = math.sqrt(sum(v * v for v in a.values()))
    b_norm = math.sqrt(sum(v * v for v in b.values()))
    if a_norm == 0 or b_norm == 0:
        return 0.0

    score = dot / (a_norm * b_norm)
    return max(0.0, min(1.0, score))


def compute_groundedness_similarity(answer_text: str, citation_texts: list[str]) -> float:
    """Compute groundedness score with embedding cosine similarity; fallback to char-gram."""
    answer = _safe_text(answer_text)
    evidence = " ".join(_safe_text(t) for t in citation_texts if _safe_text(t))
    if not answer or not evidence:
        return 0.0

    emb = _load_embeddings()
    if emb is not None:
        try:
            a_vec = emb.embed_query(answer)
            e_vec = emb.embed_query(evidence)
            if a_vec and e_vec and len(a_vec) == len(e_vec):
                dot = 0.0
                a_norm = 0.0
                e_norm = 0.0
                for i in range(len(a_vec)):
                    av = float(a_vec[i])
                    ev = float(e_vec[i])
                    dot += av * ev
                    a_norm += av * av
                    e_norm += ev * ev
                if a_norm > 0 and e_norm > 0:
                    score = dot / math.sqrt(a_norm * e_norm)
                    return max(0.0, min(1.0, float(score)))
        except Exception:
            pass

    return _compute_groundedness_similarity_chargram(answer_text, citation_texts)


def log_groundedness(
    *,
    answer_text: str,
    citation_doc_ids: list[str],
    semantic_similarity: float,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
    node_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    rag_log = paths.get("rag")
    if not rag_log:
        return

    _RAG_STATS["grounded_scores"].append(float(semantic_similarity))

    _append_lines(
        rag_log,
        [
            "[引用一致性]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"节点ID={_safe_text(node_id)}",
            f"语义相似度={float(semantic_similarity):.6f}",
            f"引用文档ID={citation_doc_ids}",
            f"回答文本={_safe_text(answer_text)[:2000]}",
            "---",
        ],
    )


def log_treatment_duration(
    *,
    visit_start_time: str = "",
    visit_end_time: str = "",
    visit_duration_minutes: float | None = None,
    simulated_duration_minutes: float | None = None,
    wall_time_seconds: float | None = None,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    perf_log = paths.get("performance")
    if not perf_log:
        return

    _append_lines(
        perf_log,
        [
            "[平均诊疗时长-单病例]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"就诊开始时间={_safe_text(visit_start_time)}",
            f"就诊结束时间={_safe_text(visit_end_time)}",
            f"就诊时长分钟={'' if visit_duration_minutes is None else f'{float(visit_duration_minutes):.3f}'}",
            f"模拟时长分钟={'' if simulated_duration_minutes is None else f'{float(simulated_duration_minutes):.3f}'}",
            f"系统运行秒数={'' if wall_time_seconds is None else f'{float(wall_time_seconds):.3f}'}",
            "---",
        ],
    )


def log_treatment_duration_summary(
    *,
    avg_duration_minutes: float,
    patient_count: int,
    run_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    perf_log = paths.get("performance")
    if not perf_log:
        return

    _append_lines(
        perf_log,
        [
            "[平均诊疗时长-汇总]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者数量={int(patient_count)}",
            f"平均诊疗时长分钟={float(avg_duration_minutes):.3f}",
            "---",
        ],
    )


def log_throughput(
    *,
    test_start: str,
    test_end: str,
    total_requests: int,
    completed_requests: int,
    test_duration_seconds: float,
    throughput_req_per_sec: float,
    peak_throughput_req_per_sec: float | None = None,
    run_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    perf_log = paths.get("performance")
    if not perf_log:
        return

    _append_lines(
        perf_log,
        [
            "[并发吞吐量]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"测试开始={_safe_text(test_start)}",
            f"测试结束={_safe_text(test_end)}",
            f"总请求数={int(total_requests)}",
            f"完成请求数={int(completed_requests)}",
            f"测试时长秒={float(test_duration_seconds):.3f}",
            f"平均吞吐量(请求/秒)={float(throughput_req_per_sec):.6f}",
            f"峰值吞吐量(请求/秒)={'' if peak_throughput_req_per_sec is None else f'{float(peak_throughput_req_per_sec):.6f}'}",
            "---",
        ],
    )


def log_consultation_quality(
    *,
    doctor_specificity: float,
    doctor_purposefulness: float,
    doctor_professionalism: float,
    doctor_information_coverage: float,
    patient_relevance: float,
    patient_faithfulness: float,
    patient_information_completeness: float,
    patient_consistency_robustness: float,
    consultation_quality_score: float,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    _append_lines(
        consultation_log,
        [
            "[问诊质量]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"医生-特异性={float(doctor_specificity):.6f}",
            f"医生-目的性={float(doctor_purposefulness):.6f}",
            f"医生-专业度={float(doctor_professionalism):.6f}",
            f"医生-信息覆盖度={float(doctor_information_coverage):.6f}",
            f"患者-相关性={float(patient_relevance):.6f}",
            f"患者-忠实度={float(patient_faithfulness):.6f}",
            f"患者-信息完整性={float(patient_information_completeness):.6f}",
            f"患者-稳健性={float(patient_consistency_robustness):.6f}",
            f"问诊质量综合分={float(consultation_quality_score):.6f}",
            "---",
        ],
    )


def log_effective_rounds(
    *,
    total_rounds: int,
    effective_rounds: int,
    avg_effective_rounds: float,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    _CONSULT_STATS["total_rounds"].append(int(total_rounds))
    _CONSULT_STATS["effective_rounds"].append(int(effective_rounds))

    _append_lines(
        consultation_log,
        [
            "[有效问诊轮次]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"总问诊轮次={int(total_rounds)}",
            f"有效问诊轮次={int(effective_rounds)}",
            f"每病例平均有效轮次={float(avg_effective_rounds):.6f}",
            "---",
        ],
    )


def log_avg_rounds(
    *,
    rounds: int,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    _append_lines(
        consultation_log,
        [
            "[问诊总轮次]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"轮次数={int(rounds)}",
            "---",
        ],
    )


def log_effective_rounds_summary(
    *,
    patient_count: int,
    avg_effective_rounds: float,
    run_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    _append_lines(
        consultation_log,
        [
            "[有效问诊轮次-汇总]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者数量={int(patient_count)}",
            f"平均有效问诊轮次={float(avg_effective_rounds):.6f}",
            "---",
        ],
    )


def log_diagnosis_accuracy(
    *,
    predicted_diagnosis: str,
    ground_truth_diagnosis: str,
    is_correct: bool,
    run_id: str = "",
    patient_id: str = "",
    case_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    _CONSULT_STATS["diag_total"] += 1
    if bool(is_correct):
        _CONSULT_STATS["diag_correct"] += 1

    _append_lines(
        consultation_log,
        [
            "[诊断准确率-单病例]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"患者ID={_safe_text(patient_id)}",
            f"病例ID={_safe_text(case_id)}",
            f"预测诊断={_safe_text(predicted_diagnosis)}",
            f"标准诊断={_safe_text(ground_truth_diagnosis)}",
            f"是否正确={str(bool(is_correct)).lower()}",
            "---",
        ],
    )


def log_diagnosis_accuracy_summary(
    *,
    total_cases: int,
    correct_cases: int,
    accuracy: float,
    run_id: str = "",
) -> None:
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    _append_lines(
        consultation_log,
        [
            "[诊断准确率-汇总]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"病例总数={int(total_cases)}",
            f"正确病例数={int(correct_cases)}",
            f"准确率={float(accuracy):.6f}",
            "---",
        ],
    )


def log_avg_rounds_summary(*, run_id: str = "") -> None:
    """按病例总问诊轮次写入 AvgRounds 汇总。"""
    paths = get_current_metrics_log_paths()
    consultation_log = paths.get("consultation")
    if not consultation_log:
        return

    totals = _CONSULT_STATS.get("total_rounds", [])
    if not totals:
        return

    avg_rounds = sum(float(x) for x in totals) / len(totals)
    _append_lines(
        consultation_log,
        [
            "[问诊总轮次-汇总]",
            f"时间戳={_now_iso()}",
            f"运行ID={_safe_text(run_id)}",
            f"病例数量={len(totals)}",
            f"AvgRounds={avg_rounds:.6f}",
            "---",
        ],
    )


def flush_rag_metric_summaries(*, run_id: str = "") -> None:
    """写入 RAG 运行级汇总：Recall@k、Groundedness、检索时延。"""
    paths = get_current_metrics_log_paths()
    rag_log = paths.get("rag")
    if not rag_log:
        return

    latencies = _RAG_STATS.get("latencies_ms", [])
    if latencies:
        avg_latency = sum(float(x) for x in latencies) / len(latencies)
        _append_lines(
            rag_log,
            [
                "[检索时延-汇总]",
                f"时间戳={_now_iso()}",
                f"运行ID={_safe_text(run_id)}",
                f"查询次数={len(latencies)}",
                f"平均检索时延毫秒={avg_latency:.6f}",
                "---",
            ],
        )

    recall_queries = int(_RAG_STATS.get("recall_queries", 0))
    recall_hits = int(_RAG_STATS.get("recall_hits", 0))
    if recall_queries > 0:
        recall_at_k = recall_hits / recall_queries
        _append_lines(
            rag_log,
            [
                "[检索召回率-汇总]",
                f"时间戳={_now_iso()}",
                f"运行ID={_safe_text(run_id)}",
                f"查询总数={recall_queries}",
                f"命中查询数={recall_hits}",
                f"Recall@k={recall_at_k:.6f}",
                "---",
            ],
        )

    grounded_scores = _RAG_STATS.get("grounded_scores", [])
    if grounded_scores:
        avg_grounded = sum(float(x) for x in grounded_scores) / len(grounded_scores)
        _append_lines(
            rag_log,
            [
                "[引用一致性-汇总]",
                f"时间戳={_now_iso()}",
                f"运行ID={_safe_text(run_id)}",
                f"样本数={len(grounded_scores)}",
                f"平均Groundedness={avg_grounded:.6f}",
                "---",
            ],
        )
