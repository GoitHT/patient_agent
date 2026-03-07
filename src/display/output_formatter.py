"""输出格式化 - 格式化诊断结果和日志输出"""

from pathlib import Path
from typing import List, Dict, Any
from utils import get_logger
from state.schema import BaseState
from display.log_formatter import get_patient_color

logger = get_logger("hospital_agent.output")


def render_summary(state: BaseState) -> str:
    """渲染诊断结果摘要
    
    Args:
        state: 状态对象
    
    Returns:
        格式化的摘要字符串
    """
    lines: List[str] = []
    lines.append(f"科室: {state.dept}  run_id: {state.run_id}")
    lines.append(f"主诉: {state.chief_complaint}")
    if state.ordered_tests:
        lines.append("检查/检验: " + ", ".join([t.get("name", "") for t in state.ordered_tests]))
    if state.test_results:
        abnormal = [r for r in state.test_results if r.get("abnormal")]
        lines.append(f"报告: {len(state.test_results)}项（异常{len(abnormal)}项）")
    lines.append(f"诊断: {state.diagnosis.get('name')}")
    if state.escalations:
        lines.append("升级建议: " + ", ".join(state.escalations))
    return "\n".join(lines)


def display_startup_banner(config: Any) -> None:
    """显示启动横幅
    
    Args:
        config: 配置对象
    """
    logger.info("\n" + "="*80)
    logger.info("🏥 医院智能体系统 - Hospital Agent System")
    logger.info("="*80)
    logger.info(f"⚙️  配置: 问诊{config.agent.max_questions}轮 | LLM={config.llm.backend}")


def display_mode_info(num_patients: int, patient_interval: float) -> None:
    """显示运行模式信息
    
    Args:
        num_patients: 患者数量
        patient_interval: 患者间隔
    """
    if num_patients == 1:
        logger.info("🏥 单患者模式")
    else:
        logger.info(f"🏥 多患者模式: {num_patients}名患者 | 间隔{patient_interval}秒")
    logger.info("="*80 + "\n")


def display_results_table(results: List[Dict[str, Any]]) -> None:
    """显示结果表格
    
    Args:
        results: 结果列表
    """
    lines: List[str] = []
    lines.append("┌" + "─"*78 + "┐")
    lines.append("│ " + "患者ID".ljust(15) + "│ " + "案例".ljust(6) + "│ " + "科室".ljust(18) + "│ " + "状态".ljust(8) + "│ " + "节点数".ljust(8) + "│")
    lines.append("├" + "─"*78 + "┤")
    
    COLOR_RESET = "\033[0m"
    
    for i, result in enumerate(results):
        status = result.get("status")
        patient_id = result.get("patient_id", "未知")
        case_id = result.get("case_id", "N/A")
        color = get_patient_color(i)
        
        if status == "completed":
            dept = result.get("dept", "N/A")
            node_count = result.get("node_count", 0)
            status_icon = f"{color}✅{COLOR_RESET}"
            lines.append(f"│ {color}{patient_id[:15].ljust(15)}{COLOR_RESET}│ {str(case_id)[:6].ljust(6)}│ {dept[:18].ljust(18)}│ {status_icon}     │ {str(node_count)[:8].ljust(8)}│")
        else:
            status_icon = f"{color}❌{COLOR_RESET}"
            lines.append(f"│ {color}{patient_id[:15].ljust(15)}{COLOR_RESET}│ {str(case_id)[:6].ljust(6)}│ {'N/A'[:18].ljust(18)}│ {status_icon}     │ {'N/A'[:8].ljust(8)}│")
    
    lines.append("└" + "─"*78 + "┘")
    logger.info("\n".join(lines) + "\n")


def display_final_statistics(results: List[Dict[str, Any]], num_patients: int) -> None:
    """显示最终统计
    
    Args:
        results: 结果列表
        num_patients: 患者总数
    """
    success_count = sum(1 for r in results if r.get("status") == "completed")
    
    if num_patients == 1:
        status_emoji = "✅" if success_count == 1 else "❌"
        status_text = "成功" if success_count == 1 else "失败"
        logger.info(f"\n{status_emoji} 诊断状态: {status_text}")
    else:
        failed_count = len(results) - success_count
        logger.info(f"\n✅ 成功: {success_count}/{len(results)} | ❌ 失败: {failed_count}/{len(results)}")


def display_log_files(num_results: int) -> None:
    """显示日志文件路径
    
    Args:
        num_results: 结果数量
    """
    patient_logs = sorted(
        Path("logs/patients").glob("*.log"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if patient_logs:
        lines = [f"📋 详细日志: {patient_logs[0].name}"]
        if num_results > 1:
            lines.append(f"   (+{num_results-1} 个其他文件)")
        logger.info("\n".join(lines))
