"""医院智能体系统 """

from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated
from dotenv import load_dotenv

from config import Config
from core import SystemInitializer
from services.workflow import MultiPatientWorkflow
from display import (
    display_startup_banner,
    display_mode_info,
    display_results_table,
    display_final_statistics,
    display_log_files
)
from utils import get_logger
from logging_utils import should_log

load_dotenv()
logger = get_logger("hospital_agent.main")

app = typer.Typer(
    help="Hospital Agent System - Multi-Agent Mode",
    add_completion=False,
)


@app.command()
def main(
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", help="配置文件路径 (默认: src/config.yaml)"),
    ] = None,
) -> None:
    """医院智能体系统 - 三智能体医疗诊断系统
    
    所有配置请在 config.yaml 中修改
    配置优先级: CLI --config > 环境变量 > config.yaml > 默认值
    """
    # 1. 加载配置
    config = Config.load(config_file=config_file)
    
    # 2. 初始化系统
    initializer = SystemInitializer(config)
    initializer.initialize_logging()
    
    # 3. 显示启动信息
    display_startup_banner(config)
    
    # 4. 检查运行模式
    if not config.mode.multi_patient:
        _show_mode_error()
        return
    
    # 5. 显示模式信息
    num_patients = config.mode.num_patients
    patient_interval = config.mode.patient_interval
    display_mode_info(num_patients, patient_interval)
    
    # 6. 初始化核心组件
    llm = initializer.initialize_llm()
    retriever = initializer.initialize_rag()
    services = initializer.initialize_business_services()
    medical_record_service = initializer.initialize_medical_record(Path("./medical_records"))
    coordinator = initializer.initialize_coordinator(medical_record_service)
    
    # 7. 创建并执行工作流
    workflow = MultiPatientWorkflow(
        config=config,
        coordinator=coordinator,
        retriever=retriever,
        llm=llm,
        services=services,
        medical_record_service=medical_record_service
    )
    
    workflow.register_doctors(num_doctors=3)
    workflow.initialize_processor(num_patients)
    
    # 8. 选择病例并调度患者
    case_ids = workflow.select_patient_cases(num_patients)
    
    if num_patients == 1:
        logger.info("🏥 准备就诊流程...\n")
    else:
        interval_display = f"{patient_interval} 秒" if patient_interval < 60 else f"{patient_interval/60:.1f} 分钟"
        logger.info(f"⏰ 患者将每隔 {interval_display} 进入系统（每个患者启动独立线程，竞争共享资源）\n")
    
    logger.info("="*80)
    workflow.schedule_patients(case_ids, patient_interval)
    
    if num_patients == 1:
        logger.info("\n" + "="*80)
        logger.info("✅ 患者已到达，开始就诊")
        logger.info("="*80 + "\n")
    else:
        logger.info("\n" + "="*80)
        logger.info(f"✅ 所有 {num_patients} 名患者已到达，各自线程正在并发执行")
        logger.info("="*80 + "\n")
    
    # 9. 启动监控并等待完成
    monitor_thread = workflow.start_monitoring()
    
    if num_patients == 1:
        logger.info("\n⏳ 等待患者完成诊断流程...")
    else:
        logger.info("\n⏳ 等待所有患者完成 LangGraph 诊断流程...")
    
    if should_log(2, "main", "monitor"):
        logger.info("💡 提示: 系统每30秒显示一次实时状态（详细模式）\n")
    else:
        logger.info("💡 提示: 系统每2分钟显示一次简要状态（详情见各患者日志文件）\n")
    
    results = workflow.wait_for_completion(num_patients)
    workflow.stop_monitoring(monitor_thread)
    
    # 10. 显示结果
    logger.info("\n" + "="*80)
    logger.info("📊 诊断结果" if num_patients == 1 else "📊 LangGraph 多患者诊断结果")
    logger.info("="*80 + "\n")
    
    display_results_table(results)
    display_final_statistics(results, num_patients)
    display_log_files(len(results))
    
    # 11. 关闭系统
    logger.info("\n" + "="*80)
    logger.info("🔚 关闭系统")
    logger.info("="*80)
    workflow.shutdown()
    
    logger.info("\n✅ 多患者模式执行完毕\n")


def _show_mode_error() -> None:
    """显示模式配置错误信息"""
    logger.error("=" * 80)
    logger.error("⚠️  配置错误：multi_patient 已设为 false")
    logger.error("=" * 80)
    logger.error("系统已统一使用多患者架构（更稳定、功能完整）")
    logger.error("")
    logger.error("💡 单患者模式请设置：")
    logger.error("   mode:")
    logger.error("     multi_patient: true")
    logger.error("     num_patients: 1        # 1个患者 = 单体模式")
    logger.error("     patient_interval: 0    # 立即开始")
    logger.error("")
    logger.error("💡 多患者并发模式请设置：")
    logger.error("   mode:")
    logger.error("     multi_patient: true")
    logger.error("     num_patients: 3        # 3个患者并发")
    logger.error("     patient_interval: 60   # 每60秒进入1个")
    logger.error("=" * 80)


if __name__ == "__main__":
    app()
