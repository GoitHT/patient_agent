"""单病例工作流 - 处理单个病例的诊断流程（已废弃，统一使用多患者模式）"""

from utils import get_logger

logger = get_logger("hospital_agent.workflow")


def process_single_case(*args, **kwargs):
    """处理单个病例（已废弃）
    
    注意：该函数已废弃，请使用多患者模式（num_patients=1）
    """
    logger.error("=" * 80)
    logger.error("⚠️  配置错误：process_single_case 已废弃")
    logger.error("=" * 80)
    logger.error("系统已统一使用多患者架构（更稳定、功能完整）")
    logger.error("")
    logger.error("💡 单患者模式请设置：")
    logger.error("   mode:")
    logger.error("     multi_patient: true")
    logger.error("     num_patients: 1        # 1个患者 = 单体模式")
    logger.error("     patient_interval: 0    # 立即开始")
    logger.error("=" * 80)
    raise NotImplementedError("单病例处理已废弃，请使用多患者模式")
