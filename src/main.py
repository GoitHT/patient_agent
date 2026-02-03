from __future__ import annotations
import json
import threading
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import typer
import logging
from typing_extensions import Annotated
from loaders import load_diagnosis_arena_case
from agents import PatientAgent, DoctorAgent, NurseAgent, LabAgent
from dotenv import load_dotenv
from environment import HospitalWorld, PhysicalState, InteractiveSession
from processing import LangGraphMultiPatientProcessor
from services.medical_record import MedicalRecordService
from services.medical_record_integration import MedicalRecordIntegration
from graphs.router import build_common_graph, build_dept_subgraphs, build_services, default_retriever
from services.llm_client import build_llm_client
from state.schema import BaseState
from utils import make_run_id, get_logger, setup_console_logging
from config import Config
from coordination import HospitalCoordinator
from logging_utils import should_log, get_output_level
from integration import get_coordinator, get_medical_record_service
load_dotenv()
# 初始化logger
logger = get_logger("hospital_agent.main")
# 患者颜色映射（用于终端显示区分）
PATIENT_COLORS = [
    "\033[96m",  # 青色
    "\033[93m",  # 黄色
    "\033[92m",  # 绿色
    "\033[95m",  # 紫色
    "\033[94m",  # 蓝色
    "\033[91m",  # 红色
    "\033[97m",  # 白色
    "\033[90m",  # 灰色
]
COLOR_RESET = "\033[0m"

def get_patient_color(patient_index: int) -> str:
    """获取患者的颜色代码"""
    return PATIENT_COLORS[patient_index % len(PATIENT_COLORS)]

def format_patient_log(patient_id: str, message: str, patient_index: int = 0) -> str:
    """格式化患者日志，添加颜色标识"""
    color = get_patient_color(patient_index)
    return f"{color}[{patient_id}]{COLOR_RESET} {message}"

# 创建 Typer 应用
app = typer.Typer(
    help="Hospital Agent System - Multi-Agent Mode ",
    add_completion=False,
)


def _render_human_summary(state: BaseState) -> str:
    lines: list[str] = []
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


@app.command()
def main(
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", help="配置文件路径 (默认: src/config.yaml)"),
    ] = None,
) -> None:
    """Hospital Agent System - 三智能体医疗诊断系统
    
    所有配置请在 config.yaml 中修改
    配置优先级: CLI --config > 环境变量 > config.yaml > 默认值
    """
    # 加载配置
    config = Config.load(config_file=config_file)
    
    verbose = config.system.verbose
    
    # 设置日志级别：verbose模式显示所有日志（DEBUG），否则显示INFO及以上
  
    console_level = logging.DEBUG if verbose else logging.INFO
    setup_console_logging(console_level=console_level)
    
    # 抑制第三方库的冗余警告日志
    logging.getLogger("urllib3").setLevel(logging.ERROR)  # 抑制SSL重试警告
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 抑制HTTP客户端详细日志
    logging.getLogger("httpcore").setLevel(logging.WARNING)  # 抑制HTTP核心库日志
    
    # 在终端显示简洁的启动信息
    logger.info("\n" + "="*80)
    logger.info("🏥 医院智能体系统 - Hospital Agent System")
    logger.info("="*80)
    
    logger.info("启动系统 ")
    
    # 显示关键配置信息
    logger.info(f"\n⚙️  核心配置:")
    logger.info(f"  • 医生问诊配额: {config.agent.max_questions} 个问题")
    logger.info(f"  • 护士分诊问题: {config.agent.max_triage_questions} 个问题")
    logger.info(f"  • LLM后端: {config.llm.backend}")
    logger.info("")
    
    # 统一使用多患者模式（num_patients=1时等同于单体模式）
    if config.mode.multi_patient:
        # 从config读取参数（CLI参数优先）
        _num_patients = config.mode.num_patients
        _patient_interval = config.mode.patient_interval
        
        # 判断是单患者还是多患者
        if _num_patients == 1:
            logger.info("🏥 启动单患者模式")
        else:
            logger.info(f"🏥 启动多患者并发模式 (共设置{_num_patients}名患者)")
        
        logger.info("="*80)
        logger.info(f"患者数量: {_num_patients}")
        if _num_patients > 1:
            logger.info(f"患者进入间隔: {_patient_interval} 秒")
        logger.info("="*80 + "\n")
        
        # 初始化 LLM
        logger.info(f"🤖 初始化大语言模型 ({config.llm.backend})...")
        try:
            llm_client = build_llm_client(config.llm.backend)
            logger.info("  ✅ 大语言模型初始化成功\n")
        except Exception as e:
            logger.error(f"❌ 大语言模型初始化失败：{e}")
            return
        
        # 初始化 RAG
        if not config.rag.skip_rag:
            logger.info(f"📂 初始化知识库检索器...")
            try:
                retriever = default_retriever(
                    persist_dir=config.rag.persist_dir,
                    collection_name=config.rag.collection_name
                )
                logger.info("  ✅ 知识库检索器初始化成功\n")
            except Exception as e:
                logger.error(f"❌ 知识库检索器初始化失败：{e}")
                return
        else:
            from rag import DummyRetriever
            logger.info("⏭️ 使用虚拟检索器（跳过RAG）\n")
            retriever = DummyRetriever()
        
        # 初始化服务
        logger.info("⚙️ 初始化服务组件...")
        services = build_services()
        logger.info("  ✅ 服务组件初始化完成\n")
        
        # 初始化医疗记录服务
        logger.info("📋 初始化病例库服务...")
        medical_record_service = get_medical_record_service(
            config=config,
            storage_dir=Path("./medical_records")
        )
        logger.info(f"  ✅ 病例库服务初始化完成")
        if hasattr(config, 'database') and config.database.enabled:
            logger.info(f"  🗄️  使用数据库存储: {config.database.connection_string.split('@')[1] if '@' in config.database.connection_string else 'MySQL'}")
            if config.database.backup_to_file:
                logger.info(f"  💾 同时备份到文件: {Path('./medical_records').absolute()}\n")
            else:
                logger.info("")
        else:
            logger.info(f"  📁 病例存储目录: {Path('./medical_records').absolute()}\n")
        
        # 初始化协调器
        logger.info("🏥 初始化医院协调器...")
        coordinator = get_coordinator(
            medical_record_service=medical_record_service
        )
        logger.info("  ✅ 协调器初始化完成\n")
        
        # 【重要】注册医生：必须在处理器初始化之前完成，否则无法预创建DoctorAgent
        logger.info("🏥 注册神经内科医生...")
        
        # 创建3名神经内科医生
        for i in range(3):
            doc_id = f"DOC{i+1:03d}"
            doc_name = f"神经内科医生{i+1}"
            coordinator.register_doctor(doc_id, doc_name, "neurology")
        
        logger.info(f"  ✅ 已注册 3 名神经内科医生")
        logger.info("")
        
        # 初始化 LangGraph 多患者处理器（必须在医生注册之后）
        logger.info("🚀 初始化 LangGraph 多患者处理器...")
        processor = LangGraphMultiPatientProcessor(
            coordinator=coordinator,
            retriever=retriever,
            llm=llm_client,
            services=services,
            medical_record_service=medical_record_service,
            max_questions=config.agent.max_questions,
            max_workers=_num_patients,  # 每个患者一个线程
        )
        logger.info("  ✅ 处理器初始化完成\n")
        
        # 准备患者数据（使用真实数据集病例，随机选择）
        
        # 加载真实数据集以获取病例总数
        logger.info("📚 检查可用的真实病例数量...")
        try:
            from loaders import _get_dataset_size
            max_case_id = _get_dataset_size(None)
            logger.info(f"  ✅ 数据集包含 {max_case_id} 个病例\n")
        except Exception as e:
            logger.warning(f"  ⚠️ 无法获取数据集大小，使用默认范围: {e}")
            max_case_id = 100  # 默认假设有100个病例
        
        # 从可用病例中随机选择
        logger.info(f"🎲 从 {max_case_id} 个病例中随机选择 {_num_patients} 名患者...\n")
        available_case_ids = list(range(max_case_id))
        random.shuffle(available_case_ids)
        selected_case_ids = available_case_ids[:_num_patients]
        
        # 使用 threading.Timer 模拟患者按时间间隔到来，每个患者到来时立即启动独立线程
        if _num_patients == 1:
            # 单患者模式：简化描述，不显示间隔信息
            logger.info(f"🏥 准备就诊流程...\n")
        else:
            # 多患者模式：显示详细的间隔和并发信息
            interval_display = f"{_patient_interval} 秒" if _patient_interval < 60 else f"{_patient_interval/60:.1f} 分钟"
            logger.info(f"⏰ 患者将每隔 {interval_display} 进入系统（每个患者启动独立线程，竞争共享资源）\n")
        logger.info("="*80)
        
        # 定义优先级计算函数
        def calculate_priority_by_symptoms(chief_complaint: str) -> int:
            """根据主诉中的症状严重程度判断优先级（1-10，数字越大越紧急）"""
            # 紧急关键词（高优先级 9-10）
            urgent_keywords = ["胸痛", "胸闷", "呼吸困难", "气促", "昏迷", "意识不清", 
                             "大出血", "出血不止", "休克", "抽搐", "癫痫发作",
                             "窒息", "严重外伤", "骨折", "剧烈头痛"]
            
            # 严重关键词（中高优先级 7-8）
            severe_keywords = ["剧烈疼痛", "持续发热", "高热", "呕血", "黑便", "便血",
                             "咯血", "晕厥", "持续呕吐", "腹痛加重", "无法忍受",
                             "突发", "急性"]
            
            # 一般关键词（中等优先级 5-6）
            moderate_keywords = ["疼痛", "不适", "发热", "咳嗽", "头晕", "乏力",
                               "腹泻", "恶心", "反酸", "烧心"]
            
            complaint_lower = chief_complaint.lower()
            
            # 紧急情况：优先级 9-10
            if any(keyword in complaint_lower for keyword in urgent_keywords):
                return random.randint(9, 10)
            # 严重情况：优先级 7-8
            elif any(keyword in complaint_lower for keyword in severe_keywords):
                return random.randint(7, 8)
            # 一般情况：优先级 5-6
            elif any(keyword in complaint_lower for keyword in moderate_keywords):
                return random.randint(5, 6)
            # 轻微情况：优先级 3-4
            else:
                return random.randint(3, 4)
        
        task_ids = []
        timers = []  # 保存所有定时器，以便等待
        
        def submit_patient_thread(i, case_id, total_patients):
            """在独立线程中提交患者（每个患者到来时立即启动）"""
            patient_id = f"patient_{case_id:03d}"
            
            # 预加载病例数据以获取主诉，用于计算优先级
            try:
                case_bundle = load_diagnosis_arena_case(case_id)
                known_case = case_bundle["known_case"]
                case_info = known_case.get("Case Information", "")
                
                # 记录病例信息以便追踪
                dataset_index = known_case.get('id', 'unknown')
                original_case_id = known_case.get('original_id', 'N/A')
                
                # 提取主诉
                if "主诉：" in case_info:
                    start_idx = case_info.find("主诉：") + 3
                    remaining = case_info[start_idx:]
                    end_markers = ["现病史：", "既往史：", "个人史：", "家族史：", "体格检查：", "\n\n"]
                    end_idx = len(remaining)
                    for marker in end_markers:
                        pos = remaining.find(marker)
                        if pos != -1 and pos < end_idx:
                            end_idx = pos
                    chief_complaint = remaining[:end_idx].strip()
                else:
                    chief_complaint = case_info[:100].strip()
                
                # 根据主诉计算优先级
                priority = calculate_priority_by_symptoms(chief_complaint)
                
            except Exception as e:
                logger.warning(f"⚠️  无法加载病例 {case_id} 的主诉，使用随机优先级: {e}")
                priority = random.randint(5, 7)  # 失败时使用中等优先级
                chief_complaint = "未知"
                dataset_index = case_id  # 使用case_id作为默认值
                original_case_id = "N/A"
            
            # 患者到来 - 使用彩色标识，显示主诉概要
            current_time = time.strftime("%H:%M:%S")
            color = get_patient_color(i)
            
            # 根据优先级显示不同的图标
            priority_icon = "🚨" if priority >= 9 else "⚠️" if priority >= 7 else "📋"
            
            logger.info(f"\n{color}{'='*80}{COLOR_RESET}")
            if total_patients == 1:
                # 单患者模式：简化显示
                logger.info(format_patient_log(patient_id, f"🚶 患者到达医院 [{current_time}]", i))
            else:
                # 多患者模式：显示序号
                logger.info(format_patient_log(patient_id, f"🚶 患者 {i+1}/{total_patients} 到达医院 [{current_time}]", i))
            logger.info(format_patient_log(patient_id, f"{priority_icon} 数据集索引={dataset_index}, 原始ID={original_case_id}, 优先级={priority}/10", i))
            # 显示主诉摘要（前50个字符）
            chief_complaint_short = chief_complaint[:50] + "..." if len(chief_complaint) > 50 else chief_complaint
            logger.info(format_patient_log(patient_id, f"💬 主诉: {chief_complaint_short}", i))
            logger.info(f"{color}{'='*80}{COLOR_RESET}\n")
            
            # 立即提交患者，启动 LangGraph 执行线程
            task_id = processor.submit_patient(
                patient_id=patient_id,
                case_id=case_id,
                dept="neurology",  # 神经内科
                priority=priority
            )
            task_ids.append(task_id)
            
            if total_patients == 1:
                # 单患者模式：简化显示
                logger.info(format_patient_log(patient_id, f"✅ 开始就诊流程", i))
            else:
                # 多患者模式：强调并发竞争
                logger.info(format_patient_log(patient_id, f"✅ 线程已启动，开始竞争资源", i))
        
        # 为每个患者创建定时器，按指定间隔触发
        for i, case_id in enumerate(selected_case_ids):
            delay = i * _patient_interval  # 第 i 个患者在 i*interval 秒后到达
            timer = threading.Timer(
                delay,
                submit_patient_thread,
                args=(i, case_id, _num_patients)
            )
            timer.start()
            timers.append(timer)
        
        # 等待所有定时器触发完成
        for timer in timers:
            timer.join()
        
        if _num_patients == 1:
            # 单患者模式：简化显示
            logger.info("\n" + "="*80)
            logger.info(f"✅ 患者已到达，开始就诊")
            logger.info("="*80 + "\n")
        else:
            # 多患者模式：强调并发
            logger.info("\n" + "="*80)
            logger.info(f"✅ 所有 {len(selected_case_ids)} 名患者已到达，各自线程正在并发执行")
            logger.info("="*80 + "\n")
        
        # 启动状态监控线程
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def monitor_status():
            """定期显示所有患者的状态（仅在详细模式下）"""
            import time
            iteration = 0
            while monitoring_active.is_set():
                time.sleep(60)  # 每60秒检查一次（降低频率）
                iteration += 1
                if not monitoring_active.is_set():
                    break
                    
                active_count = processor.get_active_count()
                if active_count == 0:
                    break
                
                # 仅在详细级别2以上或每2分钟显示一次
                if not should_log(2, "main", "monitor") and iteration % 4 != 0:
                    continue
                    
                logger.info("\n" + "┌" + "─"*78 + "┐")
                logger.info("│" + " "*25 + "\033[1m📊 实时状态监控\033[0m" + " "*28 + "│")
                logger.info("├" + "─"*78 + "┤")
                
                # 显示系统状态
                sys_stats = coordinator.get_system_stats()
                logger.info(f"│  🏥 系统状态: {active_count} 个患者处理中" + " "*(78 - 30 - len(str(active_count))) + "│")
                logger.info(f"│  👨‍⚕️  可用医生: {sys_stats['available_doctors']}/{sys_stats['total_doctors']}" + " "*(78 - 25 - len(str(sys_stats['available_doctors'])) - len(str(sys_stats['total_doctors']))) + "│")
                logger.info(f"│  ✅ 已完成: {sys_stats['total_consultations_completed']} 次" + " "*(78 - 20 - len(str(sys_stats['total_consultations_completed']))) + "│")
                
                # 显示各科室状态（显示所有有活动的科室）
                logger.info("├" + "─"*78 + "┤")
                dept_status = coordinator.get_all_dept_status()
                # 过滤有活动的科室：有等待、有医生忙碌、或有医生在问诊
                active_depts = [d for d in dept_status 
                              if d['waiting_patients'] > 0 
                              or d['busy_doctors'] > 0 
                              or d['consulting_doctors'] > 0]
                
                if active_depts:
                    # 按忙碌程度排序（等待+就诊中的患者数）
                    active_depts.sort(key=lambda x: x['waiting_patients'] + x['busy_doctors'] + x['consulting_doctors'], reverse=True)
                    
                    displayed = 0
                    for dept in active_depts:
                        if displayed >= 8:  # 最多显示8个科室
                            remaining = len(active_depts) - displayed
                            logger.info(f"│  ... 及其他 {remaining} 个科室有活动" + " "*(78 - 24 - len(str(remaining))) + "│")
                            break
                        
                        # 科室名称映射（显示中文）
                        dept_name_map = {
                            "neurology": "神经医学科",
                        }
                        dept_name = dept_name_map.get(dept['dept'], dept['dept'][:15])
                        
                        waiting = dept['waiting_patients']
                        consulting = dept['consulting_doctors']
                        busy = dept['busy_doctors']
                        avail = dept['available_doctors']
                        
                        # 构建状态行
                        status_line = f"│  {dept_name:12s}: 等待={waiting}, 问诊={consulting}, 忙碌={busy}, 空闲={avail}"
                        # 计算需要的填充空格（考虑中文字符宽度）
                        line_width = len(status_line.encode('gbk', errors='ignore'))
                        padding = max(0, 78 - line_width + len("│  "))
                        logger.info(status_line + " "*padding + "│")
                        displayed += 1
                else:
                    logger.info("│  " + " "*30 + "（所有科室空闲）" + " "*29 + "│")
                
                logger.info("└" + "─"*78 + "┘\n")
        
        monitor_thread = threading.Thread(target=monitor_status, daemon=True)
        monitor_thread.start()
        
        # 等待所有任务完成
        if _num_patients == 1:
            logger.info("\n⏳ 等待患者完成诊断流程...")
        else:
            logger.info("\n⏳ 等待所有患者完成 LangGraph 诊断流程...")
        if should_log(2, "main", "monitor"):
            logger.info("💡 提示: 系统每30秒显示一次实时状态（详细模式）")
        else:
            logger.info("💡 提示: 系统每2分钟显示一次简要状态（使用 --output-level 2 查看详细监控）\n")
        # 根据患者数量动态调整超时时间（每个患者预留10分钟）
        timeout = max(600, _num_patients * 600)
        results = processor.wait_all(timeout=timeout)
        
        # 停止监控线程
        monitoring_active.clear()
        monitor_thread.join(timeout=2)
        
        # 打印结果 - 使用表格格式
        logger.info("\n" + "="*80)
        if _num_patients == 1:
            logger.info("📊 诊断结果")
        else:
            logger.info("📊 LangGraph 多患者诊断结果")
        logger.info("="*80 + "\n")
        
        success_count = 0
        failed_count = 0
        
        # 创建结果表格
        logger.info("┌" + "─"*78 + "┐")
        logger.info("│ " + "患者ID".ljust(15) + "│ " + "案例".ljust(6) + "│ " + "科室".ljust(18) + "│ " + "状态".ljust(8) + "│ " + "节点数".ljust(8) + "│")
        logger.info("├" + "─"*78 + "┤")
        
        for i, result in enumerate(results):
            status = result.get("status")
            patient_id = result.get("patient_id", "未知")
            case_id = result.get("case_id", "N/A")
            color = get_patient_color(i)
            
            if status == "completed":
                diagnosis = result.get("diagnosis", "未明确")
                ground_truth = result.get("ground_truth", "N/A")
                dept = result.get("dept", "N/A")
                node_count = result.get("node_count", 0)
                
                # 表格行
                status_icon = f"{color}✅{COLOR_RESET}"
                logger.info(f"│ {color}{patient_id[:15].ljust(15)}{COLOR_RESET}│ {str(case_id)[:6].ljust(6)}│ {dept[:18].ljust(18)}│ {status_icon}     │ {str(node_count)[:8].ljust(8)}│")
                
                success_count += 1
            else:
                error_msg = result.get('error', result.get('reason', '未知错误'))
                status_icon = f"{color}❌{COLOR_RESET}"
                logger.info(f"│ {color}{patient_id[:15].ljust(15)}{COLOR_RESET}│ {str(case_id)[:6].ljust(6)}│ {'N/A'[:18].ljust(18)}│ {status_icon}     │ {'N/A'[:8].ljust(8)}│")
                
                failed_count += 1
        
        logger.info("└" + "─"*78 + "┘\n")
        
        # 最终统计
        logger.info("\n" + "="*80)
        logger.info("📈 最终统计")
        logger.info("="*80)
        if _num_patients == 1:
            # 单患者模式：简化统计
            logger.info(f"✅ 诊断状态: {'成功' if success_count == 1 else '失败'}")
        else:
            # 多患者模式：详细统计
            logger.info(f"✅ 成功: {success_count}/{len(results)}")
            logger.info(f"❌ 失败: {failed_count}/{len(results)}")
            logger.info(f"📊 总计: {len(results)} 名患者")
        
        # 集中输出所有日志文件路径
        logger.info("\n" + "="*80)
        logger.info("📄 输出文件汇总")
        logger.info("="*80)
        
        # 输出每个患者的详细日志
        logger.info("\n📋 患者详细日志:")
        patient_logs = sorted(Path("logs/patients").glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        # 只显示本次运行的日志（最近的N个，N为患者数）
        for log_path in patient_logs[:len(results)]:
            logger.info(f"  • {log_path}")
        
        # 关闭处理器
        logger.info("\n" + "="*80)
        logger.info("🔚 关闭系统")
        logger.info("="*80)
        processor.shutdown()
        
        logger.info("\n✅ 多患者模式执行完毕\n")
        
        return
    
    # ========================================================================
    # 注意：单体模式已统一到多患者架构中
    # 只需在 config.yaml 中设置：
    #   mode:
    #     multi_patient: true
    #     num_patients: 1        # 1个患者 = 单体模式
    #     patient_interval: 0    # 立即开始
    # ========================================================================
    else:
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
        return


def process_single_case(
    case_id: int,
    config_file: Optional[Path] = None,
    llm: Optional[str] = None,
    max_questions: Optional[int] = None,
    llm_reports: bool = False,
    save_trace: Optional[Path] = None,
    persist: Optional[Path] = None,
    collection: Optional[str] = None,
    physical_sim: bool = True,
    interactive: bool = False,
    skip_rag: bool = True,
    verbose: bool = False,
) -> dict:
    """处理单个病例
    
    Args:
        case_id: 病例ID
        其他参数: 与main函数相同
    
    Returns:
        dict: 包含病例处理结果的字典
    """
    # 构造类似 argparse 的参数对象
    from types import SimpleNamespace
    args = SimpleNamespace(
        config=config_file,
        dataset_id=case_id,
        llm=llm,
        max_questions=max_questions,
        llm_reports=llm_reports,
        save_trace=save_trace,
        persist=persist,
        collection=collection,
    )
    
    # 加载配置（优先级: CLI > 环境变量 > config.yaml > 默认值）
    config = Config.load(config_file=args.config, cli_args=args)
    
    # 输出配置摘要
    logger.info(config.summary())

    repo_root = Path(__file__).resolve().parents[1]
    
    # 从数据集加载病例
    logger.info("📚 加载病例数据...")
    
    # 从Excel文件加载患者数据（默认: patient_text.xlsx）
    case_bundle = load_diagnosis_arena_case(case_id)
    known_case = case_bundle["known_case"]
    ground_truth = case_bundle["ground_truth"]
    
    original_id = known_case.get('original_id', 'N/A')
    logger.info(f"  ✅ 数据集索引: {known_case.get('id', 'unknown')} | 原始病例ID: {original_id}")
    
    # 提取原始主诉（仅提供给患者智能体）- 改进提取逻辑，避免在句号处截断
    case_info = known_case.get("Case Information", "")
    if "主诉：" in case_info:
        # 找到主诉开始位置
        start_idx = case_info.find("主诉：") + 3
        remaining = case_info[start_idx:]
        
        # 寻找主诉结束标志（现病史、既往史等关键词，或两个连续换行）
        end_markers = ["现病史：", "既往史：", "个人史：", "家族史：", "体格检查：", "\n\n"]
        end_idx = len(remaining)
        for marker in end_markers:
            pos = remaining.find(marker)
            if pos != -1 and pos < end_idx:
                end_idx = pos
        
        original_chief_complaint = remaining[:end_idx].strip()
    else:
        # 如果没有明确的主诉标记，使用前200字符
        original_chief_complaint = case_info[:200].strip()
    
    logger.info(f"  ✅ 原始主诉（患者）: {original_chief_complaint}")
    if ground_truth.get('treatment_plan'):
        logger.info(f"  ✅ 参考治疗方案: {ground_truth['treatment_plan'][:100]}...")
    logger.info("")

    # 初始化 State（科室待护士分诊后确定）
    # 注意：run_id会在护士分诊后根据实际科室重新生成
    patient_id = "patient_001"  # 定义患者ID，用于物理环境
    
    state = BaseState(
        run_id="temp",  # 临时值，分诊后会更新
        dept="neurology",  # 神经内科
        patient_profile={"case_text": case_info},
        appointment={"channel": "APP", "timeslot": "上午"},
        original_chief_complaint=original_chief_complaint,  # 原始主诉（仅患者可见）
        chief_complaint="",  # 初始为空，医生通过问诊总结得出
        case_data=known_case,
        ground_truth=ground_truth,
        patient_id=patient_id,  # 设置患者ID
        current_location="lobby",  # 初始位置：门诊大厅
        agent_config={  # Agent配置
            "max_questions": config.agent.max_questions,
            "use_agents": True,
        },
    )
    logger.info(f"  ✅ 初始化State（科室待分诊确定，主诉待医生问诊总结）")
    
    # 初始化物理环境（总是启用，作为LangGraph的环境）
    logger.info("\n🏥 初始化物理环境...")
    world = HospitalWorld(start_time=None)  # 使用默认开始时间 8:00
    
    # 将world集成到state
    state.world_context = world
    
    # 添加患者到环境
    world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
    
    # 初始化患者物理状态（从病例信息提取）
    if patient_id in world.physical_states:
        physical_state = world.physical_states[patient_id]
        # 根据主诉设置初始症状严重程度
        # 简单解析：如果主诉包含"重"/"剧烈"等关键词，设置较高严重度
        initial_severity = 5.0  # 默认中度
        if any(keyword in original_chief_complaint for keyword in ["重", "剧烈", "严重", "无法", "难以"]):
            initial_severity = 7.5
        elif any(keyword in original_chief_complaint for keyword in ["轻微", "偶尔", "不适"]):
            initial_severity = 3.0
        
        physical_state.add_symptom("不适", severity=initial_severity)
        logger.info(f"  ✅ 患者初始症状严重度: {initial_severity}/10")
    
    # 同步物理状态到state
    state.sync_physical_state()
    
    logger.info(f"  ✅ 物理环境初始化完成")
    logger.info(f"  ✅ 患者已进入: {world.locations['lobby'].name}")
    logger.info(f"  ✅ 初始时间: {world.current_time.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"  ✅ 患者物理状态: 体力{state.physical_state_snapshot.get('energy_level', 10):.1f}/10")

    logger.info("🤖 初始化系统组件...")
    try:
        logger.info(f"\n🤖 初始化大语言模型客户端 ({config.llm.backend})...")
        llm_client = build_llm_client(config.llm.backend)
        logger.info("  ✅ 大语言模型客户端初始化成功")
    except Exception as e:  # noqa: BLE001
        logger.info(f"❌ 大语言模型初始化失败：{e}")
        logger.info("   DeepSeek模式需设置环境变量：DEEPSEEK_API_KEY")
        logger.error(f"大语言模型初始化失败：{e}")
        raise

    # 知识库检索系统初始化（可选）
    retriever = None
    if not skip_rag:
        try:
            logger.info(f"\n📂 初始化知识库...")
            retriever = default_retriever(persist_dir=config.rag.persist_dir, collection_name=config.rag.collection_name)
            logger.info("  ✅ 知识库初始化完成")
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ 知识库初始化失败：{e}")
            raise
    else:
        from rag import DummyRetriever
        logger.info("\n⏭️ 跳过知识库初始化")
        retriever = DummyRetriever()

    logger.info("\n⚙️ 初始化服务组件...")
    services = build_services()
    logger.info("  ✅ 服务组件初始化完成")
    
    # 初始化病例库服务
    logger.info("\n📋 初始化病例库系统...")
    medical_record_service = MedicalRecordService(storage_dir=Path("./medical_records"))
    medical_record_integration = MedicalRecordIntegration(medical_record_service, world)
    logger.info("  ✅ 病例库服务初始化完成")
    logger.info(f"  ✅ 病例存储目录: {medical_record_service.storage_dir.absolute()}")
    
    # 将病例库集成器添加到state
    state.medical_record_integration = medical_record_integration
    
    # 为患者创建病例
    patient_profile = {
        "name": state.case_data.get("name", "患者"),
        "age": state.case_data.get("age", 0),
        "gender": state.case_data.get("gender", "未知"),
    }
    record_id = medical_record_integration.on_patient_entry(patient_id, patient_profile)
    logger.info(f"  ✅ 病例已创建: {record_id}")
    logger.info(f"  ✅ 患者: {patient_profile['name']}, {patient_profile['age']}岁, {patient_profile['gender']}")
    
    logger.info("  ✅ 组件初始化完成\n")
    
    
    if physical_sim and interactive:
            logger.info("\n💬 启动交互式会话...")
            session = InteractiveSession(world, patient_id, agent_type="patient")
            
            logger.info("\n" + "="*60)
            logger.info("【交互式医院环境】")
            logger.info("="*60)
            logger.info("欢迎来到虚拟医院！你可以使用命令与环境交互。")
            logger.info("输入 '帮助' 或 'help' 查看可用命令")
            logger.info("输入 '退出' 或 'quit' 退出")
            logger.info("="*60 + "\n")
            
            # 显示初始观察
            initial_obs = world.get_observation(patient_id)
            logger.info(session._format_observation(initial_obs))
            logger.info("")
            
            # 交互循环
            while True:
                try:
                    prompt = session.get_prompt()
                    cmd = input(prompt).strip()
                    
                    if not cmd:
                        continue
                    
                    if cmd.lower() in ['quit', 'exit', 'q', '退出']:
                        logger.info("\n👋 感谢使用，再见！")
                        break
                    
                    response = session.execute(cmd)
                    logger.info(response + "\n")
                    
                except KeyboardInterrupt:
                    logger.info("\n\n👋 接收到中断信号，退出交互模式")
                    break
                except Exception as e:
                    logger.info(f"❌ 错误: {e}\n")
            
            logger.info("  ✅ 交互式会话结束")
            return
    
    # 初始化三智能体
    logger.info("🧑 初始化多智能体并执行分诊...")
    logger.info("\n🧑 初始化多智能体...")
    if llm_client is None:
        logger.warning("⚠️  建议使用LLM（--llm deepseek），否则对话质量较差")
    
    # 患者智能体使用原始主诉（从数据集读取的）
    patient_agent = PatientAgent(known_case=state.case_data, llm=llm_client, chief_complaint=original_chief_complaint)
    logger.info("  ✅ 患者Agent初始化完成")
    
    nurse_agent = NurseAgent(llm=llm_client, max_triage_questions=config.agent.max_triage_questions)
    logger.info(f"  ✅ 护士Agent初始化完成（最多可问{config.agent.max_triage_questions}个问题）")
    
    # 【新增】将护士添加到物理环境
    if world:
        nurse_id = "nurse_001"
        world.add_agent(nurse_id, agent_type="nurse", initial_location="triage")
        logger.info(f"  ✅ 护士已就位于: {world.locations['triage'].name}")
    
    # 初始化检验科Agent
    lab_agent = LabAgent(llm=llm_client)
    logger.info("  ✅ 检验科Agent初始化完成")
    
    # 【新增】将检验科添加到物理环境
    if world:
        lab_tech_id = "lab_tech_001"
        world.add_agent(lab_tech_id, agent_type="lab_technician", initial_location="lab")
        logger.info(f"  ✅ 检验科已就位于: {world.locations['lab'].name}")
    
    # ===== 物理环境：护士分诊流程 =====
    logger.info("\n🏥 执行护士分诊 ...")
    logger.info("\n" + "="*60)
    logger.info("👩‍⚕️ 护士分诊台 - 预检分诊")
    logger.info("="*60)
    
    if world:
        # 显示物理环境状态
        logger.info(f"\n{'─'*60}")
        logger.info(f"🏥 【物理环境 - 分诊流程开始】")
        logger.info(f"{'─'*60}")
        start_time = world.current_time.strftime('%H:%M')
        logger.info(f"⏰ 时间: {start_time}")
        
        # 患者应该已经在分诊台（由C2节点移动）
        current_loc = world.get_agent_location(patient_id)
        logger.info(f"📍 患者当前位置: {world.locations[current_loc].name}")
        
        # 确保护士在分诊台
        nurse_id = "nurse_001"
        nurse_loc = world.get_agent_location(nurse_id)
        if nurse_loc:
            logger.info(f"👩‍⚕️  护士在: {world.locations[nurse_loc].name}")
        
        logger.info(f"{'─'*60}\n")
    
    # 患者向护士描述症状
    patient_description = patient_agent.describe_to_nurse()
    logger.info(f"  👤 患者: {patient_description}\n")
    
    # 护士通过多轮对话进行分诊（如信息不足会追问）
    logger.info("  💬 护士评估中...")
    
    # 物理环境：分诊过程消耗时间
    if world:
        # 每次问答消耗约2-3分钟
        base_triage_time = 3
        logger.info(f"  ⏱️ 分诊评估开始，预计消耗 {base_triage_time} 分钟...")
    
    triaged_dept = nurse_agent.triage_with_conversation(patient_agent, patient_description)
    
    # 物理环境：记录分诊时间消耗和物理状态变化
    if world:
        # 计算分诊总时间（基础3分钟 + 每个问题2分钟）
        triage_summary_temp = nurse_agent.get_triage_summary()
        questions_asked = triage_summary_temp.get("questions_asked", 0)
        total_triage_time = 3 + (questions_asked * 2)
        
        # 等待分诊完成（时间推进）
        success, msg = world.wait(patient_id, total_triage_time)
        if success:
            logger.info(f"  ⏱️ {msg}")
        
        # 同步物理状态到state
        state.sync_physical_state()
        
        # 显示分诊后的物理状态
        end_time = world.current_time.strftime('%H:%M')
        logger.info(f"\n{'─'*60}")
        logger.info(f"🏥 【物理环境 - 分诊完成】")
        logger.info(f"{'─'*60}")
        logger.info(f"⏰ 分诊用时: {total_triage_time} 分钟 ({start_time} → {end_time})")
        
        if patient_id in world.physical_states:
            ps = world.physical_states[patient_id]
            logger.info(f"👤 患者状态:")
            logger.info(f"  💪 体力: {ps.energy_level:.1f}/10 {'🟢' if ps.energy_level > 7 else '🟡' if ps.energy_level > 4 else '🔴'}")
            logger.info(f"  😣 疼痛: {ps.pain_level:.1f}/10 {'🟢' if ps.pain_level < 3 else '🟡' if ps.pain_level < 6 else '🔴'}")
        logger.info(f"{'─'*60}\n")
    
    state.dept = triaged_dept
    triage_summary = nurse_agent.get_triage_summary()
    
    # 【修复】设置初步主诉（护士分诊时获取的患者描述）
    # 这是患者的初始描述，医生后续会通过问诊进行深入了解和总结
    state.chief_complaint = patient_description
    
    # 增强分诊记录，包含物理环境信息
    if world:
        triage_summary["physical_info"] = {
            "location": state.current_location,
            "start_time": start_time if world else None,
            "end_time": world.current_time.strftime('%H:%M') if world else None,
            "duration_minutes": total_triage_time if world else 0,
            "energy_level": state.physical_state_snapshot.get("energy_level", 10),
            "pain_level": state.physical_state_snapshot.get("pain_level", 0),
        }
    
    state.agent_interactions["nurse_triage"] = triage_summary
    
    # 【病例库】记录分诊信息
    if state.medical_record_integration:
        state.medical_record_integration.on_triage(state, nurse_id="nurse_001")
        logger.info("  📋 分诊信息已记录到病例库")
    
    # 显示分诊结果
    logger.info(f"\n  ✅ 分诊结果: {triaged_dept}")
    if triage_summary.get("history"):
        last_triage = triage_summary["history"][-1]
        logger.info(f"  📋 分诊理由: {last_triage.get('reason', 'N/A')}")
    
    if triage_summary.get("questions_asked", 0) > 0:
        logger.info(f"  💬 护士追问了 {triage_summary['questions_asked']} 个问题以明确症状")
    
    logger.info("="*80 + "\n")
    
    # 根据分诊科室生成正确的run_id
    run_id = make_run_id(triaged_dept)
    state.run_id = run_id
    logger.info(f"  ✅ 生成run_id: {run_id}")
    
    # 初始化医生Agent（需要知道科室后才能初始化）
    doctor_agent = DoctorAgent(
        dept=state.dept, 
        retriever=retriever, 
        llm=llm_client,
        max_questions=config.agent.max_questions
    )
    # 医生不直接获得主诉，需要通过问诊从患者处获得
    logger.info(f"  ✅ 医生Agent初始化完成 (科室: {state.dept}, max_questions: {config.agent.max_questions})")
    
    # 【新增】将医生添加到物理环境（根据分诊科室）
    if world:
        doctor_id = "doctor_001"
        # 医生在对应科室诊室
        dept_location_map = {
            "neurology": "neuro",  # 神经医学使用神经内科诊室
            "infectious_disease": "internal_medicine",  # 感染科使用内科诊室
            "dermatology_std": "internal_medicine",  # 皮肤性病科使用内科诊室
            "ent_ophthalmology_stomatology": "internal_medicine",  # 五官科使用内科诊室
            "psychiatry": "internal_medicine",  # 精神心理科使用内科诊室
            "rehabilitation_pain": "internal_medicine",  # 康复疼痛科使用内科诊室
            "traditional_chinese_medicine": "internal_medicine",  # 中医科使用内科诊室
        }
        doctor_location = dept_location_map.get(state.dept, "internal_medicine")
        world.add_agent(doctor_id, agent_type="doctor", initial_location=doctor_location)
        
        # 科室中文名映射
        dept_cn_names = {
            "neurology": "神经医学",
        }
        
        dept_cn = dept_cn_names.get(state.dept, state.dept)
        location_cn = world.locations[doctor_location].name
        
        # 如果科室和诊室不一致，说明使用共享诊室
        if state.dept != doctor_location:
            logger.info(f"  ✅ {dept_cn}医生已就位于:  (共享诊室)")
        else:
            logger.info(f"  ✅ {dept_cn}医生已就位于: 诊室")
    
    logger.info("\n🏭 构建专科子图...")
    dept_subgraphs = build_dept_subgraphs(
        retriever=retriever,
        llm=llm_client,
        doctor_agent=doctor_agent,
        patient_agent=patient_agent,
        max_questions=config.agent.max_questions
    )
    logger.info(f"  ✅ 已构建 {len(dept_subgraphs)} 个专科子图: {list(dept_subgraphs.keys())}")
    
    logger.info("\n🕸️ 构建执行图...")
    graph = build_common_graph(
        dept_subgraphs,
        retriever=retriever,
        services=services,
        llm=llm_client,
        llm_reports=config.llm.enable_reports,
        use_agents=True,  # 总是启用Agent模式
        patient_agent=patient_agent,
        doctor_agent=doctor_agent,
        nurse_agent=nurse_agent,
        lab_agent=lab_agent,
        max_questions=config.agent.max_questions,
        world=world,  # 传递world实例，确保节点间共享同一个world对象
    )
    logger.info("  ✅ 执行图构建完成")
    logger.info("\n" + "="*80)
    logger.info("🚀 开始执行门诊流程...")
    logger.info("="*80 + "\n")
    
    # 如果启用物理环境，模拟患者就医流程
    if physical_sim and world:
        logger.info(f"  📍 患者当前位置: {world.locations[world.agents[patient_id]].name}")
        logger.info(f"  ⏰ 当前时间: {world.current_time.strftime('%H:%M')}")
        
        # 在终端也显示初始物理状态（使用醒目的格式）
        logger.info("\n" + "╔"+"═"*78+"╗")
        logger.info("║" + " "*25 + "\033[1m🏥 物理环境初始化情况\033[0m" + " "*26 + "║")
        logger.info("╠"+"═"*78+"╣")
        logger.info(f"║  🕐 当前时间: {world.current_time.strftime('%H:%M')}" + " "*(66) + "║")
        logger.info("║" + " "*(78) + "║")
        logger.info("║  \033[1m👥 智能体初始布局\033[0m" + " "*(58) + "║")
        logger.info(f"║     👤 患者 (patient_001):  {world.locations[world.agents.get(patient_id, 'lobby')].name}" + " "*(35) + "║")
        
        # 显示医护人员位置
        nurse_id = "nurse_001"
        if nurse_id in world.agents:
            logger.info(f"║     👩‍⚕️  护士 (nurse_001):    {world.locations[world.agents[nurse_id]].name}" + " "*(39) + "║")
        
        doctor_id = "doctor_001"
        if doctor_id in world.agents:
            # 显示医生科室和位置（从患者视角显示科室诊室）
            dept_cn_names = {
                "neurology": "神经医学",
            }
            dept_cn = dept_cn_names.get(state.dept, state.dept)
            actual_location = world.locations[world.agents[doctor_id]].name
            
            # 检查是否是共享诊室（科室诊室名与实际位置不同）
            dept_location_map = {
                "neurology": "neuro",
            }
            
            if dept_location_map.get(state.dept) != state.dept:
                # 共享诊室，显示科室诊室名称并标注
                doctor_info = f"{dept_cn}医生 (doctor_001): {dept_cn}诊室（共享）"
            else:
                # 独立诊室
                doctor_info = f"{dept_cn}医生 (doctor_001): {actual_location}"
            
            # 计算空格填充（确保对齐）
            spaces_needed = 51 - len(doctor_info)
            logger.info(f"║     👨‍⚕️  {doctor_info}" + " "*(spaces_needed) + "║")
        
        lab_tech_id = "lab_tech_001"
        if lab_tech_id in world.agents:
            logger.info(f"║     🔬 检验科 (lab_tech_001): {world.locations[world.agents[lab_tech_id]].name}" + " "*(35) + "║")
        
        logger.info("║" + " "*(78) + "║")
        logger.info("║  \033[1m💊 患者初始状态\033[0m" + " "*(60) + "║")
        logger.info(f"║     ⚡ 初始体力: {state.physical_state_snapshot.get('energy_level', 10.0):.1f}/10" + " "*(52) + "║")
        
        symptoms = state.physical_state_snapshot.get('symptom_severity', {})
        if symptoms:
            logger.info("║     🩹 初始症状:" + " "*(59) + "║")
            for s_name, s_sev in list(symptoms.items())[:3]:
                symptom_line = f"        • {s_name}: {s_sev:.1f}/10"
                padding = 73 - len(symptom_line.encode('utf-8', errors='replace'))
                logger.info(f"║  {symptom_line}" + " "*max(0, padding) + "║")
        
        logger.info("╚"+"═"*78+"╝\n")
    
    logger.info("📋 执行诊断流程...")
    logger.info("-" * 80)
    
    # 使用stream模式实时显示节点执行进度
    node_count = 0
    out = None
    
    # 节点名称映射
    node_display_names = {
        "C1": "C1 开始门诊流程",
        "C2": "C2 预约挂号",
        "C3": "C3 签到候诊",
        "C4": "C4 叫号入诊室",
        "C5": "C5 准备问诊",
        "C6": "C6 专科问诊",
        "C7": "C7 判断是否需要辅助检查",
        "C8": "C8 开具检查单并解释准备",
        "C9": "C9 缴费与预约检查",
        "C10a": "C10a 获取检查报告",
        "C10b": "C10b LLM增强报告",
        "C11": "C11 复诊查看报告",
        "C12": "C12 综合分析明确诊断",
        "C13": "C13 制定治疗方案",
        "C14": "C14 文书记录",
        "C15": "C15 患者宣教与随访",
        "C16": "C16 结束门诊",
    }
    
    try:
        for chunk in graph.stream(state):
            node_count += 1
            
            # chunk 是一个字典，通常只有一个键（节点名）
            if isinstance(chunk, dict) and len(chunk) > 0:
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]
                
                # 节点内部已经有完整的logger.info输出（包含物理环境信息）
                # main.py不再重复显示，避免信息冗余和分裂
                # 物理环境状态的显示完全由各个节点自己管理
                
                # 保存最后的状态
                out = node_data
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ 诊断流程完成 (共执行 {node_count} 个节点)")
        logger.info("=" * 80 + "\n")
        
        # 显示最终物理状态
        if state.world_context and state.physical_state_snapshot:
            logger.info("\n" + "═"*80)
            logger.info("🏥 \033[1m物理环境模拟 - 就医全程回顾\033[0m")
            logger.info("═"*80)
            
            world = state.world_context
            patient_id = state.patient_id
            current_time = world.current_time
            
            # ===== 1. 时间统计 =====
            # 从movement_history获取第一次移动的时间作为开始时间
            movement_history = world.get_movement_history(patient_id)
            if movement_history:
                first_move = movement_history[0]
                start_time_str = first_move['time']
                # 解析时间字符串 "HH:MM" 或 "HH:MM:SS"
                time_parts = start_time_str.split(':')
                start_hour, start_min = int(time_parts[0]), int(time_parts[1])
                total_minutes = (current_time.hour - start_hour) * 60 + (current_time.minute - start_min)
            else:
                # 默认从08:00开始
                start_time_str = "08:00"
                total_minutes = (current_time.hour - 8) * 60 + current_time.minute
            
            logger.info(f"\n⏱️  \033[1m就医时长\033[0m: {total_minutes} 分钟 ({start_time_str} → {current_time.strftime('%H:%M')})")
            
            # ===== 2. 空间轨迹（从movement_history动态生成）=====
            if movement_history:
                # 提取完整移动路径
                path = []
                for move in movement_history:
                    if 'from' in move and move['from'] and move['from'] not in path:
                        path.append(world.get_location_name(move['from']))
                    if 'to' in move and move['to']:
                        path.append(world.get_location_name(move['to']))
                
                # 去除连续重复
                unique_path = []
                for loc in path:
                    if not unique_path or unique_path[-1] != loc:
                        unique_path.append(loc)
                
                logger.info(f"📍 \033[1m空间轨迹\033[0m: {' → '.join(unique_path)}")
                # 最终位置优先使用科室显示名称
                final_loc_name = state.dept_display_name if state.dept_display_name else world.get_location_name(state.current_location)
                logger.info(f"🏥 最终位置: {final_loc_name}")
                logger.info(f"🚶 移动次数: {len(movement_history)} 次")
            else:
                final_loc_name = state.dept_display_name if state.dept_display_name else world.get_location_name(state.current_location)
                logger.info(f"📍 \033[1m空间轨迹\033[0m: {final_loc_name}")
            
            # ===== 3. 设备使用统计 =====
            device_usage = world.get_device_usage_log(patient_id)
            if device_usage:
                logger.info(f"\n🔧 \033[1m设备使用记录\033[0m: {len(device_usage)} 次")
                # 统计各设备使用次数
                device_counts = {}
                for usage in device_usage:
                    device = usage.get('device', 'unknown')
                    device_counts[device] = device_counts.get(device, 0) + 1
                
                for device, count in device_counts.items():
                    logger.info(f"   • {device}: {count} 次")
            
            # ===== 4. 等待时间统计（从 event_log 读取）=====
            total_wait_time = 0
            wait_breakdown = {}
            
            # 从 event_log 中筛选 wait 事件
            if hasattr(world, 'event_log'):
                for event in world.event_log:
                    if event.get('type') == 'wait':
                        details = event.get('details', {})
                        # 检查是否是当前患者的事件
                        if details.get('agent_id') == patient_id:
                            duration = details.get('duration_minutes', 0)
                            location = details.get('location', 'unknown')
                            loc_name = world.get_location_name(location)
                            total_wait_time += duration
                            wait_breakdown[loc_name] = wait_breakdown.get(loc_name, 0) + duration
            
            if total_wait_time > 0:
                logger.info(f"\n⏳ \033[1m等待时间统计\033[0m: 总计 {total_wait_time} 分钟")
                for loc, duration in sorted(wait_breakdown.items(), key=lambda x: x[1], reverse=True):
                    pct = (duration / total_wait_time * 100) if total_wait_time > 0 else 0
                    logger.info(f"   • {loc}: {duration} 分钟 ({pct:.0f}%)")
            
            # ===== 5. 患者状态变化 =====
            snapshot = state.physical_state_snapshot
            initial_energy = 10.0
            final_energy = snapshot.get('energy_level', 10)
            energy_change = final_energy - initial_energy
            energy_icon = "📉" if energy_change < 0 else ("📈" if energy_change > 0 else "➡️")
            
            logger.info(f"\n👤 \033[1m患者健康状态变化\033[0m:")
            logger.info(f"   ⚡ 体力值: 10.0 → {final_energy:.1f} ({energy_change:+.1f}) {energy_icon}")
            logger.info(f"   🩹 疼痛值: 0.0 → {snapshot.get('pain_level', 0):.1f}/10")
            logger.info(f"   🧠 意识状态: {snapshot.get('consciousness_level', 'alert')}")
            
            symptoms = snapshot.get('symptom_severity', {})
            if symptoms:
                logger.info(f"\n   🩺 症状演化:")
                for name, severity in symptoms.items():
                    # 假设初始严重度为5.0
                    change = severity - 5.0
                    trend_icon = "⬆️" if change > 0 else ("⬇️" if change < 0 else "➡️")
                    status = "恶化" if change > 0 else ("改善" if change < 0 else "稳定")
                    logger.info(f"      • {name}: 5.0 → {severity:.1f} ({status}) {trend_icon}")
            
            vital_signs = snapshot.get('vital_signs', {})
            if vital_signs:
                logger.info(f"\n   📊 生命体征监测:")
                vital_display = [
                    ("heart_rate", "心率", "次/分"),
                    ("blood_pressure_systolic", "收缩压", "mmHg"),
                    ("temperature", "体温", "℃"),
                    ("oxygen_saturation", "血氧", "%")
                ]
                for key, name, unit in vital_display:
                    if key in vital_signs:
                        logger.info(f"      • {name}: {vital_signs[key]:.1f} {unit}")
            
            # ===== 新增：医护人员工作状态总结 =====
            logger.info(f"\n\n👥 \033[1m医护人员工作状态\033[0m:")
            logger.info("─"*80)
            
            # 护士状态
            if "nurse_001" in state.world_context.physical_states:
                nurse_state = state.world_context.physical_states["nurse_001"]
                logger.info(f"\n👩‍⚕️  \033[1m护士 (nurse_001)\033[0m")
                logger.info(f"   📍 位置: {state.world_context.locations[state.world_context.agents['nurse_001']].name}")
                logger.info(f"   ⚡ 体力: {nurse_state.energy_level:.1f}/10")
                logger.info(f"   📊 工作负荷: {nurse_state.work_load:.1f}/10")
                logger.info(f"   ⏱️  连续工作: {nurse_state.consecutive_work_minutes} 分钟")
                logger.info(f"   👥 今日服务: {nurse_state.patients_served_today} 人")
                logger.info(f"   🎯 工作效率: {nurse_state.get_work_efficiency()*100:.0f}%")
            
            # 医生状态
            if "doctor_001" in state.world_context.physical_states:
                doctor_state = state.world_context.physical_states["doctor_001"]
                global_qa = state.node_qa_counts.get("global_total", 0)
                max_q = config.agent.max_questions
                logger.info(f"\n👨‍⚕️  \033[1m医生 (doctor_001)\033[0m")
                logger.info(f"   📍 位置: {state.world_context.locations[state.world_context.agents['doctor_001']].name}")
                logger.info(f"   ⚡ 体力: {doctor_state.energy_level:.1f}/10")
                logger.info(f"   📊 工作负荷: {doctor_state.work_load:.1f}/10")
                logger.info(f"   ⏱️  连续工作: {doctor_state.consecutive_work_minutes} 分钟")
                logger.info(f"   👥 今日诊疗: {doctor_state.patients_served_today} 人")
                logger.info(f"   🎯 工作效率: {doctor_state.get_work_efficiency()*100:.0f}%")
                logger.info(f"   💬 问诊: {global_qa}/{max_q}（医生可主动结束）")
            
            # 检验科状态
            if "lab_tech_001" in state.world_context.physical_states:
                lab_state = state.world_context.physical_states["lab_tech_001"]
                logger.info(f"\n🔬 \033[1m检验科 (lab_tech_001)\033[0m")
                logger.info(f"   📍 位置: {state.world_context.locations[state.world_context.agents['lab_tech_001']].name}")
                logger.info(f"   ⚡ 体力: {lab_state.energy_level:.1f}/10")
                logger.info(f"   📊 工作负荷: {lab_state.work_load:.1f}/10")
                logger.info(f"   ⏱️  连续工作: {lab_state.consecutive_work_minutes} 分钟")
                logger.info(f"   🧪 今日检验: {lab_state.patients_served_today} 项")
                logger.info(f"   🎯 工作效率: {lab_state.get_work_efficiency()*100:.0f}%")
            
            # 统计信息
            tests_count = len(state.ordered_tests) if state.ordered_tests else 0
            if tests_count > 0:
                global_qa = state.node_qa_counts.get("global_total", 0)
                max_q = config.agent.max_questions
                logger.info(f"\n\n📈 \033[1m就医统计\033[0m:")
                logger.info(f"   🔬 完成检查: {tests_count} 项")
                logger.info(f"   💬 问诊轮数: {len(state.agent_interactions.get('doctor_patient_qa', []))} 轮（配额 {max_q}，医生可主动结束）")
            
                # 生成并输出完整时间线
                if state.world_context:
                    timeline_report = state.world_context.generate_timeline_report(patient_id)
                    logger.info("\n🕐 完整物理环境时间线:")
                    for entry in timeline_report:
                        logger.info(f"  [{entry['time']}] {entry['type']}: {entry['details']}")
            
        
    except Exception as e:
        logger.info(f"\n❌ 流程执行出错: {e}")
        logger.error(f"流程执行出错: {e}", exc_info=True)
        raise
    
    logger.info("\n" + "="*80)
    logger.info("✅ 门诊流程执行完成")
    logger.info("="*80)
    
    final_state = BaseState.model_validate(out)

    logger.info("\n📄 生成结果总结...")
    summary = {
        "run_id": final_state.run_id,
        "dept": final_state.dept,
        "chief_complaint": final_state.chief_complaint,
        "need_aux_tests": final_state.need_aux_tests,
        "ordered_tests": final_state.ordered_tests,
        "test_prep": final_state.test_prep,
        "test_results": final_state.test_results,
        "diagnosis": final_state.diagnosis,
        "treatment_plan": final_state.treatment_plan,
        "followup_plan": final_state.followup_plan,
        "escalations": final_state.escalations,
    }
    
    # 添加对话记录和评估
    summary["agent_interactions"] = final_state.agent_interactions
    summary["ground_truth"] = final_state.ground_truth

    # 将完整结果输出到日志文件
    logger.info("\n📄 完整诊断结果（JSON格式）:")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # 终端只显示简洁摘要
    logger.info("\n" + "="*80)
    logger.info("✅ 门诊流程执行完成")
    logger.info("="*80 + "\n")
    
    logger.info("📊 诊断结果摘要")
    logger.info("-" * 80)
    summary_lines = _render_human_summary(final_state)
    for line in summary_lines.split('\n'):
        if line.strip():
            logger.info(f"  {line}")
    logger.info("-" * 80)
    
    # 详细显示检验报告
    if final_state.test_results:
        logger.info("\n" + "="*80)
        logger.info("🔬 检验报告详细内容")
        logger.info("="*80)
        
        for idx, test_result in enumerate(final_state.test_results, 1):
            test_name = test_result.get("test_name", "未知检查")
            test_type = test_result.get("type", "未知类型")
            is_abnormal = test_result.get("abnormal", False)
            result_text = test_result.get("result", "")
            summary_text = test_result.get("summary", "")
            source = test_result.get("source", "unknown")
            
            # 状态标识
            status_icon = "⚠️ 异常" if is_abnormal else "✅ 正常"
            
            # 检查类型图标
            type_icons = {
                "lab": "🧪",
                "imaging": "📷",
                "functional": "📊",
                "endoscopy": "🔍"
            }
            type_icon = type_icons.get(test_type, "📋")
            
            logger.info(f"\n【报告 {idx}】{type_icon} {test_name} - {status_icon}")
            logger.info(f"检查类型: {test_type}")
            logger.info(f"数据来源: {source}")
            logger.info("-" * 80)
            
            # 显示完整的检查结果
            logger.info("📄 完整报告:")
            if result_text:
                # 将结果按行显示，保持格式
                for line in result_text.split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.info("   (无报告内容)")
            
            # 显示摘要
            if summary_text:
                logger.info(f"\n💡 报告摘要:")
                logger.info(f"   {summary_text}")
            
            # 显示关键发现（如果有）
            key_findings = test_result.get("key_findings", [])
            if key_findings:
                logger.info(f"\n🎯 关键发现:")
                for finding in key_findings:
                    logger.info(f"   • {finding}")
            
            # 显示临床意义（如果有）
            clinical_sig = test_result.get("clinical_significance", "")
            if clinical_sig:
                logger.info(f"\n🏥 临床意义:")
                logger.info(f"   {clinical_sig}")
            
            logger.info("-" * 80)
        
        logger.info("="*80 + "\n")
    
    # 显示评估结果
    if final_state.agent_interactions.get("evaluation"):
        eval_data = final_state.agent_interactions["evaluation"]
        logger.info("\n" + "="*80)
        logger.info("【诊断评估】")
        logger.info("="*80)
        logger.info(f"📋 医生诊断: {eval_data['doctor_diagnosis']}")
        
        # 显示多维度评估结果（如果有）
        if eval_data.get('multi_dim_scores'):
            scores = eval_data['multi_dim_scores']
            total_score = eval_data.get('total_score', 0)
            grade = eval_data.get('grade', 'F')
            
            logger.info(f"\n📊 多维度评分:")
            logger.info(f"   🎯 核心疾病识别: {scores['core_disease']['score']}/20")
            logger.info(f"      {scores['core_disease']['comment']}")
            logger.info(f"   🔗 症状关联: {scores['symptom_match']['score']}/20")
            logger.info(f"      {scores['symptom_match']['comment']}")
            logger.info(f"   🔀 鉴别诊断: {scores['differential']['score']}/20")
            logger.info(f"      {scores['differential']['comment']}")
            logger.info(f"   💊 治疗方向: {scores['treatment_direction']['score']}/20")
            logger.info(f"      {scores['treatment_direction']['comment']}")
            logger.info(f"   🎲 精确度: {scores['precision']['score']}/20")
            logger.info(f"      {scores['precision']['comment']}")
            
            # 评级符号
            grade_emoji = {
                'A': '🏆',
                'B': '✅',
                'C': '⚠️',
                'D': '❌',
                'F': '🚫'
            }.get(grade, '❓')
            
            logger.info(f"\n{grade_emoji} 综合评分: {total_score}/100 (评级: {grade})")
            logger.info(f"💭 评价: {scores.get('summary', '')}")
        
        # 显示诊断过程统计
        logger.info(f"\n📈 诊断过程:")
        logger.info(f"   💬 问诊轮数: {eval_data['questions_asked']} 轮")
        logger.info(f"   🔬 开单数量: {eval_data['tests_ordered']} 项")
        logger.info(f"   📏 评估方法: {eval_data.get('evaluation_method', '未知')}")
        logger.info("="*80)
    
    # 显示诊断质量信息
    diagnosis = final_state.diagnosis
    
    # 使用LLM生成智能诊断评估报告
    if llm and final_state.ground_truth:
        logger.info("\n" + "="*80)
        logger.info("【智能诊断质量分析】")
        logger.info("="*80)
        logger.info("\n🤖 生成智能诊断评估报告...")
        
        try:
            # 准备评估数据
            qa_count = len(final_state.agent_interactions.get('doctor_patient_qa', []))
            eval_data_for_ai = {
                "医生诊断": diagnosis.get("name", ""),
                "标准答案": final_state.ground_truth.get("Final Diagnosis", ""),
                "问诊轮数": qa_count,
                "问诊配额": config.agent.max_questions,
                "开单数量": len(final_state.ordered_tests) if final_state.ordered_tests else 0,
                "诊断推理": diagnosis.get("reasoning", "")[:300],
                "确定程度": diagnosis.get("uncertainty", ""),
            }
            
            system_prompt = "你是一位资深的临床医学专家和医学教育者，擅长评估诊断质量并提供建设性反馈。"
            
            # 构建问诊过程摘要
            qa_summary = ""
            if qa_count > 0:
                qa_list = final_state.agent_interactions.get('doctor_patient_qa', [])
                qa_summary = f"\n问诊过程（{qa_count}/{eval_data_for_ai['问诊配额']}轮）：\n"
                for i, qa in enumerate(qa_list[:3], 1):  # 只显示前3轮
                    q = qa.get('question', '')[:50]
                    a = qa.get('answer', '')[:50]
                    qa_summary += f"  [{i}] 问：{q}... 答：{a}...\n"
                if qa_count > 3:
                    qa_summary += f"  ... （共{qa_count}轮问诊）\n"
            else:
                qa_summary = "\n⚠️ 注意：医生未进行任何问诊！\n"
            
            user_prompt = (
                f"请简洁评估以下诊断：\n\n"
                f"医生诊断：{eval_data_for_ai['医生诊断']}\n"
                f"标准答案：{eval_data_for_ai['标准答案']}\n"
                f"问诊情况：{eval_data_for_ai['问诊轮数']}/{eval_data_for_ai['问诊配额']}轮"
                f"{qa_summary}\n"
                f"开单数量：{eval_data_for_ai['开单数量']}\n\n"
                f"诊断推理：{eval_data_for_ai['诊断推理']}\n\n"
                "请从以下角度简洁评估（每部分2-3句话）：\n"
                "1. 诊断准确性\n"
                "2. 过程评价\n"
                "3. 主要问题\n"
                "4. 改进建议\n\n"
                "输出格式：\n"
                "诊断准确性：[2-3句]\n"
                "过程评价：[2-3句]\n"
                "主要问题：[2-3句]\n"
                "改进建议：[2-3句]"
            )
            
            evaluation_report = llm.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # 格式化输出评估报告
            logger.info("\n" + evaluation_report)
            logger.info("\n【AI诊断评估报告】")
            logger.info(evaluation_report)
            
        except Exception as e:
            logger.warning(f"⚠️  AI评估生成失败: {e}")
            logger.info("\n⚠️  AI评估暂时不可用")
    
    logger.info("\n" + "="*80)

    if config.system.enable_trace:
        logger.info(f"\n💾 保存追踪信息到: {config.system.save_trace}")
        config.system.save_trace.parent.mkdir(parents=True, exist_ok=True)
        config.system.save_trace.write_text(
            json.dumps(final_state.audit_trail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"  ✅ Trace保存成功: {config.system.save_trace}")
        logger.info(f"\n💾 Trace已保存到: {config.system.save_trace}")
    
    # 构建返回结果
    result = {
        "case_id": case_id,
        "status": "success",
        "chief_complaint": final_state.chief_complaint,
        "dept": final_state.dept,
        "diagnosis": final_state.diagnosis.get("name", ""),
        "ground_truth": ground_truth.get("treatment_plan", "") if final_state.ground_truth else "",
        "questions_asked": sum(1 for entry in final_state.audit_trail if "interview" in entry.get("node_name", "").lower()),
        "tests_ordered": len(final_state.ordered_tests) if final_state.ordered_tests else 0,
        "escalations": final_state.escalations,
        "run_id": final_state.run_id,
    }
    
    # 如果有物理状态，添加物理环境信息
    if final_state.physical_state_snapshot:
        result["physical_state"] = {
            "final_energy": final_state.physical_state_snapshot.get("energy_level", 0),
            "final_pain": final_state.physical_state_snapshot.get("pain_level", 0),
            "total_time_minutes": final_state.physical_state_snapshot.get("elapsed_minutes", 0),
        }
    
    return result


if __name__ == "__main__":
    app()
