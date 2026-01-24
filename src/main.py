from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from loaders import load_diagnosis_arena_case
from agents import PatientAgent, DoctorAgent, NurseAgent, LabAgent
# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()  # 从当前目录或父目录查找 .env 文件
except ImportError:
    pass  # 如果没有安装 python-dotenv，跳过
from environment import HospitalWorld, PhysicalState, InteractiveSession
from langgraph_multi_patient_processor import LangGraphMultiPatientProcessor
from services.medical_record import MedicalRecordService
from services.medical_record_integration import MedicalRecordIntegration
from graphs.router import build_common_graph, build_dept_subgraphs, build_services, default_retriever
from services.llm_client import build_llm_client
from state.schema import BaseState
from utils import make_rng, make_run_id, get_logger, setup_dual_logging
from config import Config
from hospital_coordinator import HospitalCoordinator
from multi_patient_processor import MultiPatientProcessor
from monitoring_dashboard import print_simple_status

# 初始化logger
logger = get_logger("hospital_agent.main")

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
    # 核心参数
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", help="配置文件路径 (默认: config.yaml)"),
    ] = None,
    dataset_id: Annotated[
        Optional[int],
        typer.Option("--dataset-id", help="病例ID (覆盖配置文件，与batch模式互斥)"),
    ] = None,
    start_id: Annotated[
        Optional[int],
        typer.Option("--start-id", help="批量处理起始ID（默认1）"),
    ] = None,
    end_id: Annotated[
        Optional[int],
        typer.Option("--end-id", help="批量处理结束ID（默认915）"),
    ] = None,
    batch_mode: Annotated[
        bool,
        typer.Option("--batch", help="批量处理模式"),
    ] = False,
    multi_patient: Annotated[
        bool,
        typer.Option("--multi-patient", help="多患者多医生模式"),
    ] = True,
    num_patients: Annotated[
        Optional[int],
        typer.Option("--num-patients", help="多患者模式下的患者数量（默认3）"),
    ] = None,
    patient_interval: Annotated[
        Optional[int],
        typer.Option("--patient-interval", help="患者进入间隔时间（秒，默认60秒）"),
    ] = None,
    llm: Annotated[
        Optional[str],
        typer.Option("--llm", help="LLM后端: mock 或 deepseek (覆盖配置文件)"),
    ] = None,
    max_questions: Annotated[
        Optional[int],
        typer.Option("--max-questions", help="最多问题数 (覆盖配置文件)"),
    ] = None,
    
    # 可选参数
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="随机种子"),
    ] = None,
    llm_reports: Annotated[
        bool,
        typer.Option("--llm-reports", help="使用LLM增强报告"),
    ] = False,
    save_trace: Annotated[
        Optional[Path],
        typer.Option("--save-trace", help="保存追踪到指定文件"),
    ] = None,
    persist: Annotated[
        Optional[Path],
        typer.Option("--persist", help="Chroma目录"),
    ] = None,
    collection: Annotated[
        Optional[str],
        typer.Option("--collection", help="知识库集合名"),
    ] = None,
    use_hf_data: Annotated[
        Optional[bool],
        typer.Option("--use-hf-data", help="使用HuggingFace数据"),
    ] = None,
    
    # 物理环境参数
    physical_sim: Annotated[
        bool,
        typer.Option("--physical-sim", help="启用物理环境模拟"),
    ] = True,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", help="启用交互式命令模式"),
    ] = False,
    skip_rag: Annotated[
        bool,
        typer.Option("--skip-rag", help="跳过RAG系统初始化（用于测试物理环境）"),
    ] = True,
    log_file: Annotated[
        Optional[str],
        typer.Option("--log-file", help="详细日志文件路径（默认: logs/hospital_agent_运行时间.log）"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="终端显示详细日志"),
    ] = False,
) -> None:
    """Hospital Agent System - 三智能体医疗诊断系统
    
    配置优先级: CLI参数 > 环境变量 > config.yaml > 默认值
    """
    # 设置双通道日志系统
    from datetime import datetime
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir / f"hospital_agent_{timestamp}.log")
    
    # 设置日志级别：verbose模式显示所有日志（DEBUG），否则显示INFO及以上
    # 注意：所有print已改为logger.info，默认在终端显示
    import logging
    console_level = logging.DEBUG if verbose else logging.INFO
    setup_dual_logging(log_file=log_file, console_level=console_level)
    
    # 在终端显示简洁的启动信息
    logger.info("\n" + "="*80)
    logger.info("🏥 医院智能体系统 - Hospital Agent System")
    logger.info("="*80)
    
    logger.info("启动系统 ")
    logger.info(f"📝 日志输出到: {log_file}\n")
    
    # 多患者多医生模式
    if multi_patient:
        logger.info("🏥 启动多患者多医生模式 (LangGraph 集成)")
        logger.info("="*80)
        
        # 加载配置
        from types import SimpleNamespace
        temp_args = SimpleNamespace(
            config=config_file,
            dataset_id=None,
            llm=llm,
            max_questions=max_questions,
            seed=seed,
            llm_reports=llm_reports,
            save_trace=save_trace,
            persist=persist,
            collection=collection,
            use_hf_data=use_hf_data,
        )
        config = Config.load(config_file=temp_args.config, cli_args=temp_args)
        
        # 默认参数
        _num_patients = num_patients if num_patients is not None else 3
        _patient_interval = patient_interval if patient_interval is not None else 60  # 默认60秒
        
        logger.info(f"患者数量: {_num_patients}")
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
        if not skip_rag:
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
        services = build_services(seed=config.system.seed)
        logger.info("  ✅ 服务组件初始化完成\n")
        
        # 初始化医疗记录服务
        logger.info("📋 初始化病例库服务...")
        medical_record_service = MedicalRecordService(storage_dir=Path("./medical_records"))
        logger.info(f"  ✅ 病例库服务初始化完成\n")
        
        # 初始化协调器
        logger.info("🏥 初始化医院协调器...")
        coordinator = HospitalCoordinator(medical_record_service)
        logger.info("  ✅ 协调器初始化完成\n")
        
        # 初始化 LangGraph 多患者处理器
        logger.info("🚀 初始化 LangGraph 多患者处理器...")
        processor = LangGraphMultiPatientProcessor(
            coordinator=coordinator,
            retriever=retriever,
            llm=llm_client,
            services=services,
            medical_record_service=medical_record_service,
            seed=config.system.seed,
            max_questions=config.agent.max_questions,
            use_hf_data=config.agent.use_hf_data,
            max_workers=_num_patients,  # 每个患者一个线程
        )
        logger.info("  ✅ 处理器初始化完成\n")
        
        # 注册医生：为系统所有15个标准科室各配置一名医生
        logger.info("🏥 为所有标准科室注册医生...")
        
        # 15个标准科室（与 NurseAgent.VALID_DEPTS 一致）
        STANDARD_DEPTS = [
            "internal_medicine", "surgery", "orthopedics", "urology",
            "obstetrics_gynecology", "pediatrics", "neurology", "oncology",
            "infectious_disease", "dermatology_std", "ent_ophthalmology_stomatology",
            "psychiatry", "emergency", "rehabilitation_pain", "traditional_chinese_medicine"
        ]
        
        # 科室中文名称映射
        DEPT_CN_NAMES = {
            "internal_medicine": "内科",
            "surgery": "外科",
            "orthopedics": "骨科",
            "urology": "泌尿外科",
            "obstetrics_gynecology": "妇产科",
            "pediatrics": "儿科",
            "neurology": "神经医学科",
            "oncology": "肿瘤科",
            "infectious_disease": "感染性疾病科",
            "dermatology_std": "皮肤性病科",
            "ent_ophthalmology_stomatology": "眼耳鼻喉口腔科",
            "psychiatry": "精神心理科",
            "emergency": "急诊医学科",
            "rehabilitation_pain": "康复疼痛科",
            "traditional_chinese_medicine": "中医科"
        }
        
        doctor_id = 1
        for dept in STANDARD_DEPTS:
            doc_id = f"DOC{doctor_id:03d}"
            dept_cn = DEPT_CN_NAMES.get(dept, dept)
            doc_name = f"{dept_cn}医生"
            
            coordinator.register_doctor(doc_id, doc_name, dept)
            logger.info(f"  ✅ {doc_name} (ID: {doc_id}, 科室: {dept})")
            doctor_id += 1
        logger.info(f"\n已注册 {len(STANDARD_DEPTS)} 名医生（覆盖所有标准科室）\n")
        
        # 准备患者数据（使用真实数据集病例，随机选择）
        import random
        import time
        
        # 加载真实数据集以获取病例总数
        logger.info("📚 检查可用的真实病例数量...")
        try:
            from loaders import _get_dataset_size
            max_case_id = _get_dataset_size(config.dataset.cache_dir if config.dataset.use_local_cache else None)
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
        interval_display = f"{_patient_interval} 秒" if _patient_interval < 60 else f"{_patient_interval/60:.1f} 分钟"
        logger.info(f"⏰ 患者将每隔 {interval_display} 进入系统（每个患者启动独立线程，竞争共享资源）\n")
        logger.info("="*80)
        
        task_ids = []
        timers = []  # 保存所有定时器，以便等待
        
        def submit_patient_thread(i, case_id, total_patients):
            """在独立线程中提交患者（每个患者到来时立即启动）"""
            patient_id = f"patient_{case_id:03d}"
            priority = random.randint(3, 9)
            
            # 患者到来
            current_time = time.strftime("%H:%M:%S")
            logger.info(f"[{current_time}] 🚶 患者 {i+1}/{total_patients} 到达医院（启动独立处理线程）")
            logger.info(f"  📋 {patient_id}: 病例 ID={case_id} (优先级: {priority})")
            
            # 立即提交患者，启动 LangGraph 执行线程
            task_id = processor.submit_patient(
                patient_id=patient_id,
                case_id=case_id,
                dept="internal_medicine",  # 初始科室，会被护士分诊覆盖
                priority=priority
            )
            task_ids.append(task_id)
            logger.info(f"  ✅ 线程已启动: {task_id}（开始竞争资源）\n")
        
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
        
        logger.info("="*80)
        logger.info(f"✅ 所有 {len(selected_case_ids)} 名患者已到达，各自线程正在并发执行\n")
        
        # 等待所有任务完成
        logger.info("\n⏳ 等待所有患者完成 LangGraph 诊断流程...")
        results = processor.wait_all(timeout=600)  # 增加超时时间
        
        # 打印结果
        logger.info("\n" + "="*80)
        logger.info("📊 LangGraph 多患者诊断结果")
        logger.info("="*80)
        
        success_count = 0
        failed_count = 0
        
        for result in results:
            status = result.get("status")
            patient_id = result.get("patient_id", "未知")
            case_id = result.get("case_id", "N/A")
            
            if status == "completed":
                diagnosis = result.get("diagnosis", "未明确")
                ground_truth = result.get("ground_truth", "N/A")
                dept = result.get("dept", "N/A")
                node_count = result.get("node_count", 0)
                
                logger.info(f"\n✅ {patient_id} (案例 {case_id})")
                logger.info(f"   科室: {dept}")
                logger.info(f"   诊断结果: {diagnosis}")
                logger.info(f"   标准诊断: {ground_truth}")
                logger.info(f"   执行节点: {node_count} 个")
                
                success_count += 1
            else:
                error_msg = result.get('error', result.get('reason', '未知错误'))
                logger.info(f"\n❌ {patient_id} (案例 {case_id})")
                logger.info(f"   状态: {status}")
                logger.info(f"   错误: {error_msg}")
                
                failed_count += 1
        
        # 最终统计
        logger.info("\n" + "="*80)
        logger.info("📈 最终统计")
        logger.info("="*80)
        logger.info(f"✅ 成功: {success_count}/{len(results)}")
        logger.info(f"❌ 失败: {failed_count}/{len(results)}")
        logger.info(f"📊 总计: {len(results)} 名患者")
        
        # 保存结果到文件
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"multi_patient_results_{timestamp}.json"
        
        import json
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "total_patients": len(results),
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📁 结果已保存到: {results_file}")
        logger.info(f"📝 详细日志已保存到: {log_file}")
        
        # 关闭处理器
        processor.shutdown()
        
        logger.info("\n✅ LangGraph 多患者模式执行完毕\n")
        
        return
    
    # 批量处理模式
    if batch_mode:
        batch_start = start_id if start_id is not None else 1
        batch_end = end_id if end_id is not None else 915
        logger.info(f"🔄 批量处理模式: 处理病例 {batch_start} 到 {batch_end}")
        
        # 批量处理结果保存路径
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_file = results_dir / f"batch_results_{batch_start}_to_{batch_end}_{timestamp}.jsonl"
        
        logger.info(f"📊 批量结果将保存到: {batch_results_file}")
        
        # 统计信息
        success_count = 0
        fail_count = 0
        
        with open(batch_results_file, "w", encoding="utf-8") as f:
            for case_id in range(batch_start, batch_end + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"处理病例 {case_id}/{batch_end} ({case_id - batch_start + 1}/{batch_end - batch_start + 1})")
                logger.info(f"{'='*80}")
                logger.info(f"\n{'='*80}")
                logger.info(f"处理病例 {case_id}/{batch_end}")
                logger.info(f"{'='*80}")
                
                try:
                    # 调用单病例处理函数
                    result = process_single_case(
                        case_id=case_id,
                        config_file=config_file,
                        llm=llm,
                        max_questions=max_questions,
                        seed=seed,
                        llm_reports=llm_reports,
                        save_trace=save_trace,
                        persist=persist,
                        collection=collection,
                        use_hf_data=use_hf_data,
                        physical_sim=physical_sim,
                        interactive=interactive,
                        skip_rag=skip_rag,
                        verbose=verbose,
                    )
                    
                    # 保存结果
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    
                    success_count += 1
                    logger.info(f"✅ 病例 {case_id} 处理成功")
                    logger.info(f"✅ 病例 {case_id} 处理成功")
                    
                except Exception as e:
                    fail_count += 1
                    error_msg = f"❌ 病例 {case_id} 处理失败: {str(e)}"
                    logger.info(error_msg)
                    logger.error(error_msg, exc_info=True)
                    
                    # 记录失败信息
                    error_result = {
                        "case_id": case_id,
                        "status": "failed",
                        "error": str(e),
                    }
                    f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    f.flush()
        
        # 打印批量处理统计
        logger.info(f"\n{'='*80}")
        logger.info("📊 批量处理完成")
        logger.info(f"{'='*80}")
        logger.info(f"✅ 成功: {success_count}")
        logger.info(f"❌ 失败: {fail_count}")
        logger.info(f"📊 总计: {success_count + fail_count}")
        logger.info(f"📁 结果文件: {batch_results_file}")
        logger.info(f"📝 日志文件: {log_file}")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"\n{'='*80}")
        logger.info("📊 批量处理完成")
        logger.info(f"✅ 成功: {success_count}, ❌ 失败: {fail_count}")
        logger.info(f"{'='*80}")
        
        return
    
    # 单病例处理模式 - 先加载配置获取默认dataset_id
    from types import SimpleNamespace
    temp_args = SimpleNamespace(
        config=config_file,
        dataset_id=dataset_id,
        llm=llm,
        max_questions=max_questions,
        seed=seed,
        llm_reports=llm_reports,
        save_trace=save_trace,
        persist=persist,
        collection=collection,
        use_hf_data=use_hf_data,
    )
    temp_config = Config.load(config_file=temp_args.config, cli_args=temp_args)
    
    # 使用配置中的dataset_id（如果命令行未指定）
    final_dataset_id = dataset_id if dataset_id is not None else temp_config.agent.dataset_id
    
    if final_dataset_id is None:
        logger.info("❌ 错误: 请指定 --dataset-id 或在配置文件中设置 dataset_id，或使用 --batch 模式")
        logger.error("未指定dataset_id且配置文件中也没有默认值")
        return
    
    logger.info(f"📋 单病例处理模式: 病例 {final_dataset_id}")
    
    # 调用单病例处理函数
    result = process_single_case(
        case_id=final_dataset_id,
        config_file=config_file,
        llm=llm,
        max_questions=max_questions,
        seed=seed,
        llm_reports=llm_reports,
        save_trace=save_trace,
        persist=persist,
        collection=collection,
        use_hf_data=use_hf_data,
        physical_sim=physical_sim,
        interactive=interactive,
        skip_rag=skip_rag,
        verbose=verbose,
    )
    
    logger.info(f"\n📝 详细日志已保存到: {log_file}")
    logger.info("✅ 程序执行完毕\n")


def process_single_case(
    case_id: int,
    config_file: Optional[Path] = None,
    llm: Optional[str] = None,
    max_questions: Optional[int] = None,
    seed: Optional[int] = None,
    llm_reports: bool = False,
    save_trace: Optional[Path] = None,
    persist: Optional[Path] = None,
    collection: Optional[str] = None,
    use_hf_data: Optional[bool] = None,
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
        seed=seed,
        llm_reports=llm_reports,
        save_trace=save_trace,
        persist=persist,
        collection=collection,
        use_hf_data=use_hf_data,
    )
    
    # 加载配置（优先级: CLI > 环境变量 > config.yaml > 默认值）
    config = Config.load(config_file=args.config, cli_args=args)
    
    # 输出配置摘要
    logger.info(config.summary())

    repo_root = Path(__file__).resolve().parents[1]

    rng = make_rng(config.system.seed)
    
    # 从数据集加载病例
    logger.info("📚 加载病例数据...")
    logger.info(f"  🔢 数据集索引: {config.agent.dataset_id}")
    
    # 使用配置的缓存目录
    cache_dir = str(config.dataset.cache_dir) if config.dataset.use_local_cache else None
    if cache_dir:
        logger.info(f"  📂 本地缓存: {cache_dir}")
    
    case_bundle = load_diagnosis_arena_case(
        config.agent.dataset_id, 
        use_mock=not config.agent.use_hf_data,
        local_cache_dir=cache_dir or "./diagnosis_dataset"
    )
    known_case = case_bundle["known_case"]
    ground_truth = case_bundle["ground_truth"]
    
    logger.info(f"  ✅ 病例ID: {known_case.get('id', 'unknown')}（数据集第{config.agent.dataset_id}条）")
    
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
    logger.info(f"  ✅ 标准诊断: {ground_truth.get('Final Diagnosis', 'N/A')}\n")

    # 初始化 State（科室待护士分诊后确定）
    # 注意：run_id会在护士分诊后根据实际科室重新生成
    patient_id = "patient_001"  # 定义患者ID，用于物理环境
    
    state = BaseState(
        run_id="temp",  # 临时值，分诊后会更新
        dept="internal_medicine",  # 临时值，护士分诊后会更新
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
        llm = build_llm_client(config.llm.backend)
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
            logger.info(f"\n📂 初始化知识库检索器 (集合名: {config.rag.collection_name})...")
            retriever = default_retriever(persist_dir=config.rag.persist_dir, collection_name=config.rag.collection_name)
            logger.info("  ✅ 知识库检索器初始化成功")
        except Exception as e:  # noqa: BLE001
            logger.info(f"❌ 知识库检索器初始化失败：{e}")
            logger.info("   请先运行知识库构建脚本")
            logger.error(f"知识库检索器初始化失败：{e}")
            raise
    else:
        from rag import DummyRetriever
        logger.info("\n⏭️ 跳过知识库检索器初始化（使用虚拟检索器）")
        retriever = DummyRetriever()

    logger.info("\n⚙️ 初始化服务组件...")
    services = build_services(seed=config.system.seed)
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
        "dataset_id": config.agent.dataset_id,
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
    if llm is None:
        logger.warning("⚠️  建议使用LLM（--llm deepseek），否则对话质量较差")
    
    # 患者智能体使用原始主诉（从数据集读取的）
    patient_agent = PatientAgent(known_case=state.case_data, llm=llm, chief_complaint=original_chief_complaint)
    logger.info("  ✅ 患者Agent初始化完成")
    
    nurse_agent = NurseAgent(llm=llm, max_triage_questions=config.agent.max_triage_questions)
    logger.info(f"  ✅ 护士Agent初始化完成（最多可问{config.agent.max_triage_questions}个问题）")
    
    # 【新增】将护士添加到物理环境
    if world:
        nurse_id = "nurse_001"
        world.add_agent(nurse_id, agent_type="nurse", initial_location="triage")
        logger.info(f"  ✅ 护士已就位于: {world.locations['triage'].name}")
    
    # 初始化检验科Agent
    lab_agent = LabAgent(llm=llm)
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
    run_id = make_run_id(config.system.seed, triaged_dept)
    state.run_id = run_id
    logger.info(f"  ✅ 生成run_id: {run_id}")
    
    # 初始化医生Agent（需要知道科室后才能初始化）
    doctor_agent = DoctorAgent(
        dept=state.dept, 
        retriever=retriever, 
        llm=llm,
        max_questions=config.agent.max_questions
    )
    # 医生不直接获得主诉，需要通过问诊从患者处获得
    logger.info(f"  ✅ 医生Agent初始化完成 (科室: {state.dept}, max_questions: {config.agent.max_questions})")
    
    # 【新增】将医生添加到物理环境（根据分诊科室）
    if world:
        doctor_id = "doctor_001"
        # 医生在对应科室诊室（映射所有可能的分诊科室）
        # 注意：部分科室共享诊室（如皮肤科使用内科诊室）
        dept_location_map = {
            "internal_medicine": "internal_medicine",
            "surgery": "surgery", 
            "gastro": "gastro",
            "neuro": "neuro",
            "emergency": "emergency",
            "orthopedics": "surgery",  # 骨科使用外科诊室
            "urology": "surgery",  # 泌尿外科使用外科诊室
            "obstetrics_gynecology": "internal_medicine",  # 妇产科使用内科诊室
            "pediatrics": "internal_medicine",  # 儿科使用内科诊室
            "neurology": "neuro",  # 神经医学使用神经内科诊室
            "oncology": "internal_medicine",  # 肿瘤科使用内科诊室
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
            "internal_medicine": "内科",
            "surgery": "外科",
            "gastro": "消化内科",
            "neuro": "神经内科",
            "emergency": "急诊科",
            "orthopedics": "骨科",
            "urology": "泌尿外科",
            "obstetrics_gynecology": "妇产科",
            "pediatrics": "儿科",
            "neurology": "神经医学",
            "oncology": "肿瘤科",
            "infectious_disease": "感染性疾病科",
            "dermatology_std": "皮肤性病科",
            "ent_ophthalmology_stomatology": "眼耳鼻喉口腔科",
            "psychiatry": "精神心理科",
            "rehabilitation_pain": "康复疼痛科",
            "traditional_chinese_medicine": "中医科",
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
        rng=rng, 
        llm=llm,
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
        rng=rng,
        llm=llm,
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
                "internal_medicine": "内科",
                "surgery": "外科",
                "gastro": "消化内科",
                "neuro": "神经内科",
                "emergency": "急诊科",
                "orthopedics": "骨科",
                "urology": "泌尿外科",
                "obstetrics_gynecology": "妇产科",
                "pediatrics": "儿科",
                "neurology": "神经医学",
                "oncology": "肿瘤科",
                "infectious_disease": "感染科",
                "dermatology_std": "皮肤科",
                "ent_ophthalmology_stomatology": "五官科",
                "psychiatry": "精神心理科",
                "rehabilitation_pain": "康复疼痛科",
                "traditional_chinese_medicine": "中医科",
            }
            dept_cn = dept_cn_names.get(state.dept, state.dept)
            actual_location = world.locations[world.agents[doctor_id]].name
            
            # 检查是否是共享诊室（科室诊室名与实际位置不同）
            dept_location_map = {
                "internal_medicine": "internal_medicine",
                "surgery": "surgery", 
                "gastro": "gastro",
                "neuro": "neuro",
                "emergency": "emergency",
                "orthopedics": "surgery",
                "urology": "surgery",
                "obstetrics_gynecology": "internal_medicine",
                "pediatrics": "internal_medicine",
                "neurology": "neuro",
                "oncology": "internal_medicine",
                "infectious_disease": "internal_medicine",
                "dermatology_std": "internal_medicine",
                "ent_ophthalmology_stomatology": "internal_medicine",
                "psychiatry": "internal_medicine",
                "rehabilitation_pain": "internal_medicine",
                "traditional_chinese_medicine": "internal_medicine",
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
            
                
                # 将完整时间线输出到日志
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
        logger.info(f"🎯 标准答案: {eval_data['correct_diagnosis']}")
        
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
        "ground_truth": final_state.ground_truth.get("Final Diagnosis", "") if final_state.ground_truth else "",
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
