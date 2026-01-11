from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from loaders import load_diagnosis_arena_case
from agents import PatientAgent, DoctorAgent, NurseAgent
# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()  # 从当前目录或父目录查找 .env 文件
except ImportError:
    pass  # 如果没有安装 python-dotenv，跳过

from environment import HospitalWorld, PhysicalState, InteractiveSession
from graphs.router import build_common_graph, build_dept_subgraphs, build_services, default_retriever
from services.llm_client import build_llm_client
from state.schema import BaseState
from utils import make_rng, make_run_id, get_logger, setup_dual_logging
from config import Config

# 初始化logger（稍后会在main函数中设置双通道日志）
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
        typer.Option("--dataset-id", help="病例ID (覆盖配置文件)"),
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
    
    # 设置日志级别：verbose模式显示所有日志，否则只显示WARNING及以上
    import logging
    console_level = logging.INFO if verbose else logging.WARNING
    setup_dual_logging(log_file=log_file, console_level=console_level)
    
    # 在终端显示简洁的启动信息
    print("\n" + "="*80)
    print("🏥 医院智能体系统 - Hospital Agent System")
    print("="*80)
    
    logger.info("启动医院智能体系统 (三智能体模式)")
    print(f"📝 详细日志输出到: {log_file}\n")
    
    # 构造类似 argparse 的参数对象
    from types import SimpleNamespace
    args = SimpleNamespace(
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
    
    # 加载配置（优先级: CLI > 环境变量 > config.yaml > 默认值）
    config = Config.load(config_file=args.config, cli_args=args)
    
    # 输出配置摘要
    logger.info(config.summary())

    repo_root = Path(__file__).resolve().parents[1]

    rng = make_rng(config.system.seed)
    
    # 从数据集加载病例
    print("📚 加载病例数据...")
    logger.info("\n📚 加载病例数据...")
    logger.info(f"  🔢 数据集索引: {config.agent.dataset_id}")
    
    case_bundle = load_diagnosis_arena_case(config.agent.dataset_id, use_mock=not config.agent.use_hf_data)
    known_case = case_bundle["known_case"]
    ground_truth = case_bundle["ground_truth"]
    
    logger.info(f"  ✅ 病例ID: {known_case.get('id', 'unknown')}（数据集第{config.agent.dataset_id}条）")
    
    # 提取主诉
    case_info = known_case.get("Case Information", "")
    chief_complaint = case_info.split("主诉：")[1].split("。")[0] if "主诉：" in case_info else case_info[:50]
    
    print(f"  ✅ 主诉: {chief_complaint[:50]}...")
    print(f"  ✅ 标准诊断: {ground_truth.get('Final Diagnosis', 'N/A')}\n")
    logger.info(f"  ✅ 提取主诉: {chief_complaint}")
    logger.info(f"  ✅ 标准诊断: {ground_truth.get('Final Diagnosis', 'N/A')}")
    
    # 初始化 State（科室待护士分诊后确定）
    # 注意：run_id会在护士分诊后根据实际科室重新生成
    state = BaseState(
        run_id="temp",  # 临时值，分诊后会更新
        dept="internal_medicine",  # 临时值，护士分诊后会更新
        patient_profile={"case_text": case_info},
        appointment={"channel": "APP", "timeslot": "上午"},
        chief_complaint=chief_complaint,
        case_data=known_case,
        ground_truth=ground_truth,
    )
    logger.info(f"  ✅ 初始化State（科室待分诊确定）")
    
    # 用于存储物理环境引用
    patient_id = "patient_001"

    print("🤖 初始化系统组件...")
    try:
        logger.info(f"\n🤖 初始化LLM客户端 ({config.llm.backend})...")
        llm = build_llm_client(config.llm.backend)
        logger.info("  ✅ LLM客户端初始化成功")
    except Exception as e:  # noqa: BLE001
        print(f"❌ LLM初始化失败：{e}")
        print("   DeepSeek模式需设置环境变量：DEEPSEEK_API_KEY")
        logger.error(f"LLM初始化失败：{e}")
        raise

    # RAG 初始化（可选）
    retriever = None
    if not skip_rag:
        try:
            logger.info(f"\n📂 初始化RAG检索器 (collection: {config.rag.collection_name})...")
            retriever = default_retriever(persist_dir=config.rag.persist_dir, collection_name=config.rag.collection_name)
            logger.info("  ✅ RAG检索器初始化成功")
        except Exception as e:  # noqa: BLE001
            print(f"❌ RAG初始化失败：{e}")
            print("   请先运行知识库构建脚本")
            logger.error(f"RAG初始化失败：{e}")
            raise
    else:
        from rag import DummyRetriever
        logger.info("\n⏭️ 跳过RAG检索器初始化（使用虚拟检索器）")
        retriever = DummyRetriever()

    logger.info("\n⚙️ 初始化服务组件...")
    services = build_services(seed=config.system.seed)
    logger.info("  ✅ 服务组件初始化完成")
    print("  ✅ 组件初始化完成\n")
    
    # 初始化物理环境（如果启用）
    world = None
    if physical_sim:
        logger.info("\n🏥 初始化物理环境模拟...")
        world = HospitalWorld(start_time=None)  # 使用默认开始时间 8:00
        
        # 添加患者到环境
        world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
        
        # 初始化患者物理状态（从病例信息提取）
        if world.physical_states.get(patient_id):
            patient_state = world.physical_states[patient_id]
            # 可以根据主诉设置初始症状严重程度
            # 这里使用简单的默认值，后续可以从病例信息中提取
            patient_state.add_symptom("不适", severity=5.0)  # 默认中度不适
        
        logger.info(f"  ✅ 物理环境初始化完成")
        logger.info(f"  ✅ 患者已进入: {world.locations['lobby'].name}")
        logger.info(f"  ✅ 初始时间: {world.current_time.strftime('%Y-%m-%d %H:%M')}")
        
        # 如果启用交互式模式
        if interactive:
            logger.info("\n💬 启动交互式会话...")
            session = InteractiveSession(world, patient_id, agent_type="patient")
            
            print("\n" + "="*60)
            print("【交互式医院环境】")
            print("="*60)
            print("欢迎来到虚拟医院！你可以使用命令与环境交互。")
            print("输入 'help' 或 '帮助' 查看可用命令")
            print("输入 'quit' 或 'exit' 退出")
            print("="*60 + "\n")
            
            # 显示初始观察
            initial_obs = world.get_observation(patient_id)
            print(session._format_observation(initial_obs))
            print()
            
            # 交互循环
            while True:
                try:
                    prompt = session.get_prompt()
                    cmd = input(prompt).strip()
                    
                    if not cmd:
                        continue
                    
                    if cmd.lower() in ['quit', 'exit', 'q', '退出']:
                        print("\n👋 感谢使用，再见！")
                        break
                    
                    response = session.execute(cmd)
                    print(response + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 接收到中断信号，退出交互模式")
                    break
                except Exception as e:
                    print(f"❌ 错误: {e}\n")
            
            logger.info("  ✅ 交互式会话结束")
            return
    
    # 初始化三智能体
    print("🧑‍⚕️ 初始化三智能体并执行分诊...")
    logger.info("\n🧑‍⚕️ 初始化三智能体...")
    if llm is None:
        logger.warning("⚠️  建议使用LLM（--llm deepseek），否则对话质量较差")
    
    patient_agent = PatientAgent(known_case=state.case_data, llm=llm)
    logger.info("  ✅ 患者Agent初始化完成")
    
    nurse_agent = NurseAgent(llm=llm)
    logger.info("  ✅ 护士Agent初始化完成")
    
    # 护士分诊（原C0节点逻辑）
    logger.info("\n🏥 执行护士分诊...")
    triaged_dept = nurse_agent.triage(case_info)
    state.dept = triaged_dept
    triage_summary = nurse_agent.get_triage_summary()
    state.agent_interactions["nurse_triage"] = triage_summary
    print(f"  ✅ 分诊科室: {triaged_dept}")
    logger.info(f"  ✅ 分诊完成，确定科室: {triaged_dept}")
    
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
    doctor_agent.collected_info["chief_complaint"] = chief_complaint
    logger.info(f"  ✅ 医生Agent初始化完成 (科室: {state.dept}, max_questions: {config.agent.max_questions})")
    
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
        max_questions=config.agent.max_questions,
    )
    logger.info("  ✅ 执行图构建完成")

    print("\n" + "="*80)
    print("🚀 开始执行门诊流程...")
    print("="*80 + "\n")
    logger.info("\n" + "="*80)
    logger.info("🚀 开始执行门诊流程...")
    logger.info("="*80 + "\n")
    
    # 如果启用物理环境，模拟患者就医流程
    if physical_sim and world:
        logger.info("\n🎬 物理环境模拟开始...")
        logger.info(f"  📍 患者当前位置: {world.locations[world.agents[patient_id]].name}")
        logger.info(f"  ⏰ 当前时间: {world.current_time.strftime('%H:%M')}")
    
    print("📋 执行诊断流程（医生问诊、检查、诊断）...")
    print("   ⏳ 问诊中...")
    out = graph.invoke(state)
    print("\r   ✅ 诊断流程完成" + " " * 20)
    
    # 如果启用物理环境，模拟时间流逝和位置变化
    if physical_sim and world:
        print("\n" + "="*80)
        print("🎬 物理环境模拟")
        print("="*80)
        
        logger.info("\n🏥 模拟物理环境中的就医过程...")
        
        # 1. 护士分诊 -> 移动到分诊台
        print(f"📍 [08:00] 患者从门诊大厅前往分诊台...")
        logger.info("  📍 [护士分诊] 患者移动到分诊台...")
        world.move_agent(patient_id, "triage")
        world.advance_time(5)  # 分诊耗时5分钟
        
        # 显示分诊结果
        triaged_dept_name = {
            'internal_medicine': '内科',
            'surgery': '外科',
            'gastro': '消化内科',
            'neuro': '神经内科',
            'dermatology_std': '皮肤性病科',
            'orthopedics': '骨科',
            'urology': '泌尿外科',
        }.get(out.get('dept', 'internal_medicine'), out.get('dept', '内科'))
        
        print(f"   💉 分诊科室: {triaged_dept_name}")
        print(f"   ✅ 分诊完成 (耗时5分钟) - 当前时间: {world.current_time.strftime('%H:%M')}")
        logger.info(f"     💉 分诊科室: {triaged_dept_name}")
        logger.info(f"     ⏰ 分诊完成，时间: {world.current_time.strftime('%H:%M')}")
        
        # 2. 移动到诊室
        dept_location_map = {
            "internal_medicine": "internal_medicine",
            "surgery": "surgery",
            "gastro": "gastro",
            "neuro": "neuro",
            "dermatology_std": "internal_medicine",  # 皮肤科在内科区域
        }
        target_dept = dept_location_map.get(out.get('dept', 'internal_medicine'), 'internal_medicine')
        print(f"\n📍 [{world.current_time.strftime('%H:%M')}] 患者前往 {target_dept} 诊室...")
        logger.info(f"  📍 [就诊] 患者移动到 {target_dept} 诊室...")
        world.move_agent(patient_id, target_dept)
        
        # 3. 问诊过程
        qa_list = out.get('agent_interactions', {}).get('doctor_patient_qa', [])
        if qa_list:
            questions_count = len(qa_list)
            print(f"\n💬 [{world.current_time.strftime('%H:%M')}] 医生开始问诊 (共{questions_count}轮)...")
            
            # 显示前3轮问诊内容摘要
            for i, qa in enumerate(qa_list[:3], 1):
                question = qa.get('question', '')[:40]
                answer = qa.get('answer', '')[:30]
                print(f"   [{i}] 问: {question}{'...' if len(qa.get('question', '')) > 40 else ''}")
                print(f"       答: {answer}{'...' if len(qa.get('answer', '')) > 30 else ''}")
            
            if questions_count > 3:
                print(f"   ... (还有{questions_count - 3}轮问诊)")
            
            world.advance_time(questions_count * 3)  # 每轮约3分钟
            print(f"   ✅ 问诊完成 (耗时{questions_count * 3}分钟) - 当前时间: {world.current_time.strftime('%H:%M')}")
            logger.info(f"  💬 [问诊] 医生问诊 {questions_count} 轮...")
            logger.info(f"     ⏰ 问诊完成，时间: {world.current_time.strftime('%H:%M')}")
        
        # 4. 如果有检查，移动到相应科室
        if out.get('ordered_tests'):
            tests = out['ordered_tests']
            print(f"\n🔬 [{world.current_time.strftime('%H:%M')}] 医生开具 {len(tests)} 项检查单...")
            
            # 显示所有检查项目
            for idx, test in enumerate(tests, 1):
                test_name = test.get('name', '')
                test_reason = test.get('reason', '')[:50]
                print(f"   [{idx}] {test_name}")
                if test_reason:
                    print(f"       原因: {test_reason}{'...' if len(test.get('reason', '')) > 50 else ''}")
            
            logger.info(f"  🔬 [检查] 需要进行 {len(tests)} 项检查...")
            
            for idx, test in enumerate(tests, 1):
                test_name = test.get('name', '')
                test_type = test.get('type', '')
                
                # 根据检查类型移动到对应科室
                if test_type == 'imaging' or any(keyword in test_name for keyword in ['CT', 'X光', 'MRI', 'B超', '超声']):
                    print(f"\n   📍 [{world.current_time.strftime('%H:%M')}] 前往影像科 - {test_name}")
                    logger.info(f"     📍 前往影像科做 {test_name}...")
                    world.move_agent(patient_id, "imaging")
                    duration = 30
                    world.advance_time(duration)
                    print(f"       ✅ 检查完成 (耗时{duration}分钟) - {world.current_time.strftime('%H:%M')}")
                    
                elif test_type == 'lab' or any(keyword in test_name for keyword in ['血', '尿', '生化', '活检', '病理']):
                    print(f"\n   📍 [{world.current_time.strftime('%H:%M')}] 前往检验科 - {test_name}")
                    logger.info(f"     📍 前往检验科做 {test_name}...")
                    world.move_agent(patient_id, "lab")
                    duration = 20 if '活检' not in test_name else 15
                    world.advance_time(duration)
                    print(f"       ✅ 检查完成 (耗时{duration}分钟) - {world.current_time.strftime('%H:%M')}")
                    
                elif test_type == 'endoscopy' or '胃镜' in test_name or '肠镜' in test_name:
                    print(f"\n   📍 [{world.current_time.strftime('%H:%M')}] 前往内镜室 - {test_name}")
                    logger.info(f"     📍 前往内镜室做 {test_name}...")
                    world.move_agent(patient_id, "endoscopy")
                    duration = 45
                    world.advance_time(duration)
                    print(f"       ✅ 检查完成 (耗时{duration}分钟) - {world.current_time.strftime('%H:%M')}")
                else:
                    print(f"\n   🔬 [{world.current_time.strftime('%H:%M')}] {test_name}")
                    duration = 15
                    world.advance_time(duration)
                    print(f"       ✅ 完成 (耗时{duration}分钟) - {world.current_time.strftime('%H:%M')}")
                
                logger.info(f"     ✅ {test_name} 完成，时间: {world.current_time.strftime('%H:%M')}")
            
            # 返回诊室
            print(f"\n📍 [{world.current_time.strftime('%H:%M')}] 检查完毕，返回 {target_dept} 诊室...")
            logger.info(f"  📍 [复诊] 返回 {target_dept} 诊室...")
            world.move_agent(patient_id, target_dept)
            world.advance_time(5)  # 返回耗时5分钟
            print(f"   ✅ 已返回诊室 - {world.current_time.strftime('%H:%M')}")
            
            # 显示检查结果摘要
            test_results = out.get('test_results', [])
            if test_results:
                print(f"\n📊 [{world.current_time.strftime('%H:%M')}] 检查结果已出:")
                for idx, result in enumerate(test_results[:3], 1):
                    test_name = result.get('test_name', result.get('test', ''))
                    summary = result.get('summary', '')[:60]
                    abnormal = result.get('abnormal', False)
                    status = '⚠️ 异常' if abnormal else '✓ 正常'
                    print(f"   [{idx}] {test_name}: {status}")
                    if summary:
                        print(f"       {summary}{'...' if len(result.get('summary', '')) > 60 else ''}")
                
                if len(test_results) > 3:
                    print(f"   ... (还有{len(test_results) - 3}项结果)")
        
        # 5. 诊断和处方
        print(f"\n📋 [{world.current_time.strftime('%H:%M')}] 医生分析检查结果并出具诊断...")
        logger.info(f"  📋 [诊断] 医生出具诊断和治疗方案...")
        world.advance_time(10)  # 诊断和开方约10分钟
        
        # 显示诊断结果
        diagnosis = out.get('diagnosis', {})
        if diagnosis:
            diagnosis_name = diagnosis.get('name', '未知')
            print(f"   🩺 诊断结果: {diagnosis_name}")
        
        print(f"   ✅ 诊断完成 - {world.current_time.strftime('%H:%M')}")
        logger.info(f"     ⏰ 诊疗完成，时间: {world.current_time.strftime('%H:%M')}")
        
        # 6. 取药（如果有处方）
        treatment_plan = out.get('treatment_plan', {})
        if treatment_plan and (treatment_plan.get('symptomatic') or treatment_plan.get('etiology')):
            print(f"\n💊 [{world.current_time.strftime('%H:%M')}] 患者前往药房取药...")
            logger.info(f"  💊 [取药] 患者前往药房取药...")
            world.move_agent(patient_id, "pharmacy")
            world.advance_time(10)  # 取药约10分钟
            print(f"   ✅ 取药完成 - {world.current_time.strftime('%H:%M')}")
            logger.info(f"     ✅ 取药完成，时间: {world.current_time.strftime('%H:%M')}")
        
        # 最终状态
        total_minutes = (world.current_time.hour - 8) * 60 + world.current_time.minute
        final_location = world.agents.get(patient_id, "pharmacy")
        final_location_name = world.locations.get(final_location, world.locations["pharmacy"]).name if isinstance(final_location, str) else world.locations["pharmacy"].name
        
        print("\n" + "="*80)
        print("【物理环境模拟结果】")
        print("="*80)
        print(f"🏥 最终位置: {final_location_name}")
        print(f"⏰ 总耗时: {total_minutes} 分钟 (08:00 → {world.current_time.strftime('%H:%M')})")
        
        logger.info("\n" + "="*60)
        logger.info("【物理环境模拟结果】")
        logger.info("="*60)
        logger.info(f"  🏥 就诊科室: {world.locations[world.agents[patient_id]].name}")
        logger.info(f"  ⏰ 总耗时: {total_minutes} 分钟")
        logger.info(f"  🕐 结束时间: {world.current_time.strftime('%H:%M')}")
        
        # 显示患者健康状态变化
        if patient_id in world.physical_states:
            patient_state = world.physical_states[patient_id]
            print(f"\n💊 患者状态:")
            print(f"   体力: {patient_state.energy_level:.1f}/10")
            print(f"   疼痛: {patient_state.pain_level:.1f}/10")
            print(f"   意识: {patient_state.consciousness_level}")
            if patient_state.symptoms:
                symptoms_str = ', '.join([f'{name}({s.severity:.1f})' for name, s in patient_state.symptoms.items()])
                print(f"   症状: {symptoms_str}")
            
            logger.info(f"\n  💊 患者状态:")
            logger.info(f"     体力: {patient_state.energy_level:.1f}/10")
            logger.info(f"     疼痛: {patient_state.pain_level:.1f}/10")
            logger.info(f"     意识: {patient_state.consciousness_level}")
            if patient_state.symptoms:
                logger.info(f"     症状: {', '.join([f'{name}({s.severity:.1f})' for name, s in patient_state.symptoms.items()])}")
        
        print("="*80 + "\n")
        logger.info("="*60 + "\n")
    
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
    print("\n" + "="*80)
    print("✅ 门诊流程执行完成")
    print("="*80 + "\n")
    
    print("📊 诊断结果摘要")
    print("-" * 80)
    summary_lines = _render_human_summary(final_state)
    for line in summary_lines.split('\n'):
        if line.strip():
            print(f"  {line}")
    print("-" * 80)
    
    # 显示评估结果
    if final_state.agent_interactions.get("evaluation"):
        eval_data = final_state.agent_interactions["evaluation"]
        print("\n" + "="*80)
        print("【诊断评估】")
        print("="*80)
        print(f"📋 医生诊断: {eval_data['doctor_diagnosis']}")
        print(f"🎯 标准答案: {eval_data['correct_diagnosis']}")
        
        accuracy_pct = eval_data['accuracy'] * 100
        accuracy_method = eval_data.get('accuracy_method', '选项匹配')
        
        print(f"\n📊 诊断准确性:")
        if accuracy_pct >= 80:
            print(f"   ✅ 准确率: {accuracy_pct:.0f}% (优秀)")
        elif accuracy_pct >= 60:
            print(f"   ⚠️  准确率: {accuracy_pct:.0f}% (良好)")
        else:
            print(f"   ❌ 准确率: {accuracy_pct:.0f}% (需改进)")
        print(f"   📏 评估方法: {accuracy_method}")
        
        if eval_data.get('selected_option'):
            print(f"\n🔍 选项匹配: {eval_data['selected_option']} (正确: {eval_data['correct_option']})")
        
        print(f"\n📈 诊断过程:")
        print(f"   💬 问诊轮数: {eval_data['questions_asked']} 轮")
        print(f"   🔬 开单数量: {eval_data['tests_ordered']} 项")
        print("="*80)
    
    # 显示诊断质量信息
    diagnosis = final_state.diagnosis
    
    # 使用LLM生成智能诊断评估报告
    if llm and final_state.ground_truth:
        print("\n" + "="*80)
        print("【AI诊断质量分析】")
        print("="*80)
        logger.info("\n🤖 生成AI诊断评估报告...")
        
        try:
            # 准备评估数据
            eval_data_for_ai = {
                "医生诊断": diagnosis.get("name", ""),
                "标准答案": final_state.ground_truth.get("Final Diagnosis", ""),
                "问诊轮数": sum(1 for entry in final_state.audit_trail if entry.get("node_name") == "C3_specialty"),
                "开单数量": len(final_state.ordered_tests) if final_state.ordered_tests else 0,
                "诊断推理": diagnosis.get("reasoning", "")[:300],
                "确定程度": diagnosis.get("uncertainty", ""),
            }
            
            system_prompt = "你是一位资深的临床医学专家和医学教育者，擅长评估诊断质量并提供建设性反馈。"
            
            user_prompt = (
                f"请简洁评估以下诊断：\n\n"
                f"医生诊断：{eval_data_for_ai['医生诊断']}\n"
                f"标准答案：{eval_data_for_ai['标准答案']}\n"
                f"问诊轮数：{eval_data_for_ai['问诊轮数']}\n"
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
            print("\n" + evaluation_report)
            logger.info("\n【AI诊断评估报告】")
            logger.info(evaluation_report)
            
        except Exception as e:
            logger.warning(f"⚠️  AI评估生成失败: {e}")
            print("\n⚠️  AI评估暂时不可用")
    
    print("\n" + "="*80)

    if config.system.enable_trace:
        logger.info(f"\n💾 保存追踪信息到: {config.system.save_trace}")
        config.system.save_trace.parent.mkdir(parents=True, exist_ok=True)
        config.system.save_trace.write_text(
            json.dumps(final_state.audit_trail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"  ✅ Trace保存成功: {config.system.save_trace}")
        print(f"\n💾 Trace已保存到: {config.system.save_trace}")
    
    print(f"\n📝 详细日志已保存到: {log_file}")
    print("✅ 程序执行完毕\n")


if __name__ == "__main__":
    app()
