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

from graphs.router import build_common_graph, build_dept_subgraphs, build_services, default_retriever
from services.llm_client import build_llm_client
from state.schema import BaseState
from utils import make_rng, make_run_id, get_logger
from config import Config

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
) -> None:
    """Hospital Agent System - 三智能体医疗诊断系统
    
    配置优先级: CLI参数 > 环境变量 > config.yaml > 默认值
    """
    logger.info("启动医院智能体系统 (三智能体模式)")
    
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
    logger.info("\n📚 加载病例数据...")
    logger.info(f"  🔢 数据集索引: {config.agent.dataset_id}")
    
    case_bundle = load_diagnosis_arena_case(config.agent.dataset_id, use_mock=not config.agent.use_hf_data)
    known_case = case_bundle["known_case"]
    ground_truth = case_bundle["ground_truth"]
    
    logger.info(f"  ✅ 病例ID: {known_case.get('id', 'unknown')}（数据集第{config.agent.dataset_id}条）")
    
    # 提取主诉
    case_info = known_case.get("Case Information", "")
    chief_complaint = case_info.split("主诉：")[1].split("。")[0] if "主诉：" in case_info else case_info[:50]
    
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

    try:
        logger.info(f"\n🤖 初始化LLM客户端 ({config.llm.backend})...")
        llm = build_llm_client(config.llm.backend)
        logger.info("  ✅ LLM客户端初始化成功")
    except Exception as e:  # noqa: BLE001
        print(
            f"LLM 初始化失败：{e}\n"
            "DeepSeek 模式请先设置环境变量：DEEPSEEK_API_KEY（可选：DEEPSEEK_BASE_URL/DEEPSEEK_MODEL）",
            file=sys.stderr,
        )
        raise

    try:
        logger.info(f"\n📂 初始化RAG检索器 (collection: {config.rag.collection_name})...")
        retriever = default_retriever(persist_dir=config.rag.persist_dir, collection_name=config.rag.collection_name)
        logger.info("  ✅ RAG检索器初始化成功")
    except Exception as e:  # noqa: BLE001
        seed_script = repo_root / "scripts" / "seed_kb_examples.py"
        build_script = repo_root / "scripts" / "build_index.py"
        print(
            f"RAG 初始化失败：{e}\n"
            f"请先运行：python \"{seed_script}\" && python \"{build_script}\"",
            file=sys.stderr,
        )
        raise

    logger.info("\n⚙️ 初始化服务组件...")
    services = build_services(seed=config.system.seed)
    logger.info("  ✅ 服务组件初始化完成")
    
    # 初始化三智能体
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

    logger.info("\n" + "="*80)
    logger.info("🚀 开始执行门诊流程...")
    logger.info("="*80 + "\n")
    
    out = graph.invoke(state)
    
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

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n---\n")
    print(_render_human_summary(final_state))
    
    # 显示评估结果
    if final_state.agent_interactions.get("evaluation"):
        eval_data = final_state.agent_interactions["evaluation"]
        print("\n【诊断评估】")
        print(f"医生诊断: {eval_data['doctor_diagnosis']}")
        print(f"标准答案: {eval_data['correct_diagnosis']}")
        
        accuracy_pct = eval_data['accuracy'] * 100
        accuracy_method = eval_data.get('accuracy_method', '选项匹配')
        
        if accuracy_pct >= 80:
            print(f"准确率: {accuracy_pct:.0f}% ✅ (评估方法: {accuracy_method})")
        elif accuracy_pct >= 60:
            print(f"准确率: {accuracy_pct:.0f}% ⚠️  (评估方法: {accuracy_method})")
        else:
            print(f"准确率: {accuracy_pct:.0f}% ❌ (评估方法: {accuracy_method})")
        
        if eval_data.get('selected_option'):
            print(f"匹配选项: {eval_data['selected_option']} (正确选项: {eval_data['correct_option']})")
        print(f"问诊轮数: {eval_data['questions_asked']}")
        print(f"开单数量: {eval_data['tests_ordered']}")
    
    # 显示诊断质量信息
    diagnosis = final_state.diagnosis
    print("\n" + "="*60)
    print("【诊断质量分析】")
    print("="*60)
    
    # 移除所有防锚定偏差和高权重线索相关的显示
    
    # 使用LLM生成智能诊断评估报告
    if llm and final_state.ground_truth:
        logger.info("\n🤖 生成AI诊断评估报告...")
        print("\n" + "="*60)
        print("【AI诊断评估报告】")
        print("="*60)
        
        try:
            # 准备评估数据
            eval_data = {
                "医生诊断": diagnosis.get("name", ""),
                "标准答案": final_state.ground_truth.get("Final Diagnosis", ""),
                "问诊轮数": sum(1 for entry in final_state.audit_trail if entry.get("node_name") == "C3_specialty"),
                "开单数量": len(final_state.ordered_tests) if final_state.ordered_tests else 0,
                "诊断推理": diagnosis.get("reasoning", "")[:500],
                "确定程度": diagnosis.get("uncertainty", ""),
            }
            
            system_prompt = (
                "你是一位资深的临床医学专家和医学教育者，擅长评估诊断质量并提供建设性反馈。"
                "你的评估应该客观、专业、具有教育意义。"
            )
            
            user_prompt = (
                "请对以下诊断过程进行专业评估，并生成一份简洁的评估报告。\n\n"
                + "【诊断信息】\n"
                + json.dumps(eval_data, ensure_ascii=False, indent=2)
                + "\n\n【评估要求】\n"
                + "1. 诊断准确性分析：对比医生诊断与标准答案，评估准确程度和差异原因\n"
                + "2. 诊断过程评价：评估问诊效率、检查合理性、诊断推理逻辑\n"
                + "3. 质量风险识别：指出高权重线索处理、锚定风险、难解释事实等问题\n"
                + "4. 改进建议：针对发现的问题给出具体、可操作的改进建议\n\n"
                + "请用中文输出，使用专业但易懂的语言，控制在500字以内。\n"
                + "输出格式：\n"
                + "## 诊断准确性\n[分析内容]\n\n"
                + "## 过程评价\n[分析内容]\n\n"
                + "## 质量风险\n[分析内容]\n\n"
                + "## 改进建议\n[建议内容]"
            )
            
            evaluation_report = llm.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            print(evaluation_report)
            
        except Exception as e:
            logger.warning(f"⚠️  AI评估生成失败: {e}")
            print("⚠️  AI评估暂时不可用")
    
    print("\n" + "="*60)


    if config.system.enable_trace:
        logger.info(f"\n💾 保存追踪信息到: {config.system.save_trace}")
        config.system.save_trace.parent.mkdir(parents=True, exist_ok=True)
        config.system.save_trace.write_text(
            json.dumps(final_state.audit_trail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"  ✅ Trace保存成功: {config.system.save_trace}")
        print(f"\nTrace saved to: {config.system.save_trace}")


if __name__ == "__main__":
    app()
