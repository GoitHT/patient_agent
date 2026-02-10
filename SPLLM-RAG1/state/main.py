import os
import sys

# --- 关键修复：在导入任何库之前设置环境变量 ---
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制使用离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Transformers离线模式
os.environ['HF_HOME'] = './model_cache'  # 指定HuggingFace缓存目录

# 然后导入其他库
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Adaptive_RAG import app


def run_assistant(user_question: str, patient_id: str):
    """
    调用Adaptive_RAG的工作流，生成回答
    :param user_question: 用户问题
    :param patient_id: 患者ID
    :return: 模型最终回答
    """
    # 构造初始状态
    inputs = {
        "question": user_question,
        "patient_id": patient_id,
        "messages": [],
        "documents": [],
        "loop_count": 0,
        "generation": "",
        "history_context": "",
        "score": 0.0,
        "evaluation_result": {}
    }
    print(f"\n--- 正在处理患者 [{patient_id}] 的问题 ---\n")
    final_answer = None
    # 运行流
    for output in app.stream(inputs):
        for node_name, state_update in output.items():
            print(f"节点【{node_name}】处理完成")
            # 兼容处理：有些节点更新 generation，有些更新 messages
            if "generation" in state_update:
                final_answer = state_update["generation"]
    return final_answer or "未能生成有效回答，请检查流程。"


if __name__ == "__main__":
    print("=== 医疗 RAG 问诊系统启动 ===")

    # 显示缓存路径信息
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_cache")
    print(f"📂 模型缓存路径: {cache_path}")
    print(f"📂 缓存是否存在: {os.path.exists(cache_path)}")

    while True:
        # 1. 每一轮新问诊开始前，先要求输入 ID
        p_id = input("\n[开始新问诊] 请输入患者 ID (输入 'exit' 退出系统): ").strip()
        if p_id.lower() == 'exit':
            break
        print(f"\n✅ 已连接患者 {p_id} 的病历库，现在可以开始提问。")
        # 2. 进入该病人的对话循环
        while True:
            query = input(f"\n[{p_id}] 请输入问题 (输入 '结束问诊' 保存并换人): ")
            if query == "结束问诊":
                print(f"--- 患者 {p_id} 问诊结束，记录已切片存入历史库与 CSV ---")
                # ========== 新增：问诊结束后触发高质量向量库更新 ==========
                from create_database_general import init_high_quality_qa_db

                init_high_quality_qa_db()
                break
            if not query.strip():
                continue
            # 调用助手生成回答
            answer = run_assistant(query, p_id)
            print(f"\n🤖 最终回答：\n{'-' * 30}\n{answer}\n{'-' * 30}")
    print("系统已安全关闭。")

if __name__ == "__main__":
    print("=== 医疗 RAG 问诊系统启动 ===")
    while True:
        # 1. 每一轮新问诊开始前，先要求输入 ID
        p_id = input("\n[开始新问诊] 请输入患者 ID (输入 'exit' 退出系统): ").strip()
        if p_id.lower() == 'exit':
            break
        print(f"\n✅ 已连接患者 {p_id} 的病历库，现在可以开始提问。")
        # 2. 进入该病人的对话循环
        while True:
            query = input(f"\n[{p_id}] 请输入问题 (输入 '结束问诊' 保存并换人): ")
            if query == "结束问诊":
                print(f"--- 患者 {p_id} 问诊结束，记录已切片存入历史库与 CSV ---")
                # ========== 新增：问诊结束后触发高质量向量库更新 ==========
                from create_database_general import init_high_quality_qa_db
                init_high_quality_qa_db()
                break
            if not query.strip():
                continue
            # 调用助手生成回答
            answer = run_assistant(query, p_id)
            print(f"\n🤖 最终回答：\n{'-' * 30}\n{answer}\n{'-' * 30}")
    print("系统已安全关闭。")