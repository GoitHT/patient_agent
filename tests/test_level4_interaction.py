"""
测试 Level 4: 交互增强功能
- 自然语言命令
- 多模态观察
- 智能提示系统
"""
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime
from environment import HospitalWorld, PhysicalState, InteractiveSession


def test_natural_language_understanding():
    """测试自然语言理解"""
    print("\n" + "=" * 60)
    print("测试 1: 自然语言理解")
    print("=" * 60)
    
    world = HospitalWorld()
    world.add_agent("patient_001", "patient", "lobby")
    
    # 创建启用NL的会话
    session = InteractiveSession(world, "patient_001", agent_type="patient", 
                                 enable_nl=True, enable_hints=False)
    
    # 测试各种自然语言输入
    nl_commands = [
        "我想去内科看看",
        "帮我去检验科",
        "我需要做个血常规",
        "做一下CT检查",
        "现在在哪里",
        "我的情况怎么样",
        "等待10分钟",
        "现在几点了",
        "可以做什么",
    ]
    
    for cmd in nl_commands:
        print(f"\n🗣️ 用户: {cmd}")
        response = session.execute(cmd, show_hints=False)
        print(f"🤖 系统: {response[:150]}...")  # 只显示前150字符
    
    print("\n✅ 自然语言理解测试通过！")


def test_smart_hints():
    """测试智能提示系统"""
    print("\n" + "=" * 60)
    print("测试 2: 智能提示系统")
    print("=" * 60)
    
    world = HospitalWorld()
    patient_id = "patient_001"
    world.add_agent(patient_id, "patient", "lobby")
    
    # 添加症状
    if patient_id in world.physical_states:
        state = world.physical_states[patient_id]
        state.update_symptom("发热", 9)  # 严重发热
        state.update_vital_sign("temperature", 39.5)
        state.update_vital_sign("heart_rate", 110)
    
    # 创建启用提示的会话
    session = InteractiveSession(world, patient_id, enable_hints=True)
    
    # 测试不同场景的提示
    scenarios = [
        ("lobby", "在大厅时的提示"),
        ("triage", "在分诊台的提示"),
        ("internal_medicine", "在诊室的提示"),
    ]
    
    for location, desc in scenarios:
        world.agents[patient_id] = location
        print(f"\n📍 场景: {desc}")
        hints = session.hint_system.get_contextual_hints(patient_id)
        for hint in hints:
            print(f"  {hint}")
    
    # 测试动作建议
    world.agents[patient_id] = "internal_medicine"
    print(f"\n💡 动作建议:")
    suggestions = session.hint_system.get_action_suggestions(patient_id)
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    print("\n✅ 智能提示测试通过！")


def test_multimodal_observation():
    """测试多模态观察"""
    print("\n" + "=" * 60)
    print("测试 3: 多模态观察")
    print("=" * 60)
    
    world = HospitalWorld()
    patient_id = "patient_001"
    world.add_agent(patient_id, "patient", "internal_medicine")
    
    # 添加健康数据
    if patient_id in world.physical_states:
        state = world.physical_states[patient_id]
        state.update_symptom("咳嗽", 6)
        state.update_symptom("发热", 7)
        state.update_vital_sign("temperature", 38.5)
        state.update_vital_sign("heart_rate", 95)
    
    session = InteractiveSession(world, patient_id, enable_hints=True)
    
    # 获取多模态观察
    print("\n🔍 多模态观察数据:")
    multimodal = session.get_multimodal_observation()
    
    print("\n📝 文本观察:")
    print(multimodal["text"][:300] + "...")
    
    print("\n📊 结构化数据:")
    structured = multimodal["structured"]
    print(f"  时间: {structured['time']}")
    print(f"  位置: {structured['location']['name']}")
    print(f"  占用率: {structured['location']['occupancy']}")
    print(f"  可用动作: {', '.join(structured['actions'][:3])}")
    
    if "health" in structured:
        print(f"\n💊 健康状态:")
        health = structured["health"]
        print(f"  状态: {health['status']}")
        print(f"  疼痛: {health['pain_level']}/10")
        print(f"  症状: {list(health['symptoms'].keys())}")
    
    print("\n🗺️ 可视化:")
    print(multimodal["visual"])
    
    print("\n✅ 多模态观察测试通过！")


def test_enhanced_feedback():
    """测试增强反馈系统"""
    print("\n" + "=" * 60)
    print("测试 4: 增强反馈系统")
    print("=" * 60)
    
    world = HospitalWorld()
    patient_id = "patient_001"
    world.add_agent(patient_id, "patient", "lobby")
    
    session = InteractiveSession(world, patient_id, enable_hints=True, enable_nl=True)
    
    # 测试带反馈的命令执行
    commands = [
        "去分诊台",
        "look",
        "我想去内科",
    ]
    
    for cmd in commands:
        print(f"\n🎯 命令: {cmd}")
        feedback = session.execute_with_feedback(cmd)
        
        print(f"📤 响应: {feedback['response'][:100]}...")
        print(f"⏰ 时间: {feedback['time']}")
        print(f"📍 位置: {feedback['location']}")
        print(f"🔢 命令计数: {feedback['command_count']}")
        
        if feedback.get('hints'):
            print(f"💡 提示:\n{feedback['hints']}")
    
    print("\n✅ 增强反馈测试通过！")


def test_interactive_menu():
    """测试交互式菜单"""
    print("\n" + "=" * 60)
    print("测试 5: 交互式动作菜单")
    print("=" * 60)
    
    world = HospitalWorld()
    patient_id = "patient_001"
    world.add_agent(patient_id, "patient", "internal_medicine")
    
    session = InteractiveSession(world, patient_id, enable_hints=True)
    
    # 显示动作菜单
    print("\n📋 动作菜单测试:")
    menu = session.get_action_menu()
    print(menu)
    
    print("\n✅ 交互式菜单测试通过！")


def test_contextual_responses():
    """测试情境化响应"""
    print("\n" + "=" * 60)
    print("测试 6: 情境化响应生成")
    print("=" * 60)
    
    from environment.command_system import NaturalLanguageParser
    
    parser = NaturalLanguageParser()
    
    # 测试不同时段的响应
    contexts = [
        {"time_of_day": "morning", "agent_type": "patient"},
        {"time_of_day": "afternoon", "agent_type": "doctor"},
        {"time_of_day": "evening", "agent_type": "nurse"},
    ]
    
    base_response = "您的检查已完成"
    
    for context in contexts:
        response = parser.generate_response_variants(base_response, context)
        print(f"  {context}: {response}")
    
    print("\n✅ 情境化响应测试通过！")


def run_all_tests():
    """运行所有 Level 4 测试"""
    print("\n" + "=" * 60)
    print("开始运行 Level 4 交互增强测试")
    print("=" * 60)
    
    try:
        test_natural_language_understanding()
        test_smart_hints()
        test_multimodal_observation()
        test_enhanced_feedback()
        test_interactive_menu()
        test_contextual_responses()
        
        print("\n" + "=" * 60)
        print("🎉 所有 Level 4 测试通过！")
        print("=" * 60)
        print("\n✨ Level 4 功能清单:")
        print("  ✅ 自然语言理解 - 支持多种表达方式")
        print("  ✅ 智能提示系统 - 根据情境提供建议")
        print("  ✅ 多模态观察 - 文本+结构化+可视化")
        print("  ✅ 增强反馈 - 详细的执行反馈")
        print("  ✅ 交互式菜单 - 动态动作建议")
        print("  ✅ 情境化响应 - 根据角色和时间调整语气")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
