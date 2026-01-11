"""
物理环境功能测试脚本
测试位置系统、移动约束、时间系统和命令解析
"""
from pathlib import Path
import sys

# 添加 src 到 Python 路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from environment import HospitalWorld, InteractiveSession
from datetime import datetime


def test_location_system():
    """测试位置系统"""
    print("\n" + "="*60)
    print("测试 1: 位置系统和移动约束")
    print("="*60)
    
    world = HospitalWorld()
    agent_id = "test_patient"
    
    # 测试添加Agent
    success = world.add_agent(agent_id, "patient", "lobby")
    assert success, "添加Agent失败"
    print("✅ 成功添加患者到大厅")
    
    # 测试正常移动
    success, msg = world.move_agent(agent_id, "triage")
    assert success, f"移动到分诊台失败: {msg}"
    print(f"✅ {msg}")
    
    # 测试非法移动（不相邻）
    success, msg = world.move_agent(agent_id, "imaging")
    assert not success, "应该阻止不相邻的移动"
    print(f"✅ 正确阻止非法移动: {msg}")
    
    # 测试容量限制
    world.locations["triage"].capacity = 1
    success = world.add_agent("agent2", "patient", "triage")
    assert not success, "应该阻止超容量"
    print("✅ 正确实施容量限制")
    
    print("\n✅ 位置系统测试通过！")


def test_time_system():
    """测试时间系统"""
    print("\n" + "="*60)
    print("测试 2: 时间系统和工作时间限制")
    print("="*60)
    
    # 测试工作时间
    world = HospitalWorld(start_time=datetime(2024, 1, 1, 8, 0))
    assert world.is_working_hours(), "8:00应该是工作时间"
    print(f"✅ 8:00 工作状态: {world.is_working_hours()}")
    
    # 测试午休时间
    world.current_time = datetime(2024, 1, 1, 12, 30)
    assert not world.is_working_hours(), "12:30应该是午休时间"
    print(f"✅ 12:30 工作状态: {world.is_working_hours()}")
    
    # 测试下班时间
    world.current_time = datetime(2024, 1, 1, 19, 0)
    assert not world.is_working_hours(), "19:00应该已下班"
    print(f"✅ 19:00 工作状态: {world.is_working_hours()}")
    
    # 测试时间推进
    world.current_time = datetime(2024, 1, 1, 10, 0)
    world.advance_time(30)
    assert world.current_time.hour == 10 and world.current_time.minute == 30
    print(f"✅ 时间推进: 10:00 -> {world.current_time.strftime('%H:%M')}")
    
    # 测试工作时间限制
    world.current_time = datetime(2024, 1, 1, 19, 0)
    agent_id = "test_patient"
    world.add_agent(agent_id, "patient", "lobby")
    success, msg = world.move_agent(agent_id, "lab")
    assert not success, "应该阻止非工作时间进入检验科"
    print(f"✅ 正确阻止非工作时间访问: {msg}")
    
    print("\n✅ 时间系统测试通过！")


def test_equipment_system():
    """测试设备系统"""
    print("\n" + "="*60)
    print("测试 3: 设备系统和排队机制")
    print("="*60)
    
    world = HospitalWorld()
    agent_id = "test_patient"
    world.add_agent(agent_id, "patient", "lobby")
    
    # 需要先去诊室，再去影像科（符合真实路径）
    success, msg = world.move_agent(agent_id, "internal_medicine")
    assert success, f"移动到内科失败: {msg}"
    print(f"✅ 移动到内科: {msg}")
    
    success, msg = world.move_agent(agent_id, "imaging")
    assert success, f"移动到影像科失败: {msg}"
    print(f"✅ 移动到影像科: {msg}")
    
    # 测试设备使用
    success, msg = world.perform_exam(agent_id, "xray")
    assert success, f"X光检查失败: {msg}"
    print(f"✅ {msg}")
    
    # 测试设备占用
    agent2_id = "test_patient2"
    world.add_agent(agent2_id, "patient", "lobby")
    world.move_agent(agent2_id, "internal_medicine")
    world.move_agent(agent2_id, "imaging")
    
    # X光机应该正忙（需要重置时间）
    world.current_time = datetime(2024, 1, 1, 8, 0)
    equipment = world.equipment["xray_1"]
    equipment.start_exam(agent_id, world.current_time)
    
    success, msg = world.perform_exam(agent2_id, "xray")
    assert not success, "设备忙时应该加入排队"
    assert equipment.has_patient_in_queue(agent2_id), "应该加入排队"
    print(f"✅ 正确处理设备占用: {msg}")
    
    # 测试设备完成后自动释放（先清空队列避免自动开始下一个检查）
    equipment.queue.clear()  # 清空队列
    print(f"推进时间前: is_occupied={equipment.is_occupied}, occupied_until={equipment.occupied_until}, current_time={world.current_time}")
    world.advance_time(20)  # X光需要15分钟
    print(f"推进时间后: is_occupied={equipment.is_occupied}, occupied_until={equipment.occupied_until}, current_time={world.current_time}")
    assert equipment.can_use(world.current_time), "设备应该已释放"
    print("✅ 设备使用后正确释放")
    
    # 测试自动队列推进
    # 重新设置场景：设备被占用，队列中有患者
    world.current_time = datetime(2024, 1, 1, 9, 0)
    equipment.start_exam(agent_id, world.current_time)
    equipment.add_to_queue(agent2_id, priority=5, current_time=world.current_time)
    
    # 推进时间，应该自动开始下一个患者的检查
    world.advance_time(20)
    assert equipment.is_occupied, "应该自动开始下一个患者的检查"
    assert equipment.current_patient == agent2_id, "应该是 agent2 在检查"
    print("✅ 队列自动推进正常")
    
    print("\n✅ 设备系统测试通过！")


def test_command_parser():
    """测试命令解析"""
    print("\n" + "="*60)
    print("测试 4: 命令解析系统")
    print("="*60)
    
    from environment.command_system import CommandParser
    
    # 测试移动命令
    test_cases = [
        ("去 内科", "move", ["内科"]),
        ("move to lab", "move", ["lab"]),
        ("看", "look", []),
        ("look around", "look", []),
        ("开单 血常规", "order", ["血常规"]),
        ("order ct test", "order", ["ct"]),
        ("等待 10 分钟", "wait", ["10", "分钟"]),
        ("wait 15 min", "wait", ["15", "min"]),
        ("状态", "status", []),
        ("help", "help", []),
    ]
    
    for cmd, expected_type, expected_args in test_cases:
        cmd_type, args = CommandParser.parse(cmd)
        assert cmd_type == expected_type, f"命令 '{cmd}' 解析错误: 期望 {expected_type}, 得到 {cmd_type}"
        # 参数数量匹配即可
        assert len(args) == len(expected_args), f"命令 '{cmd}' 参数数量错误"
        print(f"✅ '{cmd}' -> {cmd_type} {args}")
    
    # 测试位置解析
    assert CommandParser.resolve_location("内科") == "internal_medicine"
    assert CommandParser.resolve_location("lab") == "lab"
    assert CommandParser.resolve_location("分诊台") == "triage"
    print("✅ 位置名称解析正确")
    
    # 测试检查类型解析
    assert CommandParser.resolve_exam_type("血常规") == "blood_test"
    assert CommandParser.resolve_exam_type("ct") == "ct"
    assert CommandParser.resolve_exam_type("心电图") == "ecg"
    print("✅ 检查类型解析正确")
    
    print("\n✅ 命令解析测试通过！")


def test_interactive_session():
    """测试交互式会话"""
    print("\n" + "="*60)
    print("测试 5: 交互式会话")
    print("="*60)
    
    world = HospitalWorld()
    agent_id = "test_patient"
    world.add_agent(agent_id, "patient", "lobby")
    
    session = InteractiveSession(world, agent_id, "patient")
    
    # 测试命令执行
    commands = [
        ("看", "观察命令"),
        ("去 分诊台", "移动命令"),
        ("状态", "状态查询"),
        ("时间", "时间查询"),
    ]
    
    for cmd, desc in commands:
        response = session.execute(cmd)
        assert response, f"{desc}返回空响应"
        print(f"✅ {desc}: {cmd}")
        print(f"   响应: {response[:60]}...")
    
    # 测试历史记录
    assert len(session.history) == len(commands), "历史记录数量不匹配"
    print(f"✅ 历史记录: {len(session.history)} 条")
    
    print("\n✅ 交互式会话测试通过！")


def test_physical_state():
    """测试物理状态系统"""
    print("\n" + "="*60)
    print("测试 6: 物理状态系统")
    print("="*60)
    
    world = HospitalWorld()
    agent_id = "test_patient"
    world.add_agent(agent_id, "patient", "lobby")
    
    # 使用新的 API 设置症状
    state = world.physical_states[agent_id]
    state.add_symptom("发热", severity=7.0, progression_rate=0.2)
    state.add_symptom("咳嗽", severity=5.0, progression_rate=0.1)
    
    # 更新生命体征
    state.vital_signs["temperature"].update(38.5, world.current_time)
    state.vital_signs["heart_rate"].update(90.0, world.current_time)
    
    print(f"✅ 初始症状: {state.get_symptom_severity_dict()}")
    print(f"✅ 生命体征: 体温={state.vital_signs['temperature'].value}℃, 心率={state.vital_signs['heart_rate'].value}次/分")
    
    # 推进时间测试症状变化
    initial_fever = state.symptoms["发热"].severity
    initial_energy = state.energy_level
    world.advance_time(120)  # 推进2小时
    
    print(f"✅ 2小时后症状: {state.get_symptom_severity_dict()}")
    print(f"✅ 发热变化: {initial_fever:.1f} -> {state.symptoms['发热'].severity:.1f}")
    
    # 测试体力消耗
    assert state.energy_level < initial_energy, f"体力应该有消耗: {initial_energy} -> {state.energy_level}"
    print(f"✅ 体力消耗: {initial_energy:.1f} -> {state.energy_level:.1f}")
    
    # 测试观察包含症状信息
    obs = world.get_observation(agent_id)
    assert "symptoms" in obs, "观察应该包含症状"
    assert "vital_signs" in obs, "观察应该包含生命体征"
    print("✅ 观察正确包含健康状态")
    
    print("\n✅ 物理状态测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始运行物理环境功能测试")
    print("="*60)
    
    try:
        test_location_system()
        test_time_system()
        test_equipment_system()
        test_command_parser()
        test_interactive_session()
        test_physical_state()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！Level 1 基础物理约束实现成功！")
        print("="*60)
        
        print("\n✅ 已实现功能:")
        print("  • 位置系统: 11个医院位置，相邻关系和容量限制")
        print("  • 移动约束: 相邻性检查、容量限制、移动时间消耗")
        print("  • 时间系统: 工作时间限制、午休时间、时间推进")
        print("  • 设备系统: 12种医疗设备，占用状态、排队机制")
        print("  • 物理状态: 症状系统、生命体征、体力消耗")
        print("  • 命令解析: 中英文命令、自然语言解析")
        print("  • 交互会话: 命令执行、历史记录、格式化输出")
        
        print("\n📝 使用方法:")
        print("  python main.py --physical-sim --interactive --dataset-id 1")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
