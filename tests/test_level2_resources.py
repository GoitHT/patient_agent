"""
Level 2 测试: 设备与资源管理
测试优先级队列、设备占用、资源竞争等功能
"""
from datetime import datetime
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.hospital_world import HospitalWorld


def test_priority_queue():
    """测试优先级队列"""
    print("\n" + "="*60)
    print("测试 1: 优先级队列机制")
    print("="*60)
    
    world = HospitalWorld(start_time=datetime(2026, 1, 9, 9, 0))
    
    # 添加3个患者，移动到影像科（需要先经过诊室）
    patients = ["patient_001", "patient_002", "patient_003"]
    for pid in patients:
        world.add_agent(pid, "patient", "lobby")
        # 先去内科诊室
        success, _ = world.move_agent(pid, "internal_medicine")
        assert success, f"{pid} 移动到内科失败"
        # 再去影像科
        success, _ = world.move_agent(pid, "imaging")
        assert success, f"{pid} 移动到影像科失败"
    
    # 患者1先占用CT机
    success, msg = world.perform_exam("patient_001", "ct", priority=5)
    assert success, f"患者1 CT检查失败: {msg}"
    print(f"✅ 患者1 开始CT检查（优先级5）: {msg}")
    
    # 患者2加入队列（普通优先级）
    success, msg = world.perform_exam("patient_002", "ct", priority=7)
    assert not success, "患者2应该排队"
    print(f"✅ 患者2 加入队列（优先级7）: {msg}")
    
    # 患者3加入队列（高优先级 - 急诊）
    success, msg = world.perform_exam("patient_003", "ct", priority=2)
    assert not success, "患者3应该排队"
    print(f"✅ 患者3 加入队列（优先级2 - 急诊）: {msg}")
    
    # 检查队列顺序
    ct_machine = next(eq for eq in world.equipment.values() if eq.exam_type == "ct")
    assert len(ct_machine.queue) == 2, "队列应该有2人"
    assert ct_machine.queue[0].patient_id == "patient_003", "高优先级患者应该在前面"
    assert ct_machine.queue[1].patient_id == "patient_002", "低优先级患者应该在后面"
    print(f"✅ 队列顺序正确: {[entry.patient_id for entry in ct_machine.queue]}")
    
    # 推进时间让患者1完成检查
    world.advance_time(35)  # CT需要30分钟 + 一些余量
    
    # 检查是否自动开始患者3的检查
    assert ct_machine.current_patient == "patient_003", "应该自动开始高优先级患者的检查"
    print(f"✅ 自动开始下一个检查: {ct_machine.current_patient}")
    
    print("\n✅ 优先级队列测试通过！\n")


def test_equipment_status_and_maintenance():
    """测试设备状态管理和维护"""
    print("="*60)
    print("测试 2: 设备状态管理和维护")
    print("="*60)
    
    world = HospitalWorld(start_time=datetime(2026, 1, 9, 10, 0))
    
    # 获取设备状态
    status_list = world.get_equipment_status(exam_type="ct")
    print(f"✅ CT设备数量: {len(status_list)}")
    for status in status_list:
        print(f"  - {status['name']}: {status['status']}, 使用率: {status['daily_usage']}")
    
    # 设置设备维护
    ct_machine = next(eq for eq in world.equipment.values() if eq.exam_type == "ct")
    ct_machine.set_maintenance(world.current_time, duration_minutes=30)
    assert ct_machine.status == "maintenance", "设备应该处于维护状态"
    print(f"✅ 设置 {ct_machine.name} 维护状态，持续30分钟")
    
    # 尝试使用维护中的设备
    world.add_agent("patient_001", "patient", "imaging")
    success, msg = world.perform_exam("patient_001", "ct")
    assert not success, "维护中的设备不应该可用"
    print(f"✅ 正确阻止使用维护中的设备: {msg}")
    
    # 推进时间到维护结束
    world.advance_time(35)
    assert ct_machine.status == "available", "维护应该已完成"
    print(f"✅ 维护完成，设备恢复可用")
    
    # 现在应该可以使用
    success, msg = world.perform_exam("patient_001", "ct")
    assert success, f"维护后应该可以使用: {msg}"
    print(f"✅ 维护后成功使用设备: {msg}")
    
    print("\n✅ 设备状态管理测试通过！\n")


def test_resource_competition():
    """测试资源竞争"""
    print("="*60)
    print("测试 3: 资源竞争处理")
    print("="*60)
    
    world = HospitalWorld(start_time=datetime(2026, 1, 9, 11, 0))
    
    # 创建多个患者竞争同一设备
    num_patients = 5
    for i in range(num_patients):
        pid = f"patient_{i:03d}"
        world.add_agent(pid, "patient", "lobby")
        world.move_agent(pid, "imaging")
    
    # 所有患者都想做X光检查
    xray_machine = next(eq for eq in world.equipment.values() if eq.exam_type == "xray")
    print(f"\n📊 设备信息: {xray_machine.name} (时长: {xray_machine.duration_minutes}分钟)")
    
    for i in range(num_patients):
        pid = f"patient_{i:03d}"
        priority = 5 if i < 3 else 3  # 后两个是高优先级
        success, msg = world.perform_exam(pid, "xray", priority=priority)
        status = "✅ 开始检查" if success else f"⏳ 加入队列"
        print(f"  {status} - {pid} (优先级{priority}): {msg}")
    
    # 检查队列
    print(f"\n📋 当前队列状态:")
    print(f"  - 正在检查: {xray_machine.current_patient}")
    print(f"  - 队列人数: {len(xray_machine.queue)}")
    print(f"  - 队列顺序: {[f'{e.patient_id}(P{e.priority})' for e in xray_machine.queue]}")
    
    # 生成资源竞争报告
    report = world.get_resource_competition_report()
    print(f"\n📊 资源竞争报告:")
    print(f"  - 总设备数: {report['total_equipment']}")
    print(f"  - 使用中: {report['busy_equipment']}")
    print(f"  - 总排队人数: {report['total_queue_length']}")
    
    if report['hotspots']:
        print(f"\n🔥 热点设备:")
        for hotspot in report['hotspots']:
            print(f"  - {hotspot['equipment']}: 排队{hotspot['queue']}人, 等待{hotspot['wait_time']}分钟")
    
    # 查找最佳设备
    best = world.find_best_equipment("xray")
    if best:
        print(f"\n🎯 推荐设备: {best['name']}")
        print(f"  - 位置: {best['location_name']}")
        print(f"  - 等待时间: {best['wait_time']}分钟")
        print(f"  - 预计开始: {best['estimated_start']}")
    
    print("\n✅ 资源竞争测试通过！\n")


def test_equipment_reservation():
    """测试设备预约系统"""
    print("="*60)
    print("测试 4: 设备预约系统")
    print("="*60)
    
    world = HospitalWorld(start_time=datetime(2026, 1, 9, 9, 0))
    
    # 患者1预约10:00的MRI
    success, msg = world.reserve_equipment("patient_001", "mri", "10:00")
    assert success, f"预约失败: {msg}"
    print(f"✅ 患者1预约成功: {msg}")
    
    # 患者2尝试预约同一时间槽
    success, msg = world.reserve_equipment("patient_002", "mri", "10:00")
    assert not success, "不应该能预约已被占用的时间槽"
    print(f"✅ 正确阻止重复预约: {msg}")
    
    # 患者2预约10:30
    success, msg = world.reserve_equipment("patient_002", "mri", "10:30")
    assert success, f"预约失败: {msg}"
    print(f"✅ 患者2预约成功: {msg}")
    
    # 检查预约状态
    mri_machine = next(eq for eq in world.equipment.values() if eq.exam_type == "mri")
    print(f"\n📅 MRI预约情况:")
    for time_slot, patient_id in mri_machine.reservation_slots.items():
        print(f"  - {time_slot}: {patient_id}")
    
    # 取消患者1的预约
    canceled = world.cancel_equipment_reservation("patient_001")
    assert canceled, "应该成功取消预约"
    print(f"\n✅ 成功取消患者1的预约")
    
    assert "10:00" not in mri_machine.reservation_slots, "时间槽应该被释放"
    print(f"✅ 时间槽已释放")
    
    print("\n✅ 设备预约测试通过！\n")


def test_daily_usage_limit():
    """测试每日使用限制"""
    print("="*60)
    print("测试 5: 每日使用限制")
    print("="*60)
    
    world = HospitalWorld(start_time=datetime(2026, 1, 9, 9, 0))
    
    # 获取一个设备并设置较低的每日限制
    ecg_machine = next(eq for eq in world.equipment.values() if eq.exam_type == "ecg")
    ecg_machine.max_daily_usage = 3  # 设置为只能用3次
    print(f"📋 设置 {ecg_machine.name} 每日最大使用次数: {ecg_machine.max_daily_usage}")
    
    # 添加患者到诊室
    for i in range(5):
        pid = f"patient_{i:03d}"
        world.add_agent(pid, "patient", ecg_machine.location_id)
    
    # 尝试使用4次
    successes = 0
    for i in range(4):
        pid = f"patient_{i:03d}"
        success, msg = world.perform_exam(pid, "ecg")
        if success:
            successes += 1
            print(f"✅ 第{i+1}次使用成功: {msg}")
            # 完成检查
            world.advance_time(15)
        else:
            print(f"❌ 第{i+1}次使用失败: {msg}")
    
    assert successes == 3, f"应该只能成功使用3次，实际: {successes}"
    print(f"\n✅ 正确实施每日使用限制: {successes}/3")
    
    # 检查使用计数
    assert ecg_machine.daily_usage_count == 3, "使用计数应该是3"
    print(f"✅ 使用计数正确: {ecg_machine.daily_usage_count}")
    
    # 跨天后应该重置
    world.current_time = datetime(2026, 1, 10, 9, 0)
    world._reset_daily_counters()
    assert ecg_machine.daily_usage_count == 0, "跨天后应该重置"
    print(f"✅ 跨天后计数重置: {ecg_machine.daily_usage_count}")
    
    print("\n✅ 每日使用限制测试通过！\n")


def run_all_tests():
    """运行所有Level 2测试"""
    print("\n" + "="*60)
    print("Level 2: 设备与资源管理 - 测试套件")
    print("="*60)
    
    try:
        test_priority_queue()
        test_equipment_status_and_maintenance()
        test_resource_competition()
        test_equipment_reservation()
        test_daily_usage_limit()
        
        print("="*60)
        print("🎉 所有 Level 2 测试通过！")
        print("="*60)
        print("\n✅ 实现功能:")
        print("  1. ✅ 优先级队列机制")
        print("  2. ✅ 设备占用和自动推进")
        print("  3. ✅ 资源竞争处理")
        print("  4. ✅ 设备维护状态")
        print("  5. ✅ 设备预约系统")
        print("  6. ✅ 每日使用限制")
        print("  7. ✅ 资源竞争报告")
        print()
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
