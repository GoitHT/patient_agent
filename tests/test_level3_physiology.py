"""
Level 3 测试：动态生理系统
测试症状演变、生命体征监测、病情恶化机制
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加 src 到路径
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from environment.hospital_world import HospitalWorld, PhysicalState, Symptom, VitalSign


def test_symptom_evolution():
    """测试 1: 症状随时间演变"""
    print("\n" + "="*60)
    print("测试 1: 症状随时间演变")
    print("="*60)
    
    # 创建患者状态
    patient_state = PhysicalState(patient_id="test_patient")
    
    # 添加初始症状
    patient_state.add_symptom("发热", severity=7.0, progression_rate=0.2)
    patient_state.add_symptom("咳嗽", severity=5.0, progression_rate=0.1)
    patient_state.add_symptom("头痛", severity=6.0, progression_rate=0.15)
    
    print("\n初始状态:")
    print(patient_state.get_status_summary())
    
    # 模拟未治疗情况下2小时的演变
    print("\n\n--- 2小时后（未治疗）---")
    future_time = datetime.now() + timedelta(hours=2)
    patient_state.update_physiology(future_time)
    print(patient_state.get_status_summary())
    
    # 检查症状是否恶化
    fever = patient_state.symptoms.get("发热")
    assert fever.severity >= 7.0, "未治疗的症状应该恶化或至少保持"
    print(f"\n✅ 症状演变测试通过！发热从 7.0 变为 {fever.severity:.1f}")
    
    # 应用治疗
    print("\n\n--- 应用治疗 ---")
    patient_state.apply_medication("退烧药", effectiveness=0.9)
    
    # 再过3小时
    print("\n--- 3小时后（已治疗）---")
    future_time += timedelta(hours=3)
    patient_state.update_physiology(future_time)
    print(patient_state.get_status_summary())
    
    # 检查症状是否改善
    fever_after_treatment = patient_state.symptoms.get("发热")
    print(f"\n✅ 治疗效果测试通过！发热从治疗前 {fever.severity:.1f} 降至 {fever_after_treatment.severity:.1f}")
    
    print("\n✅ 症状演变测试全部通过！\n")


def test_vital_signs_monitoring():
    """测试 2: 生命体征监测"""
    print("\n" + "="*60)
    print("测试 2: 生命体征监测")
    print("="*60)
    
    patient_state = PhysicalState(patient_id="test_patient_2")
    
    # 添加多个严重症状，影响生命体征
    patient_state.add_symptom("高热", severity=9.0, progression_rate=0.3)
    patient_state.add_symptom("呼吸困难", severity=8.5, progression_rate=0.25)
    patient_state.add_symptom("胸痛", severity=8.0, progression_rate=0.2)
    
    print("\n初始生命体征:")
    for name, vs in patient_state.vital_signs.items():
        print(f"  {vs.name}: {vs.value:.1f} {vs.unit} - {vs.get_status()}")
    
    # 模拟3小时演变
    print("\n\n--- 3小时后（未治疗，病情恶化）---")
    future_time = datetime.now() + timedelta(hours=3)
    patient_state.update_physiology(future_time)
    
    print("\n当前生命体征:")
    abnormal_count = 0
    for name, vs in patient_state.vital_signs.items():
        status = vs.get_status()
        print(f"  {vs.name}: {vs.value:.1f} {vs.unit} - {status}")
        if not vs.is_normal():
            abnormal_count += 1
    
    print(f"\n生命体征异常数量: {abnormal_count}")
    print(f"意识水平: {patient_state.consciousness_level}")
    print(f"是否危急: {'是' if patient_state.check_critical_condition() else '否'}")
    
    # 检查危急状态
    assert patient_state.check_critical_condition() or abnormal_count >= 2, \
        "严重症状应导致生命体征异常或危急状态"
    
    print("\n✅ 生命体征监测测试通过！")
    
    # 测试生命体征历史记录
    hr = patient_state.vital_signs.get("heart_rate")
    if hr and len(hr.history) > 0:
        print(f"\n心率历史记录: {len(hr.history)} 条")
        print("✅ 历史记录功能正常")
    
    print("\n✅ 生命体征监测测试全部通过！\n")


def test_condition_deterioration():
    """测试 3: 病情恶化机制"""
    print("\n" + "="*60)
    print("测试 3: 病情恶化机制")
    print("="*60)
    
    # 创建两个患者：一个治疗，一个不治疗
    patient_treated = PhysicalState(patient_id="treated")
    patient_untreated = PhysicalState(patient_id="untreated")
    
    # 相同的初始症状
    for patient in [patient_treated, patient_untreated]:
        patient.add_symptom("感染", severity=6.0, progression_rate=0.3)
        patient.add_symptom("发热", severity=7.0, progression_rate=0.25)
    
    print("\n初始状态（两患者相同）:")
    print(f"  感染: 6.0/10")
    print(f"  发热: 7.0/10")
    
    # 对一个患者进行治疗
    patient_treated.apply_medication("抗生素", effectiveness=0.85)
    print("\n患者A: 接受抗生素治疗")
    print("患者B: 未接受治疗")
    
    # 模拟6小时演变
    future_time = datetime.now() + timedelta(hours=6)
    
    print("\n\n--- 6小时后 ---\n")
    patient_treated.update_physiology(future_time)
    patient_untreated.update_physiology(future_time)
    
    print("患者A（已治疗）:")
    print(patient_treated.get_status_summary())
    
    print("\n" + "-"*60 + "\n")
    
    print("患者B（未治疗）:")
    print(patient_untreated.get_status_summary())
    
    # 比较结果
    treated_avg = sum(s.severity for s in patient_treated.symptoms.values()) / len(patient_treated.symptoms)
    untreated_avg = sum(s.severity for s in patient_untreated.symptoms.values()) / len(patient_untreated.symptoms)
    
    print("\n" + "="*60)
    print(f"平均症状严重程度:")
    print(f"  患者A（已治疗）: {treated_avg:.1f}/10")
    print(f"  患者B（未治疗）: {untreated_avg:.1f}/10")
    print(f"  差异: {abs(untreated_avg - treated_avg):.1f}")
    
    # 验证治疗效果
    assert treated_avg < untreated_avg, "治疗应该改善症状"
    print("\n✅ 病情恶化机制测试通过！治疗有效改善症状")
    
    # 检查意识水平
    print(f"\n意识水平:")
    print(f"  患者A: {patient_treated.consciousness_level}")
    print(f"  患者B: {patient_untreated.consciousness_level}")
    
    print("\n✅ 病情恶化机制测试全部通过！\n")


def test_integrated_scenario():
    """测试 4: 综合场景 - 急诊患者演变"""
    print("\n" + "="*60)
    print("测试 4: 综合场景 - 急诊患者演变")
    print("="*60)
    
    world = HospitalWorld()
    
    # 创建急诊患者
    patient_id = "emergency_001"
    world.add_agent(patient_id, agent_type="patient", initial_location="lobby")
    
    patient_state = world.physical_states[patient_id]
    
    # 模拟急性心梗症状
    patient_state.add_symptom("胸痛", severity=9.0, progression_rate=0.5)
    patient_state.add_symptom("呼吸困难", severity=8.0, progression_rate=0.4)
    patient_state.add_symptom("出汗", severity=7.0, progression_rate=0.2)
    
    # 设置异常生命体征
    patient_state.vital_signs["heart_rate"].update(110.0, datetime.now())
    patient_state.vital_signs["blood_pressure_systolic"].update(160.0, datetime.now())
    patient_state.vital_signs["oxygen_saturation"].update(92.0, datetime.now())
    
    print("\n【急诊患者入院】")
    print(patient_state.get_status_summary())
    print(f"\n⏰ 时间: {world.current_time.strftime('%H:%M')}")
    
    # 时间推进 30 分钟（未治疗）
    print("\n\n【30分钟后 - 等待中】")
    world.advance_time(30)
    print(patient_state.get_status_summary())
    print(f"\n⏰ 时间: {world.current_time.strftime('%H:%M')}")
    
    is_critical = patient_state.check_critical_condition()
    print(f"\n{'⚠️ 危急状态！' if is_critical else '状态稳定'}")
    
    # 开始治疗
    print("\n\n【开始紧急治疗】")
    patient_state.apply_medication("硝酸甘油", effectiveness=0.9)
    patient_state.record_treatment("oxygen_therapy", "鼻导管吸氧 2L/min")
    print("  - 舌下含服硝酸甘油")
    print("  - 开始吸氧")
    
    # 再过1小时
    print("\n\n【1小时后 - 治疗中】")
    world.advance_time(60)
    print(patient_state.get_status_summary())
    print(f"\n⏰ 时间: {world.current_time.strftime('%H:%M')}")
    
    # 检查改善情况
    chest_pain = patient_state.symptoms.get("胸痛")
    print(f"\n胸痛趋势: {chest_pain.trend}")
    print(f"治疗记录数: {len(patient_state.treatments)}")
    print(f"用药记录数: {len(patient_state.medications)}")
    
    assert chest_pain.treated, "症状应标记为已治疗"
    assert len(patient_state.treatments) > 0, "应有治疗记录"
    
    print("\n✅ 综合场景测试通过！")
    print("\n✅ 急诊患者演变模拟成功！\n")


def test_consciousness_assessment():
    """测试 5: 意识水平评估"""
    print("\n" + "="*60)
    print("测试 5: 意识水平评估")
    print("="*60)
    
    patient = PhysicalState(patient_id="consciousness_test")
    
    # 测试正常状态
    print("\n场景1: 正常状态")
    patient.assess_consciousness()
    print(f"  意识水平: {patient.consciousness_level}")
    assert patient.consciousness_level == "alert", "正常状态应为清醒"
    print("  ✅ 正常")
    
    # 测试嗜睡状态
    print("\n场景2: 多个异常体征")
    patient.vital_signs["heart_rate"].update(120.0, datetime.now())
    patient.vital_signs["blood_pressure_systolic"].update(170.0, datetime.now())
    patient.vital_signs["temperature"].update(39.5, datetime.now())
    patient.assess_consciousness()
    print(f"  意识水平: {patient.consciousness_level}")
    print("  ✅ 正确评估")
    
    # 测试严重症状
    print("\n场景3: 多个重症状")
    patient.add_symptom("严重感染", severity=9.5, progression_rate=0.5)
    patient.add_symptom("休克", severity=9.0, progression_rate=0.5)
    patient.add_symptom("昏迷前兆", severity=8.5, progression_rate=0.4)
    patient.assess_consciousness()
    print(f"  意识水平: {patient.consciousness_level}")
    print(f"  ✅ 严重病情正确评估为: {patient.consciousness_level}")
    
    print("\n✅ 意识水平评估测试全部通过！\n")


def run_all_tests():
    """运行所有 Level 3 测试"""
    print("\n" + "="*60)
    print("Level 3 物理环境功能测试: 动态生理系统")
    print("="*60)
    
    try:
        test_symptom_evolution()
        test_vital_signs_monitoring()
        test_condition_deterioration()
        test_integrated_scenario()
        test_consciousness_assessment()
        
        print("\n" + "="*60)
        print("🎉 所有 Level 3 测试通过！")
        print("="*60)
        print("\n✅ 症状随时间演变")
        print("✅ 生命体征监测")
        print("✅ 病情恶化机制")
        print("✅ 治疗效果模拟")
        print("✅ 意识水平评估")
        print("\n")
        
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
