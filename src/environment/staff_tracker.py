"""
医护人员工作状态追踪器
在LangGraph节点中更新医护人员的物理状态
"""
from typing import Optional
from environment import HospitalWorld


class StaffTracker:
    """医护人员状态追踪器"""
    
    @staticmethod
    def update_nurse_triage(world: Optional[HospitalWorld], duration_minutes: int = 5):
        """更新护士分诊工作状态"""
        if not world or "nurse_001" not in world.physical_states:
            return
        
        nurse_state = world.physical_states["nurse_001"]
        nurse_state.add_work_load(
            task_type="triage",
            duration_minutes=duration_minutes,
            complexity=0.3  # 分诊复杂度较低
        )
        nurse_state.serve_patient()
    
    @staticmethod
    def update_doctor_consultation(world: Optional[HospitalWorld], 
                                   duration_minutes: int = 15, 
                                   complexity: float = 0.6):
        """更新医生问诊工作状态"""
        if not world or "doctor_001" not in world.physical_states:
            return
        
        doctor_state = world.physical_states["doctor_001"]
        doctor_state.add_work_load(
            task_type="consultation",
            duration_minutes=duration_minutes,
            complexity=complexity  # 问诊复杂度中等
        )
    
    @staticmethod
    def update_doctor_diagnosis(world: Optional[HospitalWorld], 
                               duration_minutes: int = 10,
                               complexity: float = 0.8):
        """更新医生诊断工作状态"""
        if not world or "doctor_001" not in world.physical_states:
            return
        
        doctor_state = world.physical_states["doctor_001"]
        doctor_state.add_work_load(
            task_type="diagnosis",
            duration_minutes=duration_minutes,
            complexity=complexity  # 诊断复杂度高
        )
        doctor_state.serve_patient()
    
    @staticmethod
    def update_lab_technician(world: Optional[HospitalWorld], 
                             test_count: int = 1,
                             duration_per_test: int = 15):
        """更新检验技师工作状态"""
        if not world or "lab_tech_001" not in world.physical_states:
            return
        
        lab_state = world.physical_states["lab_tech_001"]
        total_duration = test_count * duration_per_test
        lab_state.add_work_load(
            task_type="lab_test",
            duration_minutes=total_duration,
            complexity=0.5  # 检验复杂度中等
        )
        # 每个检验算作服务一次
        for _ in range(test_count):
            lab_state.serve_patient()
    
    @staticmethod
    def staff_rest_break(world: Optional[HospitalWorld], agent_id: str, duration_minutes: int = 10):
        """医护人员休息"""
        if not world or agent_id not in world.physical_states:
            return
        
        staff_state = world.physical_states[agent_id]
        staff_state.apply_rest(duration_minutes=duration_minutes, quality=0.7)
    
    @staticmethod
    def get_all_staff_summary(world: Optional[HospitalWorld]) -> str:
        """获取所有医护人员状态摘要"""
        if not world:
            return ""
        
        summaries = []
        
        staff_ids = {
            "nurse_001": "护士",
            "doctor_001": "医生",
            "lab_tech_001": "检验技师"
        }
        
        for agent_id, title in staff_ids.items():
            if agent_id in world.physical_states:
                state = world.physical_states[agent_id]
                summary = (
                    f"{title}: 体力{state.energy_level:.1f} "
                    f"负荷{state.work_load:.1f} "
                    f"效率{state.get_work_efficiency()*100:.0f}% "
                    f"服务{state.patients_served_today}人"
                )
                summaries.append(summary)
        
        return " | ".join(summaries) if summaries else "无医护状态"
