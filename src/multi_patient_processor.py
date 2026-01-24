"""
多患者并发处理器 - 并发执行多个患者的就诊流程
Multi-Patient Processor - Concurrent patient consultation processing

功能：
1. 并发处理多个患者
2. 流程编排（挂号→候诊→就诊→检查→复诊→离院）
3. 异步任务管理
4. 进度跟踪
"""

import concurrent.futures
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from hospital_coordinator import HospitalCoordinator, PatientStatus, ResourceStatus
from utils import get_logger

logger = get_logger("hospital_agent.multi_patient")


class PatientProcessor:
    """单个患者的处理器"""
    
    def __init__(self, coordinator: HospitalCoordinator, patient_id: str):
        self.coordinator = coordinator
        self.patient_id = patient_id
        self.logger = get_logger(f"patient.{patient_id}")
    
    def wait_for_doctor_assignment(self, timeout: int = 300) -> bool:
        """
        等待医生分配
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否成功分配
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            session = self.coordinator.get_patient(self.patient_id)
            if session and session.assigned_doctor:
                self.logger.info(f"已分配医生: {session.assigned_doctor}")
                return True
            time.sleep(0.5)
        
        self.logger.warning(f"等待医生分配超时 ({timeout}秒)")
        return False
    
    def wait_for_lab_results(self, timeout: int = 60) -> bool:
        """
        等待检验结果
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            是否完成
        """
        self.logger.info("正在等待检验结果...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            session = self.coordinator.get_patient(self.patient_id)
            if session and session.lab_results_ready:
                self.logger.info("检验结果已就绪")
                return True
            time.sleep(1)
        
        self.logger.warning(f"等待检验结果超时 ({timeout}秒)")
        return False
    
    def simulate_consultation(self, consultation_callback: Optional[Callable] = None):
        """
        模拟就诊过程
        
        Args:
            consultation_callback: 实际的就诊处理函数
        """
        session = self.coordinator.get_patient(self.patient_id)
        if not session:
            return
        
        doctor_id = session.assigned_doctor
        doctor = self.coordinator.get_doctor(doctor_id)
        
        self.logger.info(f"开始就诊，主诊医生: {doctor.name}({doctor.dept})")
        
        try:
            if consultation_callback:
                # 调用实际的就诊逻辑
                result = consultation_callback(self.patient_id, doctor_id, self.coordinator)
            else:
                # 简单模拟
                time.sleep(5)  # 模拟就诊时间
                result = {"status": "completed"}
            
            self.logger.info(f"就诊完成")
            
            return result
            
        except Exception as e:
            self.logger.error(f"就诊过程出错: {e}")
            raise
        finally:
            # 释放医生
            self.coordinator.release_doctor(doctor_id)


class MultiPatientProcessor:
    """多患者并发处理器"""
    
    def __init__(self, coordinator: HospitalCoordinator, max_workers: int = 10):
        """
        初始化处理器
        
        Args:
            coordinator: 医院协调器
            max_workers: 最大并发数
        """
        self.coordinator = coordinator
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        
        logger.info(f"✅ 多患者处理器已启动 (最大并发: {max_workers})")
    
    def process_patient_workflow(self, patient_data: Dict[str, Any], 
                                 dept: str,
                                 consultation_callback: Optional[Callable] = None,
                                 need_lab_test: bool = False,
                                 need_imaging: bool = False,
                                 priority: int = 5) -> Dict[str, Any]:
        """
        处理单个患者的完整流程
        
        Args:
            patient_data: 患者数据
            dept: 就诊科室
            consultation_callback: 就诊处理回调函数
            need_lab_test: 是否需要检验
            need_imaging: 是否需要影像
            priority: 优先级
        
        Returns:
            处理结果
        """
        patient_id = patient_data.get("id", f"P{int(time.time() * 1000)}")
        processor = PatientProcessor(self.coordinator, patient_id)
        
        try:
            # 步骤1: 挂号
            processor.logger.info(f"========== 开始流程 ==========")
            processor.logger.info(f"步骤1: 挂号 -> {dept}科")
            self.coordinator.register_patient(patient_id, patient_data, dept, priority)
            
            # 步骤2: 加入候诊队列
            processor.logger.info(f"步骤2: 加入候诊队列")
            self.coordinator.enqueue_patient(patient_id)
            
            # 步骤3: 等待医生分配
            processor.logger.info(f"步骤3: 等待医生分配...")
            if not processor.wait_for_doctor_assignment():
                return {"status": "failed", "reason": "医生分配超时"}
            
            # 步骤4: 就诊
            processor.logger.info(f"步骤4: 就诊中...")
            consultation_result = processor.simulate_consultation(consultation_callback)
            
            # 步骤5: 检验（如果需要）
            if need_lab_test:
                processor.logger.info(f"步骤5a: 前往检验科")
                self.coordinator.send_to_lab(patient_id)
                
                # 模拟检验过程
                time.sleep(3)
                self.coordinator.complete_lab_test(patient_id)
                
                # 步骤6: 等待复诊
                processor.logger.info(f"步骤6: 等待复诊...")
                if not processor.wait_for_doctor_assignment():
                    return {"status": "failed", "reason": "复诊分配超时"}
                
                # 步骤7: 复诊
                processor.logger.info(f"步骤7: 复诊中...")
                processor.simulate_consultation(consultation_callback)
            
            # 步骤5b: 影像（如果需要）
            if need_imaging:
                processor.logger.info(f"步骤5b: 前往影像科")
                self.coordinator.send_to_imaging(patient_id)
                
                # 模拟影像过程
                time.sleep(4)
                self.coordinator.complete_imaging(patient_id)
                
                # 等待复诊
                if not processor.wait_for_doctor_assignment():
                    return {"status": "failed", "reason": "复诊分配超时"}
                
                processor.simulate_consultation(consultation_callback)
            
            # 步骤8: 离院
            processor.logger.info(f"步骤8: 离院")
            self.coordinator.discharge_patient(patient_id)
            
            processor.logger.info(f"========== 流程完成 ==========")
            
            return {
                "status": "completed",
                "patient_id": patient_id,
                "result": consultation_result
            }
            
        except Exception as e:
            processor.logger.error(f"流程失败: {e}")
            return {
                "status": "failed",
                "patient_id": patient_id,
                "error": str(e)
            }
    
    def submit_patient(self, patient_data: Dict[str, Any], dept: str, **kwargs) -> str:
        """
        提交患者任务（异步）
        
        Args:
            patient_data: 患者数据
            dept: 就诊科室
            **kwargs: 其他参数
        
        Returns:
            任务ID
        """
        patient_id = patient_data.get("id", f"P{int(time.time() * 1000)}")
        
        with self._lock:
            future = self.executor.submit(
                self.process_patient_workflow,
                patient_data,
                dept,
                **kwargs
            )
            self.active_tasks[patient_id] = future
        
        logger.info(f"✅ 任务已提交: 患者 {patient_id}")
        
        return patient_id
    
    def submit_batch(self, patients: List[Dict[str, Any]]) -> List[str]:
        """
        批量提交患者任务
        
        Args:
            patients: 患者列表，每个元素包含 patient_data, dept 等信息
        
        Returns:
            任务ID列表
        """
        task_ids = []
        
        for patient_info in patients:
            patient_data = patient_info["patient_data"]
            dept = patient_info["dept"]
            kwargs = patient_info.get("kwargs", {})
            
            task_id = self.submit_patient(patient_data, dept, **kwargs)
            task_ids.append(task_id)
        
        logger.info(f"✅ 批量提交完成: {len(task_ids)} 个患者")
        
        return task_ids
    
    def wait_for_patient(self, patient_id: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        等待单个患者任务完成
        
        Args:
            patient_id: 患者ID
            timeout: 超时时间（秒）
        
        Returns:
            任务结果
        """
        with self._lock:
            future = self.active_tasks.get(patient_id)
        
        if not future:
            return {"status": "not_found"}
        
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            return {"status": "timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def wait_all(self, timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        等待所有任务完成
        
        Args:
            timeout: 超时时间（秒）
        
        Returns:
            所有任务结果列表
        """
        results = []
        
        with self._lock:
            futures = list(self.active_tasks.values())
        
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                results.append({"status": "error", "error": str(e)})
        
        logger.info(f"✅ 所有任务完成: {len(results)} 个")
        
        return results
    
    def get_active_count(self) -> int:
        """获取活跃任务数"""
        with self._lock:
            return len([f for f in self.active_tasks.values() if not f.done()])
    
    def shutdown(self, wait: bool = True):
        """关闭处理器"""
        logger.info("关闭多患者处理器...")
        self.executor.shutdown(wait=wait)
