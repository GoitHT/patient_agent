"""
实时监控仪表板 - 显示医院系统运行状态
Real-time Monitoring Dashboard

功能：
1. 实时显示科室状态
2. 医生工作负载
3. 患者队列情况
4. 系统统计
"""

import time
import threading
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from hospital_coordinator import HospitalCoordinator
from utils import get_logger

logger = get_logger("hospital_agent.dashboard")


class MonitoringDashboard:
    """实时监控仪表板"""
    
    def __init__(self, coordinator: HospitalCoordinator, refresh_rate: float = 1.0):
        """
        初始化仪表板
        
        Args:
            coordinator: 医院协调器
            refresh_rate: 刷新频率（秒）
        """
        self.coordinator = coordinator
        self.refresh_rate = refresh_rate
        self.console = Console()
        self.running = False
        self._thread: Optional[threading.Thread] = None
    
    def generate_dept_table(self) -> Table:
        """生成科室状态表格"""
        table = Table(title="📊 科室状态", show_header=True, header_style="bold magenta")
        
        table.add_column("科室", style="cyan", width=20)
        table.add_column("医生总数", justify="center", style="white")
        table.add_column("空闲", justify="center", style="green")
        table.add_column("忙碌", justify="center", style="yellow")
        table.add_column("会诊中", justify="center", style="blue")
        table.add_column("等候患者", justify="center", style="red")
        
        dept_statuses = self.coordinator.get_all_dept_status()
        
        for status in dept_statuses:
            table.add_row(
                status["dept"],
                str(status["total_doctors"]),
                str(status["available_doctors"]),
                str(status["busy_doctors"]),
                str(status.get("consulting_doctors", 0)),
                str(status["waiting_patients"])
            )
        
        return table
    
    def generate_doctor_table(self) -> Table:
        """生成医生状态表格"""
        table = Table(title="👨‍⚕️ 医生状态", show_header=True, header_style="bold cyan")
        
        table.add_column("医生", style="cyan", width=15)
        table.add_column("科室", style="white", width=15)
        table.add_column("状态", justify="center", width=12)
        table.add_column("当前患者", style="yellow", width=15)
        table.add_column("今日接诊", justify="center", style="green")
        
        for doctor in sorted(self.coordinator.doctors.values(), key=lambda d: d.doctor_id):
            # 状态颜色
            status_text = doctor.status.value
            if doctor.status.value == "available":
                status_style = "green"
                status_text = "✅ 空闲"
            elif doctor.status.value == "busy":
                status_style = "yellow"
                status_text = "🟡 忙碌"
            elif doctor.status.value == "consulting":
                status_style = "blue"
                status_text = "🔵 会诊"
            else:
                status_style = "red"
                status_text = "⭕ 离线"
            
            table.add_row(
                doctor.name,
                doctor.dept,
                f"[{status_style}]{status_text}[/{status_style}]",
                doctor.current_patient or "-",
                str(doctor.total_patients_today)
            )
        
        return table
    
    def generate_stats_panel(self) -> Panel:
        """生成统计面板"""
        stats = self.coordinator.get_system_stats()
        
        content = f"""
[bold cyan]系统总览[/bold cyan]

📋 总注册患者: [green]{stats['total_patients_registered']}[/green]
🏥 当前活跃患者: [yellow]{stats['active_patients']}[/yellow]
👨‍⚕️ 医生总数: [cyan]{stats['total_doctors']}[/cyan]
✅ 空闲医生: [green]{stats['available_doctors']}[/green]
✔️ 完成就诊: [blue]{stats['total_consultations_completed']}[/blue]
🤝 多科会诊: [magenta]{stats['multi_consultations']}[/magenta]
⏳ 待处理会诊: [red]{stats['pending_consultation_requests']}[/red]
        """
        
        return Panel(content, title="📈 统计信息", border_style="green")
    
    def generate_layout(self) -> Layout:
        """生成完整布局"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(
            Panel(
                "[bold white]🏥 医院管理系统 - 实时监控[/bold white]",
                style="bold white on blue"
            )
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        layout["left"].split_column(
            Layout(name="dept_table"),
            Layout(name="doctor_table")
        )
        
        layout["dept_table"].update(self.generate_dept_table())
        layout["doctor_table"].update(self.generate_doctor_table())
        layout["right"].update(self.generate_stats_panel())
        
        layout["footer"].update(
            Panel(
                f"[dim]按 Ctrl+C 退出 | 刷新频率: {self.refresh_rate}秒[/dim]",
                style="dim white"
            )
        )
        
        return layout
    
    def display_snapshot(self):
        """显示一次快照（不持续刷新）"""
        self.console.clear()
        self.console.print(self.generate_layout())
    
    def _run_loop(self):
        """运行监控循环（在独立线程中）"""
        try:
            with Live(self.generate_layout(), refresh_per_second=1/self.refresh_rate, console=self.console) as live:
                while self.running:
                    time.sleep(self.refresh_rate)
                    live.update(self.generate_layout())
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
    
    def start(self):
        """启动实时监控（异步）"""
        if self.running:
            logger.warning("监控已在运行")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("✅ 监控仪表板已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            return
        
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("监控仪表板已停止")
    
    def wait(self):
        """等待监控结束（阻塞）"""
        if self._thread:
            try:
                self._thread.join()
            except KeyboardInterrupt:
                self.stop()


def print_simple_status(coordinator: HospitalCoordinator):
    """打印简单状态（用于非交互式场景）"""
    console = Console()
    
    console.print("\n" + "="*60)
    console.print("[bold cyan]医院系统状态[/bold cyan]")
    console.print("="*60 + "\n")
    
    # 科室状态
    console.print("[bold]科室状态:[/bold]")
    for status in coordinator.get_all_dept_status():
        console.print(f"  {status['dept']:20s} | "
                     f"医生: {status['total_doctors']} | "
                     f"空闲: [green]{status['available_doctors']}[/green] | "
                     f"忙碌: [yellow]{status['busy_doctors']}[/yellow] | "
                     f"等候: [red]{status['waiting_patients']}[/red]")
    
    # 系统统计
    console.print(f"\n[bold]系统统计:[/bold]")
    stats = coordinator.get_system_stats()
    console.print(f"  总患者: {stats['total_patients_registered']}")
    console.print(f"  活跃患者: {stats['active_patients']}")
    console.print(f"  完成就诊: {stats['total_consultations_completed']}")
    console.print(f"  多科会诊: {stats['multi_consultations']}")
    
    console.print("\n" + "="*60 + "\n")
