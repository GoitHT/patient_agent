"""
交互式命令系统 - 类似 ScienceWorld 的文本接口
支持自然语言命令和中文/英文混合输入
Level 4 增强功能:
- 自然语言理解
- 多模态观察
- 智能提示系统
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hospital_world import HospitalWorld


class CommandParser:
    """命令解析器 - 支持中英文命令"""
    
    # 命令模式定义（正则表达式）
    COMMANDS = {
        # 移动命令
        "move": [
            r"^(?:go|move|walk|到|去|前往)\s+(?:to\s+)?(.+)$",
            r"^进入\s*(.+)$",
        ],
        
        # 观察命令
        "look": [
            r"^(?:look|observe|check|看|观察|查看)(?:\s+around)?$",
            r"^(?:where|哪里|位置)$",
        ],
        
        # 医疗操作
        "examine": [
            r"^(?:examine|check|检查)\s+(.+)$",
        ],
        
        "order": [
            r"^(?:order|开单|申请)\s+(.+?)(?:\s+test|\s+检查)?$",
        ],
        
        "prescribe": [
            r"^(?:prescribe|开药|处方)\s+(.+)$",
        ],
        
        "consult": [
            r"^(?:consult|问诊|咨询)\s+(.+)$",
        ],
        
        # 等待命令
        "wait": [
            r"^(?:wait|等待|等)\s+(\d+)\s*(minute|min|分钟|分)?s?$",
            r"^(?:wait|等待|等)\s+(\d+)\s*(hour|hr|小时|时)s?$",
        ],
        
        # 查询命令
        "inventory": [
            r"^(?:inventory|inv|items|物品|背包)$",
        ],
        
        "status": [
            r"^(?:status|state|状态|信息)$",
        ],
        
        "help": [
            r"^(?:help|帮助|\?)$",
        ],
        
        "queue": [
            r"^(?:queue|排队|排队情况)$",
        ],
        
        "time": [
            r"^(?:time|现在|几点|时间)$",
        ],
    }
    
    # 位置名称映射
    LOCATION_MAP = {
        # 英文
        "lobby": "lobby",
        "triage": "triage",
        "neuro": "neuro",
        "lab": "lab",
        "imaging": "imaging",
        "pharmacy": "pharmacy",
        "endoscopy": "endoscopy",
        "neurophysiology": "neurophysiology",
        
        # 中文
        "大厅": "lobby",
        "门诊大厅": "lobby",
        "分诊": "triage",
        "分诊台": "triage",
        "神经科": "neuro",
        "神经内科": "neuro",
        "神经内科诊室": "neuro",
        "检验科": "lab",
        "化验室": "lab",
        "影像科": "imaging",
        "放射科": "imaging",
        "药房": "pharmacy",
        "取药": "pharmacy",
        "内镜": "endoscopy",
        "内镜中心": "endoscopy",
        "神经电生理": "neurophysiology",
        "神经电生理室": "neurophysiology",
    }
    
    # 检查类型映射
    EXAM_MAP = {
        # 影像
        "xray": "xray",
        "x光": "xray",
        "x-ray": "xray",
        "胸片": "xray",
        "ct": "ct",
        "mri": "mri",
        "核磁": "mri",
        "磁共振": "mri",
        "ultrasound": "ultrasound",
        "b超": "ultrasound",
        "超声": "ultrasound",
        
        # 检验
        "blood": "blood_test",
        "blood test": "blood_test",
        "血常规": "blood_test",
        "血液检查": "blood_test",
        "biochemistry": "biochemistry",
        "生化": "biochemistry",
        "生化检查": "biochemistry",
        
        # 功能检查
        "ecg": "ecg",
        "心电图": "ecg",
        "eeg": "eeg",
        "脑电图": "eeg",
        "emg": "emg",
        "肌电图": "emg",
        
        # 内镜
        "endoscopy": "endoscopy",
        "胃镜": "endoscopy",
        "colonoscopy": "colonoscopy",
        "肠镜": "colonoscopy",
    }
    
    @classmethod
    def parse(cls, command: str) -> Tuple[str, List[str]]:
        """解析命令
        
        Returns:
            (command_type, arguments)
        """
        if not command:
            return "unknown", []
        
        command = command.strip().lower()
        
        # 尝试匹配每个命令类型
        for cmd_type, patterns in cls.COMMANDS.items():
            for pattern in patterns:
                match = re.match(pattern, command, re.IGNORECASE)
                if match:
                    args = [g for g in match.groups() if g is not None]
                    return cmd_type, args
        
        return "unknown", [command]
    
    @classmethod
    def resolve_location(cls, location_name: str) -> str:
        """解析位置名称到ID"""
        location_name = location_name.strip().lower()
        return cls.LOCATION_MAP.get(location_name, location_name)
    
    @classmethod
    def resolve_exam_type(cls, test_name: str) -> str:
        """解析检查类型"""
        test_name = test_name.strip().lower()
        return cls.EXAM_MAP.get(test_name, test_name)


class SmartHintSystem:
    """智能提示系统 - Level 4 新功能"""
    
    def __init__(self, world: HospitalWorld):
        self.world = world
        self.hint_history: List[str] = []
    
    def get_contextual_hints(self, agent_id: str, last_action: Optional[str] = None) -> List[str]:
        """根据上下文提供智能提示"""
        hints = []
        
        location_id = self.world.agents.get(agent_id)
        if not location_id:
            return ["请先进入医院"]
        
        location = self.world.locations.get(location_id)
        if not location:
            return []
        
        # 1. 位置相关提示
        if location.type == "lobby":
            hints.append("💡 你可以前往分诊台进行登记")
        elif location.type == "triage":
            hints.append("💡 护士会为你分配科室")
        elif location.type == "clinic":
            hints.append("💡 你可以向医生描述症状，或进行体格检查")
        
        # 2. 可用设备提示
        available_equipment = [
            eq for eq in self.world.equipment.values()
            if eq.location_id == location_id and eq.can_use(self.world.current_time)
        ]
        if available_equipment:
            eq_names = [eq.name for eq in available_equipment[:3]]
            hints.append(f"🔬 可用设备: {', '.join(eq_names)}")
        
        # 3. 排队提示
        busy_equipment = [
            eq for eq in self.world.equipment.values()
            if eq.location_id == location_id and not eq.can_use(self.world.current_time)
        ]
        if busy_equipment:
            for eq in busy_equipment[:2]:
                wait_time = (eq.occupied_until - self.world.current_time).total_seconds() / 60
                hints.append(f"⏳ {eq.name} 繁忙中，还需 {int(wait_time)} 分钟")
        
        # 4. 健康状态提示
        if agent_id in self.world.physical_states:
            state = self.world.physical_states[agent_id]
            critical_symptoms = [
                name for name, symptom in state.symptoms.items()
                if symptom.severity >= 8
            ]
            if critical_symptoms:
                hints.append(f"⚠️ 严重症状: {', '.join(critical_symptoms)} - 建议尽快就医")
            
            # 生命体征异常提示
            heart_rate = state.vital_signs.get("heart_rate")
            if heart_rate and heart_rate.value > 100:
                hints.append("💓 心率偏高，建议检查")
            temperature = state.vital_signs.get("temperature")
            if temperature and temperature.value > 38.0:
                hints.append("🌡️ 体温偏高，可能有发热")
        
        # 5. 时间提示
        hour = self.world.current_time.hour
        if hour >= 17:
            hints.append("🕐 接近下班时间，部分科室即将关闭")
        elif hour < 8:
            hints.append("🌅 医院尚未开始工作")
        
        # 6. 下一步建议
        if last_action == "move":
            hints.append("💬 使用 'look' 查看当前位置信息")
        elif last_action == "order":
            hints.append("⏰ 检查需要时间，可以 'wait' 或查看 'queue'")
        
        return hints
    
    def get_action_suggestions(self, agent_id: str) -> List[str]:
        """获取可执行的动作建议"""
        location_id = self.world.agents.get(agent_id)
        if not location_id:
            return []
        
        location = self.world.locations.get(location_id)
        if not location:
            return []
        
        suggestions = []
        
        # 基于位置的动作
        for action in location.available_actions:
            if action == "move":
                nearby = [self.world.locations[lid].name for lid in location.connected_to[:3]]
                suggestions.append(f"🚶 move to {', '.join(nearby)}")
            elif action == "order_test":
                suggestions.append("📝 order <血常规|CT|X光>")
            elif action == "examine":
                suggestions.append("👨‍⚕️ examine patient")
            elif action == "look":
                suggestions.append("👀 look around")
        
        return suggestions


class NaturalLanguageParser:
    """自然语言理解器 - Level 4 新功能"""
    
    @staticmethod
    def extract_intent(text: str) -> Tuple[str, Dict[str, any]]:
        """从自然语言中提取意图和参数"""
        text = text.strip().lower()
        
        # 模式匹配：更灵活的自然语言理解
        patterns = [
            # 移动意图
            (r"(?:我想|我要|帮我)?(?:去|到|前往)(.+?)(?:看看|检查|就诊)?", "move", lambda m: {"location": m.group(1).strip()}),
            (r"(?:带我|指引|导航)(?:去|到)?(.+)", "move", lambda m: {"location": m.group(1).strip()}),
            
            # 检查意图
            (r"(?:我需要|我想做|做个|做一下)(.+?)(?:检查|测试)?", "order", lambda m: {"test": m.group(1).strip()}),
            (r"(?:帮我|给我)(?:开单|申请|安排)(.+)", "order", lambda m: {"test": m.group(1).strip()}),
            
            # 查询意图
            (r"(?:现在|目前)?(?:在哪|什么位置|我在哪里)", "look", lambda m: {}),
            (r"(?:我的)?(?:情况|状态|症状)(?:怎么样|如何)", "status", lambda m: {}),
            (r"(?:现在|当前)?(?:几点|时间)", "time", lambda m: {}),
            (r"(?:有|还有)多少人(?:在)?排队", "queue", lambda m: {}),
            
            # 等待意图
            (r"等(?:一下|待)?(\d+)(?:分钟|分|小时|时)", "wait", lambda m: {
                "duration": int(m.group(1)),
                "unit": "hour" if "时" in m.group(0) else "minute"
            }),
            
            # 帮助意图
            (r"(?:怎么|如何)(?:操作|使用|玩)", "help", lambda m: {}),
            (r"(?:有什么|可以做什么)", "help", lambda m: {}),
        ]
        
        for pattern, intent, extractor in patterns:
            match = re.search(pattern, text)
            if match:
                params = extractor(match)
                return intent, params
        
        return "unknown", {"text": text}
    
    @staticmethod
    def generate_response_variants(base_response: str, context: Dict) -> str:
        """生成更自然的响应变体"""
        # 添加情境化的语言
        time_of_day = context.get("time_of_day", "")
        if time_of_day == "morning":
            greeting = "早上好！"
        elif time_of_day == "afternoon":
            greeting = "下午好！"
        elif time_of_day == "evening":
            greeting = "晚上好！"
        else:
            greeting = ""
        
        # 根据agent类型调整语气
        agent_type = context.get("agent_type", "patient")
        if agent_type == "doctor":
            tone = "专业"
        elif agent_type == "nurse":
            tone = "温和"
        else:
            tone = "友好"
        
        return f"{greeting} {base_response}".strip()


class InteractiveSession:
    """交互式会话管理器 - Level 4 增强版"""
    
    def __init__(self, world: HospitalWorld, agent_id: str, agent_type: str = "patient", 
                 enable_hints: bool = True, enable_nl: bool = True):
        """初始化交互式会话
        
        Args:
            world: 医院世界实例
            agent_id: Agent ID
            agent_type: Agent类型 (patient, doctor, nurse)
            enable_hints: 启用智能提示
            enable_nl: 启用自然语言理解
        """
        self.world = world
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.history: List[Tuple[str, str]] = []  # (command, response)
        self.command_count = 0
        self.last_action = None
        
        # Level 4 新功能
        self.enable_hints = enable_hints
        self.enable_nl = enable_nl
        self.hint_system = SmartHintSystem(world) if enable_hints else None
        self.nl_parser = NaturalLanguageParser() if enable_nl else None
    
    def execute(self, command: str, show_hints: bool = True) -> str:
        """执行命令并返回响应
        
        Args:
            command: 用户输入的命令
            show_hints: 是否显示智能提示
        """
        self.command_count += 1
        
        # 尝试自然语言理解
        if self.enable_nl and not self._is_structured_command(command):
            intent, params = self.nl_parser.extract_intent(command)
            if intent != "unknown":
                # 转换为结构化命令
                command = self._intent_to_command(intent, params)
        
        # 解析命令
        cmd_type, args = CommandParser.parse(command)
        self.last_action = cmd_type
        
        # 执行对应操作
        if cmd_type == "move":
            response = self._handle_move(args)
        
        elif cmd_type == "look":
            response = self._handle_look()
        
        elif cmd_type == "order":
            response = self._handle_order(args)
        
        elif cmd_type == "wait":
            response = self._handle_wait(args)
        
        elif cmd_type == "status":
            response = self._handle_status()
        
        elif cmd_type == "help":
            response = self._handle_help()
        
        elif cmd_type == "queue":
            response = self._handle_queue()
        
        elif cmd_type == "time":
            response = self._handle_time()
        
        elif cmd_type == "inventory":
            response = self._handle_inventory()
        
        else:
            response = self._handle_unknown(command)
        
        # 记录历史
        self.history.append((command, response))
        
        return response
    
    def _handle_move(self, args: List[str]) -> str:
        """处理移动命令"""
        if not args:
            return "❌ 请指定目标位置。例如: '去 内科' 或 'move to lab'"
        
        target_name = args[0]
        target_id = CommandParser.resolve_location(target_name)
        
        success, message = self.world.move_agent(self.agent_id, target_id)
        
        if success:
            # 移动成功后自动观察
            obs = self.world.get_observation(self.agent_id)
            location_info = self._format_location_brief(obs)
            return f"✅ {message}\n\n{location_info}"
        else:
            return f"❌ {message}"
    
    def _handle_look(self) -> str:
        """处理观察命令"""
        obs = self.world.get_observation(self.agent_id)
        return self._format_observation(obs)
    
    def _handle_order(self, args: List[str]) -> str:
        """处理开单命令"""
        if not args:
            return "❌ 请指定检查项目。例如: '开单 血常规' 或 'order ct'"
        
        test_name = args[0]
        exam_type = CommandParser.resolve_exam_type(test_name)
        
        success, message = self.world.perform_exam(self.agent_id, exam_type)
        
        if success:
            return f"✅ {message}"
        else:
            return f"❌ {message}"
    
    def _handle_wait(self, args: List[str]) -> str:
        """处理等待命令"""
        if not args:
            return "❌ 请指定等待时间。例如: '等待 10 分钟' 或 'wait 5 min'"
        
        try:
            duration = int(args[0])
            # 检查第二个参数判断单位
            if len(args) > 1 and ('hour' in args[1] or '小时' in args[1] or '时' in args[1]):
                minutes = duration * 60
            else:
                minutes = duration
            
            if minutes > 180:  # 限制最多3小时
                return "❌ 等待时间过长，最多支持 180 分钟"
            
            self.world.advance_time(minutes)
            return f"⏰ 等待了 {minutes} 分钟（当前时间: {self.world.current_time.strftime('%H:%M')}）"
        
        except ValueError:
            return "❌ 时间格式错误，请输入数字"
    
    def _handle_status(self) -> str:
        """处理状态查询"""
        obs = self.world.get_observation(self.agent_id)
        
        lines = [
            "=" * 50,
            "【状态信息】",
            "=" * 50,
            f"时间: {obs['time']} ({obs['day_of_week']})",
            f"位置: {obs['location']}",
            f"工作状态: {'营业中' if obs['working_hours'] else '休息中'}",
        ]
        
        # 患者状态
        if self.agent_type == "patient" and "symptoms" in obs:
            lines.append("\n【健康状态】")
            
            if obs.get('vital_signs'):
                lines.append("生命体征:")
                for sign, value in obs['vital_signs'].items():
                    lines.append(f"  - {sign}: {value}")
            
            if obs.get('symptoms'):
                lines.append("症状:")
                for symptom, severity in obs['symptoms'].items():
                    status = "轻度" if severity <= 3 else ("中度" if severity <= 6 else "重度")
                    lines.append(f"  - {symptom}: {severity}/10 ({status})")
            
            if 'energy_level' in obs:
                energy = obs['energy_level']
                energy_status = "充沛" if energy >= 7 else ("一般" if energy >= 4 else "疲惫")
                lines.append(f"体力: {energy}/10 ({energy_status})")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _handle_help(self) -> str:
        """处理帮助命令"""
        help_text = """
╔══════════════════════════════════════════════════╗
║               命令帮助 (中英文)                    ║
╚══════════════════════════════════════════════════╝

【移动命令】
  去 <地点>           - 移动到指定地点
  move to <location>  - 移动到指定地点
  例: 去 内科 / move to lab

【观察命令】
  看 / look           - 观察当前位置
  哪里 / where        - 查看当前位置
  
【医疗操作】
  开单 <检查>         - 申请检查项目
  order <test>        - 申请检查项目
  例: 开单 血常规 / order ct

【时间管理】
  等待 <分钟>         - 等待指定时间
  wait <minutes>      - 等待指定时间
  例: 等待 10 / wait 15 min

【信息查询】
  状态 / status       - 查看详细状态
  时间 / time         - 查看当前时间
  排队 / queue        - 查看排队情况
  帮助 / help         - 显示此帮助

【退出】
  quit / exit / q     - 退出交互模式

════════════════════════════════════════════════════
常用地点: 大厅、分诊台、内科、外科、检验科、影像科、药房
常用检查: 血常规、CT、MRI、X光、心电图、B超
════════════════════════════════════════════════════
"""
        return help_text
    
    def _handle_queue(self) -> str:
        """处理排队查询"""
        obs = self.world.get_observation(self.agent_id)
        
        if "equipment" not in obs or not obs["equipment"]:
            return "ℹ️  当前位置没有设备"
        
        lines = [
            "=" * 50,
            "【设备排队情况】",
            "=" * 50,
        ]
        
        for eq_status in obs["equipment"]:
            lines.append(f"  {eq_status}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _handle_time(self) -> str:
        """处理时间查询"""
        obs = self.world.get_observation(self.agent_id)
        status = "营业中" if obs['working_hours'] else "休息中"
        return f"⏰ 当前时间: {obs['time']} ({obs['day_of_week']})  |  {status}"
    
    def _handle_inventory(self) -> str:
        """处理背包/物品查询"""
        # TODO: 未来可扩展为携带物品系统
        return "ℹ️  物品系统暂未实现"
    
    def _handle_unknown(self, command: str) -> str:
        """处理未知命令"""
        return f"❓ 未知命令: '{command}'\n输入 'help' 或 '帮助' 查看可用命令"
    
    def _format_observation(self, obs: Dict) -> str:
        """格式化完整观察结果"""
        lines = [
            "=" * 50,
            f"📍 {obs['location']}",
            "=" * 50,
            f"⏰ 时间: {obs['time']} ({obs['day_of_week']})  |  {'🟢 营业中' if obs['working_hours'] else '🔴 休息中'}",
            f"👥 人数: {obs['occupants_count']}/{obs['capacity']}",
        ]
        
        # 可用操作
        if obs['available_actions']:
            actions = ", ".join(obs['available_actions'])
            lines.append(f"⚡ 可用操作: {actions}")
        
        # 相邻位置
        if obs['nearby_locations']:
            nearby = " | ".join(obs['nearby_locations'])
            lines.append(f"🚪 相邻位置: {nearby}")
        
        # 设备信息
        if "equipment" in obs and obs['equipment']:
            lines.append("\n🔧 设备状态:")
            for eq_status in obs['equipment']:
                lines.append(f"  • {eq_status}")
        
        # 患者状态
        if "symptoms" in obs and obs['symptoms']:
            lines.append("\n💊 当前症状:")
            for symptom, severity in obs['symptoms'].items():
                bars = "█" * severity + "░" * (10 - severity)
                lines.append(f"  • {symptom}: {bars} {severity}/10")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def _format_location_brief(self, obs: Dict) -> str:
        """格式化简要位置信息"""
        lines = [
            f"📍 当前位置: {obs['location']} ({obs['occupants_count']}/{obs['capacity']}人)"
        ]
        
        if obs['nearby_locations']:
            nearby = " | ".join(obs['nearby_locations'])
            lines.append(f"🚪 相邻: {nearby}")
        
        return "\n".join(lines)
    
    def get_prompt(self) -> str:
        """获取命令提示符"""
        obs = self.world.get_observation(self.agent_id)
        time_str = obs.get('time', '??:??')
        location = obs.get('location', '未知位置')
        return f"[{time_str}] {location} > "    
    # ============================================================
    # Level 4: 交互增强方法
    # ============================================================
    
    def _is_structured_command(self, command: str) -> bool:
        """判断是否是结构化命令"""
        structured_keywords = ["move", "go", "order", "wait", "look", "status", "help"]
        return any(command.strip().lower().startswith(kw) for kw in structured_keywords)
    
    def _intent_to_command(self, intent: str, params: Dict) -> str:
        """将意图转换为结构化命令"""
        if intent == "move":
            return f"move to {params.get('location', '')}"
        elif intent == "order":
            return f"order {params.get('test', '')}"
        elif intent == "wait":
            duration = params.get('duration', 1)
            unit = params.get('unit', 'minute')
            return f"wait {duration} {unit}"
        elif intent == "look":
            return "look around"
        elif intent == "status":
            return "status"
        elif intent == "time":
            return "time"
        elif intent == "queue":
            return "queue"
        elif intent == "help":
            return "help"
        return ""
    
    def get_smart_hints(self) -> str:
        """获取智能提示 - Level 4 功能"""
        if not self.hint_system:
            return ""
        
        hints = self.hint_system.get_contextual_hints(self.agent_id, self.last_action)
        if not hints:
            return ""
        
        lines = ["\n" + "=" * 50, "💡 智能提示:"]
        for hint in hints[:5]:  # 最多显示5条
            lines.append(f"  {hint}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def get_action_menu(self) -> str:
        """获取可用动作菜单 - Level 4 功能"""
        if not self.hint_system:
            return ""
        
        suggestions = self.hint_system.get_action_suggestions(self.agent_id)
        if not suggestions:
            return ""
        
        lines = ["\n📋 可用动作:"]
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
        
        return "\n".join(lines)
    
    def execute_with_feedback(self, command: str) -> Dict[str, any]:
        """执行命令并返回详细反馈 - Level 4 功能
        
        Returns:
            包含响应、提示、统计等的字典
        """
        # 执行命令
        response = self.execute(command, show_hints=False)
        
        # 收集反馈信息
        feedback = {
            "response": response,
            "hints": self.get_smart_hints() if self.enable_hints else "",
            "actions": self.get_action_menu() if self.enable_hints else "",
            "command_count": self.command_count,
            "time": self.world.current_time.strftime("%H:%M"),
            "location": self.world.agents.get(self.agent_id, "unknown"),
        }
        
        # 健康状态
        if self.agent_id in self.world.physical_states:
            state = self.world.physical_states[self.agent_id]
            feedback["health_summary"] = state.get_status_summary()
        
        return feedback
    
    def get_multimodal_observation(self) -> Dict[str, any]:
        """获取多模态观察 - Level 4 功能
        
        Returns:
            包含文本、结构化数据、可视化提示等
        """
        obs = self.world.get_observation(self.agent_id)
        
        # 文本描述
        text_obs = self._format_observation(obs)
        
        # 结构化数据
        structured = {
            "time": obs.get("time"),
            "location": {
                "id": self.world.agents.get(self.agent_id),
                "name": obs.get("location"),
                "type": obs.get("location_type"),
                "occupancy": f"{obs.get('occupants_count')}/{obs.get('capacity')}",
            },
            "actions": obs.get("available_actions", []),
            "nearby": obs.get("nearby_locations", []),
        }
        
        # 设备状态
        if "equipment" in obs:
            structured["equipment"] = obs["equipment"]
        
        # 健康状态
        if self.agent_id in self.world.physical_states:
            state = self.world.physical_states[self.agent_id]
            structured["health"] = {
                "symptoms": {name: symptom.severity for name, symptom in state.symptoms.items()},
                "vital_signs": {name: vs.value for name, vs in state.vital_signs.items()},
                "status": state.consciousness_level,  # 向后兼容
                "consciousness": state.consciousness_level,
                "energy": state.energy_level,
                "pain_level": state.pain_level,
            }
        
        # 可视化提示（ASCII art）
        visual = self._generate_mini_map(obs)
        
        return {
            "text": text_obs,
            "structured": structured,
            "visual": visual,
            "hints": self.get_smart_hints() if self.enable_hints else "",
        }
    
    def _generate_mini_map(self, obs: Dict) -> str:
        """生成小地图 - ASCII艺术"""
        current_loc = obs.get("location", "")
        nearby = obs.get("nearby_locations", [])
        
        lines = [
            "🗺️  位置地图:",
            "     ",
        ]
        
        # 简单的地图布局
        if nearby:
            for i, loc in enumerate(nearby[:4]):
                direction = ["↑", "→", "↓", "←"][i % 4]
                lines.append(f"  {direction} {loc}")
        
        lines.append(f"  📍 {current_loc} (当前)")
        
        return "\n".join(lines)