"""配置加载模块 - 负责系统配置的加载和初始化"""

from pathlib import Path
from typing import Optional
from config import Config


def load_system_config(config_file: Optional[Path] = None) -> Config:
    """加载系统配置
    
    Args:
        config_file: 配置文件路径，为None时使用默认路径
    
    Returns:
        Config: 配置对象
    """
    return Config.load(config_file=config_file)
