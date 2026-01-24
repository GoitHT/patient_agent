#!/usr/bin/env python3
"""
下载 DiagnosisArena 数据集到本地

运行此脚本会将完整数据集从 HuggingFace 下载并保存到本地，
之后运行主程序时会自动使用本地缓存，无需联网。

使用方法:
    python download_dataset.py                      # 下载到默认目录 ./diagnosis_dataset
    python download_dataset.py --output ./data      # 下载到指定目录
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import get_logger

logger = get_logger("dataset_downloader")


def download_dataset(output_dir: str = "./diagnosis_dataset"):
    """
    下载数据集到本地
    
    Args:
        output_dir: 输出目录
    """
    try:
        from datasets import load_dataset
        
        output_path = Path(output_dir)
        output_json = output_path / "dataset.json"
        
        # 检查是否已存在
        if output_json.exists():
            logger.info(f"📂 数据集已存在: {output_json}")
            response = input("是否重新下载？(y/n): ")
            if response.lower() != 'y':
                logger.info("⏸️  取消下载")
                return
        
        logger.info("="*80)
        logger.info("🌐 开始从 HuggingFace 下载数据集...")
        logger.info("📦 数据集: SII-SPIRAL-MED/DiagnosisArena")
        logger.info("="*80)
        
        # 下载数据集
        try:
            dataset = load_dataset("SII-SPIRAL-MED/DiagnosisArena", split="train")
            logger.info(f"✅ 数据集下载成功")
        except (ValueError, KeyError):
            # 如果没有train split，尝试加载整个数据集
            dataset = load_dataset("SII-SPIRAL-MED/DiagnosisArena")
            # 取第一个split
            if isinstance(dataset, dict):
                split_name = list(dataset.keys())[0]
                dataset = dataset[split_name]
                logger.info(f"✅ 数据集下载成功 (使用 split: {split_name})")
        
        logger.info(f"📊 数据集信息:")
        logger.info(f"  - 病例数: {len(dataset)}")
        logger.info(f"  - 字段: {list(dataset.features.keys())}")
        
        # 保存到本地
        logger.info(f"\n💾 保存到本地...")
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.to_json(str(output_json), force_ascii=False, indent=2)
        
        # 验证文件大小
        file_size_mb = output_json.stat().st_size / (1024 * 1024)
        
        logger.info("="*80)
        logger.info("✅ 数据集下载完成！")
        logger.info(f"📁 保存位置: {output_json.absolute()}")
        logger.info(f"📦 文件大小: {file_size_mb:.2f} MB")
        logger.info(f"📚 病例数量: {len(dataset)}")
        logger.info("="*80)
        
        logger.info("\n💡 提示:")
        logger.info("  - 下次运行主程序时会自动使用本地数据，无需联网")
        logger.info("  - 如需更新数据集，请重新运行此脚本")
        
    except ImportError:
        logger.error("❌ 错误: 未安装 datasets 库")
        logger.info("请运行: pip install datasets")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        logger.info("\n💡 可能的原因:")
        logger.info("  1. 网络连接问题，无法访问 HuggingFace")
        logger.info("  2. 数据集不存在或已移除")
        logger.info("  3. HuggingFace token 配置问题")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="下载 DiagnosisArena 数据集到本地",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_dataset.py                    # 使用默认目录
  python download_dataset.py --output ./data    # 指定输出目录
        """
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./diagnosis_dataset",
        help="输出目录 (默认: ./diagnosis_dataset)"
    )
    
    args = parser.parse_args()
    
    download_dataset(args.output)


if __name__ == "__main__":
    main()
