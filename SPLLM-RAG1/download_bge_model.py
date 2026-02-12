"""
BGE-Large-ZH 模型下载脚本
首次使用前运行此脚本，自动下载模型（约1.3GB）
"""
import os
import sys

# 设置缓存路径
CACHE_FOLDER = "./model_cache"
MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# 允许联网下载
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HOME'] = CACHE_FOLDER

print("=" * 60)
print("  BGE-Large-ZH 中文嵌入模型下载工具")
print("=" * 60)
print(f"模型名称: {MODEL_NAME}")
print(f"缓存路径: {os.path.abspath(CACHE_FOLDER)}")
print(f"模型大小: 约 1.3GB")
print("=" * 60)

# 检查模型是否已存在
model_cache_path = os.path.join(CACHE_FOLDER, "models--BAAI--bge-large-zh-v1.5")
if os.path.exists(model_cache_path) and os.path.isdir(model_cache_path):
    print("✅ 模型已存在，无需重复下载")
    print(f"路径: {os.path.abspath(model_cache_path)}")
    sys.exit(0)

print("\n🔄 开始下载模型...")
print("⏳ 下载时间取决于网络速度（可能需要5-15分钟）\n")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # 下载模型
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        },
        cache_folder=CACHE_FOLDER
    )
    
    # 测试模型
    print("\n🔬 测试模型...")
    test_vec = embeddings.embed_query("测试文本")
    print(f"✅ 模型下载成功！")
    print(f"   向量维度: {len(test_vec)}")
    print(f"   向量范数: {sum(x * x for x in test_vec) ** 0.5:.4f}")
    print(f"   缓存位置: {os.path.abspath(model_cache_path)}")
    print("\n✅ 现在可以运行主程序了！")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n可能的原因:")
    print("1. 网络连接问题（需要访问 huggingface.co）")
    print("2. 磁盘空间不足（需要至少2GB可用空间）")
    print("3. 防火墙或代理设置阻止了下载")
    print("\n解决方案：")
    print("- 检查网络连接")
    print("- 尝试使用代理: export HF_ENDPOINT=https://hf-mirror.com")
    print("- 手动下载: https://huggingface.co/BAAI/bge-large-zh-v1.5")
    sys.exit(1)
