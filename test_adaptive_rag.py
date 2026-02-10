"""Adaptive RAG 系统测试脚本

用于验证 SPLLM-RAG1 集成是否成功
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_import():
    """测试 1: 导入模块"""
    print("=" * 60)
    print("测试 1: 导入模块")
    print("-" * 60)
    
    try:
        from rag.adaptive_rag_retriever import AdaptiveRAGRetriever
        print("✅ AdaptiveRAGRetriever 导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   → 请确保已安装依赖: pip install -r requirements.txt")
        return False


def test_config():
    """测试 2: 配置加载"""
    print("\n" + "=" * 60)
    print("测试 2: 配置加载")
    print("-" * 60)
    
    try:
        from config import Config
        config = Config.load()
        
        print(f"✅ 配置加载成功")
        print(f"   use_adaptive_rag: {config.rag.use_adaptive_rag}")
        print(f"   spllm_root: {config.rag.spllm_root}")
        print(f"   threshold: {config.rag.adaptive_threshold}")
        
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_spllm_path():
    """测试 3: SPLLM-RAG1 路径"""
    print("\n" + "=" * 60)
    print("测试 3: SPLLM-RAG1 路径验证")
    print("-" * 60)
    
    try:
        from config import Config
        config = Config.load()
        
        spllm_root = Path(config.rag.spllm_root)
        if not spllm_root.is_absolute():
            # 相对路径，相对于项目根目录
            root = Path(__file__).parent
            spllm_root = (root / spllm_root).resolve()
        
        print(f"   SPLLM-RAG1 路径: {spllm_root}")
        
        if not spllm_root.exists():
            print(f"❌ 路径不存在: {spllm_root}")
            print(f"   → 请确保 SPLLM-RAG1 项目在正确位置")
            return False
        
        print(f"✅ 路径存在")
        
        # 检查子目录
        chroma_path = spllm_root / "chroma"
        cache_path = spllm_root / "model_cache"
        
        if not chroma_path.exists():
            print(f"❌ chroma 目录不存在: {chroma_path}")
            print(f"   → 请运行 SPLLM-RAG1/create_database_general.py 创建向量库")
            return False
        
        print(f"✅ chroma 目录存在")
        
        # 检查向量库
        dbs = ["MedicalGuide_db", "HighQualityQA_db", "ClinicalCase_db", "UserHistory_db"]
        for db_name in dbs:
            db_path = chroma_path / db_name
            if db_path.exists():
                print(f"   ✅ {db_name}")
            else:
                print(f"   ⚠️  {db_name} 不存在（可选）")
        
        if not cache_path.exists():
            print(f"⚠️  model_cache 目录不存在（首次运行会自动创建）")
        else:
            print(f"✅ model_cache 目录存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 路径验证失败: {e}")
        return False


def test_initialization():
    """测试 4: 初始化 Adaptive RAG"""
    print("\n" + "=" * 60)
    print("测试 4: 初始化 Adaptive RAG")
    print("-" * 60)
    
    try:
        from rag.adaptive_rag_retriever import AdaptiveRAGRetriever
        from config import Config
        
        config = Config.load()
        spllm_root = Path(config.rag.spllm_root)
        if not spllm_root.is_absolute():
            root = Path(__file__).parent
            spllm_root = (root / spllm_root).resolve()
        
        print("   正在初始化检索器...")
        retriever = AdaptiveRAGRetriever(
            spllm_root=spllm_root,
            cosine_threshold=config.rag.adaptive_threshold,
            embed_model=config.rag.adaptive_embed_model,
        )
        
        print("✅ 检索器初始化成功")
        return True, retriever
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_retrieval(retriever):
    """测试 5: 执行检索"""
    print("\n" + "=" * 60)
    print("测试 5: 执行检索")
    print("-" * 60)
    
    try:
        print("   查询: '高血压患者突发头痛怎么办？'")
        print("   正在检索...")
        
        results = retriever.retrieve(
            "高血压患者突发头痛怎么办？",
            k=3
        )
        
        print(f"✅ 检索成功，返回 {len(results)} 条结果")
        
        if len(results) > 0:
            print("\n   检索结果预览：")
            for i, r in enumerate(results[:3], 1):
                source = r['meta'].get('source', 'unknown')
                score = r.get('score', 0)
                text_preview = r['text'][:80].replace('\n', ' ')
                print(f"   {i}. [{source}] 分数:{score:.2f}")
                print(f"      {text_preview}...")
        else:
            print("⚠️  未检索到结果（可能阈值设置过严格）")
        
        return True
        
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Adaptive RAG 系统集成测试" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # 测试 1: 导入
    results['import'] = test_import()
    if not results['import']:
        print("\n❌ 基础依赖缺失，测试终止")
        print("   请先运行: pip install -r requirements.txt")
        return
    
    # 测试 2: 配置
    results['config'] = test_config()
    
    # 测试 3: 路径
    results['path'] = test_spllm_path()
    if not results['path']:
        print("\n⚠️  路径配置有问题，跳过后续测试")
        print_summary(results)
        return
    
    # 测试 4: 初始化
    success, retriever = test_initialization()
    results['init'] = success
    
    if not success:
        print("\n⚠️  初始化失败，跳过检索测试")
        print_summary(results)
        return
    
    # 测试 5: 检索
    results['retrieval'] = test_retrieval(retriever)
    
    # 打印总结
    print_summary(results)


def print_summary(results):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:15s}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Adaptive RAG 系统已就绪")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
        print("\n常见问题排查：")
        print("1. 缺少依赖 → pip install -r requirements.txt")
        print("2. 路径错误 → 检查 config.yaml 中的 spllm_root")
        print("3. 向量库缺失 → 运行 SPLLM-RAG1/create_database_general.py")
        print("4. 模型未缓存 → 首次运行需要网络下载模型")
    
    print("\n完整文档: docs/adaptive_rag_integration.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
