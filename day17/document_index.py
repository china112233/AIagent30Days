"""
Day 17: 文档与索引示例

本文件演示文档处理和多种索引类型：
- 文档加载器使用
- 分块策略配置
- 多种索引类型对比（VectorStoreIndex、ListIndex、TreeIndex）
- 向量存储集成
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    ListIndex,
    TreeIndex,
    SummaryIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()


def setup_settings():
    """配置 LlamaIndex Settings"""
    llm = OpenAILike(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
        temperature=0.7,
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY", os.getenv("DEEPSEEK_API_KEY")),
        api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ Settings 已配置")


# ==========================================
# 示例 1: 文档加载方式
# ==========================================


def example_document_loading():
    """演示多种文档创建方式"""
    print("\n" + "=" * 50)
    print("示例 1: 文档加载方式")
    print("=" * 50)

    # 方式 1: 直接创建 Document
    doc1 = Document(
        text="这是直接创建的文档内容。"
        "LlamaIndex 支持多种文档创建方式。",
        metadata={"source": "direct", "id": "1"},
    )
    print(f"方式 1 - 直接创建: {doc1.text[:30]}...")

    # 方式 2: 从字典创建
    doc2 = Document(
        text="从字典创建的文档。",
        metadata={"source": "dict", "category": "demo"},
    )
    print(f"方式 2 - 字典创建: {doc2.text}")

    # 方式 3: 批量创建文档列表
    texts = [
        "文档 A: LlamaIndex 简介",
        "文档 B: 向量索引说明",
        "文档 C: 查询引擎介绍",
    ]
    documents = [Document(text=t, metadata={"batch": "true"}) for t in texts]
    print(f"方式 3 - 批量创建: {len(documents)} 个文档")

    # 模拟从文件加载（实际使用 SimpleDirectoryReader）
    print("\n实际项目中可使用:")
    print("  - SimpleDirectoryReader: 从目录加载")
    print("  - PDFReader: 加载 PDF 文件")
    print("  - JSONReader: 加载 JSON 数据")


# ==========================================
# 示例 2: 分块策略对比
# ==========================================


def example_chunking_strategies():
    """演示不同分块策略"""
    print("\n" + "=" * 50)
    print("示例 2: 分块策略对比")
    print("=" * 50)

    # 创建测试文档
    test_text = """
    LlamaIndex 提供了多种分块策略。

    SentenceSplitter 按句子分割文本，保持语义完整性。
    它会识别段落边界，避免在句子中间切断。

    TokenTextSplitter 按 token 数量分割。
    这种方式更精确控制每块的 token 数。

    SemanticSplitter 使用语义相似度分割。
    它会在语义变化点进行分割。

    选择合适的分块策略对 RAG 效果至关重要。
    """

    doc = Document(text=test_text)

    # 策略 1: SentenceSplitter
    print("\n策略 1: SentenceSplitter")
    sentence_splitter = SentenceSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    nodes1 = sentence_splitter.get_nodes_from_documents([doc])
    print(f"  节点数: {len(nodes1)}")
    for i, node in enumerate(nodes1[:2]):
        print(f"  Node {i+1}: {node.text[:40]}...")

    # 策略 2: TokenTextSplitter
    print("\n策略 2: TokenTextSplitter")
    token_splitter = TokenTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )
    nodes2 = token_splitter.get_nodes_from_documents([doc])
    print(f"  节点数: {len(nodes2)}")
    for i, node in enumerate(nodes2[:2]):
        print(f"  Node {i+1}: {node.text[:40]}...")

    # 分块参数建议
    print("\n分块参数建议:")
    print("  - 问答系统: chunk_size=256, overlap=20")
    print("  - 文档摘要: chunk_size=1024, overlap=100")
    print("  - 通用场景: chunk_size=512, overlap=50")


# ==========================================
# 示例 3: VectorStoreIndex
# ==========================================


def example_vector_store_index():
    """演示 VectorStoreIndex 的创建和使用"""
    print("\n" + "=" * 50)
    print("示例 3: VectorStoreIndex")
    print("=" * 50)

    setup_settings()

    # 创建文档
    documents = [
        Document(
            text="向量索引是最常用的索引类型。"
            "它使用嵌入向量进行语义检索。"
            "适合问答系统和 RAG 应用。",
            metadata={"type": "vector"},
        ),
        Document(
            text="创建向量索引时，LlamaIndex 会自动"
            "生成嵌入向量并存储到向量存储中。",
            metadata={"type": "vector"},
        ),
        Document(
            text="查询时，会计算查询向量与文档向量的"
            "相似度，返回最相关的结果。",
            metadata={"type": "vector"},
        ),
    ]

    # 创建向量索引
    print("创建 VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(documents)

    print("✅ VectorStoreIndex 创建成功")
    print("特点:")
    print("  - 语义检索能力强")
    print("  - 支持相似度搜索")
    print("  - 需要嵌入模型")

    # 查询测试
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("向量索引有什么特点？")
    print(f"\n查询结果: {response.response[:100]}...")


# ==========================================
# 示例 4: ListIndex
# ==========================================


def example_list_index():
    """演示 ListIndex 的创建和使用"""
    print("\n" + "=" * 50)
    print("示例 4: ListIndex")
    print("=" * 50)

    setup_settings()

    # 创建小规模文档
    documents = [
        Document(text="ListIndex 按顺序遍历所有节点。"),
        Document(text="适合小数据集或需要完整摘要的场景。"),
        Document(text="不需要嵌入模型，创建速度快。"),
        Document(text="查询时会检查每个节点。"),
    ]

    # 创建列表索引
    print("创建 ListIndex...")
    index = ListIndex.from_documents(documents)

    print("✅ ListIndex 创建成功")
    print("特点:")
    print("  - 按顺序遍历节点")
    print("  - 不需要嵌入模型")
    print("  - 适合小数据集")

    # 查询测试
    query_engine = index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query("总结 ListIndex 的特点")
    print(f"\n查询结果: {response.response[:100]}...")


# ==========================================
# 示例 5: TreeIndex
# ==========================================


def example_tree_index():
    """演示 TreeIndex 的创建和使用"""
    print("\n" + "=" * 50)
    print("示例 5: TreeIndex")
    print("=" * 50)

    setup_settings()

    # 创建多个文档用于层级结构
    documents = [
        Document(text="TreeIndex 使用层级树结构组织节点。"),
        Document(text="根节点是整个文档的摘要。"),
        Document(text="子节点是文档的不同部分。"),
        Document(text="查询时从根节点开始，逐层深入。"),
        Document(text="适合大数据集的分层查询。"),
        Document(text="可以快速定位相关内容。"),
    ]

    # 创建树索引
    print("创建 TreeIndex...")
    index = TreeIndex.from_documents(documents)

    print("✅ TreeIndex 创建成功")
    print("特点:")
    print("  - 层级树结构")
    print("  - 分层检索，效率高")
    print("  - 适合大数据集")

    # 查询测试
    query_engine = index.as_query_engine()
    response = query_engine.query("TreeIndex 的查询方式是什么？")
    print(f"\n查询结果: {response.response[:100]}...")


# ==========================================
# 示例 6: SummaryIndex
# ==========================================


def example_summary_index():
    """演示 SummaryIndex 的创建和使用"""
    print("\n" + "=" * 50)
    print("示例 6: SummaryIndex")
    print("=" * 50)

    setup_settings()

    # 创建文档
    documents = [
        Document(
            text="SummaryIndex 为每个文档生成摘要。"
            "查询时通过摘要快速定位相关文档。"
            "适合长文档的快速检索场景。"
        ),
    ]

    # 创建摘要索引
    print("创建 SummaryIndex...")
    index = SummaryIndex.from_documents(documents)

    print("✅ SummaryIndex 创建成功")
    print("特点:")
    print("  - 自动生成文档摘要")
    print("  - 通过摘要定位文档")
    print("  - 适合长文档场景")


# ==========================================
# 示例 7: 索引类型对比查询
# ==========================================


def example_index_comparison():
    """对比不同索引类型的查询效果"""
    print("\n" + "=" * 50)
    print("示例 7: 索引类型对比查询")
    print("=" * 50)

    setup_settings()

    # 创建相同文档
    documents = [
        Document(
            text="Python 是一种高级编程语言，"
            "设计哲学强调代码可读性。"
            "Python 广泛用于数据分析、Web 开发和 AI。"
        ),
        Document(
            text="JavaScript 是 Web 开发的核心语言，"
            "可以在浏览器和服务器端运行。"
            "Node.js 使 JavaScript 支持后端开发。"
        ),
        Document(
            text="Go 语言由 Google 开发，"
            "以简洁和高效著称。"
            "Go 适合并发编程和系统开发。"
        ),
    ]

    query = "哪种语言适合 Web 开发？"

    # VectorStoreIndex 查询
    print("\n--- VectorStoreIndex ---")
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_engine = vector_index.as_query_engine(similarity_top_k=2)
    vector_response = vector_engine.query(query)
    print(f"响应: {vector_response.response[:80]}...")

    # ListIndex 查询
    print("\n--- ListIndex ---")
    list_index = ListIndex.from_documents(documents)
    list_engine = list_index.as_query_engine(response_mode="compact")
    list_response = list_engine.query(query)
    print(f"响应: {list_response.response[:80]}...")

    print("\n总结:")
    print("  VectorStoreIndex: 语义检索，找到最相关节点")
    print("  ListIndex: 遍历所有节点，综合生成答案")


# ==========================================
# 示例 8: 自定义分块后创建索引
# ==========================================


def example_custom_chunking_index():
    """演示自定义分块参数创建索引"""
    print("\n" + "=" * 50)
    print("示例 8: 自定义分块后创建索引")
    print("=" * 50)

    setup_settings()

    # 设置全局分块参数
    Settings.chunk_size = 256
    Settings.chunk_overlap = 20

    # 创建文档
    doc = Document(
        text="LlamaIndex 支持自定义分块参数。"
        "chunk_size 控制每块的最大 token 数。"
        "chunk_overlap 控制相邻块的重叠 token 数。"
        "合理的分块参数可以提高检索质量。"
        "过大的分块会导致信息冗余。"
        "过小的分块可能丢失上下文。"
        "建议根据文档类型调整参数。"
    )

    print(f"全局设置: chunk_size={Settings.chunk_size}, overlap={Settings.chunk_overlap}")

    # 使用自定义分块器
    splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents([doc])
    print(f"分块后节点数: {len(nodes)}")

    # 从节点创建索引
    index = VectorStoreIndex(nodes)

    print("✅ 使用自定义分块创建索引成功")

    # 查询测试
    query_engine = index.as_query_engine()
    response = query_engine.query("分块参数如何影响检索质量？")
    print(f"\n查询结果: {response.response[:100]}...")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 17: 文档与索引示例")
    print("=" * 60)

    # 运行各示例
    example_document_loading()
    example_chunking_strategies()
    example_vector_store_index()
    example_list_index()
    example_tree_index()
    example_summary_index()
    example_index_comparison()
    example_custom_chunking_index()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()