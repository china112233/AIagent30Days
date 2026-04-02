"""
Day 17: LlamaIndex 数据索引示例

本文件演示数据索引的核心机制：
- 多种数据加载器使用
- 文档分块策略配置
- 不同索引类型创建
- 向量存储集成
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    SummaryIndex,
    KeywordTableIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
)
from llama_index.llms.openai_like import OpenAILikeLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 加载环境变量
load_dotenv()


def setup_settings():
    """配置全局设置"""
    llm = OpenAILikeLLM(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
        temperature=0.7,
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model


# ==========================================
# 示例 1: 基础分块策略
# ==========================================


def example_basic_chunking():
    """演示基础文本分块策略"""
    print("\n" + "=" * 50)
    print("示例 1: 基础分块策略")
    print("=" * 50)

    # 创建长文档
    long_text = """
    数据索引是 LlamaIndex 的核心功能之一。
    通过合理的分块策略，可以提高检索的精度和效率。

    分块策略需要考虑几个关键因素：
    1. 分块大小：影响检索粒度和上下文完整性
    2. 分块重叠：保持上下文连贯性
    3. 分块边界：按句子、段落或语义边界分割

    常用的分块器包括：
    - SentenceSplitter：按句子分割
    - TokenTextSplitter：按 token 数量分割
    - SemanticSplitterNodeParser：按语义边界分割

    选择合适的分块策略取决于：
    - 文档类型和结构
    - 查询场景需求
    - Embedding 模型的特性
    """

    document = Document(text=long_text)

    # 使用不同的分块参数
    configs = [
        {"chunk_size": 100, "chunk_overlap": 20},
        {"chunk_size": 200, "chunk_overlap": 50},
        {"chunk_size": 400, "chunk_overlap": 100},
    ]

    for config in configs:
        print(f"\n  配置: chunk_size={config['chunk_size']}, overlap={config['chunk_overlap']}")

        splitter = SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

        nodes = splitter.get_nodes_from_documents([document])

        print(f"    生成节点数: {len(nodes)}")
        print(f"    平均节点长度: {sum(len(n.text) for n in nodes) / len(nodes):.1f}")


# ==========================================
# 示例 2: Token 分块器
# ==========================================


def example_token_splitter():
    """演示 Token 级别的分块"""
    print("\n" + "=" * 50)
    print("示例 2: Token 分块器")
    print("=" * 50)

    text = """
    Token 分块器按照 token 数量进行分块，这样可以精确控制每个分块的大小。
    这对于有 token 限制的模型特别有用。

    Token 分块器会考虑：
    - 每个 token 的实际长度
    - 边界 token 的处理
    - 不会在单词中间分割

    适合用于：
    - 需要精确控制大小的场景
    - 有严格 token 限制的模型
    """

    document = Document(text=text)

    # Token 分块器
    splitter = TokenTextSplitter(
        chunk_size=50,  # 每个 chunk 最多 50 tokens
        chunk_overlap=10,
    )

    nodes = splitter.get_nodes_from_documents([document])

    print(f"  生成了 {len(nodes)} 个节点:")
    for i, node in enumerate(nodes):
        print(f"\n    Node {i+1}:")
        print(f"      内容: {node.text[:70]}...")


# ==========================================
# 示例 3: 向量索引创建
# ==========================================


def example_vector_index():
    """演示向量索引的创建和使用"""
    print("\n" + "=" * 50)
    print("示例 3: 向量索引")
    print("=" * 50)

    documents = [
        Document(text="向量索引是最常用的索引类型，使用 Embedding 向量进行相似度检索。"),
        Document(text="创建向量索引时，系统会自动将文档转换为 Embedding 向量。"),
        Document(text="向量索引适合语义检索场景，能够找到语义相似的内容。"),
        Document(text="向量索引可以与多种向量数据库集成，如 Chroma、Pinecone、Milvus。"),
    ]

    print(f"  创建向量索引，包含 {len(documents)} 个文档...")

    # 创建向量索引
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    print("  索引创建完成！")

    # 查询
    query_engine = index.as_query_engine(similarity_top_k=2)

    response = query_engine.query("向量索引适合什么场景？")
    print(f"\n  查询响应: {response.response}")

    # 查看检索分数
    print("\n  检索结果相似度分数:")
    for node in response.source_nodes:
        print(f"    分数: {node.score:.4f}, 文本: {node.node.text[:50]}...")


# ==========================================
# 示例 4: 摘要索引
# ==========================================


def example_summary_index():
    """演示摘要索引的使用"""
    print("\n" + "=" * 50)
    print("示例 4: 摘要索引")
    print("=" * 50)

    documents = [
        Document(text="产品 A 是一款高性能服务器，适合企业级应用部署。"),
        Document(text="产品 B 是一款轻量级云服务，适合个人开发者和小团队。"),
        Document(text="产品 C 是一款数据分析平台，提供可视化报表和实时监控。"),
    ]

    print("  创建摘要索引...")

    # 摘要索引会遍历所有节点
    index = SummaryIndex.from_documents(documents)

    print("  索引创建完成！")

    # 查询（摘要索引会处理所有相关内容）
    query_engine = index.as_query_engine()

    response = query_engine.query("请总结所有产品的特点")
    print(f"\n  查询响应: {response.response}")


# ==========================================
# 示例 5: 关键词索引
# ==========================================


def example_keyword_index():
    """演示关键词索引的使用"""
    print("\n" + "=" * 50)
    print("示例 5: 关键词索引")
    print("=" * 50)

    documents = [
        Document(text="Python 编程语言以其简洁的语法和丰富的库生态系统著称。"),
        Document(text="JavaScript 是 Web 开发中不可或缺的语言，支持前端和后端开发。"),
        Document(text="Java 语言在企业应用开发中广泛应用，具有跨平台和稳定性优势。"),
        Document(text="Go 语言由 Google 开发，专注于并发编程和高性能服务。"),
    ]

    print("  创建关键词索引...")

    # 关键词索引基于关键词匹配
    index = KeywordTableIndex.from_documents(documents)

    print("  索引创建完成！")

    # 查询
    query_engine = index.as_query_engine()

    # 关键词明确的查询效果更好
    response = query_engine.query("Python 语言有什么特点？")
    print(f"\n  查询响应: {response.response}")


# ==========================================
# 示例 6: 从节点创建索引
# ==========================================


def example_index_from_nodes():
    """演示从节点直接创建索引"""
    print("\n" + "=" * 50)
    print("示例 6: 从节点创建索引")
    print("=" * 50)

    from llama_index.core.schema import TextNode

    # 直接创建节点
    nodes = [
        TextNode(
            text="节点可以直接创建，不依赖于文档。",
            metadata={"source": "manual", "type": "concept"},
        ),
        TextNode(
            text="从节点创建索引提供了更灵活的控制。",
            metadata={"source": "manual", "type": "usage"},
        ),
        TextNode(
            text="可以自定义节点的元数据和关系。",
            metadata={"source": "manual", "type": "advanced"},
        ),
    ]

    print(f"  直接创建了 {len(nodes)} 个节点")

    # 从节点创建索引
    index = VectorStoreIndex(nodes)

    print("  索引创建完成！")

    # 查询
    query_engine = index.as_query_engine()

    response = query_engine.query("节点创建有什么优势？")
    print(f"\n  响应: {response.response}")


# ==========================================
# 示例 7: 分块参数对检索的影响
# ==========================================


def example_chunk_size_impact():
    """演示分块大小对检索效果的影响"""
    print("\n" + "=" * 50)
    print("示例 7: 分块大小对检索的影响")
    print("=" * 50)

    # 创建较长的文档
    full_text = """
    数据分块是 RAG 系统的关键环节。

    小分块（如 100-200 tokens）的优势：
    - 检索粒度精细，能准确定位相关信息
    - 避免无关信息干扰响应生成
    - 适合精确问答场景

    小分块的劣势：
    - 可能丢失上下文信息
    - 需要更多节点覆盖完整信息
    - 检索次数可能增加

    大分块（如 500-1000 tokens）的优势：
    - 保持更完整的上下文
    - 减少 Fragmentation 问题
    - 适合需要完整理解的场景

    大分块的劣势：
    - 可能包含过多无关信息
    - 检索精度可能下降
    - 响应可能不够精简

    最佳实践：
    - 根据查询类型选择：精确问答用小分块，理解性查询用大分块
    - 使用重叠保持上下文：推荐 10-20% 的重叠率
    - 结合多种策略：可以先粗检索再精细定位
    """

    document = Document(text=full_text)

    # 测试不同分块大小
    chunk_sizes = [100, 300, 500]

    query = "小分块有什么优势？"

    print(f"\n  测试查询: {query}")

    for chunk_size in chunk_sizes:
        print(f"\n  --- chunk_size={chunk_size} ---")

        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),
        )

        nodes = splitter.get_nodes_from_documents([document])
        index = VectorStoreIndex(nodes)

        query_engine = index.as_query_engine(similarity_top_k=2)
        response = query_engine.query(query)

        print(f"    节点数: {len(nodes)}")
        print(f"    响应: {response.response[:100]}...")


# ==========================================
# 示例 8: 模拟目录读取
# ==========================================


def example_directory_reader():
    """演示使用目录读取器加载文档"""
    print("\n" + "=" * 50)
    print("示例 8: 目录读取器（模拟）")
    print("=" * 50)

    # 创建临时示例目录
    demo_dir = "./day17/demo_docs"
    os.makedirs(demo_dir, exist_ok=True)

    # 创建示例文件
    sample_files = {
        "tech_doc.txt": """
        技术文档标题

        本文档介绍了系统的主要功能和技术架构。

        主要功能包括：
        - 用户管理
        - 数据处理
        - 报表生成

        技术架构采用分层设计，便于维护和扩展。
        """,
        "user_guide.txt": """
        用户指南

       欢迎使用本系统。

        快速入门：
        1. 登录系统
        2. 配置参数
        3. 开始使用

        如有问题，请查阅帮助文档。
        """,
    }

    for filename, content in sample_files.items():
        filepath = os.path.join(demo_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"  创建了 {len(sample_files)} 个示例文件")

    # 使用目录读取器
    print("  使用 SimpleDirectoryReader 加载文档...")

    reader = SimpleDirectoryReader(
        input_dir=demo_dir,
        required_exts=[".txt"],  # 只读取 .txt 文件
    )

    documents = reader.load_data()

    print(f"  加载了 {len(documents)} 个文档")

    # 显示文档信息
    for i, doc in enumerate(documents):
        print(f"\n    Document {i+1}:")
        print(f"      来源: {doc.metadata.get('file_name', 'unknown')}")
        print(f"      内容长度: {len(doc.text)} 字符")

    # 创建索引
    index = VectorStoreIndex.from_documents(documents)

    # 查询
    query_engine = index.as_query_engine()
    response = query_engine.query("系统有哪些主要功能？")

    print(f"\n  查询响应: {response.response}")

    # 清理
    import shutil
    shutil.rmtree(demo_dir)
    print(f"\n  已清理示例目录")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 17: LlamaIndex 数据索引示例")
    print("=" * 60)

    # 配置设置
    setup_settings()

    # 运行各示例
    example_basic_chunking()
    example_token_splitter()
    example_vector_index()
    example_summary_index()
    example_keyword_index()
    example_index_from_nodes()
    example_chunk_size_impact()
    example_directory_reader()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()