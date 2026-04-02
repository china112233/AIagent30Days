"""
Day 17: LlamaIndex 基础示例

本文件演示 LlamaIndex 的核心概念：
- LLM 和嵌入模型配置（使用 DeepSeek API）
- Document 创建与加载
- 简单向量索引创建
- 基本查询操作
"""

import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()


def setup_llm():
    """配置 LLM（使用 DeepSeek API）"""
    llm = OpenAILike(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
        temperature=0.7,
    )
    return llm


def setup_embed_model():
    """配置嵌入模型
    注意：DeepSeek 不提供嵌入 API，这里使用 OpenAI 兼容接口
    实际使用时可以换成其他嵌入服务或本地模型
    """
    # 使用 OpenAI 嵌入模型作为示例
    # 如果没有 OpenAI API Key，可以使用本地嵌入模型
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY", os.getenv("DEEPSEEK_API_KEY")),
        api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    return embed_model


def configure_settings():
    """全局配置 LlamaIndex Settings"""
    llm = setup_llm()
    embed_model = setup_embed_model()

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    print("✅ LLM 和嵌入模型已配置")
    print(f"   LLM: {Settings.llm.model}")
    print(f"   Chunk Size: {Settings.chunk_size}")


# ==========================================
# 示例 1: 创建简单 Document
# ==========================================


def example_create_document():
    """演示如何创建 Document 对象"""
    print("\n" + "=" * 50)
    print("示例 1: 创建简单 Document")
    print("=" * 50)

    # 创建单个文档
    doc = Document(
        text="LlamaIndex 是一个强大的数据框架，"
        "它可以帮助开发者将私有数据与大语言模型连接。"
        "通过构建索引，用户可以高效地检索和查询数据。",
        metadata={"source": "intro", "category": "tech"},
    )

    print(f"文档内容: {doc.text[:50]}...")
    print(f"文档元数据: {doc.metadata}")
    print(f"文档 ID: {doc.doc_id}")


# ==========================================
# 示例 2: 从文本创建索引
# ==========================================


def example_create_index_from_text():
    """演示从文本创建向量索引"""
    print("\n" + "=" * 50)
    print("示例 2: 从文本创建索引")
    print("=" * 50)

    configure_settings()

    # 创建多个文档
    documents = [
        Document(
            text="Python 是一种流行的编程语言，"
            "它以简洁和易读著称。Python 广泛用于数据分析、"
            "Web 开发和人工智能领域。",
            metadata={"topic": "python", "category": "programming"},
        ),
        Document(
            text="LlamaIndex 是一个数据框架，"
            "专注于将数据与大语言模型连接。"
            "它提供了多种索引类型和查询引擎。",
            metadata={"topic": "llamaindex", "category": "ai"},
        ),
        Document(
            text="向量数据库是存储嵌入向量的专用数据库，"
            "支持高效的相似度检索。常见的有 Pinecone、"
            "Weaviate 和 Chroma。",
            metadata={"topic": "vectordb", "category": "database"},
        ),
    ]

    print(f"创建了 {len(documents)} 个文档")

    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)

    print("✅ 向量索引已创建")
    print(f"   索引类型: {type(index).__name__}")


# ==========================================
# 示例 3: 基本查询操作
# ==========================================


def example_basic_query():
    """演示基本查询操作"""
    print("\n" + "=" * 50)
    print("示例 3: 基本查询操作")
    print("=" * 50)

    configure_settings()

    # 创建测试文档
    documents = [
        Document(
            text="LangChain 是一个 LLM 应用开发框架，"
            "提供了链式调用、Agent 和 Memory 等组件。"
            "它支持多种 LLM 后端和工具集成。",
            metadata={"topic": "langchain"},
        ),
        Document(
            text="LlamaIndex 专注于数据连接和检索，"
            "是构建 RAG 应用的首选框架。"
            "它提供了多种索引类型和查询引擎。",
            metadata={"topic": "llamaindex"},
        ),
    ]

    # 创建索引并查询
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 执行查询
    query = "什么是 LlamaIndex？"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response[:200]}...")

    # 查看检索的节点
    print(f"\n检索的节点数: {len(response.source_nodes)}")
    for i, node in enumerate(response.source_nodes):
        print(f"  Node {i+1}: {node.node.text[:50]}...")
        print(f"  相似度: {node.score:.4f}")


# ==========================================
# 示例 4: 节点解析与分块
# ==========================================


def example_node_parsing():
    """演示文档分块和节点解析"""
    print("\n" + "=" * 50)
    print("示例 4: 节点解析与分块")
    print("=" * 50)

    # 创建长文档
    long_text = """
    LlamaIndex 是一个强大的数据框架。

    核心组件包括：
    1. Document：原始数据的容器
    2. Node：文档的分块单元
    3. Index：节点的组织结构

    主要功能：
    - 数据加载：支持多种格式
    - 索引创建：向量、列表、树等
    - 查询引擎：灵活的检索配置

    应用场景：
    - RAG 问答系统
    - 文档摘要
    - 知识库构建
    """

    doc = Document(text=long_text, metadata={"source": "tutorial"})

    # 使用 SentenceSplitter 分块
    splitter = SentenceSplitter(
        chunk_size=100,  # 较小的分块便于演示
        chunk_overlap=20,
    )

    nodes = splitter.get_nodes_from_documents([doc])

    print(f"原始文档长度: {len(long_text)} 字符")
    print(f"分块后节点数: {len(nodes)}")

    for i, node in enumerate(nodes):
        print(f"\nNode {i+1}:")
        print(f"  内容: {node.text[:60]}...")
        print(f"  ID: {node.node_id}")
        print(f"  元数据: {node.metadata}")


# ==========================================
# 示例 5: 从节点创建索引
# ==========================================


def example_index_from_nodes():
    """演示从节点创建索引"""
    print("\n" + "=" * 50)
    print("示例 5: 从节点创建索引")
    print("=" * 50)

    configure_settings()

    # 创建文档并分块
    doc = Document(
        text="机器学习是人工智能的一个分支，"
        "它让计算机能够从数据中学习。"
        "深度学习是机器学习的子领域，"
        "使用神经网络进行学习。"
        "强化学习通过奖励机制训练模型。",
        metadata={"topic": "ml"},
    )

    # 分块
    splitter = SentenceSplitter(chunk_size=50, chunk_overlap=10)
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"创建了 {len(nodes)} 个节点")

    # 从节点创建索引
    index = VectorStoreIndex(nodes)

    print("✅ 从节点创建索引成功")

    # 查询
    query_engine = index.as_query_engine()
    response = query_engine.query("什么是深度学习？")

    print(f"\n查询: 什么是深度学习？")
    print(f"响应: {response.response}")


# ==========================================
# 示例 6: 流式查询
# ==========================================


def example_streaming_query():
    """演示流式查询输出"""
    print("\n" + "=" * 50)
    print("示例 6: 流式查询")
    print("=" * 50)

    configure_settings()

    # 创建文档和索引
    documents = [
        Document(
            text="RAG（检索增强生成）是一种结合检索和生成的技术。"
            "它首先从知识库检索相关文档，"
            "然后将检索结果作为上下文输入给 LLM 生成答案。"
            "RAG 可以减少幻觉，提高回答的准确性。",
            metadata={"topic": "rag"},
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # 创建流式查询引擎
    query_engine = index.as_query_engine(streaming=True)

    query = "请解释 RAG 技术的工作原理"
    print(f"\n查询: {query}")
    print("流式响应:")

    # 流式输出
    streaming_response = query_engine.query(query)
    for text in streaming_response.response_gen:
        print(text, end="", flush=True)

    print("\n\n✅ 流式查询完成")


# ==========================================
# 示例 7: 索引持久化（简单存储）
# ==========================================


def example_index_persistence():
    """演示索引的保存和加载"""
    print("\n" + "=" * 50)
    print("示例 7: 索引持久化")
    print("=" * 50)

    configure_settings()

    # 创建文档和索引
    documents = [
        Document(text="知识库是一个存储知识的系统。"),
        Document(text="向量数据库用于存储嵌入向量。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # 保存索引到本地
    storage_dir = "./day17/storage_demo"
    index.storage_context.persist(persist_dir=storage_dir)

    print(f"✅ 索引已保存到: {storage_dir}")

    # 加载索引
    from llama_index.core import StorageContext, load_index_from_storage

    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    loaded_index = load_index_from_storage(storage_context)

    print("✅ 索引已加载")

    # 验证加载的索引
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("什么是知识库？")

    print(f"\n查询结果: {response.response}")


# ==========================================
# 示例 8: 多文档查询对比
# ==========================================


def example_multi_document_query():
    """演示多个文档的查询效果"""
    print("\n" + "=" * 50)
    print("示例 8: 多文档查询对比")
    print("=" * 50)

    configure_settings()

    # 创建不同主题的文档
    documents = [
        Document(
            text="Python 是一种解释型编程语言，"
            "由 Guido van Rossum 创建。"
            "Python 的设计哲学强调代码可读性。",
            metadata={"topic": "python", "type": "language"},
        ),
        Document(
            text="JavaScript 是 Web 开发的核心语言，"
            "用于创建交互式网页。"
            "Node.js 让 JavaScript 可以运行在服务器端。",
            metadata={"topic": "javascript", "type": "language"},
        ),
        Document(
            text="SQL 是结构化查询语言，"
            "用于管理关系型数据库。"
            "SQL 可以执行查询、更新和删除操作。",
            metadata={"topic": "sql", "type": "database"},
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # 测试不同查询
    queries = [
        "哪种语言适合 Web 开发？",
        "数据库查询用什么语言？",
        "Python 有什么特点？",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        response = query_engine.query(query)
        print(f"响应: {response.response[:100]}...")
        print(f"相关节点: {len(response.source_nodes)}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 17: LlamaIndex 基础示例")
    print("=" * 60)

    # 运行各示例
    example_create_document()
    example_create_index_from_text()
    example_basic_query()
    example_node_parsing()
    example_index_from_nodes()
    example_streaming_query()
    example_index_persistence()
    example_multi_document_query()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()