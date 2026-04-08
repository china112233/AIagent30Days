"""
Day 18: LlamaIndex 高级检索策略示例

本文件演示多种高级检索技术：
- 混合检索（向量 + BM25）
- HyDE 查询变换
- 查询重写与扩展
- Cross-Encoder 重排序
- 自定义检索器组合
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleKeywordTableIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量
load_dotenv()


def setup_settings():
    """配置全局设置"""
    llm = OpenAILike(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
        temperature=0.7,
    )

    # 使用本地嵌入模型（避免 API 调用）
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    print("✅ 设置已配置")
    return llm, embed_model


def create_sample_documents():
    """创建示例文档"""
    documents = [
        Document(
            text="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
            "Python 以其简洁的语法和强大的生态系统著称，广泛用于 Web 开发、"
            "数据科学、人工智能和自动化脚本等领域。",
            metadata={"topic": "python", "category": "programming"},
        ),
        Document(
            text="机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式。"
            "常见的机器学习算法包括线性回归、决策树、神经网络和支持向量机。"
            "深度学习是机器学习的子领域，使用多层神经网络进行学习。",
            metadata={"topic": "ml", "category": "ai"},
        ),
        Document(
            text="向量数据库是专门用于存储和检索向量嵌入的数据库系统。"
            "主流向量数据库包括 Pinecone、Weaviate、Milvus 和 Chroma。"
            "向量数据库支持高效的相似度搜索，是 RAG 系统的核心组件。",
            metadata={"topic": "vectordb", "category": "database"},
        ),
        Document(
            text="LangChain 是一个 LLM 应用开发框架，提供链式调用、Agent 和 Memory 等组件。"
            "LangChain 支持多种 LLM 后端，包括 OpenAI、Anthropic 和本地模型。"
            "它可以与向量数据库、API 和各种工具进行集成。",
            metadata={"topic": "langchain", "category": "framework"},
        ),
        Document(
            text="RAG（检索增强生成）是一种结合检索和生成的技术架构。"
            "RAG 系统首先从知识库检索相关文档，然后将检索内容作为上下文输入给 LLM。"
            "这种方法可以减少幻觉，提高回答的准确性和可靠性。",
            metadata={"topic": "rag", "category": "architecture"},
        ),
        Document(
            text="知识图谱是一种结构化的知识表示方法，使用实体和关系来描述信息。"
            "知识图谱可以增强 RAG 系统的检索能力，提供更精确的语义理解。"
            "常见的知识图谱应用包括搜索引擎、推荐系统和智能问答。",
            metadata={"topic": "kg", "category": "ai"},
        ),
    ]

    return documents


# ==========================================
# 示例 1: 基础向量检索
# ==========================================


def example_basic_vector_retrieval():
    """演示基础向量检索"""
    print("\n" + "=" * 50)
    print("示例 1: 基础向量检索")
    print("=" * 50)

    documents = create_sample_documents()

    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)

    # 基础检索配置
    query_engine = index.as_query_engine(
        similarity_top_k=3,
    )

    query = "什么是 Python 语言？"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response[:150]}...")

    # 查看检索结果
    print("\n检索到的节点:")
    for i, node in enumerate(response.source_nodes):
        print(f"  Node {i+1}: 相似度 {node.score:.4f}")
        print(f"    内容: {node.node.text[:60]}...")


# ==========================================
# 示例 2: BM25 关键词检索
# ==========================================


def example_bm25_retrieval():
    """演示 BM25 关键词检索"""
    print("\n" + "=" * 50)
    print("示例 2: BM25 关键词检索")
    print("=" * 50)

    documents = create_sample_documents()

    # 分块
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=40)
    nodes = splitter.get_nodes_from_documents(documents)

    # 使用关键词表索引（类似 BM25）
    from llama_index.core.retrievers import KeywordTableSimpleRetriever

    keyword_index = SimpleKeywordTableIndex(nodes)
    keyword_retriever = KeywordTableSimpleRetriever(
        index=keyword_index,
        keyword_mode="simple",  # 简单关键词匹配
    )

    query = "向量数据库 存储 检索"
    print(f"\n查询（关键词模式）: {query}")

    retrieved_nodes = keyword_retriever.retrieve(query)
    print(f"检索到 {len(retrieved_nodes)} 个节点")

    for i, node in enumerate(retrieved_nodes):
        print(f"  Node {i+1}: {node.node.text[:60]}...")


# ==========================================
# 示例 3: 混合检索策略
# ==========================================


def example_hybrid_retrieval():
    """演示混合检索（向量 + 关键词）"""
    print("\n" + "=" * 50)
    print("示例 3: 混合检索策略")
    print("=" * 50)

    documents = create_sample_documents()

    # 分块
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=40)
    nodes = splitter.get_nodes_from_documents(documents)

    # 创建向量索引
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3,
    )

    # 创建关键词索引
    keyword_index = SimpleKeywordTableIndex(nodes)
    from llama_index.core.retrievers import KeywordTableSimpleRetriever
    keyword_retriever = KeywordTableSimpleRetriever(
        index=keyword_index,
    )

    # 混合检索：组合两个检索器
    from llama_index.core.retrievers import BaseRetriever

    class HybridRetriever(BaseRetriever):
        """自定义混合检索器"""

        def __init__(self, vector_retriever, keyword_retriever):
            self._vector_retriever = vector_retriever
            self._keyword_retriever = keyword_retriever
            super().__init__()

        def _retrieve(self, query_bundle):
            """执行混合检索"""
            # 向量检索
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            # 关键词检索
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

            # 合并并去重
            all_nodes = []
            node_ids = set()

            for node in vector_nodes + keyword_nodes:
                if node.node_id not in node_ids:
                    all_nodes.append(node)
                    node_ids.add(node.node_id)

            # 按分数排序
            all_nodes.sort(key=lambda x: x.score or 0, reverse=True)
            return all_nodes[:6]  # 返回前 6 个

    # 创建混合检索器
    hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)

    print("  混合检索配置:")
    print("    - 向量检索: top_k=3")
    print("    - 关键词检索: 全关键词匹配")
    print("    - 合并去重后返回前 6 个")

    query = "RAG 检索生成系统"
    print(f"\n查询: {query}")

    # 使用混合检索器创建查询引擎
    response_synth = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synth,
    )

    response = query_engine.query(query)
    print(f"响应: {response.response[:150]}...")

    print(f"\n检索节点数: {len(response.source_nodes)}")


# ==========================================
# 示例 4: HyDE 查询变换
# ==========================================


def example_hyde_transform():
    """演示 HyDE（假设文档嵌入）查询变换"""
    print("\n" + "=" * 50)
    print("示例 4: HyDE 查询变换")
    print("=" * 50)

    documents = create_sample_documents()

    index = VectorStoreIndex.from_documents(documents)

    print("  HyDE 工作原理:")
    print("    1. 为用户查询生成假设文档")
    print("    2. 使用假设文档的嵌入进行检索")
    print("    3. 返回实际文档节点")
    print("    4. 使用原始查询生成最终响应")

    # 创建 HyDE 变换
    hyde_transform = HyDEQueryTransform(include_original=True)

    # 应用 HyDE 的查询引擎
    base_engine = index.as_query_engine(similarity_top_k=3)
    hyde_engine = TransformQueryEngine(
        query_engine=base_engine,
        query_transform=hyde_transform,
    )

    query = "如何使用向量数据库构建搜索系统"
    print(f"\n查询: {query}")
    print("  （这是一个需要假设文档的模糊查询）")

    response = hyde_engine.query(query)
    print(f"响应: {response.response[:150]}...")


# ==========================================
# 示例 5: 查询重写
# ==========================================


def example_query_rewriting():
    """演示查询重写优化"""
    print("\n" + "=" * 50)
    print("示例 5: 查询重写")
    print("=" * 50)

    documents = create_sample_documents()

    index = VectorStoreIndex.from_documents(documents)

    print("  查询重写策略:")
    print("    - 将模糊查询转换为精确查询")
    print("    - 扩展查询关键词")
    print("    - 添加领域上下文")

    # 自定义查询重写
    from llama_index.core.indices.query.query_transform import QueryTransform

    class QueryRewriter(QueryTransform):
        """自定义查询重写器"""

        def _run(self, query_bundle):
            """重写查询"""
            original_query = query_bundle.query_str

            # 查询扩展逻辑
            rewrite_map = {
                "数据库": "向量数据库 存储 检索 嵌入",
                "AI": "人工智能 机器学习 深度学习",
                "框架": "开发框架 LangChain LlamaIndex",
            }

            rewritten = original_query
            for key, expansion in rewrite_map.items():
                if key in original_query:
                    rewritten = f"{original_query} {expansion}"

            return query_bundle.copy_with_query_str(rewritten)

    # 应用查询重写
    rewriter = QueryRewriter()
    base_engine = index.as_query_engine(similarity_top_k=4)
    rewritten_engine = TransformQueryEngine(
        query_engine=base_engine,
        query_transform=rewriter,
    )

    query = "有哪些常用的 AI 框架？"
    print(f"\n原始查询: {query}")

    response = rewritten_engine.query(query)
    print(f"响应: {response.response[:150]}...")


# ==========================================
# 示例 6: Re-ranking（重排序）
# ==========================================


def example_reranking():
    """演示检索结果重排序"""
    print("\n" + "=" * 50)
    print("示例 6: Re-ranking（重排序）")
    print("=" * 50)

    documents = create_sample_documents()

    index = VectorStoreIndex.from_documents(documents)

    print("  重排序流程:")
    print("    1. 初始检索获取较多候选节点")
    print("    2. 使用重排序模型精排")
    print("    3. 选取 top-n 高质量节点")
    print("    4. 生成最终响应")

    # 使用 LLM 进行重排序（替代 Cohere）
    from llama_index.core.postprocessor import LLMRerank

    reranker = LLMRerank(
        top_n=3,  # 重排序后保留 3 个
        choice_batch_size=5,  # 每次处理 5 个候选
    )

    # 配置重排序的查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=8,  # 先检索 8 个候选
        node_postprocessors=[reranker],  # 应用重排序
    )

    query = "请介绍 Python 和机器学习的区别"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response[:150]}...")

    print(f"\n最终节点数（重排序后）: {len(response.source_nodes)}")


# ==========================================
# 示例 7: 多查询扩展
# ==========================================


def example_multi_query_expansion():
    """演示多查询扩展检索"""
    print("\n" + "=" * 50)
    print("示例 7: 多查询扩展")
    print("=" * 50)

    documents = create_sample_documents()

    index = VectorStoreIndex.from_documents(documents)

    print("  多查询扩展策略:")
    print("    1. 将原查询分解为多个子查询")
    print("    2. 对每个子查询独立检索")
    print("    3. 合并所有检索结果")
    print("    4. 去重并生成响应")

    # 使用 LLM 生成多查询
    from llama_index.core.indices.query.query_transform import MultiQueryTransform

    multi_query = MultiQueryTransform()

    base_engine = index.as_query_engine(similarity_top_k=3)
    multi_engine = TransformQueryEngine(
        query_engine=base_engine,
        query_transform=multi_query,
    )

    query = "比较不同的数据库技术"
    print(f"\n查询: {query}")

    response = multi_engine.query(query)
    print(f"响应: {response.response[:150]}...")


# ==========================================
# 示例 8: 综合检索策略对比
# ==========================================


def example_retrieval_comparison():
    """对比不同检索策略效果"""
    print("\n" + "=" * 50)
    print("示例 8: 检索策略对比")
    print("=" * 50)

    documents = create_sample_documents()

    # 创建基础索引
    index = VectorStoreIndex.from_documents(documents)

    # 不同策略的配置
    strategies = {
        "基础向量检索": index.as_query_engine(similarity_top_k=3),
        "大 Top-K": index.as_query_engine(similarity_top_k=6),
        "重排序": index.as_query_engine(
            similarity_top_k=6,
            node_postprocessors=[LLMRerank(top_n=3)],
        ),
    }

    test_query = "什么是 RAG 技术？"
    print(f"\n测试查询: {test_query}")
    print("\n不同策略的响应结果:\n")

    for name, engine in strategies.items():
        print(f"【{name}】")
        response = engine.query(test_query)
        print(f"  响应: {response.response[:100]}...")
        print(f"  检索节点: {len(response.source_nodes)}")
        print()


# ==========================================
# 示例 9: 自定义检索权重
# ==========================================


def example_custom_retrieval_weights():
    """演示自定义检索权重"""
    print("\n" + "=" * 50)
    print("示例 9: 自定义检索权重")
    print("=" * 50)

    documents = create_sample_documents()

    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=40)
    nodes = splitter.get_nodes_from_documents(documents)

    # 向量索引
    vector_index = VectorStoreIndex(nodes)

    # 关键词索引
    keyword_index = SimpleKeywordTableIndex(nodes)

    print("  权重配置说明:")
    print("    - 向量检索权重: 语义相似度重要性")
    print("    - 关键词检索权重: 精确匹配重要性")
    print("    - alpha 参数: 平衡两者权重")

    # 使用 LlamaIndex 内置的混合检索权重
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever

    # 创建两个检索器
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=3,
    )

    bm25_retriever = BM25Retriever(
        nodes=nodes,
        similarity_top_k=3,
    )

    # 融合检索器（自动加权）
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",  # 使用 Reciprocal Rank Fusion
    )

    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever,
    )

    query = "向量数据库的特点"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response[:150]}...")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 18: LlamaIndex 高级检索策略示例")
    print("=" * 60)

    # 配置设置
    setup_settings()

    # 运行各示例
    example_basic_vector_retrieval()
    example_bm25_retrieval()
    example_hybrid_retrieval()
    example_hyde_transform()
    example_query_rewriting()
    example_reranking()
    example_multi_query_expansion()
    example_retrieval_comparison()
    example_custom_retrieval_weights()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()