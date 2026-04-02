"""
Day 17: LlamaIndex 查询引擎示例

本文件演示查询引擎的核心功能：
- 查询引擎配置与定制
- 多种检索器使用
- 响应合成策略
- 流式查询输出
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    SubQuestionQueryEngine,
)
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.retrievers import (
    VectorIndexRetriever,
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


def create_sample_index():
    """创建示例索引"""
    documents = [
        Document(
            text="RAG（检索增强生成）系统由三个核心组件组成：检索器、生成器和知识库。"
            "检索器负责从知识库中找到与查询相关的信息，"
            "生成器基于检索结果生成最终响应。"
        ),
        Document(
            text="向量检索是最常用的检索方式，通过计算查询向量与文档向量的相似度进行匹配。"
            "常用的相似度度量包括余弦相似度、欧氏距离等。"
        ),
        Document(
            text="响应合成是 RAG 的关键步骤，需要将多个检索片段整合成连贯的回答。"
            "常用的合成策略包括：精简（compact）、逐段优化（refine）、树摘要（tree_summarize）。"
        ),
        Document(
            text="查询引擎是 LlamaIndex 的核心接口，组合了检索器和响应合成器。"
            "可以配置检索参数、响应模式、流式输出等选项。"
        ),
        Document(
            text="检索参数如 similarity_top_k 控制检索的节点数量，"
            "过少可能遗漏相关信息，过多则引入噪音。"
            "通常设置在 3-10 之间。"
        ),
    ]

    return VectorStoreIndex.from_documents(documents)


# ==========================================
# 示例 1: 基础查询引擎配置
# ==========================================


def example_basic_query_engine():
    """演示基础查询引擎配置"""
    print("\n" + "=" * 50)
    print("示例 1: 基础查询引擎配置")
    print("=" * 50)

    index = create_sample_index()

    # 不同配置的查询引擎
    configs = [
        {"similarity_top_k": 2, "response_mode": "default"},
        {"similarity_top_k": 3, "response_mode": "compact"},
        {"similarity_top_k": 5, "response_mode": "refine"},
    ]

    query = "RAG 系统有哪些核心组件？"

    for config in configs:
        print(f"\n  --- 配置: top_k={config['similarity_top_k']}, mode={config['response_mode']} ---")

        query_engine = index.as_query_engine(
            similarity_top_k=config["similarity_top_k"],
            response_mode=config["response_mode"],
        )

        response = query_engine.query(query)
        print(f"  响应: {response.response[:150]}...")


# ==========================================
# 示例 2: 自定义检索器
# ==========================================


def example_custom_retriever():
    """演示自定义检索器配置"""
    print("\n" + "=" * 50)
    print("示例 2: 自定义检索器")
    print("=" * 50)

    index = create_sample_index()

    # 创建自定义检索器
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        # 可以添加其他检索参数
    )

    # 创建自定义响应合成器
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
    )

    # 组合创建查询引擎
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    print("  创建了自定义检索器和响应合成器")

    # 执行查询
    query = "什么是向量检索？"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)
    print(f"  响应: {response.response}")

    # 查看检索结果
    print("\n  检索到的节点:")
    for i, node in enumerate(response.source_nodes):
        print(f"    {i+1}. 相似度: {node.score:.4f}")
        print(f"       内容: {node.node.text[:60]}...")


# ==========================================
# 示例 3: 响应合成策略详解
# ==========================================


def example_response_modes():
    """详细演示各种响应合成策略"""
    print("\n" + "=" * 50)
    print("示例 3: 响应合成策略详解")
    print("=" * 50)

    index = create_sample_index()

    # 各种响应模式
    modes = [
        ("default", "默认模式：直接使用检索结果生成响应"),
        ("compact", "精简模式：先合并检索结果再生成响应"),
        ("refine", "优化模式：逐节点迭代优化响应"),
        ("tree_summarize", "树摘要：分层摘要合并"),
        ("no_text", "无文本模式：只返回检索结果不生成响应"),
    ]

    query = "请介绍查询引擎的功能"

    for mode_name, description in modes:
        print(f"\n  --- {mode_name} 模式 ---")
        print(f"  说明: {description}")

        query_engine = index.as_query_engine(
            response_mode=mode_name,
            similarity_top_k=3,
        )

        response = query_engine.query(query)

        if mode_name == "no_text":
            print("  只返回检索结果，不生成文本响应")
            for node in response.source_nodes:
                print(f"    - {node.node.text[:60]}...")
        else:
            print(f"  响应: {response.response[:120]}...")


# ==========================================
# 示例 4: 流式响应
# ==========================================


def example_streaming_response():
    """演示流式响应输出"""
    print("\n" + "=" * 50)
    print("示例 4: 流式响应")
    print("=" * 50)

    index = create_sample_index()

    # 配置流式查询引擎
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=3,
    )

    query = "请详细说明响应合成的各种策略"
    print(f"\n  查询: {query}")
    print("  流式响应输出:")

    # 执行流式查询
    streaming_response = query_engine.query(query)

    # 逐 token 输出
    full_response = ""
    for token in streaming_response.response_gen:
        print(token, end="", flush=True)
        full_response += token

    print(f"\n\n  总响应长度: {len(full_response)} 字符")


# ==========================================
# 示例 5: 检索参数调优
# ==========================================


def example_retriever_tuning():
    """演示检索参数对结果的影响"""
    print("\n" + "=" * 50)
    print("示例 5: 检索参数调优")
    print("=" * 50)

    index = create_sample_index()

    query = "如何设置检索参数？"

    # 测试不同的 top_k 值
    top_k_values = [1, 3, 5]

    print(f"\n  测试查询: {query}")

    for top_k in top_k_values:
        print(f"\n  --- similarity_top_k={top_k} ---")

        query_engine = index.as_query_engine(similarity_top_k=top_k)

        response = query_engine.query(query)

        print(f"  检索节点数: {len(response.source_nodes)}")
        print(f"  响应: {response.response[:100]}...")

        # 显示检索分数
        scores = [n.score for n in response.source_nodes]
        if scores:
            print(f"  相似度分数范围: {min(scores):.4f} - {max(scores):.4f}")


# ==========================================
# 示例 6: 查询响应分析
# ==========================================


def example_response_analysis():
    """演示查询响应的详细分析"""
    print("\n" + "=" * 50)
    print("示例 6: 查询响应分析")
    print("=" * 50)

    index = create_sample_index()

    query_engine = index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
    )

    query = "RAG 系统的检索器有什么功能？"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)

    # 响应信息
    print(f"\n  响应内容:")
    print(f"  {response.response}")

    # 源节点分析
    print(f"\n  检索分析:")
    print(f"  总检索节点数: {len(response.source_nodes)}")

    for i, node in enumerate(response.source_nodes):
        print(f"\n    Node {i+1}:")
        print(f"      相似度分数: {node.score:.4f}")
        print(f"      文本长度: {len(node.node.text)} 字符")
        print(f"      内容片段: {node.node.text[:80]}...")
        print(f"      Node ID: {node.node.node_id}")


# ==========================================
# 示例 7: 多轮查询
# ==========================================


def example_multi_turn_queries():
    """演示多轮查询场景"""
    print("\n" + "=" * 50)
    print("示例 7: 多轮查询")
    print("=" * 50)

    index = create_sample_index()

    query_engine = index.as_query_engine(similarity_top_k=3)

    # 模拟对话式多轮查询
    queries = [
        "什么是 RAG？",
        "它的核心组件有哪些？",
        "检索器是如何工作的？",
    ]

    print("\n  多轮查询对话:")

    for i, query in enumerate(queries):
        print(f"\n  第 {i+1} 轮:")
        print(f"  问: {query}")

        response = query_engine.query(query)
        print(f"  答: {response.response[:100]}...")


# ==========================================
# 示例 8: 获取仅检索结果
# ==========================================


def example_retrieval_only():
    """演示只获取检索结果不生成响应"""
    print("\n" + "=" * 50)
    print("示例 8: 仅检索结果")
    print("=" * 50)

    index = create_sample_index()

    # 创建检索器
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    query = "向量检索"
    print(f"\n  查询关键词: {query}")

    # 仅执行检索
    nodes = retriever.retrieve(query)

    print(f"\n  检索到 {len(nodes)} 个节点:")

    for i, node in enumerate(nodes):
        print(f"\n    Node {i+1}:")
        print(f"      相似度: {node.score:.4f}")
        print(f"      内容: {node.node.text[:100]}...")
        print(f"      元数据: {node.node.metadata}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 17: LlamaIndex 查询引擎示例")
    print("=" * 60)

    # 配置设置
    setup_settings()

    # 运行各示例
    example_basic_query_engine()
    example_custom_retriever()
    example_response_modes()
    example_streaming_response()
    example_retriever_tuning()
    example_response_analysis()
    example_multi_turn_queries()
    example_retrieval_only()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()