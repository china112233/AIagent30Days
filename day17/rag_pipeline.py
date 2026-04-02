"""
Day 17: LlamaIndex RAG 管道示例

本文件演示完整的 RAG 系统构建：
- 文档处理管道
- 检索优化配置
- 上下文增强策略
- RAG 评估与调试
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
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


def create_knowledge_base():
    """创建示例知识库"""
    documents = [
        Document(
            text="产品名称：智能数据分析平台 Pro"
            "\n产品版本：2.5.0"
            "\n发布日期：2024年1月"
            "\n核心功能：数据可视化、实时监控、报表生成、智能预警"
            "\n适用场景：企业数据分析、业务监控、决策支持"
        ),
        Document(
            text="安装要求："
            "\n- 操作系统：Windows 10+、Linux、MacOS"
            "\n- 内存：至少 8GB RAM"
            "\n- 存储：至少 50GB 可用空间"
            "\n- 网络：稳定的互联网连接"
        ),
        Document(
            text="快速入门步骤："
            "\n1. 下载安装包并运行安装程序"
            "\n2. 配置数据源连接"
            "\n3. 创建第一个分析项目"
            "\n4. 添加数据图表"
            "\n5. 配置实时监控面板"
        ),
        Document(
            text="常见问题解答："
            "\nQ: 如何添加新的数据源？"
            "\nA: 进入设置 -> 数据源管理 -> 添加新数据源，按向导完成配置。"
            "\n\nQ: 数据更新频率是多少？"
            "\nA: 默认每5分钟自动更新，可在设置中自定义更新频率。"
            "\n\nQ: 支持哪些数据格式？"
            "\nA: 支持 CSV、Excel、JSON、SQL 数据库、API 接口等多种格式。"
        ),
        Document(
            text="高级功能说明："
            "\n- 智能预警：基于机器学习算法自动检测异常数据"
            "\n- 自动报表：定时生成并发送分析报表"
            "\n- 多用户协作：支持团队共享和协作编辑"
            "\n- API 集成：提供完整的 REST API 供外部系统调用"
        ),
        Document(
            text="技术架构："
            "\n- 前端：React + TypeScript"
            "\n- 后端：Python FastAPI"
            "\n- 数据库：PostgreSQL + Redis"
            "\n- 分析引擎：Apache Spark"
            "\n- 可视化：D3.js + ECharts"
        ),
    ]

    return documents


# ==========================================
# 示例 1: 基础 RAG 管道
# ==========================================


def example_basic_rag_pipeline():
    """演示基础 RAG 管道构建"""
    print("\n" + "=" * 50)
    print("示例 1: 基础 RAG 管道")
    print("=" * 50)

    documents = create_knowledge_base()

    print("  RAG 管道步骤:")
    print("  1. 加载文档")

    # 分块
    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )

    print("  2. 文档分块")
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"     生成了 {len(nodes)} 个节点")

    # 创建索引
    print("  3. 创建向量索引")
    index = VectorStoreIndex(nodes)

    # 配置查询引擎
    print("  4. 配置查询引擎")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
    )

    # 执行查询
    print("  5. 执行查询")
    query = "这个产品有哪些核心功能？"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)
    print(f"  响应: {response.response}")


# ==========================================
# 示例 2: 自定义 RAG 提示模板
# ==========================================


def example_custom_prompt():
    """演示自定义提示模板"""
    print("\n" + "=" * 50)
    print("示例 2: 自定义提示模板")
    print("=" * 50)

    documents = create_knowledge_base()

    index = VectorStoreIndex.from_documents(documents)

    # 自定义提示模板
    custom_prompt = PromptTemplate(
        """你是一个专业的产品技术支持助手。
请基于以下检索到的信息回答用户问题。

上下文信息：
{context_str}

用户问题：{query_str}

回答要求：
1. 仅使用提供的上下文信息回答
2. 如果信息不足，明确说明
3. 回答要简洁专业

回答："""
    )

    # 使用自定义提示创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=custom_prompt,
    )

    query = "安装这个产品需要什么系统要求？"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)
    print(f"  响应: {response.response}")


# ==========================================
# 示例 3: 组件化 RAG 构建
# ==========================================


def example_component_based_rag():
    """演示组件化的 RAG 构建"""
    print("\n" + "=" * 50)
    print("示例 3: 组件化 RAG 构建")
    print("=" * 50)

    documents = create_knowledge_base()

    # 分块器
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=40)
    nodes = splitter.get_nodes_from_documents(documents)

    # 索引
    index = VectorStoreIndex(nodes)

    # 检索器（自定义配置）
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=4,
    )

    # 响应合成器（自定义配置）
    response_synth = get_response_synthesizer(
        response_mode=ResponseMode.REFINE,  # 使用 refine 模式
    )

    # 组合为查询引擎
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synth,
    )

    print("  RAG 组件配置:")
    print("    - 检索器: VectorIndexRetriever (top_k=4)")
    print("    - 合成器: Refine 模式")

    query = "如何快速开始使用这个产品？"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)
    print(f"  响应: {response.response}")


# ==========================================
# 示例 4: HyDE 查询变换
# ==========================================


def example_hyde_transform():
    """演示 HyDE（假设文档嵌入）查询变换"""
    print("\n" + "=" * 50)
    print("示例 4: HyDE 查询变换")
    print("=" * 50)

    documents = create_knowledge_base()

    index = VectorStoreIndex.from_documents(documents)

    # HyDE 变换：生成假设文档来辅助检索
    hyde_transform = HyDEQueryTransform(include_original=True)

    print("  HyDE 变换说明:")
    print("    - 为查询生成假设文档")
    print("    - 使用假设文档向量进行检索")
    print("    - 可提高语义匹配效果")

    # 应用 HyDE 的查询引擎
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine=query_engine,
        query_transform=hyde_transform,
    )

    query = "我想了解产品的技术实现细节"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)
    print(f"  响应: {response.response[:150]}...")


# ==========================================
# 示例 5: 上下文增强
# ==========================================


def example_context_enhancement():
    """演示上下文增强策略"""
    print("\n" + "=" * 50)
    print("示例 5: 上下文增强")
    print("=" * 50)

    documents = create_knowledge_base()

    # 使用较大的重叠以保持上下文
    splitter = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=100,  # 较大重叠
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print("  上下文增强配置:")
    print(f"    分块大小: 300")
    print(f"    重叠大小: 100 (33%)")

    index = VectorStoreIndex(nodes)

    # 增加检索节点数以获取更多上下文
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # 检索更多节点
        response_mode="tree_summarize",  # 树摘要模式
    )

    query = "请全面介绍这个数据分析平台"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)
    print(f"  响应: {response.response}")


# ==========================================
# 示例 6: RAG 质量调试
# ==========================================


def example_rag_debugging():
    """演示 RAG 系统调试技巧"""
    print("\n" + "=" * 50)
    print("示例 6: RAG 质量调试")
    print("=" * 50)

    documents = create_knowledge_base()

    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine(
        similarity_top_k=4,
    )

    query = "产品支持哪些数据格式？"
    print(f"\n  查询: {query}")

    response = query_engine.query(query)

    # 调试信息
    print("\n  === 调试信息 ===")

    print("\n  1. 检索分析:")
    print(f"     检索节点数: {len(response.source_nodes)}")

    for i, node in enumerate(response.source_nodes):
        print(f"\n     Node {i+1}:")
        print(f"       相似度: {node.score:.4f}")
        print(f"       内容: {node.node.text[:80]}...")

    print("\n  2. 响应质量检查:")
    print(f"     响应长度: {len(response.response)} 字符")

    # 检查响应是否引用了检索内容
    print("\n  3. 内容覆盖检查:")
    retrieved_text = " ".join([n.node.text for n in response.source_nodes])
    key_terms = ["CSV", "Excel", "JSON", "SQL"]

    for term in key_terms:
        in_retrieved = term in retrieved_text
        in_response = term in response.response
        print(f"     '{term}' - 检索: {in_retrieved}, 响应: {in_response}")


# ==========================================
# 示例 7: 完整 RAG 工作流
# ==========================================


def example_complete_rag_workflow():
    """演示完整的 RAG 工作流"""
    print("\n" + "=" * 50)
    print("示例 7: 完整 RAG 工作流")
    print("=" * 50)

    # Step 1: 准备数据
    print("\n  Step 1: 数据准备")
    documents = create_knowledge_base()
    print(f"    加载了 {len(documents)} 个文档")

    # Step 2: 数据处理
    print("\n  Step 2: 数据处理")
    splitter = SentenceSplitter(chunk_size=250, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"    分块后得到 {len(nodes)} 个节点")

    # Step 3: 索引构建
    print("\n  Step 3: 索引构建")
    index = VectorStoreIndex(nodes)
    print("    向量索引创建完成")

    # Step 4: 查询引擎配置
    print("\n  Step 4: 查询引擎配置")
    query_engine = index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact",
        streaming=False,
    )
    print("    配置完成")

    # Step 5: 执行查询
    print("\n  Step 5: 执行查询")
    queries = [
        "产品的技术架构是什么？",
        "如何配置智能预警功能？",
        "产品的安装要求有哪些？",
    ]

    for query in queries:
        print(f"\n    Q: {query}")
        response = query_engine.query(query)
        print(f"    A: {response.response[:100]}...")


# ==========================================
# 示例 8: RAG 性能优化要点
# ==========================================


def example_rag_optimization():
    """演示 RAG 性能优化要点"""
    print("\n" + "=" * 50)
    print("示例 8: RAG 优化要点")
    print("=" * 50)

    print("""
  RAG 性能优化关键点：

  1. 分块策略优化
     - 根据文档类型选择合适的分块大小
     - 保持适当的重叠以维持上下文
     - 考虑语义边界分割

  2. Embedding 模型选择
     - 选择高质量、适合语言的模型
     - 考虑模型大小与性能的平衡
     - 定期评估检索效果

  3. 检索参数调优
     - similarity_top_k: 通常 3-10
     - 过小遗漏信息，过大引入噪音
     - 可根据查询类型动态调整

  4. 响应合成策略
     - default: 简单快速
     - compact: 适合多节点
     - refine: 高质量但较慢
     - tree_summarize: 全面但耗时

  5. 提示模板优化
     - 明确回答要求
     - 添加使用限制
     - 指定回答格式

  6. 持久化策略
     - 使用向量数据库
     - 实现增量更新
     - 缓存常用查询结果
    """)


# 导入 TransformQueryEngine
from llama_index.core.query_engine import TransformQueryEngine


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 17: LlamaIndex RAG 管道示例")
    print("=" * 60)

    # 配置设置
    setup_settings()

    # 运行各示例
    example_basic_rag_pipeline()
    example_custom_prompt()
    example_component_based_rag()
    example_hyde_transform()
    example_context_enhancement()
    example_rag_debugging()
    example_complete_rag_workflow()
    example_rag_optimization()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()