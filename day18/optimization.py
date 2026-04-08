"""
Day 18: LlamaIndex 索引优化示例

本文件演示索引优化技巧：
- 分块策略优化
- 嵌入批处理
- 索引持久化优化
- 查询性能调优
- 内存管理技巧
"""

import os
import time
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
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

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ 设置已配置")
    return llm, embed_model


def create_sample_documents():
    """创建示例文档"""
    documents = [
        Document(
            text="Python 是一种高级编程语言，具有简洁的语法和强大的生态系统。"
            "Python 广泛用于 Web 开发、数据科学、人工智能等领域。"
            "Python 的设计哲学强调代码可读性和简洁性。",
            metadata={"topic": "python"},
        ),
        Document(
            text="向量数据库是存储嵌入向量的专用数据库，支持高效的相似度检索。"
            "主流向量数据库包括 Pinecone、Weaviate、Chroma 和 Milvus。"
            "向量数据库是 RAG 系统的核心组件。",
            metadata={"topic": "vectordb"},
        ),
        Document(
            text="RAG（检索增强生成）是一种结合检索和生成的技术架构。"
            "RAG 系统首先从知识库检索相关文档，然后使用 LLM 生成回答。"
            "这种方法可以减少幻觉，提高回答准确性。",
            metadata={"topic": "rag"},
        ),
        Document(
            text="LangChain 是一个 LLM 应用开发框架，提供链式调用、Agent 等组件。"
            "LangChain 支持多种 LLM 后端和工具集成。"
            "LangChain 适合构建复杂的 LLM 应用。",
            metadata={"topic": "langchain"},
        ),
        Document(
            text="LlamaIndex 是一个专注于数据索引的框架，适合构建 RAG 应用。"
            "LlamaIndex 提供多种索引类型和检索策略。"
            "LlamaIndex 支持文档管理和元数据过滤。",
            metadata={"topic": "llamaindex"},
        ),
    ]

    return documents


# ==========================================
# 示例 1: 分块策略对比
# ==========================================


def example_chunking_strategies():
    """演示不同分块策略"""
    print("\n" + "=" * 50)
    print("示例 1: 分块策略对比")
    print("=" * 50)

    documents = create_sample_documents()

    # 策略 1: 固定大小分块
    print("\n  策略 1: 固定大小分块")
    fixed_splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )
    fixed_nodes = fixed_splitter.get_nodes_from_documents(documents)
    print(f"    分块数: {len(fixed_nodes)}")
    print(f"    平均块大小: {sum(len(n.text) for n in fixed_nodes) / len(fixed_nodes):.1f}")

    # 策略 2: 小分块高重叠
    print("\n  策略 2: 小分块高重叠")
    small_splitter = SentenceSplitter(
        chunk_size=100,
        chunk_overlap=30,
    )
    small_nodes = small_splitter.get_nodes_from_documents(documents)
    print(f"    分块数: {len(small_nodes)}")
    print(f"    重叠比例: {30/100 * 100:.1f}%")

    # 策略 3: 大分块低重叠
    print("\n  策略 3: 大分块低重叠")
    large_splitter = SentenceSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    large_nodes = large_splitter.get_nodes_from_documents(documents)
    print(f"    分块数: {len(large_nodes)}")

    print("\n  分块策略选择建议:")
    print("""
    ┌─────────────────────────────────────────┐
    │           分块策略选择                   │
    ├─────────────────────────────────────────┤
    │                                         │
    │ 文档类型          推荐分块大小          │
    │ ─────────────────────────────────────── │
    │ 技术文档          512-1024 tokens       │
    │ 代码文档          256-512 tokens        │
    │ 新闻/文章         200-400 tokens        │
    │ 对话记录          100-200 tokens        │
    │ 长篇小说          1000-2000 tokens      │
    │                                         │
    │ 重叠建议：                              │
    │ - 通常 10-20% 的重叠                    │
    │ - 保持上下文连贯                        │
    │ - 防止信息断裂                          │
    │                                         │
    └─────────────────────────────────────────┘
    """)


# ==========================================
# 示例 2: 嵌入批处理优化
# ==========================================


def example_embedding_batch():
    """演示嵌入批处理"""
    print("\n" + "=" * 50)
    print("示例 2: 嵌入批处理优化")
    print("=" * 50)

    setup_settings()

    documents = create_sample_documents()

    # 分块
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=40)
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"\n  准备嵌入 {len(nodes)} 个节点")

    # 批量嵌入（LlamaIndex 默认批处理）
    print("\n  执行批量嵌入...")

    start_time = time.time()
    index = VectorStoreIndex(nodes, show_progress=True)
    elapsed = time.time() - start_time

    print(f"\n  ✅ 索引创建完成")
    print(f"    耗时: {elapsed:.2f} 秒")
    print(f"    平均每节点: {elapsed/len(nodes):.2f} 秒")

    print("\n  批处理优化建议:")
    print("""
    ┌─────────────────────────────────────────┐
    │         嵌入批处理优化                   │
    ├─────────────────────────────────────────┤
    │                                         │
    │ 1. 使用批量嵌入                          │
    │    Settings.embed_batch_size = 50       │
    │                                         │
    │ 2. 缓存嵌入结果                          │
    │    避免重复计算相同文本                  │
    │                                         │
    │ 3. 使用本地嵌入模型                      │
    │    减少 API 调用延迟                    │
    │                                         │
    │ 4. 并行处理                              │
    │    多线程嵌入大批量数据                  │
    │                                         │
    │ 5. 增量更新                              │
    │    只嵌入新增/修改的节点                 │
    │                                         │
    └─────────────────────────────────────────┘
    """)


# ==========================================
# 示例 3: 索引持久化
# ==========================================


def example_index_persistence():
    """演示索引持久化优化"""
    print("\n" + "=" * 50)
    print("示例 3: 索引持久化优化")
    print("=" * 50)

    setup_settings()

    documents = create_sample_documents()

    # 创建索引
    index = VectorStoreIndex.from_documents(documents)

    # 持久化存储
    storage_dir = "./day18/storage_demo"

    print(f"\n  保存索引到: {storage_dir}")

    start_time = time.time()
    index.storage_context.persist(persist_dir=storage_dir)
    save_time = time.time() - start_time

    print(f"  ✅ 保存完成，耗时: {save_time:.2f} 秒")

    # 加载索引
    print(f"\n  从存储加载索引...")

    start_time = time.time()
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    loaded_index = load_index_from_storage(storage_context)
    load_time = time.time() - start_time

    print(f"  ✅ 加载完成，耗时: {load_time:.2f} 秒")

    print("\n  持久化对比:")
    print(f"    保存耗时: {save_time:.2f} 秒")
    print(f"    加载耗时: {load_time:.2f} 秒")
    print(f"    加载比新建快: 嵌入从零开始需要重新计算")

    print("\n  持久化最佳实践:")
    print("""
    ┌─────────────────────────────────────────┐
    │         持久化最佳实践                   │
    ├─────────────────────────────────────────┤
    │                                         │
    │ 1. 定期保存索引                          │
    │    避免每次重建                          │
    │                                         │
    │ 2. 增量更新                              │
    │    只更新变化的文档                      │
    │                                         │
    │ 3. 使用外部向量存储                      │
    │    Chroma/Pinecone 自动持久化           │
    │                                         │
    │ 4. 版本管理                              │
    │    保存不同版本的索引                    │
    │                                         │
    │ 5. 备份策略                              │
    │    定期备份存储目录                      │
    │                                         │
    └─────────────────────────────────────────┘
    """)


# ==========================================
# 示例 4: 查询性能调优
# ==========================================


def example_query_performance():
    """演示查询性能调优"""
    print("\n" + "=" * 50)
    print("示例 4: 查询性能调优")
    print("=" * 50)

    setup_settings()

    documents = create_sample_documents()

    index = VectorStoreIndex.from_documents(documents)

    # 配置 1: 基础查询
    print("\n  配置 1: 基础查询 (top_k=3)")
    engine1 = index.as_query_engine(similarity_top_k=3)

    start = time.time()
    response1 = engine1.query("什么是 RAG？")
    time1 = time.time() - start

    print(f"    响应时间: {time1:.2f} 秒")

    # 配置 2: 更多检索节点
    print("\n  配置 2: 大 top_k (top_k=10)")
    engine2 = index.as_query_engine(similarity_top_k=10)

    start = time.time()
    response2 = engine2.query("什么是 RAG？")
    time2 = time.time() - start

    print(f"    响应时间: {time2:.2f} 秒")
    print(f"    比 top_k=3 慢: {time2 - time1:.2f} 秒")

    # 配置 3: 流式查询
    print("\n  配置 3: 流式查询")
    engine3 = index.as_query_engine(streaming=True, similarity_top_k=3)

    start = time.time()
    streaming_response = engine3.query("什么是 RAG？")
    time3 = time.time() - start

    print(f"    首字响应时间: {time3:.2f} 秒")
    print("    （流式查询让用户更快看到响应）")

    print("\n  性能调优建议:")
    print("""
    ┌─────────────────────────────────────────┐
    │         查询性能调优                     │
    ├─────────────────────────────────────────┤
    │                                         │
    │ 1. 调整 top_k                            │
    │    - 小值：快速但可能遗漏                │
    │    - 大值：全面但更慢                    │
    │    - 推荐：3-5                           │
    │                                         │
    │ 2. 使用流式响应                          │
    │    - 用户更快看到反馈                    │
    │    - 改善用户体验                        │
    │                                         │
    │ 3. 响应模式选择                          │
    │    - compact: 快速                      │
    │    - refine: 高质量但慢                 │
    │    - tree_summarize: 全面但最慢         │
    │                                         │
    │ 4. 缓存常用查询                          │
    │    - 减少重复计算                        │
    │                                         │
    │ 5. 预热索引                              │
    │    - 启动时预加载常用查询                │
    │                                         │
    └─────────────────────────────────────────┘
    """)


# ==========================================
# 示例 5: 内存管理
# ==========================================


def example_memory_management():
    """演示内存管理技巧"""
    print("\n" + "=" * 50)
    print("示例 5: 内存管理")
    print("=" * 50)

    print("""
  内存管理策略：

  ┌─────────────────────────────────────────┐
  │         内存使用分析                     │
  ├─────────────────────────────────────────┤
  │                                         │
  │ 索引大小估算：                           │
  │ ─────────────────────────────────────── │
  │                                         │
  │ 文档数 × 平均长度 × 嵌入维度 × 4 bytes  │
  │                                         │
  │ 示例：                                   │
  │ 1000 文档 × 500 tokens × 384 维 × 4    │
  │ ≈ 750 MB                                │
  │                                         │
  │ 加上节点文本存储：                       │
  │ ≈ 1000 文档 × 500 × 1 byte              │
  │ ≈ 500 KB                                │
  │                                         │
  │ 总计约：750 MB                           │
  │                                         │
  └─────────────────────────────────────────┘

  内存优化策略：

  1. 使用外部向量存储
     ┌─────────────────────────────────────┐
     │ Chroma / Pinecone / Milvus          │
     │                                     │
     │ 优势：                              │
     │ - 索引不占用应用内存                │
     │ - 支持大规模数据                    │
     │ - 自动持久化                        │
     └─────────────────────────────────────┘

  2. 分批处理大文档集
     ┌─────────────────────────────────────┐
     │ for batch in document_batches:      │
     │     process_batch(batch)            │
     │     clear_cache()                   │
     │                                     │
     │ 避免一次性处理过多数据              │
     └─────────────────────────────────────┘

  3. 清理无用索引
     ┌─────────────────────────────────────┐
     │ del old_index                       │
     │                                     │
     │ Python GC 会自动回收                │
     └─────────────────────────────────────┘

  4. 使用嵌入缓存
     ┌─────────────────────────────────────┐
     │ 缓存常用嵌入向量                    │
     │ 避免重复计算                        │
     └─────────────────────────────────────┘

  5. 增量更新而非重建
     ┌─────────────────────────────────────┐
     │ index.insert(new_document)          │
     │                                     │
     │ 只更新变化部分                      │
     └─────────────────────────────────────┘
    """)


# ==========================================
# 示例 6: 索引更新策略
# ==========================================


def example_index_update():
    """演示索引更新策略"""
    print("\n" + "=" * 50)
    print("示例 6: 索引更新策略")
    print("=" * 50)

    setup_settings()

    # 创建初始文档
    initial_docs = [
        Document(text="初始文档 1：基础内容描述"),
        Document(text="初始文档 2：另一个基础内容"),
    ]

    # 创建索引
    print("\n  创建初始索引...")
    index = VectorStoreIndex.from_documents(initial_docs)

    initial_count = len(index.docstore.docs)
    print(f"    初始文档数: {initial_count}")

    # 增量添加
    print("\n  增量添加新文档...")

    new_doc = Document(text="新增文档：最新的内容描述")
    index.insert(new_doc)

    updated_count = len(index.docstore.docs)
    print(f"    更新后文档数: {updated_count}")
    print(f"    增量添加成功")

    print("\n  索引更新策略:")
    print("""
    ┌─────────────────────────────────────────┐
    │         索引更新策略                     │
    ├─────────────────────────────────────────┤
    │                                         │
    │ 1. 增量添加                              │
    │    index.insert(new_doc)                │
    │                                         │
    │    优点：                                │
    │    - 无需重建整个索引                    │
    │    - 快速添加新文档                      │
    │                                         │
    │ 2. 批量添加                              │
    │    for doc in new_docs:                 │
    │        index.insert(doc)                │
    │                                         │
    │ 3. 删除文档                              │
    │    index.delete(doc_id)                 │
    │                                         │
    │    注意：                                │
    │    - 需要知道 doc_id                     │
    │    - 删除后需更新存储                    │
    │                                         │
    │ 4. 全量重建                              │
    │    当大量文档变化时                      │
    │                                         │
    │    场景：                                │
    │    - 嵌入模型升级                        │
    │    - 分块策略调整                        │
    │    - 大规模数据更新                      │
    │                                         │
    └─────────────────────────────────────────┘
    """)


# ==========================================
# 示例 7: 并发处理优化
# ==========================================


def example_concurrent_processing():
    """演示并发处理优化"""
    print("\n" + "=" * 50)
    print("示例 7: 并发处理优化")
    print("=" * 50)

    print("""
  并发处理策略：

  ┌─────────────────────────────────────────┐
  │         并发处理架构                     │
  ├─────────────────────────────────────────┤
  │                                         │
  │   ┌─────────────┐                       │
  │   │  文档队列   │                       │
  │   └─────────────┘                       │
  │         │                               │
  │         ▼                               │
  │   ┌─────────────────────────────────┐  │
  │   │         处理器池                │  │
  │   │                                 │  │
  │   │   Worker 1 ─── Worker 2 ─── Worker 3 │
  │   │                                 │  │
  │   │   (嵌入)    (嵌入)    (嵌入)    │  │
  │   │                                 │  │
  │   └─────────────────────────────────┘  │
  │         │                               │
  │         ▼                               │
  │   ┌─────────────┐                       │
  │   │  结果汇总   │                       │
  │   └─────────────┘                       │
  │                                         │
  └─────────────────────────────────────────┘

  实现方式：

  1. 使用 Python 多线程
     ┌─────────────────────────────────────┐
     │ from concurrent.futures import ThreadPoolExecutor │
     │                                     │
     │ with ThreadPoolExecutor(max_workers=4) as executor: │
     │     futures = [executor.submit(embed, doc) │
     │                for doc in documents] │
     │     results = [f.result() for f in futures] │
     └─────────────────────────────────────┘

  2. 使用异步处理
     ┌─────────────────────────────────────┐
     │ async def embed_batch(docs):        │
     │     tasks = [embed(doc) for doc in docs] │
     │     return await asyncio.gather(*tasks) │
     └─────────────────────────────────────┘

  3. 使用批量 API
     ┌─────────────────────────────────────┐
     │ # OpenAI 支持批量嵌入               │
     │ embeddings = client.embeddings.create( │
     │     input=texts,                    │
     │     model="text-embedding-3-small", │
     │ )                                   │
     └─────────────────────────────────────┘

  注意事项：
  - API 速率限制
  - 内存使用增长
  - 错误处理和重试
    """)


# ==========================================
# 示例 8: 综合优化配置
# ==========================================


def example_optimized_config():
    """演示综合优化配置"""
    print("\n" + "=" * 50)
    print("示例 8: 综合优化配置")
    print("=" * 50)

    print("""
  生产级优化配置示例：

  ┌─────────────────────────────────────────┐
  │         优化配置清单                     │
  ├─────────────────────────────────────────┤
  │                                         │
  │ 1. 全局设置优化                          │
  │    ──────────────────────────────────── │
  │    Settings.chunk_size = 512            │
  │    Settings.chunk_overlap = 50          │
  │    Settings.embed_batch_size = 50       │
  │                                         │
  │ 2. 分块器优化                            │
  │    ──────────────────────────────────── │
  │    splitter = SentenceSplitter(         │
  │        chunk_size=512,                  │
  │        chunk_overlap=50,                │
  │        paragraph_separator="\n\n",      │
  │    )                                    │
  │                                         │
  │ 3. 向量存储优化                          │
  │    ──────────────────────────────────── │
  │    # 使用 Chroma 持久化                 │
  │    vector_store = ChromaVectorStore(    │
  │        chroma_collection=collection,    │
  │    )                                    │
  │                                         │
  │ 4. 查询引擎优化                          │
  │    ──────────────────────────────────── │
  │    query_engine = index.as_query_engine(│
  │        similarity_top_k=3,              │
  │        response_mode="compact",         │
  │        streaming=True,                  │
  │    )                                    │
  │                                         │
  │ 5. 缓存配置                              │
  │    ──────────────────────────────────── │
  │    # 嵌入缓存                            │
  │    from llama_index.core.storage.docstore import │
  │        SimpleDocumentStore              │
  │                                         │
  │ 6. 监控指标                              │
  │    ──────────────────────────────────── │
  │    - 索引构建时间                        │
  │    - 查询响应时间                        │
  │    - 内存使用量                          │
  │    - API 调用次数                        │
  │                                         │
  └─────────────────────────────────────────┘

  推荐配置值：

  ┌─────────────────┬─────────────────┐
  │     参数        │    推荐值       │
  ├─────────────────┼─────────────────┤
  │ chunk_size      │ 512             │
  │ chunk_overlap   │ 50 (10%)        │
  │ embed_batch_size│ 50              │
  │ similarity_top_k│ 3-5             │
  │ response_mode   │ compact         │
  │ streaming       │ True            │
  └─────────────────┴─────────────────┘
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 18: LlamaIndex 索引优化示例")
    print("=" * 60)

    # 运行各示例
    example_chunking_strategies()
    example_embedding_batch()
    example_index_persistence()
    example_query_performance()
    example_memory_management()
    example_index_update()
    example_concurrent_processing()
    example_optimized_config()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()