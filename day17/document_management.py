"""
Day 17: LlamaIndex 文档管理示例

本文件演示文档管理的核心功能：
- 文档增量更新
- 元数据管理
- 文档删除与刷新
- 多源数据整合
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.schema import TextNode
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
# 示例 1: 文档增量插入
# ==========================================


def example_incremental_insert():
    """演示向现有索引增量插入文档"""
    print("\n" + "=" * 50)
    print("示例 1: 文档增量插入")
    print("=" * 50)

    # 创建初始索引
    initial_docs = [
        Document(text="初始文档 1：这是第一批加载的文档内容。"),
        Document(text="初始文档 2：包含基础的知识库信息。"),
    ]

    print("  创建初始索引...")
    index = VectorStoreIndex.from_documents(initial_docs)
    print(f"    初始文档数: {len(initial_docs)}")

    # 查询初始索引
    query_engine = index.as_query_engine()
    response = query_engine.query("有哪些文档？")
    print(f"\n  初始查询响应: {response.response}")

    # 增量插入新文档
    new_docs = [
        Document(text="新增文档 A：这是后续添加的新内容。"),
        Document(text="新增文档 B：包含更新的信息。"),
        Document(text="新增文档 C：补充的知识库条目。"),
    ]

    print("\n  增量插入新文档...")
    for doc in new_docs:
        index.insert(doc)

    print(f"    新增文档数: {len(new_docs)}")

    # 验证插入效果
    response = query_engine.query("现在知识库中有哪些内容？")
    print(f"\n  更新后查询响应: {response.response}")


# ==========================================
# 示例 2: 元数据管理
# ==========================================


def example_metadata_management():
    """演示文档元数据的管理"""
    print("\n" + "=" * 50)
    print("示例 2: 元数据管理")
    print("=" * 50)

    # 创建带丰富元数据的文档
    documents = [
        Document(
            text="技术规范文档 v1.0：定义了系统的核心架构和接口规范。",
            metadata={
                "doc_type": "specification",
                "version": "1.0",
                "category": "技术",
                "author": "技术团队",
                "created_date": "2024-01-15",
                "tags": ["架构", "接口", "规范"],
            },
        ),
        Document(
            text="用户操作手册 v2.0：详细说明系统功能和操作步骤。",
            metadata={
                "doc_type": "manual",
                "version": "2.0",
                "category": "用户",
                "author": "产品团队",
                "created_date": "2024-02-20",
                "tags": ["操作", "功能", "指南"],
            },
        ),
        Document(
            text="API 接口文档 v1.5：描述所有公开 API 的参数和返回值。",
            metadata={
                "doc_type": "api",
                "version": "1.5",
                "category": "开发",
                "author": "开发团队",
                "created_date": "2024-03-10",
                "tags": ["API", "接口", "开发"],
            },
        ),
    ]

    print("  创建带元数据的文档:")
    for doc in documents:
        print(f"\n    文档类型: {doc.metadata['doc_type']}")
        print(f"    版本: {doc.metadata['version']}")
        print(f"    分类: {doc.metadata['category']}")

    # 创建索引
    index = VectorStoreIndex.from_documents(documents)

    # 查询并查看元数据
    query_engine = index.as_query_engine()
    response = query_engine.query("有哪些技术相关的文档？")

    print("\n  查询结果:")
    for i, node in enumerate(response.source_nodes):
        print(f"\n    Node {i+1} 元数据:")
        metadata = node.node.metadata
        for key, value in metadata.items():
            print(f"      {key}: {value}")


# ==========================================
# 示例 3: 从节点管理文档
# ==========================================


def example_node_management():
    """演示从节点级别管理文档"""
    print("\n" + "=" * 50)
    print("示例 3: 节点管理")
    print("=" * 50)

    # 创建节点并添加自定义元数据
    nodes = [
        TextNode(
            text="产品功能模块 A：负责数据处理和分析。",
            metadata={
                "module": "A",
                "function": "数据处理",
                "priority": "high",
            },
        ),
        TextNode(
            text="产品功能模块 B：提供用户界面和交互。",
            metadata={
                "module": "B",
                "function": "用户界面",
                "priority": "medium",
            },
        ),
        TextNode(
            text="产品功能模块 C：实现 API 接口和集成。",
            metadata={
                "module": "C",
                "function": "API接口",
                "priority": "high",
            },
        ),
    ]

    print("  直接创建节点:")
    for node in nodes:
        print(f"    Module {node.metadata['module']}: {node.metadata['function']}")

    # 从节点创建索引
    index = VectorStoreIndex(nodes)

    # 添加新节点
    print("\n  添加新节点:")
    new_node = TextNode(
        text="产品功能模块 D：新增的报表生成功能。",
        metadata={
            "module": "D",
            "function": "报表生成",
            "priority": "medium",
        },
    )
    index.insert_nodes([new_node])
    print(f"    Module {new_node.metadata['module']}: {new_node.metadata['function']}")

    # 查询验证
    query_engine = index.as_query_engine()
    response = query_engine.query("有哪些高优先级的模块？")

    print(f"\n  查询响应: {response.response}")


# ==========================================
# 示例 4: 索引持久化与更新
# ==========================================


def example_persistence_and_update():
    """演示索引持久化与更新"""
    print("\n" + "=" * 50)
    print("示例 4: 索引持久化与更新")
    print("=" * 50)

    persist_dir = "./day17/doc_management_storage"

    # 确保目录干净
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    # 创建并保存初始索引
    initial_docs = [
        Document(
            text="文档 1：基础知识库内容。",
            metadata={"id": "doc1", "status": "active"},
        ),
        Document(
            text="文档 2：进阶知识库内容。",
            metadata={"id": "doc2", "status": "active"},
        ),
    ]

    print("  创建初始索引...")
    index = VectorStoreIndex.from_documents(initial_docs)

    # 保存索引
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"  索引已保存到: {persist_dir}")

    # 加载索引
    print("\n  从存储加载索引...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    loaded_index = load_index_from_storage(storage_context)

    # 验证加载
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("知识库有什么内容？")
    print(f"  加载后查询: {response.response[:80]}...")

    # 向加载的索引添加新文档
    print("\n  向加载的索引添加新文档...")
    new_doc = Document(
        text="文档 3：新增的知识库条目。",
        metadata={"id": "doc3", "status": "active"},
    )
    loaded_index.insert(new_doc)

    # 重新保存
    loaded_index.storage_context.persist(persist_dir=persist_dir)
    print("  索引已更新保存")

    # 再次加载验证
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    final_index = load_index_from_storage(storage_context)

    response = final_index.as_query_engine().query("现在有哪些文档？")
    print(f"\n  最终查询: {response.response[:80]}...")

    # 清理
    import shutil
    shutil.rmtree(persist_dir)
    print(f"\n  已清理存储目录")


# ==========================================
# 示例 5: 文档版本管理
# ==========================================


def example_version_management():
    """演示文档版本管理"""
    print("\n" + "=" * 50)
    print("示例 5: 文档版本管理")
    print("=" * 50)

    # 创建不同版本的文档
    docs_v1 = [
        Document(
            text="配置指南 v1：基础配置步骤。",
            metadata={"doc_id": "config_guide", "version": "1.0", "deprecated": False},
        ),
    ]

    print("  初始版本 (v1.0)")
    index = VectorStoreIndex.from_documents(docs_v1)

    # 添加新版本
    print("\n  添加新版本 (v2.0)")
    docs_v2 = [
        Document(
            text="配置指南 v2：更新后的配置步骤，包含新功能配置。",
            metadata={"doc_id": "config_guide", "version": "2.0", "deprecated": False},
        ),
    ]

    for doc in docs_v2:
        index.insert(doc)

    # 查询时查看版本
    query_engine = index.as_query_engine()
    response = query_engine.query("配置指南有哪些版本？")

    print(f"\n  查询响应: {response.response}")

    # 查看所有版本信息
    print("\n  检索到的版本:")
    for node in response.source_nodes:
        print(f"    版本 {node.node.metadata['version']}: {node.node.text[:50]}...")


# ==========================================
# 示例 6: 多源数据整合
# ==========================================


def example_multi_source_integration():
    """演示多数据源整合"""
    print("\n" + "=" * 50)
    print("示例 6: 多源数据整合")
    print("=" * 50)

    # 模拟来自不同数据源的文档
    sources = {
        "internal_docs": [
            Document(
                text="内部文档：公司内部的技术规范和流程。",
                metadata={"source": "internal", "access": "restricted"},
            ),
        ],
        "public_docs": [
            Document(
                text="公开文档：面向用户的产品介绍和指南。",
                metadata={"source": "public", "access": "open"},
            ),
        ],
        "api_docs": [
            Document(
                text="API 文档：开发者接口参考。",
                metadata={"source": "api", "access": "developer"},
            ),
        ],
        "training_docs": [
            Document(
                text="培训文档：员工培训和学习材料。",
                metadata={"source": "training", "access": "internal"},
            ),
        ],
    }

    print("  整合多数据源:")

    # 合并所有文档
    all_documents = []
    for source_name, docs in sources.items():
        print(f"    来源: {source_name}, 文档数: {len(docs)}")
        all_documents.extend(docs)

    # 创建统一索引
    print("\n  创建统一索引...")
    index = VectorStoreIndex.from_documents(all_documents)

    # 查询
    query_engine = index.as_query_engine()
    response = query_engine.query("有哪些可用的文档？")

    print(f"\n  查询响应: {response.response}")

    # 查看来源分布
    print("\n  检索结果的来源:")
    sources_count = {}
    for node in response.source_nodes:
        source = node.node.metadata.get("source", "unknown")
        sources_count[source] = sources_count.get(source, 0) + 1

    for source, count in sources_count.items():
        print(f"    {source}: {count} 个节点")


# ==========================================
# 示例 7: 文档状态管理
# ==========================================


def example_document_status():
    """演示文档状态管理"""
    print("\n" + "=" * 50)
    print("示例 7: 文档状态管理")
    print("=" * 50)

    # 创建带状态标记的文档
    documents = [
        Document(
            text="活跃文档 A：当前有效的知识库内容。",
            metadata={"status": "active", "reviewed": True},
        ),
        Document(
            text="待审核文档 B：新提交的内容等待审核。",
            metadata={"status": "pending", "reviewed": False},
        ),
        Document(
            text="已归档文档 C：历史内容已归档。",
            metadata={"status": "archived", "reviewed": True},
        ),
        Document(
            text="活跃文档 D：另一个有效内容。",
            metadata={"status": "active", "reviewed": True},
        ),
    ]

    print("  创建带状态标记的文档:")
    status_count = {}
    for doc in documents:
        status = doc.metadata["status"]
        status_count[status] = status_count.get(status, 0) + 1

    for status, count in status_count.items():
        print(f"    {status}: {count} 个文档")

    # 创建索引
    index = VectorStoreIndex.from_documents(documents)

    # 查询活跃文档
    query_engine = index.as_query_engine()
    response = query_engine.query("有哪些有效的内容？")

    print(f"\n  查询响应: {response.response}")

    # 过滤分析
    print("\n  检索结果状态:")
    for node in response.source_nodes:
        status = node.node.metadata.get("status")
        reviewed = node.node.metadata.get("reviewed")
        print(f"    状态: {status}, 已审核: {reviewed}")


# ==========================================
# 示例 8: 文档管理最佳实践
# ==========================================


def example_best_practices():
    """演示文档管理最佳实践"""
    print("\n" + "=" * 50)
    print("示例 8: 文档管理最佳实践")
    print("=" * 50)

    print("""
  文档管理最佳实践：

  1. 元数据设计
     - 定义统一的元数据字段
     - 包含：来源、类型、版本、状态、时间戳
     - 便于后续过滤和管理

  2. 增量更新策略
     - 使用 insert 方法添加新文档
     - 避免每次重建整个索引
     - 定期持久化保存

  3. 版本控制
     - 为文档维护版本号
     - 旧版本标记为 deprecated
     - 保留历史记录便于回溯

  4. 数据源追踪
     - 记录每个文档的来源
     - 便于数据更新和同步
     - 支持按来源过滤

  5. 状态管理
     - 定义清晰的状态流转
     - active -> pending -> archived
     - 支持审核流程

  6. 持久化策略
     - 定期保存索引状态
     - 备份重要版本
     - 实现增量持久化

  7. 清理策略
     - 定期清理过期文档
     - 删除无效节点
     - 优化索引性能

  8. 监控与日志
     - 记录文档变更历史
     - 监控索引大小和性能
     - 异常情况告警
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 17: LlamaIndex 文档管理示例")
    print("=" * 60)

    # 配置设置
    setup_settings()

    # 运行各示例
    example_incremental_insert()
    example_metadata_management()
    example_node_management()
    example_persistence_and_update()
    example_version_management()
    example_multi_source_integration()
    example_document_status()
    example_best_practices()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()