"""
Day 18: LlamaIndex 知识图谱集成示例

本文件演示知识图谱功能：
- 知识图谱索引创建
- 实体关系抽取
- 图谱查询与遍历
- KG + Vector 组合检索

注意：完整功能需要知识图谱存储（如 Neo4j、NebulaGraph）
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    KnowledgeGraphIndex,
)
from llama_index.core.node_parser import SentenceSplitter
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


# ==========================================
# 示例 1: 知识图谱概念
# ==========================================


def example_knowledge_graph_concept():
    """演示知识图谱基本概念"""
    print("\n" + "=" * 50)
    print("示例 1: 知识图谱概念")
    print("=" * 50)

    print("""
  知识图谱结构：

  ┌─────────────────────────────────────────────┐
  │              知识图谱示例                    │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   ┌─────────┐                               │
  │   │ Python  │                               │
  │   │ (实体)  │                               │
  │   └─────────┘                               │
  │        │                                    │
  │        │ created_by (关系)                  │
  │        │                                    │
  │        ▼                                    │
  │   ┌─────────────────┐                       │
  │   │ Guido van Rossum│                       │
  │   │     (实体)      │                       │
  │   └─────────────────┘                       │
  │                                             │
  │   ┌─────────┐    used_for    ┌───────────┐ │
  │   │ Python  │ ────────────→ │ AI/ML     │ │
  │   │         │               │ (实体)    │ │
  │   └─────────┘               └───────────┘ │
  │                                             │
  └─────────────────────────────────────────────┘

  三元组（Triplet）：
  (主语, 关系, 宾语) = (Python, created_by, Guido)
  (Python, used_for, AI/ML)
  (Python, is_type, Programming Language)

  知识图谱优势：
  1. 结构化知识表示
  2. 支持复杂关系查询
  3. 可发现隐含关系
  4. 增强语义理解
    """)


# ==========================================
# 示例 2: 知识图谱索引创建
# ==========================================


def example_kg_index_creation():
    """演示知识图谱索引创建"""
    print("\n" + "=" * 50)
    print("示例 2: 知识图谱索引创建")
    print("=" * 50)

    setup_settings()

    # 创建包含实体关系的文档
    documents = [
        Document(
            text="Python 是一种高级编程语言，由 Guido van Rossum 创建。"
            "Python 广泛用于 Web 开发、数据科学和人工智能领域。"
            "流行的 Python 库包括 NumPy、Pandas 和 TensorFlow。",
        ),
        Document(
            text="LangChain 是一个 LLM 应用开发框架。"
            "LangChain 支持多种 LLM 后端，包括 OpenAI 和 Anthropic。"
            "LangChain 可以与向量数据库集成，构建 RAG 应用。",
        ),
        Document(
            text="LlamaIndex 是一个专注于数据索引的框架。"
            "LlamaIndex 由 LlamaIndex 团队开发。"
            "LlamaIndex 可以创建多种索引类型，包括向量索引和知识图谱索引。",
        ),
    ]

    print("  创建知识图谱索引...")
    print("  （使用 LLM 抽取实体和关系）")

    # 创建知识图谱索引
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=5,  # 每个分块最多 5 个三元组
        include_embeddings=True,  # 包含向量嵌入
        show_progress=True,
    )

    print("✅ 知识图谱索引已创建")

    # 查看抽取的三元组
    print("\n  抽取的知识三元组示例:")
    # 注意：实际三元组由 LLM 动态生成
    print("    (Python, is, Programming Language)")
    print("    (Python, created_by, Guido van Rossum)")
    print("    (LangChain, supports, OpenAI)")
    print("    (LlamaIndex, can_create, KG Index)")


# ==========================================
# 示例 3: 知识图谱查询
# ==========================================


def example_kg_query():
    """演示知识图谱查询"""
    print("\n" + "=" * 50)
    print("示例 3: 知识图谱查询")
    print("=" * 50)

    setup_settings()

    documents = [
        Document(
            text="OpenAI 是一家 AI 公司，开发了 GPT 系列模型。"
            "GPT-4 是 OpenAI 最先进的语言模型。"
            "GPT-4 可以处理文本和图像输入。",
        ),
        Document(
            text="DeepSeek 是中国 AI 公司，开发了 DeepSeek 模型。"
            "DeepSeek 模型支持中文和英文。"
            "DeepSeek 提供 API 接口供开发者使用。",
        ),
    ]

    # 创建 KG 索引
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=5,
    )

    # 知识图谱查询
    query_engine = kg_index.as_query_engine()

    query = "GPT-4 是由哪家公司开发的？"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response}")


# ==========================================
# 示例 4: KG + Vector 组合检索
# ==========================================


def example_kg_vector_combined():
    """演示 KG + Vector 组合检索"""
    print("\n" + "=" * 50)
    print("示例 4: KG + Vector 组合检索")
    print("=" * 50)

    setup_settings()

    print("""
  组合检索架构：

  ┌─────────────────────────────────────────────┐
  │              组合检索流程                    │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   Query ───→ 实体识别 ───→ KG 检索        │
  │            │                               │
  │            └──→ 向量检索                   │
  │                                             │
  │   KG Results ───┐                           │
  │                 ├──→ 融合 ───→ Response    │
  │   Vector Results─┘                          │
  │                                             │
  └─────────────────────────────────────────────┘

  优势：
  1. KG：精确关系查询，发现隐含知识
  2. Vector：语义相似度，补充上下文
  3. 组合：更全面的检索覆盖
    """)

    documents = [
        Document(
            text="机器学习是人工智能的核心技术。"
            "深度学习是机器学习的子领域。"
            "神经网络是深度学习的基础架构。"
            "卷积神经网络用于图像处理。"
            "循环神经网络用于序列数据。",
        ),
        Document(
            text="TensorFlow 是 Google 开发的机器学习框架。"
            "PyTorch 是 Facebook 开发的深度学习框架。"
            "这两个框架都支持神经网络训练。",
        ),
    ]

    # 创建 KG 索引（包含向量）
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=10,
        include_embeddings=True,
    )

    # 组合查询引擎
    query_engine = kg_index.as_query_engine(
        similarity_top_k=3,
        include_text=True,  # 包含文本检索
    )

    query = "深度学习和机器学习有什么关系？"
    print(f"\n查询: {query}")

    response = query_engine.query(query)
    print(f"响应: {response.response[:150]}...")


# ==========================================
# 示例 5: 实体关系抽取
# ==========================================


def example_entity_relation_extraction():
    """演示实体关系抽取"""
    print("\n" + "=" * 50)
    print("示例 5: 实体关系抽取")
    print("=" * 50)

    print("""
  实体关系抽取流程：

  ┌─────────────────────────────────────────────┐
  │           实体关系抽取                       │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   输入文本：                                 │
  │   "Python 由 Guido 创建，用于 Web 开发"     │
  │                                             │
  │   Step 1: 实体识别                          │
  │   ┌───────────────────────────────┐        │
  │   │ Python ─── Entity             │        │
  │   │ Guido  ─── Entity             │        │
  │   │ Web Development ─── Entity    │        │
  │   └───────────────────────────────┘        │
  │                                             │
  │   Step 2: 关系识别                          │
  │   ┌───────────────────────────────┐        │
  │   │ created_by (Python → Guido)   │        │
  │   │ used_for (Python → Web Dev)   │        │
  │   └───────────────────────────────┘        │
  │                                             │
  │   Step 3: 三元组生成                        │
  │   ┌───────────────────────────────┐        │
  │   │ (Python, created_by, Guido)   │        │
  │   │ (Python, used_for, Web Dev)   │        │
  │   └───────────────────────────────┘        │
  │                                             │
  └─────────────────────────────────────────────┘

  抽取方法：
  1. LLM 抽取：使用 GPT 等模型
  2. 规则抽取：基于模板匹配
  3. NER + 关系模型：专业抽取工具

  LlamaIndex 默认使用 LLM 抽取
    """)


# ==========================================
# 示例 6: 图谱存储选项
# ==========================================


def example_kg_storage_options():
    """演示知识图谱存储选项"""
    print("\n" + "=" * 50)
    print("示例 6: 知识图谱存储选项")
    print("=" * 50)

    print("""
  知识图谱存储后端：

  1. 简单内存存储（默认）
     ┌─────────────────────────────────────┐
     │ SimpleGraphStore                    │
     │                                     │
     │ - 内存中存储三元组                  │
     │ - 适合小规模知识图谱                │
     │ - 无法持久化                        │
     └─────────────────────────────────────┘

  2. Neo4j（推荐）
     ┌─────────────────────────────────────┐
     │ Neo4j Graph Store                   │
     │                                     │
     │ - 专业图数据库                      │
     │ - 支持复杂图查询                    │
     │ - 可持久化                          │
     │                                     │
     │ pip install llama-index-graph-stores-neo4j │
     └─────────────────────────────────────┘

  3. NebulaGraph
     ┌─────────────────────────────────────┐
     │ NebulaGraph Store                   │
     │                                     │
     │ - 分布式图数据库                    │
     │ - 适合大规模图谱                    │
     │ - 高性能                            │
     │                                     │
     │ pip install llama-index-graph-stores-nebula │
     └─────────────────────────────────────┘

  配置示例：
  ┌─────────────────────────────────────────┐
  │ from llama_index.graph_stores.neo4j import Neo4jGraphStore │
  │                                         │
  │ graph_store = Neo4jGraphStore(         │
  │     username="neo4j",                  │
  │     password="password",               │
  │     url="bolt://localhost:7687",       │
  │ )                                      │
  │                                         │
  │ kg_index = KnowledgeGraphIndex.from_documents( │
  │     documents,                         │
  │     graph_store=graph_store,           │
  │ )                                      │
  └─────────────────────────────────────────┘
    """)


# ==========================================
# 示例 7: 知识图谱应用场景
# ==========================================


def example_kg_use_cases():
    """演示知识图谱应用场景"""
    print("\n" + "=" * 50)
    print("示例 7: 知识图谱应用场景")
    print("=" * 50)

    print("""
  知识图谱 RAG 应用场景：

  1. 企业知识库
     ┌─────────────────────────────────────┐
     │ 场景：企业内部知识管理             │
     │                                     │
     │ 知识类型：                          │
     │ - 组织架构关系                      │
     │ - 项目与团队关联                    │
     │ - 技术栈依赖关系                    │
     │                                     │
     │ 查询示例：                          │
     │ Q: 负责项目 X 的团队有哪些人？     │
     │ A: 通过组织关系图查询              │
     └─────────────────────────────────────┘

  2. 学术文献
     ┌─────────────────────────────────────┐
     │ 场景：学术研究辅助                  │
     │                                     │
     │ 知识类型：                          │
     │ - 论文引用关系                      │
     │ - 作者合作关系                      │
     │ - 研究领域关联                      │
     │                                     │
     │ 查询示例：                          │
     │ Q: 作者 A 的合作者有哪些？         │
     │ A: 通过合作关系图查询              │
     └─────────────────────────────────────┘

  3. 产品知识
     ┌─────────────────────────────────────┐
     │ 场景：电商产品推荐                  │
     │                                     │
     │ 知识类型：                          │
     │ - 产品属性关系                      │
     │ - 品牌与产品关联                    │
     │ - 用户购买行为                      │
     │                                     │
     │ 查询示例：                          │
     │ Q: 与产品 A 相似的有哪些？         │
     │ A: 通过产品关系图推荐              │
     └─────────────────────────────────────┘

  4. 医疗知识
     ┌─────────────────────────────────────┐
     │ 场景：医疗诊断辅助                  │
     │                                     │
     │ 知识类型：                          │
     │ - 症状与疾病关联                    │
     │ - 药物与治疗关系                    │
     │ - 疾病风险因素                      │
     │                                     │
     │ 查询示例：                          │
     │ Q: 症状 A 可能是哪些疾病？         │
     │ A: 通过症状-疾病关系查询           │
     └─────────────────────────────────────┘
    """)


# ==========================================
# 示例 8: KG vs Vector 对比
# ==========================================


def example_kg_vs_vector():
    """对比知识图谱和向量检索"""
    print("\n" + "=" * 50)
    print("示例 8: KG vs Vector 对比")
    print("=" * 50)

    print("""
  知识图谱 vs 向量检索对比：

  ┌─────────────────┬─────────────────┬─────────────────┐
  │     特性        │   知识图谱      │    向量检索     │
  ├─────────────────┼─────────────────┼─────────────────┤
  │ 知识表示        │ 结构化三元组    │ 语义向量        │
  │ 查询类型        │ 关系查询        │ 语义相似        │
  │ 精确度          │ 高（结构化）    │ 中（相似性）    │
  │ 发现隐含知识    │ ✅ 支持         │ ❌ 不支持       │
  │ 复杂查询        │ ✅ 支持         │ ❌ 有限         │
  │ 构建成本        │ 高（需抽取）    │ 低（直接嵌入）  │
  │ 适用数据        │ 结构化/实体密集 │ 任意文本        │
  │ 多语言          │ 需要适配        │ 自动支持        │
  └─────────────────┴─────────────────┴─────────────────┘

  选择建议：

  ✅ 使用知识图谱：
  - 实体关系明确的数据
  - 需要复杂关系查询
  - 需要发现隐含关联
  - 结构化程度高的领域

  ✅ 使用向量检索：
  - 大规模文本数据
  - 语义相似度查询
  - 快速原型开发
  - 非结构化数据

  ✅ 组合使用：
  - 复杂知识密集领域
  - 需要全面覆盖
  - 高质量问答系统
    """)


# ==========================================
# 示例 9: 知识图谱最佳实践
# ==========================================


def example_kg_best_practices():
    """演示知识图谱最佳实践"""
    print("\n" + "=" * 50)
    print("示例 9: 知识图谱最佳实践")
    print("=" * 50)

    print("""
  知识图谱构建最佳实践：

  ✅ 推荐做法：

  1. 控制三元组数量
     ┌─────────────────────────────────────┐
     │ max_triplets_per_chunk = 10        │
     │                                     │
     │ 避免过多三元组：                    │
     │ - 增加存储成本                      │
     │ - 降低查询效率                      │
     │ - 增加噪音                          │
     └─────────────────────────────────────┘

  2. 结合向量检索
     ┌─────────────────────────────────────┐
     │ include_embeddings = True          │
     │                                     │
     │ KG + Vector 组合：                  │
     │ - KG：精确关系                      │
     │ - Vector：语义补充                  │
     └─────────────────────────────────────┘

  3. 使用专业图数据库
     ┌─────────────────────────────────────┐
     │ Neo4j / NebulaGraph                 │
     │                                     │
     │ 优势：                              │
     │ - 可持久化                          │
     │ - 支持复杂查询                      │
     │ - 性能优化                          │
     └─────────────────────────────────────┘

  4. 预定义实体类型
     ┌─────────────────────────────────────┐
     │ 为常见实体定义类型：                │
     │ - Person（人物）                    │
     │ - Organization（组织）              │
     │ - Product（产品）                   │
     │ - Concept（概念）                   │
     └─────────────────────────────────────┘

  ❌ 避免做法：

  1. 过度抽取三元组
     - max_triplets_per_chunk 过高
     - 导致图过于复杂

  2. 仅使用 KG 检索
     - 缺失语义上下文
     - 无法处理非结构化内容

  3. 忽略图数据库配置
     - 使用内存存储
     - 无法持久化

  4. 不验证三元组质量
     - LLM 可能生成错误关系
     - 需要人工审核重要领域
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 18: LlamaIndex 知识图谱集成示例")
    print("=" * 60)

    # 运行各示例
    example_knowledge_graph_concept()
    example_kg_index_creation()
    example_kg_query()
    example_kg_vector_combined()
    example_entity_relation_extraction()
    example_kg_storage_options()
    example_kg_use_cases()
    example_kg_vs_vector()
    example_kg_best_practices()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()