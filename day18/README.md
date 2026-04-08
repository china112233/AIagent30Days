# Day 18: LlamaIndex 进阶

## 概述

在 Day 17 学习了 LlamaIndex 的基础概念后，本天深入探索 LlamaIndex 的进阶功能，包括高级检索策略、多模态支持、知识图谱集成、索引优化和外部工具集成。这些高级特性可以帮助构建更智能、更高效的 RAG 应用。

## 学习目标

- 掌握多种高级检索策略（混合检索、重排序、HyDE）
- 了解多模态 RAG 的实现方式
- 学会知识图谱与 LlamaIndex 的集成
- 掌握索引性能优化技巧
- 了解外部工具和服务的集成方式

## 核心概念

### 1. 高级检索策略架构

```
┌─────────────────────────────────────────────────────────────┐
│                    高级检索策略架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              查询处理阶段                            │  │
│   │                                                     │  │
│   │   Query ───→ Query Transform ───→ Multi-Retrieval  │  │
│   │   (查询)    (查询变换)         (多路检索)           │  │
│   │                                                     │  │
│   │   - HyDE (假设文档嵌入)                              │  │
│   │   - Query Rewriting (查询重写)                       │  │
│   │   - Multi-Query (多查询扩展)                         │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              检索融合阶段                            │  │
│   │                                                     │  │
│   │   Vector Retrieval ───→ Keyword Retrieval           │  │
│   │   (向量检索)          (关键词检索)                   │  │
│   │         │                    │                      │  │
│   │         ▼                    ▼                      │  │
│   │   ──────────────── Fusion ────────────────         │  │
│   │                     (融合)                          │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              重排序阶段                              │  │
│   │                                                     │  │
│   │   Retrieved Nodes ───→ Re-ranker ───→ Top-K Nodes   │  │
│   │   (检索节点)        (重排序器)    (最终节点)         │  │
│   │                                                     │  │
│   │   - Cohere Rerank                                   │  │
│   │   - Cross-Encoder                                    │  │
│   │   - LLM-based Rerank                                 │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 检索策略对比

| 策略 | 说明 | 适用场景 | 优点 | 缺点 |
|------|------|----------|------|------|
| Vector Retrieval | 纯向量相似度检索 | 语义匹配 | 语义理解强 | 精确匹配差 |
| Keyword Retrieval | BM25 关键词检索 | 精确匹配 | 快速精确 | 无语义理解 |
| Hybrid Retrieval | 向量 + 关键词融合 | 综合场景 | 高召回率 | 计算成本高 |
| HyDE | 假设文档嵌入检索 | 模糊查询 | 提高召回 | 增加延迟 |
| Re-ranking | 二次重排序 | 高精度需求 | 高准确性 | 增加成本 |

### 3. 多模态支持架构

```
┌─────────────────────────────────────────────────────────────┐
│                    多模态 RAG 架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              多模态数据输入                          │  │
│   │                                                     │  │
│   │    Text ───┐                                        │  │
│   │    Image ──┼──→ MultiModal Document ───→ Index      │  │
│   │    Audio ──┘                                        │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              多模态处理                              │  │
│   │                                                     │  │
│   │   ┌───────────┐   ┌───────────┐   ┌───────────┐   │  │
│   │   │ Text      │   │ Image     │   │ Audio     │   │  │
│   │   │ Embedder  │   │ Encoder   │   │ Encoder   │   │  │
│   │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   │  │
│   │         │               │               │         │  │
│   │         └───────┬───────┴───────┬───────┘         │  │
│   │                 │               │                 │  │
│   │                 ▼               ▼                 │  │
│   │           MultiModal Vector Store                 │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              多模态查询与响应                        │  │
│   │                                                     │  │
│   │   Query ───→ MultiModal Retrieval ───→ Response    │  │
│   │   (可能包含图像/音频)                               │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. 知识图谱集成

```
┌─────────────────────────────────────────────────────────────┐
│                    知识图谱 RAG 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              知识图谱构建                            │  │
│   │                                                     │  │
│   │   Documents ───→ KG Extraction ───→ KnowledgeGraph │  │
│   │   (文档)        (知识抽取)        (知识图谱)        │  │
│   │                                                     │  │
│   │   抽取内容：                                         │  │
│   │   - Entities (实体)                                 │  │
│   │   - Relations (关系)                                │  │
│   │   - Properties (属性)                               │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              知识图谱索引                            │  │
│   │                                                     │  │
│   │   ┌───────────┐       ┌───────────┐               │  │
│   │   │ KG Index  │       │ Vector    │               │  │
│   │   │ (图谱索引) │ ←───→ │ Index     │               │  │
│   │   │           │       │ (向量索引) │               │  │
│   │   └─────┬─────┘       └─────┬─────┘               │  │
│   │         │                   │                     │  │
│   │         └───────┬───────────┘                     │  │
│   │                 ▼                                 │  │
│   │         Combined Retrieval                        │  │
│   │         (组合检索)                                 │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              图谱增强响应                            │  │
│   │                                                     │  │
│   │   Query ───→ Entity Linking ───→ Graph Traversal   │  │
│   │   (查询)    (实体链接)       (图谱遍历)             │  │
│   │                                                     │  │
│   │   Graph Context ───→ Vector Context ───→ Response  │  │
│   │   (图谱上下文)     (向量上下文)     (响应)          │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. 索引优化策略

| 优化类型 | 说明 | 效果 |
|----------|------|------|
| 分块优化 | 动态分块、语义分割 | 提高检索精度 |
| 嵌入优化 | 批量嵌入、缓存机制 | 降低延迟成本 |
| 索引压缩 | 量化、降维 | 减少存储空间 |
| 增量更新 | 智能更新策略 | 减少重建成本 |
| 缓存策略 | 结果缓存、嵌入缓存 | 提高响应速度 |

### 6. 外部集成架构

```
┌─────────────────────────────────────────────────────────────┐
│                    外部集成架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              LlamaIndex 核心                         │  │
│   │                                                     │  │
│   │         ┌───────────────────────────────┐          │  │
│   │         │      Query Engine             │          │  │
│   │         └───────────────────────────────┘          │  │
│   │                        │                            │  │
│   └────────────────────────│────────────────────────────┘  │
│                            │                                │
│       ┌────────────────────┼────────────────────┐          │
│       │                    │                    │          │
│       ▼                    ▼                    ▼          │
│   ┌───────────┐     ┌───────────┐     ┌───────────┐        │
│   │ Vector    │     │ Tools     │     │ LLMs      │        │
│   │ Stores    │     │ Agents    │     │ Services  │        │
│   ├───────────┤     ├───────────┤     ├───────────┤        │
│   │ • Chroma  │     │ • Query   │     │ • OpenAI  │        │
│   │ • Pinecone│     │   Tools   │     │ • DeepSeek│        │
│   │ • Weaviate│     │ • API     │     │ • Claude  │        │
│   │ • Milvus  │     │   Tools   │     │ • Local   │        │
│   │ • Qdrant  │     │ • Web     │     │   Models  │        │
│   │           │     │   Search  │     │           │        │
│   └───────────┘     └───────────┘     └───────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装依赖

```bash
# LlamaIndex 核心
pip install llama-index llama-index-core

# LLM 和嵌入
pip install llama-index-llms-openai-like
pip install llama-index-embeddings-huggingface

# 向量存储
pip install llama-index-vector-stores-chroma

# 重排序
pip install llama-index-postprocessor-cohere-rerank

# 知识图谱
pip install llama-index-graph-stores-nebula

# 多模态（可选）
pip install llama-index-multi-modal-llms-openai

# 工具集成
pip install llama-index-tools-wikipedia
pip install llama-index-tools-google
```

### 配置环境

参考 `.env.example` 创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

## 练习文件说明

### `advanced_retrieval.py` - 高级检索策略

演示多种检索技术：
- 混合检索（向量 + BM25）
- HyDE 查询变换
- 查询重写与扩展
- Cross-Encoder 重排序
- 自定义检索器组合

### `multimodal.py` - 多模态支持

展示多模态处理：
- 图像文档处理
- 多模态嵌入生成
- 图像检索与问答
- 多模态 RAG 管道

### `knowledge_graph.py` - 知识图谱集成

演示知识图谱功能：
- 知识图谱索引创建
- 实体关系抽取
- 图谱查询与遍历
- KG + Vector 组合检索

### `optimization.py` - 索引优化

展示优化技巧：
- 分块策略优化
- 嵌入批处理
- 索引持久化优化
- 查询性能调优
- 内存管理技巧

### `integrations.py` - 外部工具集成

演示集成功能：
- 向量数据库集成（Chroma/Pinecone）
- 外部 API 工具集成
- Wikipedia 搜索集成
- Web 搜索集成
- 多 LLM 后端配置

## 运行示例

```bash
# 运行高级检索示例
python day18/advanced_retrieval.py

# 运行多模态示例
python day18/multimodal.py

# 运行知识图谱示例
python day18/knowledge_graph.py

# 运行优化示例
python day18/optimization.py

# 运行集成示例
python day18/integrations.py
```

## 最佳实践

### 1. 高级检索策略选择

```python
# ✅ 根据场景选择检索策略
# 精确匹配场景：使用 KeywordRetriever 或混合检索
# 语义理解场景：使用 VectorRetriever + HyDE
# 高精度需求：添加 Re-ranking 层

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# 混合检索配置
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
bm25_retriever = BM25Retriever(nodes=nodes, similarity_top_k=5)

# ❌ 避免：盲目使用单一检索策略
# 不同场景需要不同的检索组合
```

### 2. 重排序器使用

```python
# ✅ 使用重排序提高精度
from llama_index.postprocessor.cohere_rerank import CohereRerank

query_engine = index.as_query_engine(
    similarity_top_k=10,  # 先检索更多节点
    node_postprocessors=[CohereRerank(top_n=3)]  # 重排序后取 top 3
)

# ❌ 避免：检索后直接使用，不进行重排序
# 对于高精度场景，重排序可显著提高质量
```

### 3. 查询变换优化

```python
# ✅ 使用 HyDE 提高召回
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

hyde = HyDEQueryTransform(include_original=True)
query_engine = TransformQueryEngine(
    query_engine=index.as_query_engine(),
    query_transform=hyde
)

# 适用场景：查询表述模糊、语义匹配困难

# ❌ 避免：对所有查询使用 HyDE
# HyDE 会增加延迟，只在必要时使用
```

### 4. 多模态数据处理

```python
# ✅ 正确处理多模态数据
from llama_index.core import SimpleDirectoryReader

# 指定文件类型
reader = SimpleDirectoryReader(
    input_dir="./images",
    required_exts=[".png", ".jpg", ".pdf"],
)

documents = reader.load_data()

# ❌ 避免：混合处理时忽略格式
# 不同模态需要不同的处理管道
```

### 5. 知识图谱索引

```python
# ✅ 合理配置知识图谱
from llama_index.core import KnowledgeGraphIndex

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,  # 控制三元组数量
    include_embeddings=True,  # 结合向量检索
)

# ❌ 避免：过度抽取三元组
# 三元组过多会增加图复杂度，影响查询效率
```

### 6. 性能优化配置

```python
# ✅ 使用批量嵌入和缓存
from llama_index.core import Settings

Settings.embed_model = embed_model
Settings.chunk_size = 512

# 批量嵌入
nodes = splitter.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes, show_progress=True)

# 使用缓存
from llama_index.core.storage.docstore import SimpleDocumentStore
docstore = SimpleDocumentStore()

# ❌ 避免：逐个嵌入，无缓存
# 这会导致大量重复计算，浪费时间和成本
```

### 7. 向量数据库集成

```python
# ✅ 使用持久化向量存储
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("my_docs")
vector_store = ChromaVectorStore(chroma_collection=collection)

# ❌ 避免：使用内存存储处理大量数据
# 内存存储无法持久化，重启后需要重建索引
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 混合检索效果差 | 权重配置不当 | 调整 alpha 参数平衡向量和关键词 |
| HyDE 响应慢 | 需额外调用 LLM | 减少假设文档数量或缓存结果 |
| 重排序失败 | API 配置错误 | 检查 Cohere API Key 和配额 |
| 知识图谱构建慢 | 三元组抽取耗时 | 减少 max_triplets_per_chunk |
| 多模态处理失败 | 缺少必要依赖 | 安装 Pillow、torch 等库 |
| 索引更新慢 | 全量重建 | 使用增量更新策略 |
| 内存溢出 | 索引过大 | 使用外部向量数据库 |

## LlamaIndex 高级功能对比

| 功能 | 基础版 | 进阶版 |
|------|--------|--------|
| 检索 | 单一向量检索 | 混合检索 + 重排序 |
| 数据 | 纯文本 | 多模态（文本/图像/音频） |
| 知识 | 向量索引 | KG + Vector 组合 |
| 性能 | 基本配置 | 批处理 + 缓存 + 增量更新 |
| 存储 | 内存/本地 | 外部向量数据库 |
| 扩展 | 单一 LLM | 多 LLM + 工具集成 |

## 学习成果

完成本天学习后，你将能够：
- ✅ 实现混合检索和重排序策略
- ✅ 构建多模态 RAG 应用
- ✅ 集成知识图谱增强检索
- ✅ 优化索引性能和资源使用
- ✅ 集成外部工具和服务

## 下一步

Day 19 将探索其他框架，包括：Semantic Kernel、Dify、AutoGen 简介。

## 参考资料

- [LlamaIndex 高级检索](https://docs.llamaindex.ai/en/stable/examples/retrievers/)
- [LlamaIndex 多模态](https://docs.llamaindex.ai/en/stable/examples/multi_modal/)
- [LlamaIndex 知识图谱](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/)
- [LlamaIndex 优化指南](https://docs.llamaindex.ai/en/stable/optimizing_rag/)