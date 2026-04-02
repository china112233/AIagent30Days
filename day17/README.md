# Day 17: LlamaIndex 基础

## 概述

LlamaIndex（原名 GPT Index）是一个专注于数据连接的 LLM 框架，旨在让用户能够轻松地将私有或特定领域数据与大模型连接。相比 LangChain 的通用链式编排，LlamaIndex 在数据索引和检索方面更加专业，是构建 RAG 应用的首选框架之一。

## 学习目标

- 理解 LlamaIndex 的核心概念：Document、Node、Index
- 掌握多种索引类型的创建和使用
- 学会配置和优化查询引擎
- 实现完整的 RAG 管道
- 了解文档管理和元数据过滤

## 核心概念

### 1. LlamaIndex 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    LlamaIndex 架构图                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                  数据处理层                          │  │
│   │                                                     │  │
│   │    Document ───→ Node ───→ Index                    │  │
│   │    (文档)        (节点)     (索引)                   │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                  查询处理层                          │  │
│   │                                                     │  │
│   │    Query ───→ retriever ───→ response synthesizer   │  │
│   │    (查询)    (检索器)       (响应合成器)              │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                  存储层                              │  │
│   │                                                     │  │
│   │    Vector Store │ Index Store │ Doc Store           │  │
│   │    (向量存储)    (索引存储)   (文档存储)              │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 核心组件对比

| 组件 | 说明 | 类比理解 |
|------|------|----------|
| Document | 原始数据容器 | 一本书 |
| Node | 文档的分块单元 | 书中的一页/段落 |
| Index | 节点的组织结构 | 图书目录/索引 |
| Query Engine | 查询处理入口 | 图书管理员 |
| Retriever | 检索相关节点 | 查找书籍 |
| Response Synthesizer | 生成最终响应 | 总结答案 |

### 3. 索引类型对比

| 索引类型 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| VectorStoreIndex | 语义检索、RAG | 高召回率、语义理解 | 需要嵌入模型 |
| ListIndex | 小数据集、顺序处理 | 简单、无需嵌入 | 不支持语义检索 |
| TreeIndex | 层级摘要、大文档 | 分层检索效率高 | 构建成本较高 |
| KeywordTableIndex | 关键词匹配、精确检索 | 快速、精确 | 无语义理解 |

### 4. 数据处理流程

```
┌─────────────────────────────────────────────────────────────┐
│                   数据处理流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 加载文档                                                │
│     ┌───────────────────────────────────────────────────┐ │
│     │ Document = SimpleDirectoryReader("./data").load() │ │
│     └───────────────────────────────────────────────────┘ │
│                         │                                   │
│                         ▼                                   │
│  2. 解析为节点                                              │
│     ┌───────────────────────────────────────────────────┐ │
│     │ Nodes = Parser.split_documents(documents)         │ │
│     │ - SentenceSplitter (按句子分割)                    │ │
│     │ - TokenTextSplitter (按 Token 分割)               │ │
│     │ - SemanticSplitter (语义分割)                     │ │
│     └───────────────────────────────────────────────────┘ │
│                         │                                   │
│                         ▼                                   │
│  3. 创建索引                                                │
│     ┌───────────────────────────────────────────────────┐ │
│     │ Index = VectorStoreIndex(nodes)                   │ │
│     │ - 生成嵌入向量                                     │ │
│     │ - 存储到向量数据库                                 │ │
│     └───────────────────────────────────────────────────┘ │
│                         │                                   │
│                         ▼                                   │
│  4. 查询                                                    │
│     ┌───────────────────────────────────────────────────┐ │
│     │ Engine = Index.as_query_engine()                  │ │
│     │ Response = Engine.query("问题")                   │ │
│     └───────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. Response Mode（响应模式）

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| tree_summarize | 递归总结所有节点 | 需要完整答案 |
| compact | 压缩节点后生成响应 | 节点较多时 |
| refine | 逐节点优化响应 | 需要精确答案 |
| simple_summarize | 简单汇总 | 快速响应 |
| no_text | 只返回节点，不生成响应 | 仅检索场景 |

### 6. 检索策略

```python
# 相似度检索（默认）
retriever = index.as_retriever(similarity_top_k=3)

# 混合检索（向量 + 关键词）
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import KeywordTableSimpleRetriever

# 元数据过滤
retriever = index.as_retriever(
    similarity_top_k=3,
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="category", value="tech")]
    )
)
```

## 快速开始

### 安装依赖

```bash
pip install llama-index llama-index-core
pip install llama-index-llms-openai  # OpenAI 兼容 LLM
pip install llama-index-embeddings-openai  # OpenAI 兼容嵌入
pip install llama-index-vector-stores-chroma  # 向量存储（可选）
pip install python-dotenv  # 环境变量管理
```

### 配置环境

参考 `.env.example` 创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

## 练习文件说明

### `llamaindex_basics.py` - 基础入门

演示 LlamaIndex 的核心概念：
- LLM 和嵌入模型配置
- Document 创建与加载
- 简单向量索引创建
- 基本查询操作

### `document_index.py` - 文档与索引

深入文档处理和索引类型：
- 文档加载器使用
- 分块策略配置
- 多种索引类型对比
- 向量存储集成

### `query_engine.py` - 查询引擎

展示查询引擎配置：
- 查询引擎参数配置
- Response Mode 对比
- 检索器自定义
- 相似度检索优化

### `rag_pipeline.py` - RAG 管道

实现完整 RAG 系统：
- 端到端 RAG 管道构建
- 自定义检索与合成
- 元数据过滤
- 性能优化技巧

## 运行示例

```bash
# 运行基础示例
python day17/llamaindex_basics.py

# 运行文档索引示例
python day17/document_index.py

# 运行查询引擎示例
python day17/query_engine.py

# 运行 RAG 管道示例
python day17/rag_pipeline.py
```

## 最佳实践

### 1. 分块策略选择

```python
# ✅ 根据文档类型选择分块策略
from llama_index.core.node_parser import SentenceSplitter

# 技术文档：保持代码完整性
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    paragraph_separator="\n\n"
)

# 长文档：使用语义分割
from llama_index.core.node_parser import SemanticSplitterNodeParser
semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95
)

# ❌ 避免：不分块直接处理长文档
```

### 2. 索引选择

```python
# ✅ 根据场景选择索引类型
# RAG 应用：VectorStoreIndex（语义检索）
# 小数据集：ListIndex（简单高效）
# 层级数据：TreeIndex（分层摘要）

# ❌ 避免：盲目使用 VectorStoreIndex
# 如果数据量小且需要精确匹配，KeywordTableIndex 更合适
```

### 3. 查询引擎配置

```python
# ✅ 合理配置检索参数
query_engine = index.as_query_engine(
    similarity_top_k=3,  # 检索 3 个最相关节点
    response_mode="compact",  # 压缩模式，减少 Token 消耗
    streaming=True  # 流式响应，提升用户体验
)

# ❌ 避免：检索过多节点
# similarity_top_k=20 会导致响应变慢、成本增加
```

### 4. 元数据管理

```python
# ✅ 为文档添加元数据，支持过滤
documents = [
    Document(
        text="内容...",
        metadata={
            "category": "tech",
            "date": "2026-01-15",
            "author": "张三"
        }
    )
]

# 查询时过滤
filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="category", value="tech")
    ]
)
query_engine = index.as_query_engine(filters=filters)

# ❌ 避免：忽略元数据
# 元数据可以大幅提升检索精度
```

### 5. 嵌入模型选择

```python
# ✅ 选择合适的嵌入模型
from llama_index.embeddings.openai import OpenAIEmbedding

# 高质量：text-embedding-3-large
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# 平衡成本：text-embedding-3-small
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 使用 DeepSeek 兼容的嵌入服务
embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_base="https://api.deepseek.com"
)
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 嵌入失败 | API 配置错误 | 检查 base_url 和 api_key |
| 检索结果不相关 | 分块策略不当 | 调整 chunk_size 和 overlap |
| 响应太慢 | 检索节点过多 | 减少 similarity_top_k |
| 内存不足 | 索引过大 | 使用外部向量数据库 |
| 响应质量差 | 嵌入模型不适配 | 尝试其他嵌入模型 |
| 中文效果差 | 嵌入模型不支持中文 | 使用多语言嵌入模型 |

## LlamaIndex vs LangChain

| 特性 | LlamaIndex | LangChain |
|------|------------|-----------|
| 核心定位 | 数据索引与检索 | 通用链式编排 |
| RAG 支持 | 专业、深度 | 基础支持 |
| 索引类型 | 多种专业索引 | 基本向量存储 |
| 学习曲线 | 中等 | 较陡 |
| 文档处理 | 强大、灵活 | 基础 |
| Agent 支持 | 有 | 强大 |
| 生态系统 | 数据连接丰富 | 工具集成丰富 |

## 学习成果

完成本天学习后，你将能够：
- ✅ 理解 LlamaIndex 的核心架构和数据流
- ✅ 创建和管理不同类型的索引
- ✅ 配置查询引擎进行高效检索
- ✅ 实现完整的 RAG 应用管道
- ✅ 使用元数据过滤和分块优化

## 下一步

Day 18 将学习 LlamaIndex 进阶，内容包括：高级检索策略、多模态支持、知识图谱集成。

## 参考资料

- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [LlamaIndex 示例](https://docs.llamaindex.ai/en/latest/examples/)
- [RAG 最佳实践](https://docs.llamaindex.ai/en/latest/optimizing_rag/)