# Day 5: RAG（检索增强生成）

## 概述

第五天学习 RAG（Retrieval-Augmented Generation，检索增强生成），让 AI 能够基于私有知识库进行回答，解决大模型的"知识截止"问题。

## 学习目标

- 理解 RAG 的工作原理和应用场景
- 掌握向量数据库的基本操作
- 学会文档分块和嵌入向量化
- 构建完整的知识库问答系统

## 核心概念

### 1. 什么是 RAG？

RAG 是一种将"检索"与"生成"结合的技术架构：

```
用户问题 → 向量化 → 相似度检索 → 获取相关文档 → 组合 Prompt → LLM 生成回答
```

**解决的问题**：
- 知识时效性：LLM 训练数据有截止日期
- 私有数据：企业内部文档、个人笔记等
- 减少幻觉：基于真实文档回答，更准确
- 降低成本：无需微调模型即可更新知识

### 2. 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                        RAG 系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【知识库构建阶段】                                           │
│                                                             │
│   文档 ─→ 分块 ─→ 嵌入模型 ─→ 向量化 ─→ 向量数据库            │
│                      ↓                                      │
│              text-embedding-3-small                         │
│                      ↓                                      │
│              [0.1, 0.2, 0.3, ...]                          │
│                                                             │
│  【查询阶段】                                                 │
│                                                             │
│   问题 ─→ 嵌入模型 ─→ 向量化 ─→ 相似度搜索                    │
│                                    ↓                        │
│                            找到 Top-K 相关文档               │
│                                    ↓                        │
│                            组合 Context + 问题              │
│                                    ↓                        │
│                            LLM 生成回答                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. 向量数据库

向量数据库专门存储和检索高维向量：

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| ChromaDB | 轻量级、易上手、支持持久化 | 开发测试、小型项目 |
| FAISS | Meta 开源、高性能、纯内存 | 大规模向量检索 |
| Pinecone | 云托管、零运维 | 生产环境 |
| Milvus | 分布式、可扩展 | 企业级应用 |

**本教程使用 ChromaDB**，原因：
- 安装简单（pip install chromadb）
- 内置嵌入模型支持
- 支持持久化存储
- API 简洁易学

### 4. 文本嵌入（Embedding）

将文本转换为高维向量，语义相似的文本向量距离更近：

```python
# 文本转向量示例
"机器学习是AI的子领域" → [0.12, -0.34, 0.56, ..., 0.78]  # 1536维
"深度学习使用神经网络"   → [0.11, -0.32, 0.58, ..., 0.76]  # 相似向量

# 相似度计算（余弦相似度）
similarity = cosine_similarity(vec1, vec2)  # 0.92 (很相似)
```

常用嵌入模型：
- `text-embedding-3-small`：OpenAI，性价比高
- `text-embedding-3-large`：OpenAI，效果更好
- `bge-large-zh`：本地中文模型
- `m3e-base`：本地多语言模型

### 5. 文档分块策略

长文档需要切分成小块存储：

```python
# 分块参数
chunk_size = 500      # 每块最大字符数
chunk_overlap = 50    # 相邻块重叠字符数

# 分块示例
原文档（1000字）→ [块1(500字)] + [块2(500字)]
                    ↑____重叠50字____↓
```

**分块原则**：
- 块太小：语义不完整
- 块太大：检索精度下降
- 重叠区：保证上下文连贯

## 快速开始

### 安装依赖

```bash
pip install chromadb sentence-transformers openai python-dotenv
```

### 基础示例

```python
import chromadb
from chromadb.utils import embedding_functions

# 1. 创建向量数据库
client = chromadb.Client()
collection = client.create_collection("my_knowledge")

# 2. 添加文档
collection.add(
    documents=["Python是一种编程语言", "机器学习是AI的核心技术"],
    ids=["doc1", "doc2"]
)

# 3. 查询
results = collection.query(
    query_texts=["什么是Python？"],
    n_results=1
)
print(results['documents'])  # ['Python是一种编程语言']
```

## 练习文件

### `basic_rag.py`
基础 RAG 实现：
- 创建向量数据库
- 添加文档和嵌入
- 语义相似度搜索
- 结合 LLM 生成回答

### `vector_store.py`
向量数据库操作：
- ChromaDB 完整 CRUD
- 持久化存储
- 元数据过滤
- 批量操作优化

### `document_processor.py`
文档处理模块：
- 文本分块算法
- 多格式文档加载
- 嵌入向量化
- 分块策略优化

### `rag_chatbot.py`
完整 RAG 聊天机器人：
- 知识库管理
- 智能检索
- 多轮对话支持
- 来源引用显示

## RAG 流程详解

### 1. 知识库构建

```python
# 步骤1：加载文档
documents = load_documents("./knowledge/")

# 步骤2：文本分块
chunks = split_documents(documents, chunk_size=500, overlap=50)

# 步骤3：生成嵌入
embeddings = embedding_model.encode(chunks)

# 步骤4：存储到向量数据库
vector_store.add(
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"source": doc.source} for doc in chunks]
)
```

### 2. 查询流程

```python
def rag_query(question: str) -> str:
    # 1. 问题向量化
    question_embedding = embedding_model.encode(question)

    # 2. 相似度检索
    results = vector_store.search(
        query_embedding=question_embedding,
        top_k=3
    )

    # 3. 构建 Context
    context = "\n\n".join(results.documents)

    # 4. 生成回答
    prompt = f"""根据以下信息回答问题：

相关信息：
{context}

问题：{question}

请基于以上信息回答，如果信息不足请说明。"""

    response = llm.generate(prompt)
    return response
```

## 最佳实践

### 1. 文档分块优化

```python
# 根据文档类型选择分块策略
# 结构化文档（API文档）
chunk_size = 300, overlap = 0

# 非结构化文档（文章、报告）
chunk_size = 500, overlap = 50

# 长文档（书籍）
chunk_size = 1000, overlap = 100
```

### 2. 检索优化

```python
# 混合检索：关键词 + 向量
def hybrid_search(query, k=5):
    # 向量检索
    vector_results = vector_store.search(query, k=k*2)

    # 关键词检索
    keyword_results = keyword_search(query, k=k*2)

    # 融合排序
    return rerank(vector_results, keyword_results, k)
```

### 3. Prompt 优化

```python
# 使用结构化 Prompt
RAG_PROMPT = """你是一个知识问答助手。请根据提供的参考信息回答用户问题。

## 参考信息
{context}

## 用户问题
{question}

## 回答要求
1. 优先使用参考信息中的内容回答
2. 如果参考信息不足，明确说明
3. 引用具体来源（如：根据文档[1]...）
4. 回答要简洁准确

## 回答"""
```

### 4. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 检索不相关 | 向量质量差 | 更换嵌入模型 |
| 回答不准确 | Context 太长 | 优化分块策略 |
| 速度慢 | 向量库太大 | 添加索引、过滤 |
| 知识缺失 | 文档未收录 | 补充知识库 |

## 进阶主题

### 1. 重排序（Reranking）

```python
# 先粗检索，再精细排序
candidates = vector_store.search(query, k=20)
reranked = reranker.rerank(query, candidates, top_k=5)
```

### 2. 多轮对话 RAG

```python
# 结合对话历史进行检索
def chat_with_rag(message, history):
    # 结合历史重写问题
    rewritten = rewrite_query(message, history)

    # 检索相关文档
    docs = retrieve(rewritten)

    # 生成回答
    return generate(message, docs, history)
```

### 3. 知识库更新

```python
# 增量更新知识库
def update_knowledge(doc_id, new_content):
    # 删除旧文档
    vector_store.delete(doc_id)

    # 添加新文档
    chunks = split(new_content)
    vector_store.add(chunks, ids=[f"{doc_id}_{i}" for i in range(len(chunks))])
```

## 学习成果

完成本天学习后，你将能够：
- 理解 RAG 的核心原理和流程
- 使用 ChromaDB 构建向量数据库
- 实现文档分块和嵌入向量化
- 构建基于知识库的问答系统

## 下一步

第六天将学习 AI Agent 开发，让 AI 能够自主规划和执行复杂任务。

## 参考资料

- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)