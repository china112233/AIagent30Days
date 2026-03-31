"""
Day 15: 完整 RAG 实现示例

本文件演示完整的 RAG（检索增强生成）应用：
- 文档加载与分割
- 向量嵌入与存储
- 多种检索策略
- RAG 链构建
- 问答系统实现
"""

import os
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.vectorstores import Chroma


# 加载环境变量
load_dotenv()


def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7,
    )


def create_embeddings():
    """创建 Embeddings 实例"""
    # 使用 OpenAI Embeddings（需要 OpenAI API Key）
    # 或者使用其他嵌入模型如 HuggingFace
    try:
        return OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    except Exception:
        # 如果 OpenAI 不可用，使用本地嵌入
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


# ==========================================
# 示例 1: 文档加载
# ==========================================


def example_document_loading():
    """演示文档加载"""
    print("\n" + "=" * 50)
    print("示例 1: 文档加载")
    print("=" * 50)

    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
LangChain 简介

LangChain 是一个用于开发大语言模型应用的框架。
它提供了一套完整的工具链，帮助开发者快速构建复杂的 LLM 应用。

核心组件包括：
- LLM Model: 各种大语言模型的统一接口
- Prompt Template: 灵活的提示词模板系统
- Chain: 可组合的处理链
- Memory: 对话记忆管理
- Tool: 工具定义和调用
- Agent: 智能代理系统

LangChain 支持多种语言模型后端，包括 OpenAI、Anthropic、DeepSeek 等。
""")
        text_file = f.name

    # 加载文本文件
    print("\n加载文本文件:")
    loader = TextLoader(text_file)
    docs = loader.load()
    print(f"加载了 {len(docs)} 个文档")
    print(f"第一个文档内容 (前100字符): {docs[0].page_content[:100]}")
    print(f"元数据: {docs[0].metadata}")

    # 清理临时文件
    os.unlink(text_file)

    # 模拟 PDF 加载（需要实际 PDF 文件）
    print("\n--- PDF 加载示例 ---")
    print("PyPDFLoader 可以加载 PDF 文件:")
    print("loader = PyPDFLoader('document.pdf')")
    print("docs = loader.load()")
    print("每个页面会成为一个独立的 Document")

    # 模拟目录加载
    print("\n--- 目录加载示例 ---")
    print("DirectoryLoader 可以批量加载目录中的文件:")
    print("loader = DirectoryLoader('./documents', glob='**/*.txt')")
    print("docs = loader.load()")


# ==========================================
# 示例 2: 文档分割
# ==========================================


def example_document_splitting():
    """演示文档分割"""
    print("\n" + "=" * 50)
    print("示例 2: 文档分割")
    print("=" * 50)

    # 创建测试文档
    long_text = """
# LangChain 架构

LangChain 采用模块化架构设计，主要包括以下层次：

## 应用层
应用层是最终用户接触的界面，包括聊天机器人、RAG 系统、Agent 等。

## Chain 层
Chain 层负责组件的编排和组合，实现复杂的处理流程。

## 核心组件层
核心组件层提供基础功能模块：
- LLM: 语言模型接口
- Prompt: 提示词管理
- Memory: 状态记忆
- Tool: 工具定义

## 集成层
集成层连接外部服务和模型提供商。

# RAG 技术

RAG（Retrieval-Augmented Generation）是一种重要的技术架构：

## RAG 流程
1. 用户提出问题
2. 系统从知识库检索相关文档
3. 将检索结果与问题组合
4. LLM 生成答案

## RAG 优势
- 减少幻觉
- 支持知识更新
- 提供来源追溯

# Agent 系统

Agent 是能够自主决策的 AI 系统。

## Agent 类型
- ReAct Agent: 推理+行动模式
- Plan-and-Execute: 规划执行模式
- Conversational Agent: 对话型代理
"""
    doc = Document(page_content=long_text)

    # 1. RecursiveCharacterTextSplitter（推荐）
    print("\n--- RecursiveCharacterTextSplitter ---")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents([doc])
    print(f"分割为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1} (长度 {len(chunk.page_content)}):")
        print(chunk.page_content[:100] + "...")

    # 2. CharacterTextSplitter（简单分割）
    print("\n--- CharacterTextSplitter ---")
    splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="\n",
    )
    chunks = splitter.split_documents([doc])
    print(f"分割为 {len(chunks)} 个块")

    # 3. MarkdownHeaderTextSplitter（按标题分割）
    print("\n--- MarkdownHeaderTextSplitter ---")
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(long_text)
    print(f"分割为 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1}:")
        print(f"元数据: {chunk.metadata}")
        print(f"内容: {chunk.page_content[:100]}...")

    # 最佳实践
    print("\n--- 分割策略选择 ---")
    print("选择分割策略的原则：")
    print("1. 通用文本：RecursiveCharacterTextSplitter")
    print("2. Markdown：MarkdownHeaderTextSplitter")
    print("3. 代码：按函数/类分割")
    print("4. PDF：按页面分割后用 RecursiveCharacterTextSplitter")

    print("\n参数选择原则：")
    print("- chunk_size: 500-1000 字符（平衡检索精度和上下文）")
    print("- chunk_overlap: 100-200 字符（保持上下文连贯）")


# ==========================================
# 示例 3: 向量嵌入与存储
# ==========================================


def example_vector_store():
    """演示向量嵌入与存储"""
    print("\n" + "=" * 50)
    print("示例 3: 向量嵌入与存储")
    print("=" * 50)

    # 创建测试文档
    docs = [
        Document(page_content="LangChain 是一个 LLM 应用开发框架。", metadata={"id": 1}),
        Document(page_content="RAG 技术结合了检索和生成能力。", metadata={"id": 2}),
        Document(page_content="向量数据库用于存储文本的嵌入表示。", metadata={"id": 3}),
        Document(page_content="Agent 可以自主决策和执行任务。", metadata={"id": 4}),
        Document(page_content="LCEL 是 LangChain 的表达式语言。", metadata={"id": 5}),
    ]

    # 创建临时向量存储目录
    persist_dir = tempfile.mkdtemp()

    print("\n创建向量存储...")
    print("注意：如果 OpenAI API 不可用，将使用本地嵌入模型")

    try:
        embeddings = create_embeddings()

        # 创建 Chroma 向量存储
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

        print(f"向量存储创建成功，包含 {len(docs)} 个文档")

        # 测试相似度检索
        print("\n--- 相似度检索 ---")
        query = "什么是 LangChain"
        results = vectorstore.similarity_search(query, k=3)
        print(f"查询: '{query}'")
        print(f"找到 {len(results)} 个相关文档:")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content}")

        # 带分数的相似度检索
        print("\n--- 带分数的相似度检索 ---")
        results = vectorstore.similarity_search_with_score(query, k=3)
        print(f"查询: '{query}'")
        for doc, score in results:
            print(f"  分数 {score:.4f}: {doc.page_content}")

        # 最大边际相关性检索（MMR）
        print("\n--- MMR 检索（多样性） ---")
        results = vectorstore.max_marginal_relevance_search(query, k=3)
        print(f"MMR 检索结果:")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content}")

        # 创建检索器
        print("\n--- 检索器 ---")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        results = retriever.invoke(query)
        print(f"检索器结果数量: {len(results)}")

    except Exception as e:
        print(f"向量存储创建失败: {e}")
        print("这可能是因为嵌入模型不可用")

    finally:
        # 清理临时目录
        import shutil
        try:
            shutil.rmtree(persist_dir)
        except:
            pass


# ==========================================
# 示例 4: 检索策略
# ==========================================


def example_retrieval_strategies():
    """演示多种检索策略"""
    print("\n" + "=" * 50)
    print("示例 4: 检索策略")
    print("=" * 50)

    # 创建模拟的向量存储
    docs = [
        Document(page_content="Python 是一种流行的编程语言，广泛用于数据科学和 AI 开发。"),
        Document(page_content="LangChain 提供了构建 LLM 应用的工具和框架。"),
        Document(page_content="RAG 系统通过检索相关文档来增强 LLM 的回答质量。"),
        Document(page_content="向量数据库可以高效存储和检索文本嵌入。"),
        Document(page_content="深度学习是机器学习的一个分支，使用神经网络模型。"),
        Document(page_content="自然语言处理让计算机理解人类语言。"),
    ]

    print("\n--- 检索策略对比 ---")

    # 策略 1: 相似度检索
    print("\n1. Similarity（相似度检索）")
    print("特点: 纯粹基于向量相似度，返回最相似的文档")
    print("适用: 基础 RAG，简单场景")

    # 策略 2: MMR 检索
    print("\n2. MMR（最大边际相关性）")
    print("特点: 在相似度和多样性之间平衡，减少重复内容")
    print("适用: 需要多样化答案的场景")
    print("参数: fetch_k（候选数量）, lambda_mult（相似度权重）")

    # 策略 3: 带阈值过滤
    print("\n3. Similarity Score Threshold")
    print("特点: 只返回相似度高于阈值的文档")
    print("适用: 对答案质量要求高的场景")
    print("参数: score_threshold（阈值，如 0.8）")

    # 策略 4: Multi-Query
    print("\n4. Multi-Query（多查询扩展）")
    print("特点: 将用户查询扩展为多个相关查询，提高召回率")
    print("适用: 用户查询可能不完整的场景")

    # 策略 5: Self-Query
    print("\n5. Self-Query（自查询）")
    print("特点: LLM 分析查询，提取过滤条件")
    print("适用: 需要按元数据过滤的场景（如时间、类别）")

    # 示例：使用不同策略
    print("\n--- 实际示例 ---")

    try:
        embeddings = create_embeddings()
        vectorstore = Chroma.from_documents(docs, embeddings)

        # 相似度检索
        retriever_sim = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        results = retriever_sim.invoke("AI 开发")
        print(f"相似度检索 'AI 开发': {len(results)} 个结果")

        # MMR 检索
        retriever_mmr = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
        )
        results = retriever_mmr.invoke("AI 开发")
        print(f"MMR 检索 'AI 开发': {len(results)} 个结果")

        # 带阈值检索
        retriever_threshold = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.7},
        )
        results = retriever_threshold.invoke("AI 开发")
        print(f"阈值检索 'AI 开发': {len(results)} 个结果（阈值 0.7）")

    except Exception as e:
        print(f"检索示例失败（嵌入模型可能不可用）: {e}")


# ==========================================
# 示例 5: 完整 RAG 链构建
# ==========================================


def example_rag_chain():
    """演示完整 RAG 链构建"""
    print("\n" + "=" * 50)
    print("示例 5: 完整 RAG 链构建")
    print("=" * 50)

    # 创建知识库文档
    docs = [
        Document(page_content="LangChain 是一个用于开发大语言模型应用的框架。它提供了 Chain、Memory、Tool、Agent 等核心组件。"),
        Document(page_content="LCEL（LangChain Expression Language）是一种声明式语法，用于组合处理链。它支持管道操作符 | 和流式输出。"),
        Document(page_content="RAG（检索增强生成）是一种架构模式，通过检索相关文档来增强 LLM 的回答能力。主要流程包括：检索、构建上下文、生成回答。"),
        Document(page_content="Agent 是能够自主决策的 AI 系统。它可以使用工具、规划任务、执行多步操作。常见的 Agent 类型有 ReAct、Plan-and-Execute。"),
        Document(page_content="向量数据库是存储文本嵌入的专门数据库。常见的有 Chroma、Pinecone、Weaviate、Milvus 等。它们支持高效的相似度检索。"),
    ]

    # 分割文档
    print("\n分割文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"分割为 {len(chunks)} 个块")

    try:
        # 创建向量存储
        print("\n创建向量存储...")
        embeddings = create_embeddings()
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 创建 LLM
        llm = create_llm()

        # 定义提示词模板
        prompt = ChatPromptTemplate.from_template(
            """你是一个知识助手。请基于以下参考信息回答问题。
如果参考信息中没有相关内容，请说明无法从知识库中找到答案。

参考信息：
{context}

问题：{question}

回答："""
        )

        # 格式化文档
        def format_docs(docs: List[Document]) -> str:
            return "\n\n---\n\n".join([d.page_content for d in docs])

        # 构建 RAG 链
        print("\n构建 RAG 链...")
        rag_chain = (
            RunnablePassthrough.assign(context=retriever | format_docs)
            | prompt
            | llm
            | StrOutputParser()
        )

        # 测试问答
        print("\n--- 测试问答 ---")

        questions = [
            "什么是 LangChain？",
            "LCEL 有什么特点？",
            "请解释 RAG 的工作原理。",
        ]

        for question in questions:
            print(f"\n问题: {question}")
            print("回答: ", end="")
            result = rag_chain.invoke({"question": question})
            print(result[:200] + "...")

        # 流式输出示例
        print("\n--- 流式输出 ---")
        question = "什么是向量数据库？"
        print(f"问题: {question}")
        print("回答: ", end="")
        for chunk in rag_chain.stream({"question": question}):
            print(chunk, end="", flush=True)
        print()

    except Exception as e:
        print(f"RAG 链示例失败: {e}")
        print("这可能是因为 API 或嵌入模型不可用")

        # 使用模拟组件演示链结构
        print("\n--- 模拟演示链结构 ---")
        from custom_components import MockLLM, SimpleRetriever

        mock_llm = MockLLM()
        mock_retriever = SimpleRetriever documents=chunks)

        rag_chain = (
            RunnablePassthrough.assign(context=mock_retriever | format_docs)
            | prompt
            | mock_llm
        )

        result = rag_chain.invoke({"question": "什么是 LangChain？"})
        print(f"模拟结果: {result}")


# ==========================================
# 示例 6: 高级 RAG 技术
# ==========================================


def example_advanced_rag():
    """演示高级 RAG 技术"""
    print("\n" + "=" * 50)
    print("示例 6: 高级 RAG 技术")
    print("=" * 50)

    # 1. 多查询扩展
    print("\n--- 多查询扩展 ---")
    print("""
多查询扩展的思路：
- 用户查询可能表述不完整
- 用 LLM 生成多个相关查询
- 合并所有查询的检索结果
- 提高召回率

示例代码结构：
```python
def generate_queries(question):
    # 使用 LLM 生成 3-5 个相关查询
    query_gen_prompt = ChatPromptTemplate.from_template(
        "生成 3 个与以下问题相关的搜索查询：{question}"
    )
    chain = query_gen_prompt | llm | StrOutputParser()
    queries = chain.invoke({"question": question})
    return queries.split('\n')

# 对每个查询检索，合并结果
all_docs = []
for query in queries:
    docs = retriever.invoke(query)
    all_docs.extend(docs)
# 去重
unique_docs = list(set(all_docs))
```
""")

    # 2. 重排序
    print("\n--- 重排序（Re-ranking） ---")
    print("""
重排序思路：
- 先用向量检索获取候选文档（数量较多）
- 使用 Cross-Encoder 模型重新计算相关性分数
- 选择最相关的文档

重排序优势：
- 向量检索速度快但精度有限
- Cross-Encoder 精度高但速度慢
- 结合两者优势

示例代码：
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 创建重排序检索器
compressor = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```
""")

    # 3. 混合检索
    print("\n--- 混合检索（Hybrid Search） ---")
    print("""
混合检索思路：
- 结合向量检索（语义相似）和关键词检索（精确匹配）
- BM25 + 向量检索
- 综合排序

适用场景：
- 用户查询包含特定术语或名称
- 需要精确匹配的场景

示例代码：
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 创建 BM25 检索器
bm25_retriever = BM25Retriever.from_documents(docs)

# 创建向量检索器
vector_retriever = vectorstore.as_retriever()

# 组合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```
""")

    # 4. 上下文窗口管理
    print("\n--- 上下文窗口管理 ---")
    print("""
问题：检索文档太多会超出 LLM 的 token 限制

解决方案：
1. 动态调整 k 值：根据文档长度调整返回数量
2. 文档压缩：使用 LLM 压缩文档内容
3. 选择性加载：只加载最相关的部分
4. 摘要代替：对长文档先摘要

示例代码：
```python
def adaptive_k(query, max_tokens=2000):
    docs = retriever.invoke(query)
    total_length = sum(len(d.page_content) for d in docs)
    # 根据总长度动态调整
    if total_length > max_tokens:
        # 按分数排序，截取
        docs = docs[:int(len(docs) * max_tokens / total_length)]
    return docs
```
""")

    # 5. 来源追溯
    print("\n--- 来源追溯 ---")
    print("""
RAG 应用应该提供来源信息，增强可信度。

实现方式：
- 在回答中引用文档来源
- 返回来源文档列表
- 提供原文链接

示例代码：
```python
prompt = ChatPromptTemplate.from_template(
    """基于以下参考信息回答问题。
请在回答中引用来源编号（如[来源1]）。

参考信息：
{context_with_sources}

问题：{question}"""
)

# 格式化带来源
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '未知')
        formatted.append(f"[来源{i+1}] {source}\n{doc.page_content}")
    return "\n\n".join(formatted)
```""")


# ==========================================
# 示例 7: RAG 评估
# ==========================================


def example_rag_evaluation():
    """演示 RAG 评估"""
    print("\n" + "=" * 50)
    print("示例 7: RAG 评估")
    print("=" * 50)

    print("\n--- RAG 评估指标 ---")

    # 1. 检索质量评估
    print("\n1. 检索质量评估")
    print("""
指标：
- Precision: 检索文档中相关文档的比例
- Recall: 所有相关文档中被检索到的比例
- MRR (Mean Reciprocal Rank): 第一个相关文档的位置排名
- Hit Rate: 查询中至少返回一个相关文档的比例

评估方法：
- 使用标注的问答对
- 计算检索结果与标准答案的匹配度
""")

    # 2. 生成质量评估
    print("\n2. 生成质量评估")
    print("""
指标：
- Faithfulness: 回答是否忠实于检索内容（无幻觉）
- Answer Relevance: 回答是否与问题相关
- Context Relevance: 检索内容是否与问题相关
- Groundedness: 回答是否有检索内容支撑

评估方法：
- LLM-as-Judge: 用 GPT-4 等模型评估
- 人工评估: 专家打分
- 自动化测试: 标准测试集
""")

    # 3. 评估框架
    print("\n3. 评估框架示例")
    print("""
Ragas 框架（推荐）：
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_relevance,
)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevance, context_relevance]
)
```

LangSmith 评估：
```python
from langsmith import Client

client = Client()
# 创建评估数据集
# 运行评估
# 查看结果
```
""")

    # 4. 持续改进
    print("\n4. 持续改进策略")
    print("""
发现问题后：
- 检索质量低：优化文档分割、调整检索参数、添加重排序
- 生成质量低：优化提示词、调整模型参数、添加约束
- 幻觉问题：添加来源引用、限制回答范围、验证机制
""")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 15: 完整 RAG 实现示例")
    print("=" * 60)

    example_document_loading()
    example_document_splitting()
    example_vector_store()
    example_retrieval_strategies()
    example_rag_chain()
    example_advanced_rag()
    example_rag_evaluation()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
    print("\n提示：")
    print("- 确保 API Key 配置正确（在 .env 文件中）")
    print("- 如无 OpenAI API Key，部分示例会使用本地嵌入模型")
    print("- RAG 需要向量数据库支持，Chroma 是推荐的入门选择")


if __name__ == "__main__":
    main()