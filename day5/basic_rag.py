"""
RAG 基础示例
学习检索增强生成的基本流程
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 知识库数据 ====================

# 模拟知识库文档
KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "content": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python以简洁、易读的语法著称，广泛应用于Web开发、数据分析、人工智能等领域。",
        "metadata": {"source": "python_intro.txt", "category": "编程语言"}
    },
    {
        "id": "doc2",
        "content": "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需显式编程。常见的机器学习算法包括线性回归、决策树、神经网络等。",
        "metadata": {"source": "ml_basics.txt", "category": "人工智能"}
    },
    {
        "id": "doc3",
        "content": "深度学习使用多层神经网络来学习数据的表示。它在图像识别、自然语言处理、语音识别等领域取得了突破性进展。常用的深度学习框架有TensorFlow和PyTorch。",
        "metadata": {"source": "deep_learning.txt", "category": "人工智能"}
    },
    {
        "id": "doc4",
        "content": "RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它首先从知识库中检索相关文档，然后利用这些文档作为上下文，让大语言模型生成更准确的回答。",
        "metadata": {"source": "rag_intro.txt", "category": "AI技术"}
    },
    {
        "id": "doc5",
        "content": "向量数据库是专门用于存储和检索向量嵌入的数据库。常见的向量数据库包括ChromaDB、Pinecone、Milvus等。它们支持高效的相似度搜索，是RAG系统的核心组件。",
        "metadata": {"source": "vector_db.txt", "category": "数据库"}
    },
    {
        "id": "doc6",
        "content": "OpenAI的GPT系列模型是大语言模型的代表。GPT-4是目前最先进的版本，具有强大的语言理解和生成能力，支持多模态输入，可以处理文本和图像。",
        "metadata": {"source": "gpt_intro.txt", "category": "AI模型"}
    },
    {
        "id": "doc7",
        "content": "自然语言处理（NLP）是人工智能的重要分支，研究计算机如何理解和处理人类语言。NLP应用包括机器翻译、情感分析、问答系统、文本摘要等。",
        "metadata": {"source": "nlp_intro.txt", "category": "人工智能"}
    },
    {
        "id": "doc8",
        "content": "LangChain是一个开源框架，用于构建基于大语言模型的应用程序。它提供了链式调用、提示模板、向量存储等组件，简化了AI应用的开发流程。",
        "metadata": {"source": "langchain.txt", "category": "AI框架"}
    }
]


# ==================== 基础 RAG 实现 ====================

class BasicRAG:
    """基础 RAG 系统实现"""

    def __init__(self, collection_name: str = "knowledge_base"):
        """
        初始化 RAG 系统

        Args:
            collection_name: 向量数据库集合名称
        """
        # 创建内存向量数据库
        self.db_client = chromadb.Client()

        # 使用默认的嵌入函数（sentence-transformers）
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # 创建集合
        self.collection = self.db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: list):
        """
        添加文档到知识库

        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
        """
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        print(f"已添加 {len(documents)} 篇文档到知识库")

    def search(self, query: str, top_k: int = 3) -> list:
        """
        语义搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回最相关的 k 个结果

        Returns:
            搜索结果列表
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })

        return formatted_results

    def generate_answer(self, question: str, context_docs: list) -> str:
        """
        基于检索到的文档生成回答

        Args:
            question: 用户问题
            context_docs: 检索到的相关文档

        Returns:
            生成的回答
        """
        # 构建上下文
        context = "\n\n".join([
            f"[文档{i+1}] {doc['content']}"
            for i, doc in enumerate(context_docs)
        ])

        # 构建 Prompt
        prompt = f"""你是一个知识问答助手。请根据以下参考信息回答用户的问题。
如果参考信息中没有相关内容，请说明"参考信息中没有找到相关内容"。

## 参考信息
{context}

## 用户问题
{question}

## 回答（请简洁准确地回答，可以引用文档编号）"""

        # 调用 LLM 生成回答
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的知识问答助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> str:
        """
        RAG 查询：检索 + 生成

        Args:
            question: 用户问题
            top_k: 检索的相关文档数量
            verbose: 是否打印详细信息

        Returns:
            生成的回答
        """
        if verbose:
            print(f"\n问题: {question}")
            print("-" * 40)

        # 步骤1：检索相关文档
        context_docs = self.search(question, top_k)

        if verbose:
            print(f"检索到 {len(context_docs)} 篇相关文档:")
            for i, doc in enumerate(context_docs):
                print(f"  [{i+1}] {doc['metadata'].get('source', 'unknown')} "
                      f"(相似度: {1 - (doc['distance'] or 0):.2f})")

        # 步骤2：生成回答
        answer = self.generate_answer(question, context_docs)

        if verbose:
            print("-" * 40)
            print(f"回答: {answer}")

        return answer


# ==================== 演示函数 ====================

def demo_basic_rag():
    """基础 RAG 演示"""
    print("\n" + "=" * 50)
    print("示例：基础 RAG 流程")
    print("=" * 50)

    # 1. 创建 RAG 系统
    rag = BasicRAG()

    # 2. 添加知识库文档
    print("\n[步骤1] 构建知识库...")
    rag.add_documents(KNOWLEDGE_BASE)

    # 3. 执行查询
    print("\n[步骤2] 执行 RAG 查询...")

    questions = [
        "什么是Python？",
        "RAG技术有什么用？",
        "深度学习和机器学习有什么关系？"
    ]

    for question in questions:
        rag.query(question)
        print()


def demo_semantic_search():
    """语义搜索演示"""
    print("\n" + "=" * 50)
    print("示例：语义搜索能力")
    print("=" * 50)

    rag = BasicRAG()
    rag.add_documents(KNOWLEDGE_BASE)

    # 测试不同表述的相同问题
    print("\n测试：不同表述，相同语义")
    print("-" * 40)

    queries = [
        "Python是谁发明的？",
        "Python的创始人是谁？",
        "Guido van Rossum创造了什么？"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        results = rag.search(query, top_k=1)
        print(f"最相关文档: {results[0]['metadata']['source']}")
        print(f"内容片段: {results[0]['content'][:50]}...")


def demo_compare_with_without_rag():
    """对比有无 RAG 的回答效果"""
    print("\n" + "=" * 50)
    print("示例：RAG vs 无 RAG 对比")
    print("=" * 50)

    # 问题：知识库中没有的内容
    question = "Claude和GPT-4哪个更强？"

    # 无 RAG 的回答
    print("\n【无 RAG】直接询问 LLM:")
    print("-" * 40)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": question}
        ]
    )
    print(f"回答: {response.choices[0].message.content}")

    # 有 RAG 的回答
    print("\n【有 RAG】基于知识库回答:")
    print("-" * 40)
    rag = BasicRAG()
    rag.add_documents(KNOWLEDGE_BASE)
    rag.query(question)

    print("\n说明: 当知识库中没有相关信息时，RAG 会明确告知，避免幻觉。")


def demo_interactive():
    """交互式问答"""
    print("\n" + "=" * 50)
    print("RAG 问答系统 - 交互模式")
    print("输入 'quit' 退出")
    print("=" * 50)

    # 初始化
    rag = BasicRAG()
    rag.add_documents(KNOWLEDGE_BASE)

    print("\n知识库已加载，包含以下主题:")
    categories = set(doc["metadata"]["category"] for doc in KNOWLEDGE_BASE)
    print(f"  {', '.join(categories)}")

    while True:
        user_input = input("\n你的问题: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break

        if not user_input:
            continue

        rag.query(user_input)


if __name__ == "__main__":
    print("=" * 50)
    print("RAG 基础示例")
    print("=" * 50)

    # 基础 RAG 演示
    demo_basic_rag()

    # 语义搜索演示
    demo_semantic_search()

    # RAG vs 无 RAG 对比
    demo_compare_with_without_rag()

    # 交互模式
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()