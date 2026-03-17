"""
完整 RAG 聊天机器人
实现知识库管理、智能检索、多轮对话、来源引用
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
from dataclasses import dataclass

# 导入文档处理模块
from document_processor import DocumentProcessor, Document, DocumentLoader

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 配置 ====================

@dataclass
class RAGConfig:
    """RAG 系统配置"""
    collection_name: str = "rag_knowledge"
    persist_directory: str = "./chroma_db"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
    max_history: int = 10


# ==================== RAG 聊天机器人 ====================

class RAGChatbot:
    """
    完整的 RAG 聊天机器人
    支持知识库管理、智能检索、多轮对话
    """

    def __init__(self, config: RAGConfig = None):
        """
        初始化 RAG 聊天机器人

        Args:
            config: 配置对象
        """
        self.config = config or RAGConfig()

        # 初始化向量数据库
        if self.config.persist_directory:
            self.db_client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.db_client = chromadb.Client()

        # 嵌入函数
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # 获取或创建集合
        self.collection = self.db_client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=self.embedding_function
        )

        # 文档处理器
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # 对话历史
        self.conversation_history: List[Dict] = []

        # 系统提示词
        self.system_prompt = """你是一个专业的知识问答助手。请根据提供的参考信息回答用户的问题。

## 回答要求
1. 优先使用参考信息中的内容回答
2. 如果参考信息不足或不相关，请明确说明
3. 在回答中引用信息来源（如：根据文档[1]...）
4. 保持回答简洁准确
5. 如果问题超出参考信息范围，可以基于你的知识回答，但要说明这部分不在知识库中"""

    # ==================== 知识库管理 ====================

    def add_text(self, text: str, doc_id: str = None, metadata: dict = None) -> int:
        """
        添加文本到知识库

        Args:
            text: 文本内容
            doc_id: 文档ID（可选）
            metadata: 元数据（可选）

        Returns:
            添加的文档块数量
        """
        # 处理文档
        chunks = self.doc_processor.process_text(text, doc_id, metadata)

        # 添加到向量数据库
        for chunk in chunks:
            self.collection.upsert(
                ids=[chunk.id],
                documents=[chunk.content],
                metadatas=[chunk.metadata]
            )

        print(f"已添加 {len(chunks)} 个文档块到知识库")
        return len(chunks)

    def add_document(self, document: Document) -> int:
        """
        添加文档对象到知识库

        Args:
            document: Document 对象

        Returns:
            添加的文档块数量
        """
        chunks = self.doc_processor.process_document(document)

        for chunk in chunks:
            self.collection.upsert(
                ids=[chunk.id],
                documents=[chunk.content],
                metadatas=[chunk.metadata]
            )

        print(f"已添加文档 '{document.id}' 的 {len(chunks)} 个块")
        return len(chunks)

    def add_file(self, file_path: str, metadata: dict = None) -> int:
        """
        从文件添加到知识库

        Args:
            file_path: 文件路径
            metadata: 额外元数据

        Returns:
            添加的文档块数量
        """
        # 加载文件
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.md':
            document = DocumentLoader.load_markdown_file(file_path)
        else:
            document = DocumentLoader.load_text_file(file_path)

        # 合并元数据
        if metadata:
            document.metadata.update(metadata)

        return self.add_document(document)

    def add_directory(self, directory: str, extensions: List[str] = None) -> int:
        """
        从目录添加所有文档

        Args:
            directory: 目录路径
            extensions: 文件扩展名列表

        Returns:
            添加的总块数
        """
        documents = DocumentLoader.load_directory(directory, extensions)

        total_chunks = 0
        for doc in documents:
            chunks = self.add_document(doc)
            total_chunks += chunks

        print(f"从目录添加了 {len(documents)} 个文档，共 {total_chunks} 个块")
        return total_chunks

    def remove_document(self, doc_id: str):
        """
        从知识库删除文档

        Args:
            doc_id: 文档ID
        """
        # 获取该文档的所有块
        results = self.collection.get(
            where={"doc_id": doc_id}
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"已删除文档 '{doc_id}' 的 {len(results['ids'])} 个块")
        else:
            print(f"未找到文档 '{doc_id}'")

    def clear_knowledge_base(self):
        """清空知识库"""
        all_docs = self.collection.get()
        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
            print(f"已清空知识库，删除 {len(all_docs['ids'])} 个块")
        else:
            print("知识库已经是空的")

    def get_knowledge_base_info(self) -> dict:
        """获取知识库信息"""
        return {
            "name": self.collection.name,
            "document_count": self.collection.count(),
            "persist_directory": self.config.persist_directory
        }

    # ==================== 检索功能 ====================

    def search(self, query: str, top_k: int = None, metadata_filter: dict = None) -> List[dict]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量
            metadata_filter: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        top_k = top_k or self.config.top_k

        if metadata_filter:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=metadata_filter
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

        # 格式化结果
        documents = []
        for i in range(len(results["ids"][0])):
            doc = {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if "distances" in results else 0
            }
            documents.append(doc)

        return documents

    # ==================== 对话功能 ====================

    def chat(self, user_message: str, show_sources: bool = True) -> str:
        """
        与聊天机器人对话

        Args:
            user_message: 用户消息
            show_sources: 是否显示来源

        Returns:
            助手回复
        """
        # 检索相关文档
        relevant_docs = self.search(user_message)

        # 构建上下文
        context = self._build_context(relevant_docs)

        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 添加对话历史
        messages.extend(self.conversation_history[-self.config.max_history:])

        # 添加当前问题
        user_prompt = f"""## 参考信息
{context}

## 用户问题
{user_message}"""

        messages.append({"role": "user", "content": user_prompt})

        # 调用 LLM
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # 更新对话历史
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # 显示来源
        if show_sources and relevant_docs:
            sources = self._format_sources(relevant_docs)
            assistant_message += f"\n\n---\n**来源引用:**\n{sources}"

        return assistant_message

    def _build_context(self, documents: List[dict]) -> str:
        """构建上下文"""
        if not documents:
            return "（未找到相关参考信息）"

        context_parts = []
        for i, doc in enumerate(documents):
            source = doc["metadata"].get("source", "未知来源")
            context_parts.append(f"[文档{i+1}] 来源: {source}\n{doc['content']}")

        return "\n\n".join(context_parts)

    def _format_sources(self, documents: List[dict]) -> str:
        """格式化来源引用"""
        sources = []
        seen_sources = set()

        for i, doc in enumerate(documents):
            source = doc["metadata"].get("source", "未知来源")
            if source not in seen_sources:
                similarity = 1 - doc["distance"]
                sources.append(f"- [{i+1}] {source} (相关度: {similarity:.2f})")
                seen_sources.add(source)

        return "\n".join(sources)

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        print("对话历史已重置")

    # ==================== 高级功能 ====================

    def chat_with_suggestions(self, user_message: str) -> dict:
        """
        对话并提供建议问题

        Args:
            user_message: 用户消息

        Returns:
            包含回复和建议的字典
        """
        # 获取回答
        answer = self.chat(user_message, show_sources=True)

        # 生成建议问题
        suggestion_prompt = f"""基于以下对话，生成3个用户可能感兴趣的后续问题。

用户问题: {user_message}
助手回答: {answer[:500]}...

请生成3个相关的后续问题，每个问题一行，不要编号。"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个问题生成助手。"},
                {"role": "user", "content": suggestion_prompt}
            ],
            max_tokens=200
        )

        suggestions = response.choices[0].message.content.strip().split("\n")
        suggestions = [s.strip() for s in suggestions if s.strip()]

        return {
            "answer": answer,
            "suggestions": suggestions[:3]
        }

    def export_conversation(self, file_path: str):
        """
        导出对话历史

        Args:
            file_path: 导出文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for msg in self.conversation_history:
                role = "用户" if msg["role"] == "user" else "助手"
                f.write(f"【{role}】\n{msg['content']}\n\n")

        print(f"对话历史已导出到: {file_path}")


# ==================== 演示函数 ====================

def demo_basic_usage():
    """基础用法演示"""
    print("\n" + "=" * 50)
    print("示例：基础用法")
    print("=" * 50)

    # 创建聊天机器人
    config = RAGConfig(persist_directory=None)  # 内存模式
    chatbot = RAGChatbot(config)

    # 添加知识
    print("\n[步骤1] 添加知识到知识库...")
    chatbot.add_text(
        "Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python语法简洁，易于学习。",
        doc_id="python_intro",
        metadata={"topic": "编程语言"}
    )

    chatbot.add_text(
        "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。常见应用包括图像识别、自然语言处理等。",
        doc_id="ml_intro",
        metadata={"topic": "人工智能"}
    )

    # 获取知识库信息
    info = chatbot.get_knowledge_base_info()
    print(f"知识库: {info['name']}, 文档数: {info['document_count']}")

    # 问答
    print("\n[步骤2] 开始问答...")
    questions = [
        "Python是什么？",
        "机器学习有什么应用？"
    ]

    for question in questions:
        print(f"\n用户: {question}")
        answer = chatbot.chat(question)
        print(f"助手: {answer}")


def demo_multi_turn_conversation():
    """多轮对话演示"""
    print("\n" + "=" * 50)
    print("示例：多轮对话")
    print("=" * 50)

    chatbot = RAGChatbot(RAGConfig(persist_directory=None))

    # 添加知识
    chatbot.add_text(
        "深度学习使用多层神经网络来学习数据的表示。主要框架包括TensorFlow和PyTorch。"
        "深度学习在图像识别、语音识别、自然语言处理等领域取得了巨大成功。",
        doc_id="dl_intro"
    )

    # 多轮对话
    conversation = [
        "什么是深度学习？",
        "它有哪些应用？",
        "常用的框架有哪些？"
    ]

    for question in conversation:
        print(f"\n用户: {question}")
        answer = chatbot.chat(question, show_sources=False)
        print(f"助手: {answer}")


def demo_knowledge_management():
    """知识库管理演示"""
    print("\n" + "=" * 50)
    print("示例：知识库管理")
    print("=" * 50)

    chatbot = RAGChatbot(RAGConfig(persist_directory=None))

    # 添加多个文档
    print("\n[添加文档]")
    chatbot.add_text("文档1: Java是一种面向对象的编程语言。", doc_id="java")
    chatbot.add_text("文档2: Go语言由Google开发，适合并发编程。", doc_id="go")
    chatbot.add_text("文档3: Rust注重安全性和性能。", doc_id="rust")

    print(f"\n当前文档数: {chatbot.get_knowledge_base_info()['document_count']}")

    # 搜索
    print("\n[搜索文档]")
    results = chatbot.search("编程语言")
    for r in results:
        print(f"  - {r['content'][:30]}...")

    # 删除文档
    print("\n[删除文档 'java']")
    chatbot.remove_document("java")

    print(f"删除后文档数: {chatbot.get_knowledge_base_info()['document_count']}")


def demo_with_suggestions():
    """带建议问题的对话演示"""
    print("\n" + "=" * 50)
    print("示例：智能建议")
    print("=" * 50)

    chatbot = RAGChatbot(RAGConfig(persist_directory=None))

    chatbot.add_text(
        "RAG（检索增强生成）是一种AI技术，结合了信息检索和文本生成。"
        "它能让AI基于特定知识库回答问题，减少幻觉，提高准确性。",
        doc_id="rag"
    )

    result = chatbot.chat_with_suggestions("什么是RAG？")

    print(f"\n回答:\n{result['answer']}")
    print(f"\n建议问题:")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"  {i}. {suggestion}")


def demo_interactive():
    """交互式聊天"""
    print("\n" + "=" * 50)
    print("RAG 聊天机器人 - 交互模式")
    print("=" * 50)
    print("命令:")
    print("  /info  - 显示知识库信息")
    print("  /reset - 重置对话")
    print("  /export - 导出对话")
    print("  /quit  - 退出")
    print("=" * 50)

    # 创建聊天机器人
    config = RAGConfig(persist_directory="./chroma_db")  # 持久化存储
    chatbot = RAGChatbot(config)

    # 添加默认知识
    default_knowledge = [
        ("Python是一种高级编程语言，由Guido van Rossum创建。它以简洁的语法和丰富的库著称。", "python"),
        ("机器学习是AI的核心技术，让计算机从数据中学习模式。", "ml"),
        ("深度学习使用神经网络，在视觉和语言任务上表现出色。", "dl"),
    ]

    for text, doc_id in default_knowledge:
        chatbot.add_text(text, doc_id)

    print(f"\n知识库已加载，文档数: {chatbot.get_knowledge_base_info()['document_count']}")

    while True:
        user_input = input("\n你: ").strip()

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith('/'):
            cmd = user_input.lower()
            if cmd == '/quit':
                print("再见！")
                break
            elif cmd == '/info':
                info = chatbot.get_knowledge_base_info()
                for k, v in info.items():
                    print(f"  {k}: {v}")
            elif cmd == '/reset':
                chatbot.reset_conversation()
            elif cmd == '/export':
                chatbot.export_conversation("conversation_export.txt")
            else:
                print("未知命令")
            continue

        # 普通对话
        response = chatbot.chat(user_input)
        print(f"\n助手: {response}")


if __name__ == "__main__":
    print("=" * 50)
    print("RAG 聊天机器人演示")
    print("=" * 50)

    # 基础用法
    demo_basic_usage()

    # 多轮对话
    demo_multi_turn_conversation()

    # 知识库管理
    demo_knowledge_management()

    # 智能建议
    demo_with_suggestions()

    # 交互模式
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()