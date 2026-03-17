"""
文档处理模块
实现文本分块、文档加载、嵌入向量化
"""

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from chromadb.utils import embedding_functions


# ==================== 数据结构 ====================

@dataclass
class TextChunk:
    """文本块数据结构"""
    id: str
    content: str
    metadata: dict
    start_index: int = 0
    end_index: int = 0


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    source: str
    metadata: dict = None


# ==================== 文本分块器 ====================

class TextSplitter:
    """
    文本分块器
    支持多种分块策略
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        初始化分块器

        Args:
            chunk_size: 每块最大字符数
            chunk_overlap: 相邻块重叠字符数
            separator: 分隔符（优先按分隔符分割）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """
        分割文本

        Args:
            text: 原始文本

        Returns:
            分割后的文本块列表
        """
        if not text:
            return []

        # 首先按分隔符分割
        if self.separator in text:
            splits = text.split(self.separator)
        else:
            splits = [text]

        # 合并小块，切分大块
        chunks = []
        current_chunk = ""

        for split in splits:
            # 如果当前块+新分割不超过限制，则合并
            if len(current_chunk) + len(split) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + split
                else:
                    current_chunk = split
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 处理超大块
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_large_text(split)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # 添加重叠内容
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap + self.separator + split
                    else:
                        current_chunk = split

        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """
        处理超大文本块

        Args:
            text: 超大文本

        Returns:
            分割后的文本块
        """
        chunks = []

        # 尝试按句子分割
        sentences = re.split(r'([。！？.!?])', text)
        sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]

        if len(sentences) <= 1:
            # 按字符强制分割
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                chunks.append(chunk)
        else:
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk)

        return chunks


class RecursiveTextSplitter:
    """
    递归文本分块器
    按优先级尝试不同分隔符
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        初始化递归分块器

        Args:
            chunk_size: 每块最大字符数
            chunk_overlap: 相邻块重叠字符数
            separators: 分隔符优先级列表（从高到低）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 默认分隔符优先级
        self.separators = separators or [
            "\n\n",    # 段落
            "\n",      # 换行
            "。",      # 中文句号
            ".",       # 英文句号
            " ",       # 空格
            ""         # 字符
        ]

    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归分割实现"""
        if not text:
            return []

        # 如果文本足够小，直接返回
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # 尝试按分隔符分割
        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                chunks = []
                current_chunk = ""

                for split in splits:
                    if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                        if current_chunk:
                            current_chunk += separator + split
                        else:
                            current_chunk = split
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        # 递归处理大块
                        if len(split) > self.chunk_size:
                            sub_chunks = self._split_text_recursive(
                                split,
                                separators[separators.index(separator)+1:]
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split

                if current_chunk:
                    chunks.append(current_chunk.strip())

                return self._merge_small_chunks(chunks)

        # 无法分割，强制切分
        return self._force_split(text)

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的块"""
        if not chunks:
            return []

        merged = []
        current = ""

        for chunk in chunks:
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                if current:
                    current += " " + chunk
                else:
                    current = chunk
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        return merged

    def _force_split(self, text: str) -> List[str]:
        """强制按字符分割"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


# ==================== 文档处理器 ====================

class DocumentProcessor:
    """
    文档处理器
    负责文档加载、分块、向量化
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_recursive: bool = True
    ):
        """
        初始化文档处理器

        Args:
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            use_recursive: 是否使用递归分块器
        """
        if use_recursive:
            self.splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)
        else:
            self.splitter = TextSplitter(chunk_size, chunk_overlap)

        # 嵌入函数
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def process_text(
        self,
        text: str,
        doc_id: str = None,
        metadata: dict = None
    ) -> List[TextChunk]:
        """
        处理文本

        Args:
            text: 原始文本
            doc_id: 文档ID
            metadata: 元数据

        Returns:
            文本块列表
        """
        if doc_id is None:
            doc_id = f"doc_{id(text)}"

        # 分块
        chunks = self.splitter.split_text(text)

        # 创建 TextChunk 对象
        text_chunks = []
        current_index = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            start_idx = text.find(chunk, current_index)
            end_idx = start_idx + len(chunk)

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["doc_id"] = doc_id
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            text_chunks.append(TextChunk(
                id=chunk_id,
                content=chunk,
                metadata=chunk_metadata,
                start_index=start_idx,
                end_index=end_idx
            ))

            current_index = end_idx

        return text_chunks

    def process_document(self, document: Document) -> List[TextChunk]:
        """
        处理文档对象

        Args:
            document: 文档对象

        Returns:
            文本块列表
        """
        return self.process_text(
            text=document.content,
            doc_id=document.id,
            metadata=document.metadata
        )

    def process_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        批量处理文档

        Args:
            documents: 文档列表

        Returns:
            所有文档的文本块列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        embeddings = self.embedding_function(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings


# ==================== 文档加载器 ====================

class DocumentLoader:
    """
    文档加载器
    支持从文件、目录加载文档
    """

    @staticmethod
    def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
        """
        加载文本文件

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            Document 对象
        """
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()

        filename = os.path.basename(file_path)
        doc_id = os.path.splitext(filename)[0]

        return Document(
            id=doc_id,
            content=content,
            source=file_path,
            metadata={"filename": filename, "type": "txt"}
        )

    @staticmethod
    def load_markdown_file(file_path: str, encoding: str = "utf-8") -> Document:
        """
        加载 Markdown 文件

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            Document 对象
        """
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()

        filename = os.path.basename(file_path)
        doc_id = os.path.splitext(filename)[0]

        return Document(
            id=doc_id,
            content=content,
            source=file_path,
            metadata={"filename": filename, "type": "md"}
        )

    @staticmethod
    def load_directory(
        directory: str,
        extensions: List[str] = None,
        encoding: str = "utf-8"
    ) -> List[Document]:
        """
        加载目录下所有文档

        Args:
            directory: 目录路径
            extensions: 文件扩展名列表
            encoding: 文件编码

        Returns:
            Document 列表
        """
        if extensions is None:
            extensions = ['.txt', '.md']

        documents = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        if ext == '.md':
                            doc = DocumentLoader.load_markdown_file(file_path, encoding)
                        else:
                            doc = DocumentLoader.load_text_file(file_path, encoding)
                        documents.append(doc)
                    except Exception as e:
                        print(f"加载文件失败 {file_path}: {e}")

        return documents


# ==================== 演示函数 ====================

def demo_text_splitter():
    """文本分块演示"""
    print("\n" + "=" * 50)
    print("示例：文本分块")
    print("=" * 50)

    sample_text = """
    Python是一种广泛使用的高级编程语言。它由Guido van Rossum于1991年首次发布。

    Python的设计哲学强调代码的可读性和简洁性。它的语法允许程序员用更少的代码行表达概念。

    Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。它具有自动内存管理和大型标准库。

    Python常用于Web开发、数据分析、人工智能、科学计算等领域。流行的框架包括Django、Flask、NumPy、Pandas等。
    """

    # 基础分块器
    print("\n[基础分块器] chunk_size=100, overlap=20")
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"  块{i+1}: {chunk[:50]}...")

    # 递归分块器
    print("\n[递归分块器] chunk_size=100, overlap=20")
    recursive_splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = recursive_splitter.split_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"  块{i+1}: {chunk[:50]}...")


def demo_document_processor():
    """文档处理演示"""
    print("\n" + "=" * 50)
    print("示例：文档处理")
    print("=" * 50)

    # 创建处理器
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)

    # 处理文本
    text = """
    人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。

    机器学习是AI的核心技术之一，它使计算机能够从数据中学习而无需显式编程。

    深度学习使用多层神经网络，在图像识别、自然语言处理等领域取得了突破。
    """

    print("\n处理文本...")
    chunks = processor.process_text(text, doc_id="ai_intro", metadata={"topic": "AI"})

    print(f"生成 {len(chunks)} 个文本块:")
    for chunk in chunks:
        print(f"  ID: {chunk.id}")
        print(f"  内容: {chunk.content[:50]}...")
        print(f"  元数据: {chunk.metadata}")
        print()


def demo_chunk_strategies():
    """不同分块策略对比"""
    print("\n" + "=" * 50)
    print("示例：分块策略对比")
    print("=" * 50)

    text = """
    RAG系统需要合理的文档分块策略。块太小会导致语义不完整，块太大会降低检索精度。

    常见的分块策略包括：
    1. 固定大小分块：简单直接，但可能切断语义
    2. 句子分块：保持句子完整，语义更连贯
    3. 段落分块：保持段落完整，上下文更丰富
    4. 递归分块：按优先级尝试不同分隔符
    """

    strategies = [
        ("小分块", 50, 10),
        ("中等分块", 100, 20),
        ("大分块", 200, 40)
    ]

    for name, size, overlap in strategies:
        print(f"\n[{name}] chunk_size={size}, overlap={overlap}")
        splitter = RecursiveTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        print(f"  生成 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks[:2]):  # 只显示前2个
            print(f"  块{i+1}: {chunk[:40]}...")


def demo_embedding_generation():
    """嵌入向量化演示"""
    print("\n" + "=" * 50)
    print("示例：嵌入向量化")
    print("=" * 50)

    processor = DocumentProcessor()

    texts = [
        "Python是一种编程语言",
        "Java也是一种编程语言",
        "苹果是一种水果"
    ]

    print("\n生成嵌入向量...")
    embeddings = processor.generate_embeddings(texts)

    print(f"向量维度: {len(embeddings[0])}")
    print(f"向量数量: {len(embeddings)}")

    # 显示向量前几维
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"\n文本: {text}")
        print(f"向量[:5]: {emb[:5]}")


def demo_full_pipeline():
    """完整处理流程演示"""
    print("\n" + "=" * 50)
    print("示例：完整处理流程")
    print("=" * 50)

    # 1. 创建文档
    documents = [
        Document(
            id="doc1",
            content="Python是一种流行的编程语言，广泛应用于数据科学和Web开发。它具有简洁的语法和丰富的库。",
            source="intro.txt",
            metadata={"category": "编程"}
        ),
        Document(
            id="doc2",
            content="机器学习是AI的重要分支，通过数据训练模型来实现预测和分类任务。常用算法包括决策树、神经网络等。",
            source="ml.txt",
            metadata={"category": "AI"}
        )
    ]

    # 2. 创建处理器
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # 3. 处理文档
    print("\n[步骤1] 处理文档...")
    all_chunks = processor.process_documents(documents)
    print(f"生成 {len(all_chunks)} 个文本块")

    # 4. 生成嵌入
    print("\n[步骤2] 生成嵌入向量...")
    texts = [chunk.content for chunk in all_chunks]
    embeddings = processor.generate_embeddings(texts)
    print(f"生成了 {len(embeddings)} 个向量")

    # 5. 输出结果
    print("\n[结果] 文本块信息:")
    for chunk in all_chunks:
        print(f"  {chunk.id}: {chunk.content[:30]}...")


if __name__ == "__main__":
    print("=" * 50)
    print("文档处理模块演示")
    print("=" * 50)

    # 文本分块演示
    demo_text_splitter()

    # 文档处理演示
    demo_document_processor()

    # 分块策略对比
    demo_chunk_strategies()

    # 嵌入向量化演示
    demo_embedding_generation()

    # 完整流程演示
    demo_full_pipeline()

    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)