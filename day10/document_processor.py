"""
文档处理模块

实现文档加载、切分等预处理功能。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import re
import os


@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理：确保 metadata 是字典"""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def content_length(self) -> int:
        """内容长度"""
        return len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建"""
        return cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {})
        )


class BaseLoader(ABC):
    """文档加载器基类"""
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        加载文档
        
        Args:
            source: 文档源（路径、URL等）
            
        Returns:
            文档列表
        """
        pass
    
    def load_batch(self, sources: List[str]) -> List[Document]:
        """批量加载文档"""
        documents = []
        for source in sources:
            documents.extend(self.load(source))
        return documents


class TextLoader(BaseLoader):
    """文本文件加载器"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, source: str) -> List[Document]:
        """加载文本文件"""
        try:
            with open(source, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            return [Document(
                content=content,
                metadata={
                    "source": source,
                    "filename": os.path.basename(source),
                    "type": "text"
                }
            )]
        except Exception as e:
            print(f"加载文件失败 {source}: {e}")
            return []


class MarkdownLoader(BaseLoader):
    """Markdown 文件加载器"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, source: str) -> List[Document]:
        """加载 Markdown 文件"""
        try:
            with open(source, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            # 提取标题作为元数据
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else os.path.basename(source)
            
            return [Document(
                content=content,
                metadata={
                    "source": source,
                    "filename": os.path.basename(source),
                    "type": "markdown",
                    "title": title
                }
            )]
        except Exception as e:
            print(f"加载文件失败 {source}: {e}")
            return []


class DirectoryLoader(BaseLoader):
    """目录加载器"""
    
    def __init__(
        self,
        loader: Optional[BaseLoader] = None,
        glob_pattern: str = "**/*.txt",
        recursive: bool = True
    ):
        """
        初始化目录加载器
        
        Args:
            loader: 文件加载器
            glob_pattern: 文件匹配模式
            recursive: 是否递归
        """
        self.loader = loader or TextLoader()
        self.glob_pattern = glob_pattern
        self.recursive = recursive
    
    def load(self, source: str) -> List[Document]:
        """加载目录中的所有文件"""
        import glob
        
        pattern = os.path.join(source, self.glob_pattern)
        files = glob.glob(pattern, recursive=self.recursive)
        
        documents = []
        for file_path in files:
            if os.path.isfile(file_path):
                documents.extend(self.loader.load(file_path))
        
        return documents


class DocumentLoader:
    """
    统一的文档加载器
    
    支持多种文件格式和加载方式。
    """
    
    def __init__(self):
        self._loaders: Dict[str, BaseLoader] = {
            ".txt": TextLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
        }
    
    def register_loader(self, extension: str, loader: BaseLoader):
        """注册加载器"""
        self._loaders[extension] = loader
    
    def load(self, source: str) -> List[Document]:
        """
        加载文档
        
        Args:
            source: 文件路径或目录
            
        Returns:
            文档列表
        """
        if os.path.isdir(source):
            return self._load_directory(source)
        else:
            return self._load_file(source)
    
    def _load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        ext = os.path.splitext(file_path)[1].lower()
        loader = self._loaders.get(ext, TextLoader())
        return loader.load(file_path)
    
    def _load_directory(self, dir_path: str) -> List[Document]:
        """加载目录"""
        loader = DirectoryLoader()
        return loader.load(dir_path)
    
    def load_from_text(self, text: str, metadata: Optional[Dict] = None) -> Document:
        """从文本创建文档"""
        return Document(content=text, metadata=metadata or {})
    
    def load_from_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """从文本列表创建文档"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(content=text, metadata=metadata))
        return documents


# ============ 文本切分器 ============

class BaseTextSplitter(ABC):
    """文本切分器基类"""
    
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """切分文本"""
        pass
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            切分后的文档列表
        """
        chunks = []
        
        for doc in documents:
            text_chunks = self.split(doc.content)
            
            for i, chunk in enumerate(text_chunks):
                # 继承原文档的元数据，添加切分信息
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(text_chunks)
                
                chunks.append(Document(
                    content=chunk,
                    metadata=metadata
                ))
        
        return chunks


class CharacterTextSplitter(BaseTextSplitter):
    """
    字符切分器
    
    按固定字符数切分，支持重叠。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        初始化
        
        Args:
            chunk_size: 每块大小
            chunk_overlap: 重叠大小
            separator: 分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split(self, text: str) -> List[str]:
        """切分文本"""
        # 先按分隔符分割
        if self.separator:
            splits = text.split(self.separator)
        else:
            splits = [text]
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # 如果当前块加上新分割不超过大小，则添加
            if len(current_chunk) + len(split) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + split
                else:
                    current_chunk = split
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果单个分割超过大小，需要进一步切分
                if len(split) > self.chunk_size:
                    # 按字符数切分
                    for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                        chunk = split[i:i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk.strip())
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # 添加最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    递归字符切分器
    
    按层级分隔符递归切分，优先保持语义完整性。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        初始化
        
        Args:
            chunk_size: 每块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表（按优先级排序）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分隔符（按优先级）
        self.separators = separators or [
            "\n\n",  # 段落
            "\n",    # 行
            "。",    # 句号
            "！",    # 感叹号
            "？",    # 问号
            "；",    # 分号
            "，",    # 逗号
            " ",     # 空格
            ""       # 字符
        ]
    
    def split(self, text: str) -> List[str]:
        """切分文本"""
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分文本"""
        if not text:
            return []
        
        # 找到合适的分隔符
        separator = separators[-1]
        for sep in separators:
            if sep in text:
                separator = sep
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = [text]
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # 去除空分割
            if not split:
                continue
            
            # 检查是否需要进一步切分
            if len(split) > self.chunk_size:
                # 如果还有更多分隔符，递归切分
                if len(separators) > 1:
                    sub_chunks = self._split_text(split, separators[1:])
                    chunks.extend(sub_chunks)
                else:
                    # 已经是最后一级，强制切分
                    for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                        chunk = split[i:i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk.strip())
                continue
            
            # 检查是否可以添加到当前块
            new_length = len(current_chunk) + len(split) + len(separator)
            
            if current_chunk and new_length <= self.chunk_size:
                current_chunk += separator + split
            elif not current_chunk:
                current_chunk = split
            else:
                # 保存当前块，开始新块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 添加重叠
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap + separator + split
                else:
                    current_chunk = split
        
        # 添加最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class SentenceTextSplitter(BaseTextSplitter):
    """
    句子切分器
    
    按句子切分，适合需要保持句子完整性的场景。
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        language: str = "zh"
    ):
        """
        初始化
        
        Args:
            chunk_size: 每块大小
            chunk_overlap: 重叠大小
            language: 语言（zh/en）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        
        # 句子分隔符
        if language == "zh":
            self.sentence_endings = ["。", "！", "？", "…"]
        else:
            self.sentence_endings = [".", "!", "?"]
    
    def split(self, text: str) -> List[str]:
        """切分文本"""
        # 分割句子
        sentences = self._split_sentences(text)
        
        # 合并为块
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in self.sentence_endings:
                sentences.append(current)
                current = ""
        
        if current.strip():
            sentences.append(current)
        
        return sentences


class TextSplitter:
    """
    统一的文本切分器
    
    提供便捷的切分接口。
    """
    
    @staticmethod
    def create_splitter(
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ) -> BaseTextSplitter:
        """
        创建切分器
        
        Args:
            strategy: 切分策略（character/recursive/sentence）
            chunk_size: 每块大小
            chunk_overlap: 重叠大小
            
        Returns:
            切分器实例
        """
        if strategy == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        elif strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        elif strategy == "sentence":
            return SentenceTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        else:
            raise ValueError(f"未知的切分策略: {strategy}")
    
    @staticmethod
    def split_text(
        text: str,
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[str]:
        """便捷方法：切分文本"""
        splitter = TextSplitter.create_splitter(strategy, chunk_size, chunk_overlap)
        return splitter.split(text)
    
    @staticmethod
    def split_documents(
        documents: List[Document],
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """便捷方法：切分文档"""
        splitter = TextSplitter.create_splitter(strategy, chunk_size, chunk_overlap)
        return splitter.split_documents(documents)


# ============ 演示 ============

def demo():
    """演示文档处理"""
    print("=" * 60)
    print("文档处理演示")
    print("=" * 60)
    
    # 创建示例文档
    sample_text = """
    # RAG 系统介绍
    
    RAG（Retrieval-Augmented Generation）是一种将检索与生成结合的技术。
    它通过检索相关文档来增强大模型的生成能力。
    
    ## 核心组件
    
    RAG 系统包含以下核心组件：
    
    1. 文档处理：加载和切分文档
    2. 嵌入模型：将文本转换为向量
    3. 向量数据库：存储和检索向量
    4. 检索策略：查找相关文档
    5. 生成模型：基于上下文生成答案
    
    ## 应用场景
    
    RAG 适用于以下场景：
    - 企业知识库问答
    - 技术文档助手
    - 客服机器人
    - 法律文档分析
    """
    
    # 创建文档
    loader = DocumentLoader()
    doc = loader.load_from_text(sample_text, {"source": "demo.md"})
    
    print(f"\n原始文档长度: {doc.content_length} 字符")
    
    # 测试不同切分策略
    strategies = ["character", "recursive", "sentence"]
    
    for strategy in strategies:
        print(f"\n--- {strategy} 切分 ---")
        chunks = TextSplitter.split_text(
            sample_text,
            strategy=strategy,
            chunk_size=200,
            chunk_overlap=30
        )
        
        print(f"切分数量: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
            print(f"\nChunk {i+1} ({len(chunk)} 字符):")
            print(f"  {chunk[:100]}...")
    
    # 测试文档切分
    print("\n" + "=" * 60)
    print("文档切分演示")
    print("=" * 60)
    
    splitter = TextSplitter.create_splitter("recursive", chunk_size=200)
    chunked_docs = splitter.split_documents([doc])
    
    print(f"\n切分后文档数量: {len(chunked_docs)}")
    for i, chunk_doc in enumerate(chunked_docs[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  内容长度: {chunk_doc.content_length}")
        print(f"  元数据: {chunk_doc.metadata}")


if __name__ == "__main__":
    demo()