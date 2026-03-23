"""
RAG 引擎模块

实现完整的检索增强生成流程。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import time

# 导入本地模块
try:
    from document_processor import Document, TextSplitter
    from embedding_utils import EmbeddingModel
    from vector_store import VectorStore, VectorDocument, SearchResult
except ImportError:
    from document_processor import Document, TextSplitter
    from embedding_utils import EmbeddingModel
    from vector_store import VectorStore, VectorDocument, SearchResult


@dataclass
class RAGConfig:
    """RAG 配置"""
    # 嵌入配置
    embedding_model: str = "mock"
    embedding_dimensions: int = 384
    
    # 切分配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    split_strategy: str = "recursive"
    
    # 检索配置
    top_k: int = 5
    min_score: float = 0.0
    
    # 存储配置
    vector_store: str = "simple"
    persist_directory: Optional[str] = None
    
    # 生成配置
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # 提示词模板
    system_prompt: str = """你是一个有帮助的助手。请基于以下上下文回答用户的问题。
如果上下文中没有相关信息，请诚实地说你不知道，不要编造答案。
回答时请注明引用的来源。"""

    context_prompt: str = """上下文：
{context}

问题：{question}

请基于以上上下文回答问题："""


@dataclass
class RAGResponse:
    """RAG 响应"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "scores": self.scores,
            "metadata": self.metadata
        }


class RAGEngine:
    """
    RAG 引擎
    
    实现完整的检索增强生成流程。
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化
        
        Args:
            config: RAG 配置
        """
        self.config = config or RAGConfig()
        
        # 初始化组件
        self._embedding = EmbeddingModel(
            model_type=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions
        )
        
        self._vector_store = VectorStore.create(
            store_type=self.config.vector_store,
            persist_directory=self.config.persist_directory
        )
        
        self._splitter = TextSplitter.create_splitter(
            strategy=self.config.split_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def index(self, documents: List[Union[Document, Dict]]) -> int:
        """
        索引文档
        
        Args:
            documents: 文档列表
            
        Returns:
            索引的文档块数量
        """
        # 转换格式
        processed_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                doc = Document(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
            processed_docs.append(doc)
        
        # 切分文档
        chunks = self._splitter.split_documents(processed_docs)
        
        # 生成嵌入
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embedding.embed_batch(texts)
        
        # 创建向量文档
        vector_docs = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vector_docs.append(VectorDocument(
                id=f"chunk_{i}_{hash(chunk.content) % 10000}",
                content=chunk.content,
                embedding=emb,
                metadata=chunk.metadata
            ))
        
        # 添加到向量存储
        self._vector_store.add(vector_docs)
        
        return len(vector_docs)
    
    def index_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        """
        索引文本列表
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            索引的文档块数量
        """
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(content=text, metadata=metadata))
        
        return self.index(documents)
    
    def index_file(self, file_path: str) -> int:
        """
        索引文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            索引的文档块数量
        """
        from document_processor import DocumentLoader
        loader = DocumentLoader()
        documents = loader.load(file_path)
        return self.index(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter: 过滤条件
            
        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.top_k
        
        # 查询嵌入
        query_embedding = self._embedding.embed(query)
        
        # 检索
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter
        )
        
        # 过滤低分结果
        if self.config.min_score > 0:
            results = [r for r in results if r.score >= self.config.min_score]
        
        return results
    
    def build_context(self, results: List[SearchResult]) -> str:
        """
        构建上下文
        
        Args:
            results: 检索结果
            
        Returns:
            上下文字符串
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source = result.document.metadata.get("source", "未知来源")
            context_parts.append(f"[{i}] 来源: {source}\n{result.document.content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        context: str
    ) -> str:
        """
        生成答案
        
        Args:
            query: 用户查询
            context: 上下文
            
        Returns:
            生成的答案
        """
        # 构建提示词
        prompt = self.config.context_prompt.format(
            context=context,
            question=query
        )
        
        # 模拟 LLM 响应（实际应调用 LLM API）
        return self._simulate_llm_response(query, context)
    
    def _simulate_llm_response(self, query: str, context: str) -> str:
        """
        模拟 LLM 响应
        
        实际应用中应调用真实的 LLM API。
        """
        # 简单的模拟响应
        if context:
            return f"""根据检索到的信息，我来回答您的问题：

{context[:500]}...

以上是相关信息的摘要。如果您需要更详细的解答，请告诉我具体的问题。

注意：这是一个模拟响应。实际使用时请配置真实的 LLM API。"""
        else:
            return "抱歉，我没有找到相关信息来回答您的问题。"
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> RAGResponse:
        """
        查询并生成答案
        
        Args:
            question: 问题
            top_k: 检索数量
            filter: 过滤条件
            
        Returns:
            RAG 响应
        """
        start_time = time.time()
        
        # 检索
        results = self.retrieve(question, top_k, filter)
        
        # 构建上下文
        context = self.build_context(results)
        
        # 生成答案
        answer = self.generate(question, context)
        
        # 构建响应
        sources = [
            {
                "content": r.document.content,
                "score": r.score,
                "metadata": r.document.metadata
            }
            for r in results
        ]
        
        elapsed = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            scores=[r.score for r in results],
            metadata={
                "elapsed_time": elapsed,
                "num_sources": len(results)
            }
        )
    
    def query_with_sources(
        self,
        question: str,
        top_k: Optional[int] = None
    ) -> tuple:
        """
        查询并返回答案和来源
        
        Args:
            question: 问题
            top_k: 检索数量
            
        Returns:
            (答案, 来源列表)
        """
        response = self.query(question, top_k)
        return response.answer, response.sources
    
    @property
    def document_count(self) -> int:
        """文档数量"""
        return self._vector_store.count()
    
    def clear(self):
        """清空索引"""
        if hasattr(self._vector_store, 'clear'):
            self._vector_store.clear()


class SimpleRAG:
    """
    简单 RAG 接口
    
    提供更简洁的使用方式。
    """
    
    def __init__(
        self,
        embedding_model: str = "mock",
        chunk_size: int = 500
    ):
        """
        初始化
        
        Args:
            embedding_model: 嵌入模型类型
            chunk_size: 文档块大小
        """
        config = RAGConfig(
            embedding_model=embedding_model,
            chunk_size=chunk_size
        )
        self._engine = RAGEngine(config)
    
    def add(self, texts: List[str]):
        """添加文本"""
        self._engine.index_texts(texts)
    
    def ask(self, question: str) -> str:
        """提问"""
        response = self._engine.query(question)
        return response.answer
    
    def ask_with_sources(self, question: str) -> tuple:
        """提问并返回来源"""
        return self._engine.query_with_sources(question)


# ============ 演示 ============

def demo():
    """演示 RAG 引擎"""
    print("=" * 60)
    print("RAG 引擎演示")
    print("=" * 60)
    
    # 创建 RAG 引擎
    rag = RAGEngine()
    
    # 示例文档
    documents = [
        Document(
            content="""
            # 向量数据库介绍
            
            向量数据库是一种专门用于存储和检索高维向量的数据库系统。
            它支持高效的相似度搜索，广泛应用于推荐系统、语义搜索、RAG 等场景。
            
            常见的向量数据库包括：Chroma、Pinecone、Milvus、Weaviate 等。
            选择向量数据库时需要考虑：性能、扩展性、易用性、成本等因素。
            """,
            metadata={"source": "vector_db.md", "topic": "database"}
        ),
        Document(
            content="""
            # RAG 技术概述
            
            RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
            
            RAG 的工作流程：
            1. 用户提出问题
            2. 系统检索相关文档
            3. 将文档作为上下文
            4. LLM 基于上下文生成答案
            
            RAG 的优势：知识可更新、减少幻觉、可引用来源、数据隐私。
            """,
            metadata={"source": "rag_intro.md", "topic": "rag"}
        ),
        Document(
            content="""
            # 嵌入模型
            
            嵌入模型将文本转换为数值向量（Embedding）。
            好的嵌入模型能让语义相似的文本在向量空间中距离更近。
            
            常用嵌入模型：
            - OpenAI: text-embedding-3-small/large
            - 开源: BGE, M3E, E5, Sentence-BERT
            
            选择嵌入模型时需要考虑：语言支持、维度大小、性能、成本。
            """,
            metadata={"source": "embedding.md", "topic": "embedding"}
        )
    ]
    
    # 索引文档
    print("\n--- 索引文档 ---")
    num_chunks = rag.index(documents)
    print(f"索引了 {num_chunks} 个文档块")
    print(f"向量存储中共有 {rag.document_count} 个文档")
    
    # 查询
    print("\n--- 查询测试 ---")
    
    queries = [
        "什么是向量数据库？",
        "RAG 技术有什么优势？",
        "如何选择嵌入模型？"
    ]
    
    for query in queries:
        print(f"\n问题: {query}")
        print("-" * 40)
        
        response = rag.query(query)
        
        print(f"答案:\n{response.answer[:300]}...")
        print(f"\n来源数量: {len(response.sources)}")
        print(f"检索耗时: {response.metadata['elapsed_time']:.3f}s")
        
        if response.sources:
            print("\n来源:")
            for i, source in enumerate(response.sources[:2], 1):
                print(f"  [{i}] 相似度: {source['score']:.4f}")
                print(f"      文件: {source['metadata'].get('source', '未知')}")
    
    # 简单接口演示
    print("\n" + "=" * 60)
    print("简单 RAG 接口演示")
    print("=" * 60)
    
    simple_rag = SimpleRAG()
    
    # 添加文本
    simple_rag.add([
        "Python 是一种流行的编程语言，易于学习和使用。",
        "机器学习是人工智能的一个重要分支。",
        "深度学习使用神经网络来学习数据表示。"
    ])
    
    # 提问
    question = "Python 是什么？"
    print(f"\n问题: {question}")
    answer = simple_rag.ask(question)
    print(f"答案: {answer}")


if __name__ == "__main__":
    demo()