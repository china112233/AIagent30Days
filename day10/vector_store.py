"""
向量存储模块

实现向量数据库的存储和检索功能。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import json
import os

import numpy as np

# 导入本地模块
try:
    from embedding_utils import EmbeddingModel
except ImportError:
    from embedding_utils import EmbeddingModel


@dataclass
class VectorDocument:
    """向量文档"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDocument":
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=data["embedding"],
            metadata=data.get("metadata", {})
        )


@dataclass
class SearchResult:
    """检索结果"""
    document: VectorDocument
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.document.content,
            "score": self.score,
            "metadata": self.document.metadata
        }


class BaseVectorStore(ABC):
    """向量存储基类"""
    
    @abstractmethod
    def add(self, documents: List[VectorDocument]) -> List[str]:
        """
        添加文档
        
        Args:
            documents: 文档列表
            
        Returns:
            文档 ID 列表
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        检索文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter: 元数据过滤条件
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def get(self, id: str) -> Optional[VectorDocument]:
        """获取文档"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """文档数量"""
        pass


class SimpleVectorStore(BaseVectorStore):
    """
    简单向量存储
    
    使用内存存储，适合开发和小规模数据。
    """
    
    def __init__(self):
        self._documents: Dict[str, VectorDocument] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._counter = 0
    
    def _generate_id(self) -> str:
        """生成文档 ID"""
        self._counter += 1
        return f"doc_{self._counter}"
    
    def add(self, documents: List[VectorDocument]) -> List[str]:
        """添加文档"""
        ids = []
        for doc in documents:
            if not doc.id:
                doc.id = self._generate_id()
            
            self._documents[doc.id] = doc
            self._embeddings[doc.id] = np.array(doc.embedding)
            ids.append(doc.id)
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """检索文档"""
        if not self._documents:
            return []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        # 计算相似度
        scores = []
        for doc_id, doc in self._documents.items():
            # 元数据过滤
            if filter and not self._match_filter(doc.metadata, filter):
                continue
            
            doc_vec = self._embeddings[doc_id]
            doc_norm = np.linalg.norm(doc_vec)
            
            if doc_norm == 0:
                continue
            
            similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
            scores.append((doc, similarity))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [SearchResult(document=doc, score=score) for doc, score in scores[:top_k]]
    
    def _match_filter(self, metadata: Dict, filter: Dict) -> bool:
        """匹配过滤条件"""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if isinstance(value, dict):
                # 范围查询
                if "$eq" in value and metadata[key] != value["$eq"]:
                    return False
                if "$ne" in value and metadata[key] == value["$ne"]:
                    return False
                if "$gt" in value and metadata[key] <= value["$gt"]:
                    return False
                if "$lt" in value and metadata[key] >= value["$lt"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def delete(self, ids: List[str]) -> bool:
        """删除文档"""
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._embeddings[doc_id]
        return True
    
    def get(self, id: str) -> Optional[VectorDocument]:
        """获取文档"""
        return self._documents.get(id)
    
    def count(self) -> int:
        """文档数量"""
        return len(self._documents)
    
    def clear(self):
        """清空存储"""
        self._documents.clear()
        self._embeddings.clear()
    
    def save(self, path: str):
        """保存到文件"""
        data = {
            "documents": [doc.to_dict() for doc in self._documents.values()],
            "counter": self._counter
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """从文件加载"""
        if not os.path.exists(path):
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._documents = {}
        self._embeddings = {}
        
        for doc_data in data.get("documents", []):
            doc = VectorDocument.from_dict(doc_data)
            self._documents[doc.id] = doc
            self._embeddings[doc.id] = np.array(doc.embedding)
        
        self._counter = data.get("counter", 0)


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma 向量存储
    
    使用 ChromaDB 作为后端。
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        初始化
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_model: 嵌入模型
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or EmbeddingModel()
        self._client = None
        self._collection = None
    
    def _init_client(self):
        """初始化客户端"""
        if self._client is None:
            try:
                import chromadb
                
                if self.persist_directory:
                    self._client = chromadb.PersistentClient(path=self.persist_directory)
                else:
                    self._client = chromadb.Client()
                
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            except ImportError:
                print("请安装 chromadb: pip install chromadb")
                raise
        return self._collection
    
    def add(self, documents: List[VectorDocument]) -> List[str]:
        """添加文档"""
        collection = self._init_client()
        
        ids = []
        contents = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.id)
            contents.append(doc.content)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata)
        
        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """检索文档"""
        collection = self._init_client()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        search_results = []
        for i in range(len(results["ids"][0])):
            doc = VectorDocument(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                embedding=results["embeddings"][0][i] if results.get("embeddings") else [],
                metadata=results["metadatas"][0][i] if results.get("metadatas") else {}
            )
            score = 1.0 - results["distances"][0][i]  # Chroma 返回距离，转换为相似度
            search_results.append(SearchResult(document=doc, score=score))
        
        return search_results
    
    def delete(self, ids: List[str]) -> bool:
        """删除文档"""
        collection = self._init_client()
        collection.delete(ids=ids)
        return True
    
    def get(self, id: str) -> Optional[VectorDocument]:
        """获取文档"""
        collection = self._init_client()
        results = collection.get(ids=[id])
        
        if not results["ids"]:
            return None
        
        return VectorDocument(
            id=results["ids"][0],
            content=results["documents"][0],
            embedding=results["embeddings"][0] if results.get("embeddings") else [],
            metadata=results["metadatas"][0] if results.get("metadatas") else {}
        )
    
    def count(self) -> int:
        """文档数量"""
        collection = self._init_client()
        return collection.count()


class VectorStore:
    """
    统一的向量存储接口
    
    提供便捷的向量存储操作。
    """
    
    @staticmethod
    def create(
        store_type: str = "simple",
        **kwargs
    ) -> BaseVectorStore:
        """
        创建向量存储
        
        Args:
            store_type: 存储类型（simple/chroma）
            **kwargs: 其他参数
            
        Returns:
            向量存储实例
        """
        if store_type == "simple":
            return SimpleVectorStore()
        elif store_type == "chroma":
            return ChromaVectorStore(**kwargs)
        else:
            raise ValueError(f"未知的存储类型: {store_type}")
    
    @staticmethod
    def from_documents(
        documents: List[Dict],
        embedding_model: EmbeddingModel,
        store_type: str = "simple"
    ) -> BaseVectorStore:
        """
        从文档创建向量存储
        
        Args:
            documents: 文档列表（包含 content 和 metadata）
            embedding_model: 嵌入模型
            store_type: 存储类型
            
        Returns:
            向量存储实例
        """
        store = VectorStore.create(store_type)
        
        # 生成嵌入
        texts = [doc.get("content", "") for doc in documents]
        embeddings = embedding_model.embed_batch(texts)
        
        # 创建向量文档
        vector_docs = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            vector_docs.append(VectorDocument(
                id=doc.get("id", f"doc_{i}"),
                content=doc["content"],
                embedding=emb,
                metadata=doc.get("metadata", {})
            ))
        
        # 添加到存储
        store.add(vector_docs)
        
        return store


# ============ 演示 ============

def demo():
    """演示向量存储"""
    print("=" * 60)
    print("向量存储演示")
    print("=" * 60)
    
    # 创建向量存储
    store = SimpleVectorStore()
    
    # 创建嵌入模型
    embedding = EmbeddingModel(model_type="mock")
    
    # 示例文档
    documents = [
        {"content": "向量数据库是一种专门用于存储和检索向量的数据库", "metadata": {"topic": "database"}},
        {"content": "RAG 是检索增强生成的缩写，用于增强 LLM 能力", "metadata": {"topic": "rag"}},
        {"content": "嵌入是将文本转换为数值向量的过程", "metadata": {"topic": "embedding"}},
        {"content": "语义搜索使用向量相似度来查找相关内容", "metadata": {"topic": "search"}},
    ]
    
    print("\n--- 添加文档 ---")
    
    # 生成嵌入并添加到存储
    vector_docs = []
    for i, doc in enumerate(documents):
        emb = embedding.embed(doc["content"])
        vector_docs.append(VectorDocument(
            id=f"doc_{i}",
            content=doc["content"],
            embedding=emb,
            metadata=doc["metadata"]
        ))
    
    ids = store.add(vector_docs)
    print(f"添加了 {len(ids)} 个文档")
    
    # 检索
    print("\n--- 检索测试 ---")
    
    queries = [
        "什么是向量数据库？",
        "如何进行语义搜索？",
        "RAG 技术介绍"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        query_emb = embedding.embed(query)
        results = store.search(query_emb, top_k=2)
        
        for i, result in enumerate(results):
            print(f"  [{i+1}] 相似度: {result.score:.4f}")
            print(f"      内容: {result.document.content}")
            print(f"      元数据: {result.document.metadata}")
    
    # 过滤检索
    print("\n--- 过滤检索 ---")
    query_emb = embedding.embed("数据库相关")
    results = store.search(
        query_emb,
        top_k=5,
        filter={"topic": "database"}
    )
    print(f"查询 topic=database:")
    for result in results:
        print(f"  - {result.document.content}")
    
    # 统计
    print(f"\n--- 统计 ---")
    print(f"文档总数: {store.count()}")
    
    # 保存和加载
    print("\n--- 保存和加载 ---")
    save_path = "vector_store_demo.json"
    store.save(save_path)
    print(f"已保存到: {save_path}")
    
    new_store = SimpleVectorStore()
    new_store.load(save_path)
    print(f"加载后文档数: {new_store.count()}")


if __name__ == "__main__":
    demo()