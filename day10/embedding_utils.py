"""
嵌入工具模块

实现文本嵌入功能，支持多种嵌入模型。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import hashlib
import json
import numpy as np


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    embedding: List[float]
    text: str
    model: str
    dimensions: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding,
            "text": self.text,
            "model": self.model,
            "dimensions": self.dimensions,
            "metadata": self.metadata
        }


class BaseEmbedding(ABC):
    """嵌入模型基类"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        嵌入单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度"""
        pass
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数
        """
        a = np.array(vec1)
        b = np.array(vec2)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))


class MockEmbedding(BaseEmbedding):
    """
    模拟嵌入模型
    
    用于测试和开发，不需要实际 API 调用。
    """
    
    def __init__(self, dimensions: int = 384):
        super().__init__("mock-embedding")
        self._dimensions = dimensions
    
    def embed(self, text: str) -> List[float]:
        """生成模拟嵌入"""
        # 使用文本哈希生成确定性向量
        text_hash = hashlib.md5(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16))
        embedding = np.random.randn(self._dimensions).tolist()
        # 归一化
        embedding = np.array(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成模拟嵌入"""
        return [self.embed(text) for text in texts]
    
    @property
    def dimensions(self) -> int:
        return self._dimensions


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI 嵌入模型
    
    使用 OpenAI API 生成嵌入。
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化
        
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
        """
        super().__init__(model)
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        
        # 模型维度映射
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    def embed(self, text: str) -> List[float]:
        """生成嵌入"""
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            return response.data[0].embedding
            
        except ImportError:
            print("请安装 openai: pip install openai")
            return MockEmbedding(self.dimensions).embed(text)
        except Exception as e:
            print(f"OpenAI API 调用失败: {e}")
            return MockEmbedding(self.dimensions).embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        try:
            import openai
            
            client = openAI.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            print(f"OpenAI API 调用失败: {e}")
            return MockEmbedding(self.dimensions).embed_batch(texts)
    
    @property
    def dimensions(self) -> int:
        return self._model_dimensions.get(self.model_name, 1536)


class LocalEmbedding(BaseEmbedding):
    """
    本地嵌入模型
    
    使用 sentence-transformers 等本地模型。
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        初始化
        
        Args:
            model_name: 模型名称或路径
            device: 设备（cpu/cuda）
        """
        super().__init__(model_name)
        self.device = device
        self._model = None
        self._dimensions = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimensions = self._model.get_sentence_embedding_dimension()
            except ImportError:
                print("请安装 sentence-transformers: pip install sentence-transformers")
                return None
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """生成嵌入"""
        model = self._load_model()
        if model is None:
            return MockEmbedding().embed(text)
        
        embedding = model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        model = self._load_model()
        if model is None:
            return MockEmbedding().embed_batch(texts)
        
        embeddings = model.encode(texts)
        return embeddings.tolist()
    
    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            model = self._load_model()
            if model is None:
                return 384
        return self._dimensions


class EmbeddingCache:
    """
    嵌入缓存
    
    避免重复计算相同文本的嵌入。
    """
    
    def __init__(self, max_size: int = 10000):
        """
        初始化
        
        Args:
            max_size: 最大缓存数量
        """
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
    
    def _get_key(self, text: str, model: str) -> str:
        """生成缓存键"""
        return hashlib.md5(f"{model}:{text}".encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """获取缓存的嵌入"""
        key = self._get_key(text, model)
        return self._cache.get(key)
    
    def set(self, text: str, model: str, embedding: List[float]):
        """设置缓存"""
        if len(self._cache) >= self.max_size:
            # 简单的 LRU：删除一半
            keys = list(self._cache.keys())[:self.max_size // 2]
            for k in keys:
                del self._cache[k]
        
        key = self._get_key(text, model)
        self._cache[key] = embedding
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()


class EmbeddingModel:
    """
    统一的嵌入模型接口
    
    提供便捷的嵌入功能，支持缓存。
    """
    
    def __init__(
        self,
        model_type: str = "mock",
        model_name: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """
        初始化
        
        Args:
            model_type: 模型类型（mock/openai/local）
            model_name: 模型名称
            use_cache: 是否使用缓存
            **kwargs: 其他参数
        """
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None
        
        # 创建嵌入模型
        if model_type == "mock":
            dimensions = kwargs.get("dimensions", 384)
            self._model = MockEmbedding(dimensions=dimensions)
        elif model_type == "openai":
            self._model = OpenAIEmbedding(
                model=model_name or "text-embedding-3-small",
                **kwargs
            )
        elif model_type == "local":
            self._model = LocalEmbedding(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
                **kwargs
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def embed(self, text: str) -> List[float]:
        """
        嵌入文本
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 检查缓存
        if self.use_cache and self.cache:
            cached = self.cache.get(text, self._model.model_name)
            if cached is not None:
                return cached
        
        # 计算嵌入
        embedding = self._model.embed(text)
        
        # 缓存结果
        if self.use_cache and self.cache:
            self.cache.set(text, self._model.model_name, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入"""
        # 对于批量，直接使用模型（简化缓存逻辑）
        return self._model.embed_batch(texts)
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        嵌入文档
        
        Args:
            documents: 文档列表（包含 content 字段）
            
        Returns:
            带有 embedding 字段的文档列表
        """
        texts = [doc.get("content", "") for doc in documents]
        embeddings = self.embed_batch(texts)
        
        result = []
        for doc, embedding in zip(documents, embeddings):
            new_doc = doc.copy()
            new_doc["embedding"] = embedding
            result.append(new_doc)
        
        return result
    
    @property
    def dimensions(self) -> int:
        """向量维度"""
        return self._model.dimensions
    
    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._model.model_name
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算相似度"""
        return self._model.similarity(vec1, vec2)
    
    def most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        找出最相似的向量
        
        Args:
            query_embedding: 查询向量
            embeddings: 候选向量列表
            top_k: 返回数量
            
        Returns:
            (index, similarity) 列表
        """
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self.similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# ============ 演示 ============

def demo():
    """演示嵌入功能"""
    print("=" * 60)
    print("嵌入模型演示")
    print("=" * 60)
    
    # 创建模拟嵌入模型
    embedding = EmbeddingModel(model_type="mock", dimensions=384)
    
    print(f"\n模型: {embedding.model_name}")
    print(f"维度: {embedding.dimensions}")
    
    # 测试嵌入
    texts = [
        "向量数据库是一种专门存储和检索向量的数据库",
        "RAG 是检索增强生成的缩写",
        "嵌入是将文本转换为数值向量的过程"
    ]
    
    print("\n--- 嵌入文本 ---")
    embeddings = embedding.embed_batch(texts)
    
    for text, emb in zip(texts, embeddings):
        print(f"\n文本: {text}")
        print(f"向量维度: {len(emb)}")
        print(f"向量前5位: {emb[:5]}")
    
    # 测试相似度
    print("\n--- 相似度计算 ---")
    query = "什么是向量数据库？"
    query_emb = embedding.embed(query)
    
    print(f"\n查询: {query}")
    print("\n相似度结果:")
    
    most_similar = embedding.most_similar(query_emb, embeddings, top_k=3)
    for idx, sim in most_similar:
        print(f"  [{sim:.4f}] {texts[idx]}")
    
    # 测试缓存
    print("\n--- 缓存测试 ---")
    print("第二次嵌入相同文本（应该使用缓存）...")
    
    import time
    start = time.time()
    embedding.embed(texts[0])
    elapsed1 = time.time() - start
    
    start = time.time()
    embedding.embed(texts[0])
    elapsed2 = time.time() - start
    
    print(f"第一次: {elapsed1:.6f}s")
    print(f"第二次: {elapsed2:.6f}s (缓存)")


if __name__ == "__main__":
    demo()