"""
Day 11: 混合检索实现

实现向量检索和关键词检索的融合，提高检索召回率和精确率。
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Document):
            return self.id == other.id
        return False


@dataclass
class SearchResult:
    """检索结果"""
    document: Document
    score: float
    source: str = "unknown"  # 检索来源: "vector", "bm25", "hybrid"


# ==================== BM25 检索器 ====================

class BM25Retriever:
    """
    BM25 关键词检索器
    
    BM25 评分公式:
    Score(Q, D) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        language: str = "chinese"
    ):
        """
        初始化 BM25 检索器
        
        Args:
            k1: 词频饱和参数，控制词频对得分的影响
            b: 文档长度归一化参数，控制文档长度惩罚
            language: 语言，用于分词
        """
        self.k1 = k1
        self.b = b
        self.language = language
        
        # 文档存储
        self.documents: List[Document] = []
        self.doc_tokens: List[List[str]] = []
        
        # BM25 统计信息
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # 词频（文档级别）
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0  # 文档总数
        
        # IDF 缓存
        self.idf_cache: Dict[str, float] = {}
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 待分词文本
            
        Returns:
            分词结果列表
        """
        text = text.lower()
        
        if self.language == "chinese":
            # 简单的中文分词（按字符）
            # 实际应用中应使用 jieba 等专业分词工具
            tokens = []
            # 提取中文字符和英文单词
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
            english_pattern = re.compile(r'[a-z0-9]+')
            
            # 提取中文
            for match in chinese_pattern.finditer(text):
                # 中文按字切分（简单处理）
                tokens.extend(list(match.group()))
            
            # 提取英文和数字
            for match in english_pattern.finditer(text):
                tokens.append(match.group())
            
            return tokens
        else:
            # 英文分词
            return re.findall(r'\b\w+\b', text)
    
    def index(self, documents: List[Document]) -> None:
        """
        建立索引
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_tokens = []
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.idf_cache = {}
        
        # 统计文档频率
        for doc in documents:
            tokens = self.tokenize(doc.content)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # 统计每个词在多少文档中出现
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        self.N = len(documents)
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
    
    def _get_idf(self, token: str) -> float:
        """
        计算 IDF 值
        
        IDF = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        
        Args:
            token: 词
            
        Returns:
            IDF 值
        """
        if token in self.idf_cache:
            return self.idf_cache[token]
        
        n = self.doc_freqs.get(token, 0)
        idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
        self.idf_cache[token] = idf
        return idf
    
    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        计算单个文档的 BM25 分数
        
        Args:
            query_tokens: 查询词列表
            doc_idx: 文档索引
            
        Returns:
            BM25 分数
        """
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        # 计算文档中每个词的频率
        term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token not in term_freqs:
                continue
            
            tf = term_freqs[token]
            idf = self._get_idf(token)
            
            # BM25 评分公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        检索文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 元数据过滤条件
            
        Returns:
            检索结果列表
        """
        if not self.documents:
            return []
        
        query_tokens = self.tokenize(query)
        
        # 计算所有文档的分数
        scores = []
        for idx, doc in enumerate(self.documents):
            # 应用过滤器
            if filters:
                skip = False
                for key, value in filters.items():
                    if doc.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            score = self._score_document(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))
        
        # 排序并返回 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            results.append(SearchResult(
                document=self.documents[idx],
                score=score,
                source="bm25"
            ))
        
        return results
    
    def get_doc_freq(self, token: str) -> int:
        """获取词的文档频率"""
        return self.doc_freqs.get(token, 0)


# ==================== 向量检索器 ====================

class VectorRetriever:
    """
    向量检索器
    
    使用余弦相似度进行向量检索
    """
    
    def __init__(self, embedding_model: Optional[Any] = None):
        """
        初始化向量检索器
        
        Args:
            embedding_model: 嵌入模型，需要有 embed 方法
        """
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        
        Args:
            text: 文本
            
        Returns:
            嵌入向量
        """
        if self.embedding_model is None:
            # 使用简单的 TF-IDF 向量作为占位
            # 实际应用中应使用真实的嵌入模型
            words = text.lower().split()
            vector = np.zeros(100)  # 假设维度为 100
            for i, word in enumerate(set(words)):
                # 简单的哈希映射
                idx = hash(word) % 100
                vector[idx] = 1.0
            return vector / (np.linalg.norm(vector) + 1e-8)
        
        return np.array(self.embedding_model.embed(text))
    
    def index(self, documents: List[Document]) -> None:
        """
        建立向量索引
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        
        # 计算所有文档的嵌入
        embeddings = []
        for doc in documents:
            emb = self._get_embedding(doc.content)
            embeddings.append(emb)
        
        self.embeddings = np.array(embeddings) if embeddings else None
        
        # 归一化
        if self.embeddings is not None:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / (norms + 1e-8)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        检索文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 元数据过滤条件
            
        Returns:
            检索结果列表
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # 获取查询向量
        query_emb = self._get_embedding(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_emb)
        
        # 应用过滤器
        valid_indices = []
        for idx, doc in enumerate(self.documents):
            if filters:
                skip = False
                for key, value in filters.items():
                    if doc.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            valid_indices.append(idx)
        
        # 获取有效文档的分数
        scores = [(idx, similarities[idx]) for idx in valid_indices]
        
        # 排序并返回 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            results.append(SearchResult(
                document=self.documents[idx],
                score=float(score),
                source="vector"
            ))
        
        return results


# ==================== 混合检索器 ====================

class HybridRetriever:
    """
    混合检索器
    
    融合向量检索和关键词检索的结果
    """
    
    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60
    ):
        """
        初始化混合检索器
        
        Args:
            bm25_retriever: BM25 检索器
            vector_retriever: 向量检索器
            fusion_method: 融合方法 ("rrf", "weighted", "simple")
            alpha: 加权融合的权重（向量检索的权重）
            rrf_k: RRF 融合的 k 参数
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.rrf_k = rrf_k
    
    def _rrf_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        RRF (Reciprocal Rank Fusion) 融合
        
        Score = Σ 1/(k + rank)
        
        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            top_k: 返回结果数量
            
        Returns:
            融合后的结果
        """
        # 计算每个文档的 RRF 分数
        rrf_scores: Dict[str, Tuple[Document, float]] = {}
        
        # BM25 结果
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.document.id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = (result.document, 0.0)
            rrf_scores[doc_id] = (
                rrf_scores[doc_id][0],
                rrf_scores[doc_id][1] + 1.0 / (self.rrf_k + rank)
            )
        
        # 向量检索结果
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document.id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = (result.document, 0.0)
            rrf_scores[doc_id] = (
                rrf_scores[doc_id][0],
                rrf_scores[doc_id][1] + 1.0 / (self.rrf_k + rank)
            )
        
        # 排序
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1][1],
            reverse=True
        )
        
        # 构建结果
        results = []
        for doc_id, (doc, score) in sorted_results[:top_k]:
            results.append(SearchResult(
                document=doc,
                score=score,
                source="hybrid"
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        加权融合
        
        Score = α * vector_score + (1-α) * bm25_score
        
        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            top_k: 返回结果数量
            
        Returns:
            融合后的结果
        """
        # 归一化分数
        def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r.document.id: 1.0 for r in results}
            return {
                r.document.id: (r.score - min_s) / (max_s - min_s)
                for r in results
            }
        
        bm25_normalized = normalize_scores(bm25_results)
        vector_normalized = normalize_scores(vector_results)
        
        # 合并并计算加权分数
        all_docs: Dict[str, Document] = {}
        for result in bm25_results + vector_results:
            all_docs[result.document.id] = result.document
        
        weighted_scores = {}
        for doc_id in all_docs:
            bm25_s = bm25_normalized.get(doc_id, 0.0)
            vector_s = vector_normalized.get(doc_id, 0.0)
            weighted_scores[doc_id] = self.alpha * vector_s + (1 - self.alpha) * bm25_s
        
        # 排序
        sorted_results = sorted(
            weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建结果
        results = []
        for doc_id, score in sorted_results[:top_k]:
            results.append(SearchResult(
                document=all_docs[doc_id],
                score=score,
                source="hybrid"
            ))
        
        return results
    
    def _simple_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        简单合并
        
        取两个检索结果的并集，按各自排名加权
        
        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            top_k: 返回结果数量
            
        Returns:
            融合后的结果
        """
        seen = set()
        results = []
        
        # 交替取结果
        i, j = 0, 0
        while len(results) < top_k and (i < len(bm25_results) or j < len(vector_results)):
            if i < len(bm25_results):
                result = bm25_results[i]
                if result.document.id not in seen:
                    seen.add(result.document.id)
                    results.append(SearchResult(
                        document=result.document,
                        score=result.score,
                        source="hybrid"
                    ))
                i += 1
            
            if len(results) >= top_k:
                break
            
            if j < len(vector_results):
                result = vector_results[j]
                if result.document.id not in seen:
                    seen.add(result.document.id)
                    results.append(SearchResult(
                        document=result.document,
                        score=result.score,
                        source="hybrid"
                    ))
                j += 1
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            bm25_top_k: BM25 检索数量（默认为 top_k * 2）
            vector_top_k: 向量检索数量（默认为 top_k * 2）
            filters: 元数据过滤条件
            
        Returns:
            检索结果
        """
        bm25_top_k = bm25_top_k or top_k * 2
        vector_top_k = vector_top_k or top_k * 2
        
        # 获取各检索器结果
        bm25_results = []
        vector_results = []
        
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, bm25_top_k, filters)
        
        if self.vector_retriever:
            vector_results = self.vector_retriever.retrieve(query, vector_top_k, filters)
        
        # 如果只有一个检索器有结果，直接返回
        if not bm25_results:
            return vector_results[:top_k]
        if not vector_results:
            return bm25_results[:top_k]
        
        # 融合结果
        if self.fusion_method == "rrf":
            return self._rrf_fusion(bm25_results, vector_results, top_k)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(bm25_results, vector_results, top_k)
        elif self.fusion_method == "simple":
            return self._simple_fusion(bm25_results, vector_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


# ==================== 查询优化 ====================

class QueryOptimizer:
    """
    查询优化器
    
    提供查询扩展、重写等功能
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        初始化查询优化器
        
        Args:
            llm: 大语言模型，用于查询重写和扩展
        """
        self.llm = llm
        
        # 同义词词典（示例）
        self.synonyms = {
            "向量数据库": ["向量库", "vector database", "向量存储"],
            "检索": ["搜索", "查找", "查询", "retrieval", "search"],
            "嵌入": ["向量表示", "embedding", "向量化"],
            "大模型": ["LLM", "大语言模型", "语言模型"],
        }
    
    def expand_query(self, query: str) -> str:
        """
        查询扩展
        
        添加同义词和相关信息
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询
        """
        expanded_terms = []
        
        for term, syns in self.synonyms.items():
            if term in query:
                expanded_terms.extend(syns[:2])  # 只添加前2个同义词
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        生成多个相关查询
        
        使用 LLM 生成不同角度的查询
        
        Args:
            query: 原始查询
            num_queries: 生成的查询数量
            
        Returns:
            查询列表
        """
        if self.llm is None:
            # 简单规则生成
            return [query]
        
        prompt = f"""请根据以下问题，生成{num_queries}个不同角度的相似问题。
要求：保持原意，但用不同的表达方式。

原问题：{query}

请直接输出问题，每行一个："""
        
        # 这里应该调用 LLM
        # response = self.llm.generate(prompt)
        # return response.strip().split('\n')
        
        return [query]
    
    def rewrite_query(self, query: str) -> str:
        """
        查询重写
        
        优化查询表达，使其更适合检索
        
        Args:
            query: 原始查询
            
        Returns:
            重写后的查询
        """
        # 移除疑问词
        question_words = ["请问", "如何", "怎么", "什么是", "怎样", "为什么"]
        rewritten = query
        for word in question_words:
            rewritten = rewritten.replace(word, "")
        
        # 移除语气词
        tone_words = ["呢", "啊", "吧", "吗", "的"]
        for word in tone_words:
            rewritten = rewritten.replace(word, "")
        
        return rewritten.strip() or query


# ==================== 使用示例 ====================

def demo_hybrid_retrieval():
    """演示混合检索"""
    
    # 准备示例文档
    documents = [
        Document(
            id="doc1",
            content="向量数据库是一种专门用于存储和检索向量嵌入的数据库系统。它支持高效的相似度搜索，广泛应用于推荐系统、图像检索等领域。",
            metadata={"category": "database", "topic": "vector"}
        ),
        Document(
            id="doc2",
            content="RAG（检索增强生成）是一种结合检索和生成的技术。它首先从知识库中检索相关文档，然后将这些文档作为上下文传递给大语言模型生成回答。",
            metadata={"category": "ai", "topic": "rag"}
        ),
        Document(
            id="doc3",
            content="BM25是一种经典的关键词检索算法，它考虑了词频和逆文档频率，在信息检索领域有着广泛应用。BM25对长文档有自然的惩罚机制。",
            metadata={"category": "algorithm", "topic": "retrieval"}
        ),
        Document(
            id="doc4",
            content="混合检索结合了向量检索和关键词检索的优势。向量检索擅长语义理解，关键词检索擅长精确匹配。两者的融合可以显著提高检索质量。",
            metadata={"category": "ai", "topic": "hybrid"}
        ),
        Document(
            id="doc5",
            content="重排序是在初步检索后对结果进行精细化排序的过程。常用的重排序模型包括Cross-Encoder，它可以更准确地计算查询和文档的相关性。",
            metadata={"category": "ai", "topic": "rerank"}
        ),
    ]
    
    print("=" * 60)
    print("Day 11: 混合检索演示")
    print("=" * 60)
    
    # 创建 BM25 检索器
    print("\n1. 创建 BM25 检索器...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(documents)
    print(f"   已索引 {len(documents)} 个文档")
    print(f"   平均文档长度: {bm25.avgdl:.2f}")
    
    # 创建向量检索器
    print("\n2. 创建向量检索器...")
    vector = VectorRetriever()
    vector.index(documents)
    print(f"   已计算 {len(documents)} 个文档的嵌入向量")
    
    # 创建混合检索器
    print("\n3. 创建混合检索器...")
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        fusion_method="rrf",
        rrf_k=60
    )
    
    # 测试查询
    queries = [
        "什么是向量数据库？",
        "BM25 算法原理",
        "如何提高检索质量？"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print("-" * 60)
        
        # BM25 检索
        print("\nBM25 检索结果:")
        bm25_results = bm25.retrieve(query, top_k=3)
        for i, result in enumerate(bm25_results, 1):
            print(f"  {i}. [{result.source}] 分数: {result.score:.4f}")
            print(f"     内容: {result.document.content[:50]}...")
        
        # 向量检索
        print("\n向量检索结果:")
        vector_results = vector.retrieve(query, top_k=3)
        for i, result in enumerate(vector_results, 1):
            print(f"  {i}. [{result.source}] 分数: {result.score:.4f}")
            print(f"     内容: {result.document.content[:50]}...")
        
        # 混合检索
        print("\n混合检索结果 (RRF 融合):")
        hybrid_results = hybrid.retrieve(query, top_k=3)
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. [{result.source}] 分数: {result.score:.4f}")
            print(f"     内容: {result.document.content[:50]}...")
    
    # 测试不同融合方法
    print("\n" + "=" * 60)
    print("融合方法对比")
    print("=" * 60)
    
    query = "检索技术"
    
    for method in ["rrf", "weighted", "simple"]:
        hybrid = HybridRetriever(
            bm25_retriever=bm25,
            vector_retriever=vector,
            fusion_method=method,
            alpha=0.6,
            rrf_k=60
        )
        
        results = hybrid.retrieve(query, top_k=3)
        print(f"\n{method.upper()} 融合:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result.document.id}, 分数: {result.score:.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_hybrid_retrieval()