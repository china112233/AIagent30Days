"""
Day 11: 进阶 RAG 引擎

整合混合检索、重排序、知识库优化等技术，实现生产级 RAG 系统。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 导入其他模块
from hybrid_retrieval import BM25Retriever, VectorRetriever, HybridRetriever, Document, SearchResult
from reranker import CrossEncoderReranker, MockCrossEncoderReranker, LLMReranker, RerankResult
from knowledge_base_optimizer import (
    DocumentCleaner, DocumentDeduplicator, AdaptiveChunker,
    QualityScorer, IncrementalUpdater, Chunk
)


@dataclass
class AdvancedRAGConfig:
    """进阶 RAG 配置"""
    # 嵌入配置
    embedding_model: str = "text-embedding-3-small"
    
    # 向量存储配置
    vector_store: str = "chroma"
    
    # 检索配置
    use_hybrid_retrieval: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    fusion_method: str = "rrf"  # "rrf", "weighted", "simple"
    rrf_k: int = 60
    fusion_alpha: float = 0.6  # 向量检索权重
    
    # 重排序配置
    use_reranking: bool = True
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cpu"
    
    # 切分配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # 检索参数
    initial_top_k: int = 20  # 初检数量
    final_top_k: int = 5     # 最终返回数量
    
    # 生成配置
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # 优化配置
    enable_cleaning: bool = True
    enable_dedup: bool = True
    enable_quality_scoring: bool = True
    quality_threshold: float = 0.3
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 秒


@dataclass
class RAGResponse:
    """RAG 响应"""
    answer: str
    sources: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 性能指标
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0


class AdvancedRAGEngine:
    """
    进阶 RAG 引擎
    
    整合所有进阶技术：
    - 混合检索（向量 + BM25）
    - Cross-Encoder 重排序
    - 知识库优化
    - 查询优化
    - 评估支持
    """
    
    def __init__(self, config: Optional[AdvancedRAGConfig] = None):
        """
        初始化进阶 RAG 引擎
        
        Args:
            config: 配置对象
        """
        self.config = config or AdvancedRAGConfig()
        
        # 组件
        self.cleaner = DocumentCleaner()
        self.deduplicator = DocumentDeduplicator()
        self.chunker = AdaptiveChunker(
            default_chunk_size=self.config.chunk_size,
            default_overlap=self.config.chunk_overlap
        )
        self.quality_scorer = QualityScorer()
        self.updater = IncrementalUpdater()
        
        # 检索器
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.vector_retriever: Optional[VectorRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        
        # 重排序器
        self.reranker = None
        
        # 文档存储
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self.chunk_index: Dict[str, Chunk] = {}
        
        # 缓存
        self.query_cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        
        # 初始化
        self._init_retrievers()
        self._init_reranker()
    
    def _init_retrievers(self):
        """初始化检索器"""
        self.bm25_retriever = BM25Retriever(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )
        
        self.vector_retriever = VectorRetriever()
        
        if self.config.use_hybrid_retrieval:
            self.hybrid_retriever = HybridRetriever(
                bm25_retriever=self.bm25_retriever,
                vector_retriever=self.vector_retriever,
                fusion_method=self.config.fusion_method,
                alpha=self.config.fusion_alpha,
                rrf_k=self.config.rrf_k
            )
    
    def _init_reranker(self):
        """初始化重排序器"""
        if not self.config.use_reranking:
            return
        
        # 使用模拟重排序器（实际应用中使用真实模型）
        self.reranker = MockCrossEncoderReranker()
    
    def _preprocess_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        文档预处理
        
        Args:
            documents: 原始文档
            
        Returns:
            处理后的文档
        """
        processed = documents
        
        # 清洗
        if self.config.enable_cleaning:
            processed = [self.cleaner.clean_document(doc) for doc in processed]
        
        # 去重
        if self.config.enable_dedup:
            processed, _ = self.deduplicator.exact_deduplicate(processed)
        
        # 质量评分
        if self.config.enable_quality_scoring:
            processed = self.quality_scorer.score_documents(processed)
            # 过滤低质量文档
            processed = [
                doc for doc in processed
                if doc.quality_score >= self.config.quality_threshold
            ]
        
        return processed
    
    def _chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        切分文档
        
        Args:
            documents: 文档列表
            
        Returns:
            文档块列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def add_documents(
        self,
        documents: List[Document],
        incremental: bool = False
    ) -> Dict[str, Any]:
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            incremental: 是否增量更新
            
        Returns:
            添加结果统计
        """
        start_time = time.time()
        
        if incremental:
            # 增量更新
            new_docs, updated_docs, deleted_ids = self.updater.check_update(documents)
            stats = {
                "new": len(new_docs),
                "updated": len(updated_docs),
                "deleted": len(deleted_ids)
            }
            documents = new_docs + updated_docs
        else:
            stats = {"new": len(documents), "updated": 0, "deleted": 0}
        
        # 预处理
        processed_docs = self._preprocess_documents(documents)
        stats["after_preprocessing"] = len(processed_docs)
        
        # 切分
        chunks = self._chunk_documents(processed_docs)
        stats["chunks"] = len(chunks)
        
        # 更新存储
        self.documents.extend(processed_docs)
        self.chunks.extend(chunks)
        self.chunk_index = {c.id: c for c in self.chunks}
        
        # 更新检索器索引
        # 将 Chunk 转换为 Document 格式
        chunk_docs = [
            Document(
                id=c.id,
                content=c.content,
                metadata=c.metadata
            )
            for c in chunks
        ]
        
        self.bm25_retriever.index(chunk_docs)
        self.vector_retriever.index(chunk_docs)
        
        stats["indexing_time"] = time.time() - start_time
        
        return stats
    
    def _check_cache(self, query: str) -> Optional[List[SearchResult]]:
        """检查缓存"""
        if not self.config.enable_cache:
            return None
        
        cache_key = query.lower().strip()
        if cache_key in self.query_cache:
            results, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return results
        
        return None
    
    def _update_cache(self, query: str, results: List[SearchResult]):
        """更新缓存"""
        if not self.config.enable_cache:
            return
        
        cache_key = query.lower().strip()
        self.query_cache[cache_key] = (results, time.time())
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            检索结果
        """
        top_k = top_k or self.config.final_top_k
        initial_k = self.config.initial_top_k
        
        # 检查缓存
        cached = self._check_cache(query)
        if cached:
            return cached[:top_k]
        
        # 执行检索
        if self.config.use_hybrid_retrieval and self.hybrid_retriever:
            results = self.hybrid_retriever.retrieve(
                query,
                top_k=initial_k,
                filters=filters
            )
        elif self.vector_retriever:
            results = self.vector_retriever.retrieve(
                query,
                top_k=initial_k,
                filters=filters
            )
        else:
            results = []
        
        # 更新缓存
        self._update_cache(query, results)
        
        return results[:top_k]
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        重排序检索结果
        
        Args:
            query: 查询文本
            results: 检索结果
            top_k: 返回数量
            
        Returns:
            重排序后的结果
        """
        if not self.config.use_reranking or not self.reranker:
            return []
        
        top_k = top_k or self.config.final_top_k
        
        # 转换格式
        candidates = [
            {
                "id": r.document.id,
                "content": r.document.content,
                "score": r.score,
                "metadata": r.document.metadata
            }
            for r in results
        ]
        
        # 重排序
        return self.reranker.rerank(query, candidates, top_k=top_k)
    
    def generate_answer(
        self,
        query: str,
        context: List[str]
    ) -> str:
        """
        生成答案
        
        Args:
            query: 查询文本
            context: 上下文文档列表
            
        Returns:
            生成的答案
        """
        # 构建提示词
        context_text = "\n\n".join([
            f"[文档{i+1}]\n{c}"
            for i, c in enumerate(context)
        ])
        
        prompt = f"""你是一个专业的助手。请根据以下上下文回答用户的问题。

上下文:
{context_text}

用户问题: {query}

请基于上下文回答问题，如果上下文中没有相关信息，请说明。回答要准确、简洁。
"""
        
        # 实际应用中应调用 LLM
        # response = self.llm.generate(prompt)
        
        # 模拟响应
        return f"基于检索到的 {len(context)} 个相关文档，我为您回答：{query} 的问题..." \
               f"\n\n相关内容已在上下文中提供。"
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        with_sources: bool = True
    ) -> RAGResponse:
        """
        执行完整 RAG 查询
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            filters: 过滤条件
            with_sources: 是否返回来源
            
        Returns:
            RAG 响应
        """
        start_time = time.time()
        
        # 1. 检索
        retrieval_start = time.time()
        results = self.retrieve(query, top_k=self.config.initial_top_k, filters=filters)
        retrieval_time = time.time() - retrieval_start
        
        # 2. 重排序
        rerank_start = time.time()
        if self.config.use_reranking and self.reranker:
            reranked = self.rerank(query, results, top_k=top_k or self.config.final_top_k)
            final_results = reranked
        else:
            final_results = results[:top_k or self.config.final_top_k]
        rerank_time = time.time() - rerank_start
        
        # 3. 构建上下文
        if isinstance(final_results[0], RerankResult) if final_results else False:
            context = [r.content for r in final_results]
            scores = [r.score for r in final_results]
            sources = [
                {
                    "id": r.document_id,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score,
                    "original_rank": r.original_rank
                }
                for r in final_results
            ]
        else:
            context = [r.document.content for r in final_results]
            scores = [r.score for r in final_results]
            sources = [
                {
                    "id": r.document.id,
                    "content": r.document.content[:200] + "...",
                    "score": r.score,
                    "source": r.source
                }
                for r in final_results
            ]
        
        # 4. 生成答案
        generation_start = time.time()
        answer = self.generate_answer(query, context)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            sources=sources if with_sources else [],
            scores=scores,
            metadata={
                "query": query,
                "num_results": len(final_results),
                "filters": filters
            },
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation_time=generation_time,
            total_time=total_time
        )
    
    def evaluate(
        self,
        eval_data: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        评估 RAG 系统
        
        Args:
            eval_data: 评估数据集
            metrics: 评估指标
            
        Returns:
            评估结果
        """
        if metrics is None:
            metrics = ["faithfulness", "answer_relevance", "context_precision"]
        
        results = {m: [] for m in metrics}
        
        for item in eval_data:
            query = item["question"]
            ground_truth = item.get("ground_truth", "")
            
            # 执行查询
            response = self.query(query)
            
            # 计算各项指标
            # 实际应用中应使用 RAGAS 等评估框架
            
            # 模拟评估
            if "faithfulness" in metrics:
                # 忠实度：答案是否基于上下文
                results["faithfulness"].append(0.85)
            
            if "answer_relevance" in metrics:
                # 答案相关性
                results["answer_relevance"].append(0.80)
            
            if "context_precision" in metrics:
                # 上下文精确率
                results["context_precision"].append(0.75)
        
        # 计算平均值
        return {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in results.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "cache_size": len(self.query_cache),
            "config": {
                "hybrid_retrieval": self.config.use_hybrid_retrieval,
                "reranking": self.config.use_reranking,
                "chunk_size": self.config.chunk_size,
                "initial_top_k": self.config.initial_top_k,
                "final_top_k": self.config.final_top_k
            },
            "updater_stats": self.updater.get_stats()
        }
    
    def clear_cache(self):
        """清空查询缓存"""
        self.query_cache.clear()


# ==================== 使用示例 ====================

def demo_advanced_rag():
    """演示进阶 RAG 系统"""
    
    print("=" * 70)
    print("Day 11: 进阶 RAG 系统演示")
    print("=" * 70)
    
    # 创建配置
    config = AdvancedRAGConfig(
        use_hybrid_retrieval=True,
        use_reranking=True,
        chunk_size=300,
        chunk_overlap=30,
        initial_top_k=10,
        final_top_k=3,
        enable_cleaning=True,
        enable_dedup=True,
        enable_quality_scoring=True
    )
    
    print("\n配置信息:")
    print(f"  混合检索: {config.use_hybrid_retrieval}")
    print(f"  重排序: {config.use_reranking}")
    print(f"  块大小: {config.chunk_size}")
    print(f"  初检数量: {config.initial_top_k}")
    print(f"  最终数量: {config.final_top_k}")
    
    # 创建引擎
    print("\n创建 RAG 引擎...")
    engine = AdvancedRAGEngine(config)
    
    # 准备文档
    documents = [
        Document(
            id="doc1",
            content="向量数据库是一种专门用于存储和检索向量嵌入的数据库系统。它支持高效的相似度搜索，广泛应用于推荐系统、图像检索和自然语言处理等领域。主流的向量数据库包括 Pinecone、Milvus、Chroma 等。",
            metadata={"category": "database", "topic": "vector"}
        ),
        Document(
            id="doc2",
            content="RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它首先从知识库中检索相关文档，然后将这些文档作为上下文传递给大语言模型，生成更准确、更有依据的回答。RAG 可以有效减少模型幻觉，提高回答的可信度。",
            metadata={"category": "ai", "topic": "rag"}
        ),
        Document(
            id="doc3",
            content="BM25 是一种经典的关键词检索算法，它考虑了词频（TF）和逆文档频率（IDF）。BM25 对长文档有自然的惩罚机制，避免了长文档在检索中的优势。它是现代搜索引擎的核心算法之一。",
            metadata={"category": "algorithm", "topic": "retrieval"}
        ),
        Document(
            id="doc4",
            content="混合检索结合了向量检索和关键词检索的优势。向量检索擅长语义理解，可以找到语义相似但词汇不同的内容；关键词检索擅长精确匹配，可以准确找到包含特定词汇的文档。两者的融合可以显著提高检索的召回率和精确率。",
            metadata={"category": "ai", "topic": "hybrid"}
        ),
        Document(
            id="doc5",
            content="重排序是在初步检索后对结果进行精细化排序的过程。常用的重排序模型是 Cross-Encoder，它将查询和文档一起输入模型，可以更准确地计算相关性。重排序可以显著提高检索结果的质量，但会增加一定的计算开销。",
            metadata={"category": "ai", "topic": "rerank"}
        ),
    ]
    
    # 添加文档
    print("\n添加文档到知识库...")
    stats = engine.add_documents(documents)
    print(f"添加统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 显示系统统计
    print("\n系统统计:")
    system_stats = engine.get_stats()
    print(f"  文档数: {system_stats['documents']}")
    print(f"  文档块数: {system_stats['chunks']}")
    
    # 执行查询
    queries = [
        "什么是混合检索？",
        "如何提高检索质量？",
        "RAG 技术有什么优势？"
    ]
    
    for query in queries:
        print("\n" + "=" * 70)
        print(f"查询: {query}")
        print("-" * 70)
        
        response = engine.query(query)
        
        print(f"\n答案:\n{response.answer}")
        
        print(f"\n来源文档 ({len(response.sources)} 个):")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. [ID: {source['id']}] 分数: {source['score']:.4f}")
            print(f"     内容: {source['content'][:60]}...")
        
        print(f"\n性能指标:")
        print(f"  检索时间: {response.retrieval_time*1000:.2f} ms")
        print(f"  重排时间: {response.rerank_time*1000:.2f} ms")
        print(f"  生成时间: {response.generation_time*1000:.2f} ms")
        print(f"  总时间: {response.total_time*1000:.2f} ms")
    
    # 演示评估
    print("\n" + "=" * 70)
    print("系统评估")
    print("-" * 70)
    
    eval_data = [
        {
            "question": "什么是向量数据库？",
            "ground_truth": "向量数据库是存储向量嵌入的数据库..."
        },
        {
            "question": "RAG 的优势是什么？",
            "ground_truth": "RAG 可以减少幻觉..."
        },
    ]
    
    eval_results = engine.evaluate(eval_data)
    print("\n评估结果:")
    for metric, score in eval_results.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
    
    # 打印使用说明
    print("""
使用说明:
=========

# 1. 创建 RAG 引擎
from advanced_rag_engine import AdvancedRAGEngine, AdvancedRAGConfig

config = AdvancedRAGConfig(
    use_hybrid_retrieval=True,
    use_reranking=True,
    chunk_size=500,
    initial_top_k=20,
    final_top_k=5
)
engine = AdvancedRAGEngine(config)

# 2. 添加文档
from hybrid_retrieval import Document

documents = [
    Document(id="doc1", content="...", metadata={"source": "file1.pdf"}),
    Document(id="doc2", content="...", metadata={"source": "file2.pdf"}),
]
engine.add_documents(documents)

# 3. 查询
response = engine.query("你的问题")
print(response.answer)
print(response.sources)

# 4. 增量更新
new_docs = [...]
engine.add_documents(new_docs, incremental=True)

# 5. 评估
eval_data = [
    {"question": "...", "ground_truth": "..."},
    ...
]
results = engine.evaluate(eval_data)
""")


if __name__ == "__main__":
    demo_advanced_rag()