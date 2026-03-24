"""
Day 11: 重排序实现

实现基于 Cross-Encoder 和 LLM 的重排序模型。
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RerankResult:
    """重排序结果"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    original_rank: int  # 原始排名


class Reranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        重排序候选文档
        
        Args:
            query: 查询文本
            candidates: 候选文档列表，每个文档包含 id, content, score, metadata
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果列表
        """
        pass


# ==================== Cross-Encoder 重排序器 ====================

class CrossEncoderReranker(Reranker):
    """
    Cross-Encoder 重排序器
    
    使用 Cross-Encoder 模型计算 query-document 对的精确相关性分数。
    与 Bi-Encoder 不同，Cross-Encoder 将 query 和 document 一起输入模型，
    可以捕获更深层的交互信息。
    
    常用模型:
    - BAAI/bge-reranker-large (推荐中文)
    - BAAI/bge-reranker-base
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (英文)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        初始化 Cross-Encoder 重排序器
        
        Args:
            model_name: 模型名称
            device: 运行设备 ("cpu" 或 "cuda")
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """加载模型（延迟加载）"""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            print(f"正在加载重排序模型: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print("模型加载完成!")
            
        except ImportError:
            raise ImportError(
                "请安装 transformers 和 torch: "
                "pip install transformers torch"
            )
    
    def _compute_score(self, query: str, document: str) -> float:
        """
        计算单个 query-document 对的相关性分数
        
        Args:
            query: 查询文本
            document: 文档文本
            
        Returns:
            相关性分数
        """
        import torch
        
        # 编码输入
        inputs = self.tokenizer(
            query,
            document,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 计算分数
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits[0].item()
        
        return score
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        使用 Cross-Encoder 重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        # 加载模型
        self._load_model()
        
        # 计算每个候选文档的分数
        scores = []
        for i, candidate in enumerate(candidates):
            score = self._compute_score(query, candidate["content"])
            scores.append((i, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        for rank, (original_idx, score) in enumerate(scores[:top_k], 1):
            candidate = candidates[original_idx]
            results.append(RerankResult(
                document_id=candidate["id"],
                content=candidate["content"],
                score=score,
                metadata=candidate.get("metadata", {}),
                original_rank=original_idx + 1
            ))
        
        return results


class MockCrossEncoderReranker(Reranker):
    """
    模拟的 Cross-Encoder 重排序器
    
    用于在没有真实模型时进行测试
    """
    
    def __init__(self):
        """初始化模拟重排序器"""
        pass
    
    def _compute_score(self, query: str, document: str) -> float:
        """
        模拟计算相关性分数
        
        基于简单的词重叠率
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & doc_words)
        return overlap / len(query_words) + np.random.uniform(-0.1, 0.1)
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """模拟重排序"""
        if not candidates:
            return []
        
        # 计算分数
        scores = []
        for i, candidate in enumerate(candidates):
            score = self._compute_score(query, candidate["content"])
            scores.append((i, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        for rank, (original_idx, score) in enumerate(scores[:top_k], 1):
            candidate = candidates[original_idx]
            results.append(RerankResult(
                document_id=candidate["id"],
                content=candidate["content"],
                score=score,
                metadata=candidate.get("metadata", {}),
                original_rank=original_idx + 1
            ))
        
        return results


# ==================== LLM 重排序器 ====================

class LLMReranker(Reranker):
    """
    基于大语言模型的重排序器
    
    使用 LLM 对候选文档进行相关性判断和重排序。
    优点：
    - 可以理解复杂的语义关系
    - 可以给出重排序理由
    缺点：
    - 成本较高
    - 速度较慢
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        初始化 LLM 重排序器
        
        Args:
            llm: 大语言模型实例
            model: 模型名称
        """
        self.llm = llm
        self.model = model
    
    def _build_prompt(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> str:
        """
        构建重排序提示词
        
        Args:
            query: 查询文本
            candidates: 候选文档
            
        Returns:
            提示词
        """
        prompt = f"""请根据查询，对以下文档按相关性进行排序。

查询: {query}

文档列表:
"""
        for i, candidate in enumerate(candidates, 1):
            prompt += f"\n[{i}] {candidate['content'][:200]}...\n"
        
        prompt += """
请输出排序后的文档编号，从最相关到最不相关，格式如: 3, 1, 4, 2, 5
只输出编号列表，不要其他内容。"""
        
        return prompt
    
    def _parse_response(
        self,
        response: str,
        num_candidates: int
    ) -> List[int]:
        """
        解析 LLM 响应
        
        Args:
            response: LLM 响应
            num_candidates: 候选文档数量
            
        Returns:
            排序后的索引列表
        """
        # 提取数字
        numbers = re.findall(r'\d+', response)
        
        # 转换为索引（从1开始）
        indices = []
        seen = set()
        for n in numbers:
            idx = int(n) - 1  # 转换为0索引
            if 0 <= idx < num_candidates and idx not in seen:
                indices.append(idx)
                seen.add(idx)
        
        # 如果解析失败，返回原始顺序
        if len(indices) < num_candidates:
            for i in range(num_candidates):
                if i not in seen:
                    indices.append(i)
        
        return indices
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        使用 LLM 重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        if not candidates:
            return []
        
        if self.llm is None:
            # 返回原始顺序
            return [
                RerankResult(
                    document_id=c["id"],
                    content=c["content"],
                    score=c.get("score", 1.0 - i * 0.1),
                    metadata=c.get("metadata", {}),
                    original_rank=i + 1
                )
                for i, c in enumerate(candidates[:top_k])
            ]
        
        # 构建提示词
        prompt = self._build_prompt(query, candidates)
        
        # 调用 LLM
        # response = self.llm.generate(prompt)
        # indices = self._parse_response(response, len(candidates))
        
        # 模拟响应（实际应用中应调用真实 LLM）
        indices = list(range(len(candidates)))
        
        # 构建结果
        results = []
        for rank, idx in enumerate(indices[:top_k], 1):
            candidate = candidates[idx]
            results.append(RerankResult(
                document_id=candidate["id"],
                content=candidate["content"],
                score=len(candidates) - rank + 1,  # 简单的分数
                metadata=candidate.get("metadata", {}),
                original_rank=idx + 1
            ))
        
        return results


# ==================== 多路重排序器 ====================

class MultiReranker(Reranker):
    """
    多路重排序器
    
    组合多个重排序器的结果
    """
    
    def __init__(
        self,
        rerankers: List[Tuple[Reranker, float]],
        aggregation: str = "weighted"
    ):
        """
        初始化多路重排序器
        
        Args:
            rerankers: 重排序器列表及其权重 [(reranker, weight), ...]
            aggregation: 聚合方法 ("weighted", "max", "avg")
        """
        self.rerankers = rerankers
        self.aggregation = aggregation
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        多路重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        if not candidates or not self.rerankers:
            return []
        
        # 收集所有重排序结果
        all_scores: Dict[str, List[float]] = {}
        doc_info: Dict[str, Dict[str, Any]] = {}
        
        for reranker, weight in self.rerankers:
            results = reranker.rerank(query, candidates, top_k=len(candidates))
            
            for i, result in enumerate(results):
                doc_id = result.document_id
                if doc_id not in all_scores:
                    all_scores[doc_id] = []
                    doc_info[doc_id] = {
                        "content": result.content,
                        "metadata": result.metadata,
                        "original_rank": result.original_rank
                    }
                
                # 根据排名计算分数
                rank_score = len(candidates) - i
                all_scores[doc_id].append(rank_score * weight)
        
        # 聚合分数
        final_scores = {}
        for doc_id, scores in all_scores.items():
            if self.aggregation == "weighted":
                final_scores[doc_id] = sum(scores)
            elif self.aggregation == "max":
                final_scores[doc_id] = max(scores)
            elif self.aggregation == "avg":
                final_scores[doc_id] = sum(scores) / len(scores)
        
        # 排序
        sorted_docs = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建结果
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1):
            info = doc_info[doc_id]
            results.append(RerankResult(
                document_id=doc_id,
                content=info["content"],
                score=score,
                metadata=info["metadata"],
                original_rank=info["original_rank"]
            ))
        
        return results


# ==================== 重排序优化策略 ====================

class RerankOptimizer:
    """
    重排序优化策略
    
    提供一些重排序相关的优化方法
    """
    
    @staticmethod
    def diversity_rerank(
        results: List[RerankResult],
        diversity_threshold: float = 0.8,
        similarity_func: Optional[callable] = None
    ) -> List[RerankResult]:
        """
        多样性重排序
        
        确保返回结果具有一定多样性，避免过于相似的内容
        
        Args:
            results: 重排序结果
            diversity_threshold: 多样性阈值
            similarity_func: 相似度计算函数
            
        Returns:
            多样化后的结果
        """
        if len(results) <= 1:
            return results
        
        if similarity_func is None:
            # 默认使用简单的词重叠
            def similarity_func(text1: str, text2: str) -> float:
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                if not words1 or not words2:
                    return 0.0
                return len(words1 & words2) / min(len(words1), len(words2))
        
        diverse_results = [results[0]]
        
        for candidate in results[1:]:
            # 检查与已选结果的相似度
            max_sim = 0.0
            for selected in diverse_results:
                sim = similarity_func(candidate.content, selected.content)
                max_sim = max(max_sim, sim)
            
            # 如果相似度低于阈值，加入结果
            if max_sim < diversity_threshold:
                diverse_results.append(candidate)
        
        return diverse_results
    
    @staticmethod
    def relevance_diversity_balance(
        results: List[RerankResult],
        alpha: float = 0.7,
        diversity_threshold: float = 0.8
    ) -> List[RerankResult]:
        """
        平衡相关性和多样性
        
        Args:
            results: 重排序结果
            alpha: 相关性权重 (1-alpha 为多样性权重)
            diversity_threshold: 多样性阈值
            
        Returns:
            平衡后的结果
        """
        if len(results) <= 1:
            return results
        
        def simple_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / min(len(words1), len(words2))
        
        # 贪心选择
        final_results = []
        remaining = list(results)
        
        # 首先选择相关性最高的
        final_results.append(remaining.pop(0))
        
        while remaining and len(final_results) < len(results):
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in remaining:
                # 相关性分数
                relevance_score = candidate.score
                
                # 多样性分数 (与已选结果的最大差异)
                min_sim = 1.0
                for selected in final_results:
                    sim = simple_similarity(candidate.content, selected.content)
                    min_sim = min(min_sim, sim)
                diversity_score = 1 - min_sim
                
                # 综合分数
                combined_score = alpha * relevance_score + (1 - alpha) * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                final_results.append(best_candidate)
                remaining.remove(best_candidate)
        
        return final_results


# ==================== 使用示例 ====================

def demo_reranking():
    """演示重排序功能"""
    
    print("=" * 60)
    print("Day 11: 重排序演示")
    print("=" * 60)
    
    # 准备候选文档
    candidates = [
        {
            "id": "doc1",
            "content": "向量数据库是一种专门用于存储和检索向量嵌入的数据库系统，支持高效的相似度搜索。",
            "score": 0.85,
            "metadata": {"category": "database"}
        },
        {
            "id": "doc2",
            "content": "RAG（检索增强生成）结合了检索和生成，首先检索相关文档，然后使用大语言模型生成回答。",
            "score": 0.82,
            "metadata": {"category": "ai"}
        },
        {
            "id": "doc3",
            "content": "BM25是一种经典的关键词检索算法，考虑了词频和逆文档频率，在信息检索领域有广泛应用。",
            "score": 0.78,
            "metadata": {"category": "algorithm"}
        },
        {
            "id": "doc4",
            "content": "混合检索结合了向量检索和关键词检索的优势，可以显著提高检索质量。",
            "score": 0.75,
            "metadata": {"category": "ai"}
        },
        {
            "id": "doc5",
            "content": "重排序是在初步检索后对结果进行精细化排序的过程，使用Cross-Encoder计算精确相关性。",
            "score": 0.72,
            "metadata": {"category": "ai"}
        },
    ]
    
    query = "什么是重排序技术？"
    
    print(f"\n查询: {query}")
    print("-" * 60)
    
    # 显示原始顺序
    print("\n原始检索结果:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. [{c['id']}] 分数: {c['score']:.2f}")
        print(f"     内容: {c['content'][:40]}...")
    
    # 使用模拟的 Cross-Encoder 重排序
    print("\n" + "=" * 60)
    print("Cross-Encoder 重排序 (模拟)")
    print("=" * 60)
    
    reranker = MockCrossEncoderReranker()
    results = reranker.rerank(query, candidates, top_k=5)
    
    print("\n重排序结果:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.document_id}] 分数: {result.score:.4f}")
        print(f"     原始排名: {result.original_rank}")
        print(f"     内容: {result.content[:40]}...")
    
    # 测试多样性重排序
    print("\n" + "=" * 60)
    print("多样性重排序")
    print("=" * 60)
    
    diverse_results = RerankOptimizer.diversity_rerank(
        results,
        diversity_threshold=0.5
    )
    
    print("\n多样化后的结果:")
    for i, result in enumerate(diverse_results, 1):
        print(f"  {i}. [{result.document_id}]")
        print(f"     内容: {result.content[:40]}...")
    
    # 测试平衡相关性和多样性
    print("\n" + "=" * 60)
    print("平衡相关性和多样性")
    print("=" * 60)
    
    balanced_results = RerankOptimizer.relevance_diversity_balance(
        results,
        alpha=0.7,
        diversity_threshold=0.5
    )
    
    print("\n平衡后的结果:")
    for i, result in enumerate(balanced_results, 1):
        print(f"  {i}. [{result.document_id}] 分数: {result.score:.4f}")
        print(f"     内容: {result.content[:40]}...")
    
    # 比较不同重排序方法
    print("\n" + "=" * 60)
    print("重排序方法对比")
    print("=" * 60)
    
    # LLM 重排序
    llm_reranker = LLMReranker()
    llm_results = llm_reranker.rerank(query, candidates, top_k=5)
    
    print("\nLLM 重排序结果:")
    for i, result in enumerate(llm_results, 1):
        print(f"  {i}. [{result.document_id}] 原始排名: {result.original_rank}")
    
    # 多路重排序
    multi_reranker = MultiReranker(
        rerankers=[
            (MockCrossEncoderReranker(), 0.7),
            (LLMReranker(), 0.3)
        ],
        aggregation="weighted"
    )
    multi_results = multi_reranker.rerank(query, candidates, top_k=5)
    
    print("\n多路重排序结果 (CrossEncoder 0.7 + LLM 0.3):")
    for i, result in enumerate(multi_results, 1):
        print(f"  {i}. [{result.document_id}] 分数: {result.score:.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    # 打印使用说明
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("""
# 使用真实 Cross-Encoder 模型
from reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="BAAI/bge-reranker-large",
    device="cuda"  # 或 "cpu"
)
results = reranker.rerank(query, candidates, top_k=5)

# 使用 LLM 重排序
from reranker import LLMReranker

reranker = LLMReranker(llm=your_llm_instance)
results = reranker.rerank(query, candidates, top_k=5)

# 多路重排序
from reranker import MultiReranker

reranker = MultiReranker(
    rerankers=[
        (CrossEncoderReranker(), 0.7),
        (LLMReranker(), 0.3)
    ]
)
results = reranker.rerank(query, candidates, top_k=5)
""")


if __name__ == "__main__":
    demo_reranking()