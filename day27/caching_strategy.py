# day27/caching_strategy.py
"""
Day 27: 性能优化 - 缓存策略

演示多级缓存系统：
1. 结果缓存 - 相同请求直接返回
2. 嵌入缓存 - 向量缓存避免重复计算
3. 会话缓存 - 上下文缓存减少重复
4. 分层缓存 - 内存 + Redis + 磁盘
"""

import time
import json
import hashlib
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps


# ============================================================
# 数据模型定义
# ============================================================

@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl: float = 3600  # 秒
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)


# ============================================================
# 结果缓存
# ============================================================

class ResultCache:
    """结果缓存 - 相同请求直接返回"""

    def __init__(self, maxsize: int = 1000, default_ttl: float = 3600):
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self.stats = CacheStats()

    def _hash_request(self, prompt: str, params: Dict) -> str:
        """生成请求唯一标识"""
        # 规范化参数
        normalized_params = json.dumps(params, sort_keys=True, ensure_ascii=False)
        content = f"{prompt}:{normalized_params}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get(self, prompt: str, params: Dict) -> Optional[Any]:
        """获取缓存结果"""
        key = self._hash_request(prompt, params)

        if key in self.cache:
            entry = self.cache[key]

            # 检查过期
            if entry.is_expired():
                del self.cache[key]
                self.stats.misses += 1
                return None

            # 更新访问信息
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            # LRU：移到末尾
            self.cache.move_to_end(key)

            self.stats.hits += 1
            return entry.value

        self.stats.misses += 1
        return None

    def set(self, prompt: str, params: Dict, result: Any, ttl: Optional[float] = None):
        """设置缓存结果"""
        key = self._hash_request(prompt, params)

        # LRU 淘汰
        if len(self.cache) >= self.maxsize:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats.evictions += 1

        # 计算大小（估算）
        size_bytes = len(json.dumps(result, ensure_ascii=False).encode())

        self.cache[key] = CacheEntry(
            key=key,
            value=result,
            ttl=ttl or self.default_ttl,
            size_bytes=size_bytes
        )

    def invalidate(self, prompt: str, params: Dict):
        """失效特定缓存"""
        key = self._hash_request(prompt, params)
        if key in self.cache:
            del self.cache[key]

    def clear_expired(self) -> int:
        """清理过期条目"""
        expired_keys = [
            k for k, v in self.cache.items() if v.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        self.stats.size = len(self.cache)
        total_size = sum(e.size_bytes for e in self.cache.values())

        return {
            "size": self.stats.size,
            "maxsize": self.maxsize,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "hit_rate": f"{self.stats.hit_rate():.2%}",
            "total_size_bytes": total_size,
            "avg_size_bytes": total_size / max(1, len(self.cache)),
            "avg_access_count": sum(e.access_count for e in self.cache.values()) / max(1, len(self.cache))
        }


# ============================================================
# 嵌入缓存
# ============================================================

class EmbeddingCache:
    """嵌入缓存 - 向量缓存避免重复计算"""

    def __init__(self, maxsize: int = 1000, similarity_threshold: float = 0.95):
        self.cache: Dict[str, List[float]] = {}
        self.maxsize = maxsize
        self.similarity_threshold = similarity_threshold
        self.stats = CacheStats()

    def _hash_text(self, text: str) -> str:
        """文本哈希"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional[List[float]]:
        """获取嵌入向量"""
        hash_key = self._hash_text(text)

        if hash_key in self.cache:
            self.stats.hits += 1
            return self.cache[hash_key]

        self.stats.misses += 1
        return None

    def set(self, text: str, embedding: List[float]):
        """设置嵌入向量"""
        hash_key = self._hash_text(text)

        if len(self.cache) >= self.maxsize:
            # 淘汰最少使用的
            oldest_key = min(self.cache.keys())
            del self.cache[oldest_key]
            self.stats.evictions += 1

        self.cache[hash_key] = embedding

    async def get_or_compute(self, text: str, compute_func) -> Tuple[List[float], bool]:
        """获取或计算嵌入（带缓存标记）"""
        cached = self.get(text)

        if cached is not None:
            return cached, True  # 从缓存获取

        # 计算新嵌入
        embedding = await compute_func(text)
        self.set(text, embedding)

        return embedding, False  # 新计算

    def find_similar(self, embedding: List[float]) -> Optional[str]:
        """查找相似缓存"""
        best_match = None
        best_similarity = 0.0

        for hash_key, cached_embedding in self.cache.items():
            similarity = self._cosine_similarity(embedding, cached_embedding)

            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = hash_key

        return best_match

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_stats(self) -> Dict:
        """获取统计"""
        self.stats.size = len(self.cache)
        return {
            "size": self.stats.size,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate():.2%}",
            "similarity_threshold": self.similarity_threshold
        }


# ============================================================
# 会话缓存
# ============================================================

class SessionCache:
    """会话缓存 - 上下文缓存减少重复"""

    def __init__(self, max_sessions: int = 1000, session_ttl: float = 3600):
        self.sessions: Dict[str, Dict] = {}
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self.stats = CacheStats()

    def create_session(self, session_id: str) -> Dict:
        """创建会话"""
        if len(self.sessions) >= self.max_sessions:
            # 淘汰最旧会话
            oldest = min(self.sessions.keys(), key=lambda k: self.sessions[k]["created_at"])
            del self.sessions[oldest]
            self.stats.evictions += 1

        self.sessions[session_id] = {
            "context": [],
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "metadata": {}
        }

        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # 检查过期
            if datetime.now() - session["created_at"] > timedelta(seconds=self.session_ttl):
                del self.sessions[session_id]
                self.stats.misses += 1
                return None

            session["last_accessed"] = datetime.now()
            self.stats.hits += 1
            return session

        self.stats.misses += 1
        return None

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """添加消息到会话"""
        session = self.get_session(session_id)

        if session is None:
            session = self.create_session(session_id)

        session["context"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        return True

    def get_context(self, session_id: str, max_messages: Optional[int] = None) -> List[Dict]:
        """获取会话上下文"""
        session = self.get_session(session_id)

        if session is None:
            return []

        context = session["context"]

        if max_messages:
            context = context[-max_messages:]

        return context

    def compress_context(self, session_id: str, max_messages: int = 10) -> bool:
        """压缩历史上下文"""
        session = self.get_session(session_id)

        if session is None:
            return False

        context = session["context"]

        if len(context) <= max_messages:
            return True

        # 保留最近消息，压缩旧消息
        old_messages = context[:-max_messages]
        compressed_summary = self._summarize_messages(old_messages)

        session["context"] = [
            {
                "role": "system",
                "content": f"[历史摘要] {compressed_summary}",
                "compressed": True,
                "original_count": len(old_messages)
            },
            *context[-max_messages:]
        ]

        return True

    def _summarize_messages(self, messages: List[Dict]) -> str:
        """摘要消息（简化版）"""
        # 实际应调用 LLM 进行摘要
        topics = []
        for msg in messages:
            content = msg.get("content", "")
            # 简化：取前50字作为主题
            topic = content[:50] + "..." if len(content) > 50 else content
            topics.append(f"{msg.get('role')}: {topic}")

        return " | ".join(topics[:3])  # 最多显示3条

    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_stats(self) -> Dict:
        """获取统计"""
        self.stats.size = len(self.sessions)

        total_messages = sum(
            len(s["context"]) for s in self.sessions.values()
        )

        avg_age_seconds = sum(
            (datetime.now() - s["created_at"]).total_seconds()
            for s in self.sessions.values()
        ) / max(1, len(self.sessions))

        return {
            "active_sessions": self.stats.size,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "total_messages_cached": total_messages,
            "avg_messages_per_session": total_messages / max(1, self.stats.size),
            "avg_session_age_seconds": avg_age_seconds
        }


# ============================================================
# 多级缓存
# ============================================================

class MultiTierCache:
    """多级缓存 - 内存 + Redis + 磁盘"""

    def __init__(self):
        # L1: 内存缓存（最快）
        self.l1_cache = ResultCache(maxsize=100, default_ttl=60)

        # L2: Redis 缓存（分布式）
        # self.l2_cache = RedisCache()  # 实际应连接 Redis

        # L3: 磁盘缓存（持久化）
        # self.l3_cache = DiskCache()  # 实际应使用 diskcache

        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0
        }

    def get(self, prompt: str, params: Dict) -> Tuple[Optional[Any], str]:
        """多级查询"""
        # L1 查询
        result = self.l1_cache.get(prompt, params)
        if result is not None:
            self.stats["l1_hits"] += 1
            return result, "L1"

        self.stats["l1_misses"] += 1

        # L2 查询（模拟）
        # result = self.l2_cache.get(prompt, params)
        # if result is not None:
        #     self.stats["l2_hits"] += 1
        #     # 回填 L1
        #     self.l1_cache.set(prompt, params, result)
        #     return result, "L2"

        # self.stats["l2_misses"] += 1

        # L3 查询（模拟）
        # result = self.l3_cache.get(prompt, params)
        # if result is not None:
        #     self.stats["l3_hits"] += 1
        #     # 回填 L1, L2
        #     self.l1_cache.set(prompt, params, result)
        #     self.l2_cache.set(prompt, params, result)
        #     return result, "L3"

        # self.stats["l3_misses"] += 1

        return None, "miss"

    def set(self, prompt: str, params: Dict, result: Any, ttl_levels: Dict = None):
        """多级设置"""
        default_ttls = {
            "l1": 60,
            "l2": 3600,
            "l3": 86400
        }

        ttl_levels = ttl_levels or default_ttls

        # L1 设置
        self.l1_cache.set(prompt, params, result, ttl_levels.get("l1"))

        # L2 设置（模拟）
        # self.l2_cache.set(prompt, params, result, ttl_levels.get("l2"))

        # L3 设置（模拟）
        # self.l3_cache.set(prompt, params, result, ttl_levels.get("l3"))

    def get_stats(self) -> Dict:
        """获取统计"""
        total_requests = sum([
            self.stats["l1_hits"] + self.stats["l1_misses"]
        ])

        l1_hit_rate = self.stats["l1_hits"] / max(1, total_requests)

        return {
            "l1_hit_rate": f"{l1_hit_rate:.2%}",
            "l1_stats": self.l1_cache.get_stats(),
            "tier_stats": self.stats,
            "recommendation": self._generate_recommendation(l1_hit_rate)
        }

    def _generate_recommendation(self, hit_rate: float) -> str:
        """生成优化建议"""
        if hit_rate > 0.8:
            return "缓存效率高，可考虑降低 TTL 提高数据新鲜度"
        elif hit_rate > 0.5:
            return "缓存效率中等，建议增加缓存容量或调整 TTL"
        else:
            return "缓存效率低，建议分析请求模式或增加语义相似缓存"


# ============================================================
# 缓存装饰器
# ============================================================

def cached(cache: ResultCache, ttl: float = 3600):
    """缓存装饰器"""

    def decorator(func):

        @wraps(func)
        async def wrapper(prompt: str, **kwargs):
            # 检查缓存
            cached_result = cache.get(prompt, kwargs)
            if cached_result is not None:
                print(f"  [缓存命中] {func.__name__}")
                return cached_result

            # 执行函数
            print(f"  [缓存未命中] 执行 {func.__name__}")
            result = await func(prompt, **kwargs)

            # 设置缓存
            cache.set(prompt, kwargs, result, ttl)

            return result

        return wrapper

    return decorator


# ============================================================
# 演示函数
# ============================================================

async def demo_result_cache():
    """演示结果缓存"""
    print("\n=== 结果缓存演示 ===")

    cache = ResultCache(maxsize=10, default_ttl=60)

    # 模拟请求
    requests = [
        {"prompt": "你好", "params": {"max_tokens": 50}},
        {"prompt": "你好", "params": {"max_tokens": 50}},  # 相同请求
        {"prompt": "请介绍一下自己", "params": {"max_tokens": 100}},
        {"prompt": "你好", "params": {"max_tokens": 100}},  # 不同参数
        {"prompt": "你好", "params": {"max_tokens": 50}},  # 再次相同
    ]

    for i, req in enumerate(requests):
        print(f"\n请求 {i+1}: '{req['prompt']}' (max_tokens={req['params']['max_tokens']})")

        result = cache.get(req["prompt"], req["params"])

        if result:
            print(f"  ✅ 缓存命中: {result[:30]}...")
        else:
            print(f"  ❌ 缓存未命中，执行推理...")
            # 模拟推理
            result = f"响应: {req['prompt']}..."
            cache.set(req["prompt"], req["params"], result)
            print(f"  ✅ 结果已缓存")

    print(f"\n缓存统计: {cache.get_stats()}")


async def demo_embedding_cache():
    """演示嵌入缓存"""
    print("\n=== 嵌入缓存演示 ===")

    cache = EmbeddingCache(maxsize=10, similarity_threshold=0.9)

    async def compute_embedding(text: str) -> List[float]:
        """模拟嵌入计算"""
        print(f"  计算嵌入向量...")
        await asyncio.sleep(0.1)  # 模拟计算延迟
        # 简单模拟：基于文本长度生成向量
        return [len(text) / 100, len(text.split()) / 10, text.count(" ") / 10]

    texts = [
        "你好，请问有什么可以帮助你？",
        "你好，请问有什么可以帮助你？",  # 完全相同
        "你好，有什么可以帮助你？",  # 相似
        "请介绍一下人工智能",
        "你好，请问有什么可以帮助你？",  # 再次相同
    ]

    for text in texts:
        print(f"\n文本: '{text}'")

        embedding, was_cached = await cache.get_or_compute(text, compute_embedding)

        if was_cached:
            print(f"  ✅ 缓存命中")
        else:
            print(f"  ❌ 新计算，已缓存")

        print(f"  嵌入向量: {embedding[:3]}")

    # 查找相似
    print("\n相似性查询:")
    test_embedding = [0.5, 0.3, 0.2]
    similar_key = cache.find_similar(test_embedding)
    print(f"  测试向量: {test_embedding}")
    print(f"  最相似缓存: {similar_key or '无'}")

    print(f"\n缓存统计: {cache.get_stats()}")


async def demo_session_cache():
    """演示会话缓存"""
    print("\n=== 会话缓存演示 ===")

    cache = SessionCache(max_sessions=5, session_ttl=3600)

    session_id = "user_001"

    # 创建并填充会话
    print(f"创建会话: {session_id}")
    cache.create_session(session_id)

    # 添加消息
    messages = [
        ("user", "你好"),
        ("assistant", "你好！有什么可以帮助你？"),
        ("user", "请介绍一下自己"),
        ("assistant", "我是一个AI助手，可以帮助你解决问题。"),
        ("user", "你能做什么？"),
        ("assistant", "我可以回答问题、提供建议、帮助分析等。")
    ]

    for role, content in messages:
        cache.add_message(session_id, role, content)
        print(f"  添加消息: [{role}] {content[:30]}...")

    # 查看上下文
    context = cache.get_context(session_id)
    print(f"\n完整上下文 ({len(context)} 条消息):")
    for msg in context:
        print(f"  [{msg['role']}] {msg['content'][:50]}...")

    # 压缩上下文
    print("\n压缩上下文（保留最近 3 条）:")
    cache.compress_context(session_id, max_messages=3)

    compressed_context = cache.get_context(session_id)
    for msg in compressed_context:
        if msg.get("compressed"):
            print(f"  [摘要] {msg['content'][:50]}...")
        else:
            print(f"  [{msg['role']}] {msg['content'][:50]}...")

    print(f"\n缓存统计: {cache.get_stats()}")


async def demo_multi_tier_cache():
    """演示多级缓存"""
    print("\n=== 多级缓存演示 ===")

    cache = MultiTierCache()

    requests = [
        {"prompt": "你好", "params": {}},
        {"prompt": "你好", "params": {}},
        {"prompt": "请介绍", "params": {}},
        {"prompt": "你好", "params": {}},
    ]

    for i, req in enumerate(requests):
        print(f"\n请求 {i+1}: '{req['prompt']}'")

        result, tier = cache.get(req["prompt"], req["params"])

        if result:
            print(f"  ✅ 缓存命中 ({tier})")
        else:
            print(f"  ❌ 缓存未命中，执行推理...")
            result = f"响应: {req['prompt']}"
            cache.set(req["prompt"], req["params"], result)
            print(f"  ✅ 结果已缓存到所有层级")

    print(f"\n多级缓存统计: {cache.get_stats()}")


import asyncio


async def main():
    """主函数"""
    print("=" * 70)
    print("Day 27: 性能优化 - 缓存策略演示")
    print("=" * 70)

    # 1. 结果缓存演示
    await demo_result_cache()

    # 2. 嵌入缓存演示
    await demo_embedding_cache()

    # 3. 会话缓存演示
    await demo_session_cache()

    # 4. 多级缓存演示
    await demo_multi_tier_cache()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())