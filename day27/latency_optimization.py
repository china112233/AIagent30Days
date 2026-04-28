# day27/latency_optimization.py
"""
Day 27: 性能优化 - 延迟优化

演示推理延迟优化技术：
1. 流式输出 - 降低首 Token 延迟
2. KV Cache - 缓存注意力计算结果
3. 模型预热 - 减少冷启动延迟
4. 批处理优化 - 并行推理提高吞吐
"""

import asyncio
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
import statistics
import hashlib


# ============================================================
# 数据模型定义
# ============================================================

@dataclass
class LatencyMetrics:
    """延迟指标"""
    ttft: float = 0.0  # Time to First Token
    total_latency: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0


# ============================================================
# 流式输出模拟
# ============================================================

class StreamingGenerator:
    """流式输出生成器"""

    def __init__(self, avg_ttft: float = 0.1, avg_token_latency: float = 0.02):
        self.avg_ttft = avg_ttft
        self.avg_token_latency = avg_token_latency

    async def generate_stream(self, prompt: str, max_tokens: int = 100) -> Dict:
        """流式生成演示"""
        print("\n=== 流式输出演示 ===")

        start_time = time.time()
        first_token_time = None
        tokens = []

        # 模拟生成过程
        for i in range(max_tokens):
            # 首个 token 延迟较高
            if i == 0:
                await asyncio.sleep(self.avg_ttft)
                first_token_time = time.time()
                print(f"[TTFT] 首个 Token 响应时间: {first_token_time - start_time:.3f}s")
            else:
                await asyncio.sleep(self.avg_token_latency)

            token = f"Token_{i}"
            tokens.append(token)

            # 实时输出（模拟）
            if i < 5:
                print(f"  生成: {token}")

        total_time = time.time() - start_time

        return {
            "ttft": first_token_time - start_time,
            "total_latency": total_time,
            "tokens_generated": len(tokens),
            "tokens_per_second": len(tokens) / total_time
        }

    async def generate_batch(self, prompt: str, max_tokens: int = 100) -> Dict:
        """批量生成对比（非流式）"""
        print("\n=== 批量输出对比 ===")

        start_time = time.time()

        # 模拟批量生成（必须等待全部完成）
        await asyncio.sleep(self.avg_ttft + max_tokens * self.avg_token_latency)

        total_time = time.time() - start_time

        return {
            "ttft": total_time,  # 批量模式 TTFT = 总延迟
            "total_latency": total_time,
            "tokens_generated": max_tokens,
            "tokens_per_second": max_tokens / total_time
        }


# ============================================================
# KV Cache 管理
# ============================================================

class KVCacheManager:
    """KV Cache 管理 - 缓存注意力计算结果"""

    def __init__(self, max_cache_size: int = 100):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_cache_size
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def _hash_context(self, context: List[str]) -> str:
        """生成上下文哈希"""
        content = "".join(context)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_cached_attention(self, session_id: str, context: List[str]) -> Optional[Any]:
        """获取缓存的注意力状态"""
        key = f"{session_id}:{self._hash_context(context)}"

        if key in self.cache:
            self.stats["hits"] += 1
            entry = self.cache[key]
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            # 移到末尾（LRU）
            self.cache.move_to_end(key)
            return entry.value

        self.stats["misses"] += 1
        return None

    def cache_attention(self, session_id: str, context: List[str], attention_state: Any):
        """缓存注意力状态"""
        key = f"{session_id}:{self._hash_context(context)}"

        # LRU 淘汰策略
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1

        self.cache[key] = CacheEntry(
            key=key,
            value=attention_state
        )

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(1, total_requests)

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": f"{hit_rate:.2%}"
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()


# ============================================================
# 模型预热
# ============================================================

class ModelPreheater:
    """模型预热 - 减少冷启动延迟"""

    def __init__(self):
        self.is_preheated = False
        self.preheat_time = 0.0
        self.warmup_prompts = [
            "你好",
            "请介绍一下自己",
            "帮我分析一下这个问题",
            "总结以下内容",
            "解释一下这个概念"
        ]

    async def preheat(self) -> Dict:
        """预热模型"""
        print("\n=== 模型预热 ===")

        if self.is_preheated:
            print("模型已预热，跳过")
            return {"status": "already_preheated"}

        start_time = time.time()
        print("开始模型预热...")

        # 模拟预热过程
        for i, prompt in enumerate(self.warmup_prompts):
            # 模拟推理（首次较慢）
            latency = 0.5 if i < 2 else 0.2  # 首次推理较慢
            await asyncio.sleep(latency)
            print(f"  预热 {i+1}/{len(self.warmup_prompts)}: {prompt[:20]}... ({latency:.2f}s)")

        self.preheat_time = time.time() - start_time
        self.is_preheated = True

        print(f"预热完成，总耗时: {self.preheat_time:.2f}s")

        return {
            "status": "preheated",
            "preheat_time": self.preheat_time,
            "warmup_count": len(self.warmup_prompts)
        }

    async def benchmark_preheated_vs_cold(self) -> Dict:
        """对比预热前后性能"""
        print("\n=== 预热前后对比 ===")

        # 冷启动延迟
        cold_latencies = []
        print("测试冷启动性能...")
        for i in range(3):
            start = time.time()
            await asyncio.sleep(0.5 + 0.1 * i)  # 模拟冷启动延迟
            cold_latencies.append(time.time() - start)

        # 预热后延迟
        warm_latencies = []
        print("测试预热后性能...")
        for i in range(3):
            start = time.time()
            await asyncio.sleep(0.2)  # 模拟预热后延迟（更快）
            warm_latencies.append(time.time() - start)

        return {
            "cold_avg": statistics.mean(cold_latencies),
            "cold_p95": max(cold_latencies),
            "warm_avg": statistics.mean(warm_latencies),
            "warm_p95": max(warm_latencies),
            "improvement": f"{(statistics.mean(cold_latencies) - statistics.mean(warm_latencies)) / statistics.mean(cold_latencies):.1%}"
        }


# ============================================================
# 批处理优化
# ============================================================

class BatchProcessor:
    """批处理优化 - 并行推理提高吞吐"""

    def __init__(self, batch_size: int = 8, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: List = []
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "single_requests": 0,
            "avg_batch_size": 0.0
        }

    async def add_request(self, prompt: str) -> Dict:
        """添加请求"""
        self.stats["total_requests"] += 1
        request = {
            "prompt": prompt,
            "timestamp": time.time(),
            "result": None
        }
        self.pending_requests.append(request)

        # 检查是否需要处理
        if len(self.pending_requests) >= self.batch_size:
            await self._process_batch()

        return request

    async def _process_batch(self):
        """处理当前批次"""
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]

        self.stats["batched_requests"] += len(batch)

        print(f"\n处理批次: {len(batch)} 个请求")

        # 模拟批量推理（比逐个处理更快）
        start_time = time.time()

        # 批量处理模拟
        await asyncio.sleep(0.3)  # 批量处理固定时间

        batch_time = time.time() - start_time

        # 为每个请求设置结果
        for req in batch:
            req["result"] = f"响应: {req['prompt'][:20]}..."
            req["batch_time"] = batch_time

        # 更新统计
        avg_size = self.stats["batched_requests"] / max(1, self.stats["total_requests"])
        self.stats["avg_batch_size"] = avg_size

    async def process_remaining(self):
        """处理剩余请求"""
        if self.pending_requests:
            self.stats["single_requests"] += len(self.pending_requests)
            await self._process_batch()

    async def compare_sequential_vs_batch(self, prompts: List[str]) -> Dict:
        """对比顺序处理和批处理"""
        print("\n=== 批处理 vs 顺序处理对比 ===")

        # 顺序处理
        print("顺序处理...")
        seq_start = time.time()
        seq_latencies = []
        for prompt in prompts:
            start = time.time()
            await asyncio.sleep(0.2)  # 模拟单独推理
            seq_latencies.append(time.time() - start)
        seq_total = time.time() - seq_start

        # 批处理
        print("批处理...")
        batch_start = time.time()

        # 分批处理
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            await asyncio.sleep(0.3)  # 批量处理更快

        batch_total = time.time() - batch_start

        return {
            "request_count": len(prompts),
            "batch_size": self.batch_size,
            "sequential_total": seq_total,
            "sequential_avg": statistics.mean(seq_latencies),
            "batch_total": batch_total,
            "batch_avg": batch_total / len(prompts),
            "throughput_improvement": f"{seq_total / batch_total:.1f}x",
            "time_saved": seq_total - batch_total
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats


# ============================================================
# 综合延迟分析
# ============================================================

class LatencyAnalyzer:
    """延迟分析器"""

    def __init__(self):
        self.latencies: List[float] = []

    def record(self, latency: float):
        """记录延迟"""
        self.latencies.append(latency)

    def analyze(self) -> Dict:
        """分析延迟分布"""
        if not self.latencies:
            return {"error": "无数据"}

        return {
            "count": len(self.latencies),
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "std": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
            "min": min(self.latencies),
            "max": max(self.latencies),
            "p50": statistics.median(self.latencies),
            "p90": self._percentile(90),
            "p95": self._percentile(95),
            "p99": self._percentile(99)
        }

    def _percentile(self, p: float) -> float:
        """计算百分位数"""
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    def generate_report(self) -> str:
        """生成分析报告"""
        analysis = self.analyze()

        report = f"""
延迟分析报告
================
样本数量: {analysis['count']}
平均延迟: {analysis['mean']:.3f}s
中位数:   {analysis['median']:.3f}s
标准差:   {analysis['std']:.3f}s
最小值:   {analysis['min']:.3f}s
最大值:   {analysis['max']:.3f}s
P90:     {analysis['p90']:.3f}s
P95:     {analysis['p95']:.3f}s
P99:     {analysis['p99']:.3f}s
"""
        return report


# ============================================================
# 演示函数
# ============================================================

async def demo_streaming():
    """演示流式输出"""
    generator = StreamingGenerator()

    # 流式生成
    stream_result = await generator.generate_stream("你好，请介绍一下自己", max_tokens=20)
    print(f"\n流式结果: TTFT={stream_result['ttft']:.3f}s, 总延迟={stream_result['total_latency']:.3f}s")

    # 批量生成对比
    batch_result = await generator.generate_batch("你好，请介绍一下自己", max_tokens=20)
    print(f"\n批量结果: TTFT={batch_result['ttft']:.3f}s, 总延迟={batch_result['total_latency']:.3f}s")

    improvement = (batch_result['ttft'] - stream_result['ttft']) / batch_result['ttft']
    print(f"\nTTFT 优化: {improvement:.1%}")


async def demo_kv_cache():
    """演示 KV Cache"""
    cache_manager = KVCacheManager(max_cache_size=10)

    print("\n模拟对话场景...")
    session_id = "session_001"

    # 第一轮对话（无缓存）
    context1 = ["用户: 你好"]
    attention1 = {"layer_0": "attention_state_1"}

    result = cache_manager.get_cached_attention(session_id, context1)
    print(f"第一次查询: {'命中' if result else '未命中'}")
    cache_manager.cache_attention(session_id, context1, attention1)

    # 第二轮对话（相同上下文）
    result = cache_manager.get_cached_attention(session_id, context1)
    print(f"第二次查询相同上下文: {'命中' if result else '未命中'}")

    # 第三轮对话（新增内容）
    context2 = ["用户: 你好", "AI: 你好！有什么可以帮助你？", "用户: 请介绍自己"]
    result = cache_manager.get_cached_attention(session_id, context2)
    print(f"第三次查询新上下文: {'命中' if result else '未命中'}")

    print(f"\n缓存统计: {cache_manager.get_stats()}")


async def demo_preheat():
    """演示模型预热"""
    preheater = ModelPreheater()

    # 预热
    await preheater.preheat()

    # 对比预热前后
    comparison = await preheater.benchmark_preheated_vs_cold()
    print(f"\n对比结果: 冷启动平均 {comparison['cold_avg']:.3f}s, 预热后平均 {comparison['warm_avg']:.3f}s")
    print(f"性能提升: {comparison['improvement']}")


async def demo_batch():
    """演示批处理"""
    processor = BatchProcessor(batch_size=5)

    prompts = [f"问题 {i}: 请解释概念{i}" for i in range(15)]

    # 对比
    comparison = await processor.compare_sequential_vs_batch(prompts)

    print(f"\n顺序处理总时间: {comparison['sequential_total']:.3f}s")
    print(f"批处理总时间: {comparison['batch_total']:.3f}s")
    print(f"吞吐量提升: {comparison['throughput_improvement']}")
    print(f"节省时间: {comparison['time_saved']:.3f}s")


async def main():
    """主函数"""
    print("=" * 70)
    print("Day 27: 性能优化 - 延迟优化演示")
    print("=" * 70)

    # 1. 流式输出演示
    await demo_streaming()

    # 2. KV Cache 演示
    await demo_kv_cache()

    # 3. 模型预热演示
    await demo_preheat()

    # 4. 批处理演示
    await demo_batch()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())