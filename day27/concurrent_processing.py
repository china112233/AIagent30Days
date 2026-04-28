# day27/concurrent_processing.py
"""
Day 27: 性能优化 - 并发处理

演示并发控制和调度系统：
1. 连接池管理
2. 请求队列与调度
3. 限流策略
4. 异步处理模式
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
import random


# ============================================================
# 数据模型定义
# ============================================================

class Priority(Enum):
    """优先级"""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Request:
    """请求"""
    id: str
    prompt: str
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    params: Dict = field(default_factory=dict)
    result: Optional[Any] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None


@dataclass
class WorkerStats:
    """Worker 统计"""
    processed: int = 0
    failed: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0


# ============================================================
# 连接池管理
# ============================================================

class ConnectionPool:
    """连接池管理 - 复用连接减少开销"""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active_connections = 0
        self.idle_connections = deque()
        self.connection_history: List[Dict] = []
        self.stats = {
            "total_borrowed": 0,
            "total_returned": 0,
            "max_concurrent": 0,
            "avg_wait_time": 0.0
        }

    async def borrow(self) -> str:
        """借用连接"""
        start_time = time.time()

        # 等待可用连接
        while self.active_connections >= self.max_connections:
            await asyncio.sleep(0.01)

        # 获取连接
        if self.idle_connections:
            connection = self.idle_connections.popleft()
            reuse = True
        else:
            connection = f"conn_{self.stats['total_borrowed']}"
            reuse = False

        self.active_connections += 1
        self.stats["total_borrowed"] += 1
        self.stats["max_concurrent"] = max(
            self.stats["max_concurrent"],
            self.active_connections
        )

        wait_time = time.time() - start_time
        self.connection_history.append({
            "connection": connection,
            "action": "borrow",
            "reuse": reuse,
            "wait_time": wait_time,
            "timestamp": datetime.now()
        })

        return connection

    async def release(self, connection: str):
        """归还连接"""
        self.idle_connections.append(connection)
        self.active_connections -= 1
        self.stats["total_returned"] += 1

        self.connection_history.append({
            "connection": connection,
            "action": "release",
            "timestamp": datetime.now()
        })

    def get_stats(self) -> Dict:
        """获取统计"""
        avg_wait = sum(
            h["wait_time"] for h in self.connection_history
            if h["action"] == "borrow"
        ) / max(1, self.stats["total_borrowed"])

        reuse_rate = sum(
            1 for h in self.connection_history
            if h.get("reuse", False)
        ) / max(1, self.stats["total_borrowed"])

        return {
            "max_connections": self.max_connections,
            "active_connections": self.active_connections,
            "idle_connections": len(self.idle_connections),
            "total_borrowed": self.stats["total_borrowed"],
            "reuse_rate": f"{reuse_rate:.2%}",
            "avg_wait_time": avg_wait,
            "max_concurrent": self.stats["max_concurrent"]
        }


# ============================================================
# 请求队列
# ============================================================

class RequestQueue:
    """请求队列 - 排队处理高并发"""

    def __init__(self, max_size: int = 100, num_workers: int = 5):
        self.max_size = max_size
        self.num_workers = num_workers

        # 按优先级的队列
        self.high_priority_queue: deque = deque()
        self.normal_priority_queue: deque = deque()
        self.low_priority_queue: deque = deque()

        self.pending_futures: Dict[str, asyncio.Future] = {}

        self.semaphore = asyncio.Semaphore(num_workers)
        self.active_workers = 0

        self.stats = {
            "total_enqueued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "avg_wait_time": 0.0,
            "avg_process_time": 0.0
        }

        self.process_history: List[Dict] = []

    async def enqueue(self, request: Request) -> asyncio.Future:
        """入队"""
        if self.stats["total_enqueued"] >= self.max_size:
            raise Exception("队列已满")

        future = asyncio.Future()
        self.pending_futures[request.id] = future

        # 根据优先级入队
        if request.priority == Priority.HIGH:
            self.high_priority_queue.append(request)
        elif request.priority == Priority.NORMAL:
            self.normal_priority_queue.append(request)
        else:
            self.low_priority_queue.append(request)

        self.stats["total_enqueued"] += 1

        # 尝试处理
        asyncio.create_task(self._try_process())

        return future

    async def _try_process(self):
        """尝试处理"""
        async with self.semaphore:
            request = self._get_next_request()

            if request is None:
                return

            self.active_workers += 1

            start_time = time.time()
            request.status = "processing"

            try:
                # 模拟处理
                process_time = random.uniform(0.1, 0.5)
                await asyncio.sleep(process_time)

                request.result = f"响应: {request.prompt}"
                request.status = "completed"

                wait_time = (datetime.now() - request.created_at).total_seconds()

                # 记录统计
                self.stats["total_processed"] += 1
                self.process_history.append({
                    "request_id": request.id,
                    "wait_time": wait_time,
                    "process_time": process_time,
                    "priority": request.priority.name,
                    "success": True
                })

                # 完成 Future
                if request.id in self.pending_futures:
                    self.pending_futures[request.id].set_result(request)
                    del self.pending_futures[request.id]

            except Exception as e:
                request.status = "failed"
                request.error = str(e)

                self.stats["total_failed"] += 1

                if request.id in self.pending_futures:
                    self.pending_futures[request.id].set_exception(e)
                    del self.pending_futures[request.id]

            finally:
                self.active_workers -= 1

    def _get_next_request(self) -> Optional[Request]:
        """获取下一个请求（优先级优先）"""
        if self.high_priority_queue:
            return self.high_priority_queue.popleft()
        elif self.normal_priority_queue:
            return self.normal_priority_queue.popleft()
        elif self.low_priority_queue:
            return self.low_priority_queue.popleft()

        return None

    def get_queue_length(self) -> Dict:
        """获取队列长度"""
        return {
            "high": len(self.high_priority_queue),
            "normal": len(self.normal_priority_queue),
            "low": len(self.low_priority_queue),
            "total": len(self.high_priority_queue) + len(self.normal_priority_queue) + len(self.low_priority_queue)
        }

    def get_stats(self) -> Dict:
        """获取统计"""
        avg_wait = sum(h["wait_time"] for h in self.process_history) / max(1, len(self.process_history))
        avg_process = sum(h["process_time"] for h in self.process_history) / max(1, len(self.process_history))

        success_rate = self.stats["total_processed"] / max(1, self.stats["total_enqueued"])

        return {
            "max_size": self.max_size,
            "num_workers": self.num_workers,
            "active_workers": self.active_workers,
            "queue_length": self.get_queue_length(),
            "total_enqueued": self.stats["total_enqueued"],
            "total_processed": self.stats["total_processed"],
            "total_failed": self.stats["total_failed"],
            "success_rate": f"{success_rate:.2%}",
            "avg_wait_time": avg_wait,
            "avg_process_time": avg_process
        }


# ============================================================
# 限流策略
# ============================================================

class RateLimiter:
    """限流策略"""

    def __init__(self, requests_per_second: int = 10):
        self.rps = requests_per_second
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
        self.stats = {
            "total_requests": 0,
            "allowed": 0,
            "rejected": 0,
            "wait_time": 0.0
        }

    async def check(self) -> Tuple[bool, float]:
        """检查是否允许请求"""
        async with self.lock:
            now = time.time()

            # 清理过期记录
            self.requests = [t for t in self.requests if now - t < 1.0]

            self.stats["total_requests"] += 1

            # 检查当前速率
            if len(self.requests) >= self.rps:
                self.stats["rejected"] += 1

                # 计算需要等待的时间
                oldest = self.requests[0]
                wait_time = 1.0 - (now - oldest)

                return False, wait_time

            self.requests.append(now)
            self.stats["allowed"] += 1

            return True, 0.0

    async def wait_for_slot(self) -> float:
        """等待可用槽位"""
        total_wait = 0.0

        while True:
            allowed, wait_time = await self.check()

            if allowed:
                return total_wait

            total_wait += wait_time
            await asyncio.sleep(wait_time)

        return total_wait


class TokenBucketLimiter:
    """令牌桶限流"""

    def __init__(self, rate: float = 10.0, capacity: int = 20):
        self.rate = rate  # 令牌生成速率
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens_needed: int = 1) -> bool:
        """获取令牌"""
        async with self.lock:
            now = time.time()

            # 补充令牌
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # 检查是否有足够令牌
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True

            return False

    async def wait_for_tokens(self, tokens_needed: int = 1) -> float:
        """等待足够令牌"""
        wait_time = 0.0

        while True:
            if await self.acquire(tokens_needed):
                return wait_time

            # 计算需要等待的时间
            tokens_needed_remaining = tokens_needed - self.tokens
            wait = tokens_needed_remaining / self.rate

            wait_time += wait
            await asyncio.sleep(wait)


class MultiTierRateLimiter:
    """多级限流"""

    def __init__(self):
        self.global_limiter = RateLimiter(100)  # 全局
        self.user_limiters: Dict[str, RateLimiter] = {}  # 每用户
        self.api_limiters: Dict[str, RateLimiter] = {}  # 每API

    async def check_all(self, user_id: str, api_key: str) -> Tuple[bool, str]:
        """多级检查"""
        # 全局检查
        allowed, wait = await self.global_limiter.check()
        if not allowed:
            return False, "global_limit"

        # 用户检查
        if user_id not in self.user_limiters:
            self.user_limiters[user_id] = RateLimiter(20)

        allowed, wait = await self.user_limiters[user_id].check()
        if not allowed:
            return False, "user_limit"

        return True, "allowed"


# ============================================================
# 异步处理
# ============================================================

class AsyncProcessor:
    """异步处理"""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            "processed": 0,
            "failed": 0,
            "total_time": 0.0
        }

    async def process_batch(self, requests: List[Request]) -> List[Request]:
        """批量异步处理"""
        tasks = [self._process_one(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in results:
            if isinstance(result, Exception):
                failed_req = Request(id="failed", prompt="", status="failed")
                failed_req.error = str(result)
                processed.append(failed_req)
                self.stats["failed"] += 1
            else:
                processed.append(result)
                self.stats["processed"] += 1

        return processed

    async def _process_one(self, request: Request) -> Request:
        """处理单个请求"""
        async with self.semaphore:
            start = time.time()

            # 模拟处理
            await asyncio.sleep(random.uniform(0.1, 0.3))

            request.result = f"响应: {request.prompt}"
            request.status = "completed"

            self.stats["total_time"] += time.time() - start

            return request

    async def process_with_timeout(self, request: Request, timeout: float = 5.0) -> Request:
        """带超时的处理"""
        try:
            result = await asyncio.wait_for(
                self._process_one(request),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            request.status = "failed"
            request.error = "超时"
            return request

    async def process_with_retry(
        self,
        request: Request,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Request:
        """带重试的处理"""
        for attempt in range(max_retries):
            try:
                result = await self._process_one(request)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    request.status = "failed"
                    request.error = str(e)
                    return request

                # 指数退避
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)

        return request


# ============================================================
# 演示函数
# ============================================================

async def demo_connection_pool():
    """演示连接池"""
    print("\n=== 连接池演示 ===")

    pool = ConnectionPool(max_connections=5)

    # 模拟并发请求
    async def make_request(req_id: int):
        conn = await pool.borrow()
        print(f"  请求 {req_id}: 获取连接 {conn}")
        await asyncio.sleep(0.2)  # 模拟使用
        await pool.release(conn)
        print(f"  请求 {req_id}: 释放连接 {conn}")

    # 同时发起 10 个请求
    tasks = [make_request(i) for i in range(10)]
    await asyncio.gather(*tasks)

    print(f"\n连接池统计: {pool.get_stats()}")


async def demo_request_queue():
    """演示请求队列"""
    print("\n=== 请求队列演示 ===")

    queue = RequestQueue(max_size=50, num_workers=3)

    # 创建不同优先级的请求
    requests = [
        Request(id="req_1", prompt="高优先级任务", priority=Priority.HIGH),
        Request(id="req_2", prompt="普通任务", priority=Priority.NORMAL),
        Request(id="req_3", prompt="低优先级任务", priority=Priority.LOW),
        Request(id="req_4", prompt="高优先级任务2", priority=Priority.HIGH),
        Request(id="req_5", prompt="普通任务2", priority=Priority.NORMAL),
    ]

    # 入队
    futures = []
    for req in requests:
        print(f"入队: {req.id} ({req.priority.name})")
        future = await queue.enqueue(req)
        futures.append(future)

    # 等待完成
    results = await asyncio.gather(*futures)

    print("\n处理结果:")
    for result in results:
        print(f"  {result.id}: {result.status}")

    print(f"\n队列统计: {queue.get_stats()}")


async def demo_rate_limiter():
    """演示限流"""
    print("\n=== 限流演示 ===")

    limiter = RateLimiter(requests_per_second=5)

    # 快速发起 10 个请求
    results = []
    for i in range(10):
        allowed, wait_time = await limiter.check()
        results.append((i, allowed, wait_time))
        if allowed:
            print(f"  请求 {i}: ✅ 允许")
        else:
            print(f"  请求 {i}: ❌ 拒绝 (等待 {wait_time:.2f}s)")

        await asyncio.sleep(0.1)

    allowed_count = sum(1 for _, a, _ in results if a)
    print(f"\n允许: {allowed_count}/10")

    # 令牌桶演示
    print("\n令牌桶限流:")
    token_limiter = TokenBucketLimiter(rate=5.0, capacity=10)

    for i in range(15):
        acquired = await token_limiter.acquire()
        print(f"  请求 {i}: {'✅ 获取令牌' if acquired else '❌ 令牌不足'} (剩余: {token_limiter.tokens:.1f})")


async def demo_async_processor():
    """演示异步处理"""
    print("\n=== 异步处理演示 ===")

    processor = AsyncProcessor(max_concurrent=5)

    # 创建请求
    requests = [
        Request(id=f"req_{i}", prompt=f"任务{i}")
        for i in range(10)
    ]

    # 批量处理
    print("批量异步处理 10 个请求:")
    start_time = time.time()
    results = await processor.process_batch(requests)
    batch_time = time.time() - start_time

    print(f"  总耗时: {batch_time:.2f}s")
    print(f"  成功: {sum(1 for r in results if r.status == 'completed')}")

    # 超时处理演示
    print("\n超时处理:")
    slow_request = Request(id="slow", prompt="慢任务")
    result = await processor.process_with_timeout(slow_request, timeout=0.05)
    print(f"  结果: {result.status} ({result.error or '完成'})")

    # 重试演示
    print("\n重试处理:")
    retry_request = Request(id="retry", prompt="重试任务")
    result = await processor.process_with_retry(retry_request, max_retries=3)
    print(f"  结果: {result.status}")

    print(f"\n处理统计: {processor.stats}")


async def demo_full_pipeline():
    """完整流程演示"""
    print("\n=== 完整并发处理流程 ===")

    # 创建组件
    pool = ConnectionPool(max_connections=10)
    queue = RequestQueue(max_size=100, num_workers=5)
    limiter = RateLimiter(requests_per_second=20)
    processor = AsyncProcessor(max_concurrent=10)

    # 模拟完整请求流程
    async def full_request(req_id: int, priority: Priority):
        # 限流检查
        allowed, wait_time = await limiter.check()

        if not allowed:
            print(f"请求 {req_id}: 被限流")
            return

        # 获取连接
        conn = await pool.borrow()

        # 创建请求
        request = Request(id=f"req_{req_id}", prompt=f"任务{req_id}", priority=priority)

        # 入队
        future = await queue.enqueue(request)

        # 等待处理
        result = await future

        # 归还连接
        await pool.release(conn)

        print(f"请求 {req_id}: 完成 ({result.status})")

    # 发起多个请求
    tasks = [
        full_request(i, Priority.HIGH if i % 3 == 0 else Priority.NORMAL)
        for i in range(15)
    ]

    start_time = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    print(f"\n总耗时: {total_time:.2f}s")
    print(f"\n连接池: {pool.get_stats()}")
    print(f"\n队列: {queue.get_stats()}")


async def main():
    """主函数"""
    print("=" * 70)
    print("Day 27: 性能优化 - 并发处理演示")
    print("=" * 70)

    # 1. 连接池演示
    await demo_connection_pool()

    # 2. 请求队列演示
    await demo_request_queue()

    # 3. 限流演示
    await demo_rate_limiter()

    # 4. 异步处理演示
    await demo_async_processor()

    # 5. 完整流程演示
    await demo_full_pipeline()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())