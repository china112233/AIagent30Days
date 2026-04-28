# Day 27: 性能优化

## 概述

性能优化是将 Agent 从原型转化为生产级服务的核心环节。通过延迟优化、成本控制、缓存策略和并发处理，可以实现高性能、低成本、高可用的 Agent 服务。

### 学习目标

- 掌握推理延迟优化技术
- 实现智能成本控制和预算管理
- 学习缓存策略设计和实现
- 理解并发处理和资源调度

---

## 核心概念

### 1. 性能优化架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent 性能优化架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   请求入口                                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────┐                                                │
│  │  缓存层      │  ◄── 结果缓存 / 嵌入缓存 / 会话缓存             │
│  │ (Caching)   │                                                │
│  └─────────────┘                                                │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  并发调度    │ -> │  成本控制   │ -> │  延迟优化   │        │
│  │ (Scheduler) │    │ (Cost Ctrl) │    │ (Latency)   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│      │                   │                   │                 │
│      └───────────────────┼───────────────────┘                 │
│                          │                                      │
│                          ▼                                      │
│              ┌─────────────────────┐                           │
│              │   Agent 核心处理    │                           │
│              │   • 模型推理        │                           │
│              │   • 工具执行        │                           │
│              │   • 状态管理        │                           │
│              └─────────────────────┘                           │
│                          │                                      │
│                          ▼                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│   │ 响应优化 │    │ 监控统计 │    │ 反馈调整 │                │
│   │ (Output) │    │ (Metrics)│    │ (Feedback)│               │
│   └──────────┘    └──────────┘    └──────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 性能指标体系

| 指标类别 | 指标名称 | 说明 | 目标值 |
|----------|----------|------|--------|
| 延迟指标 | TTFT (Time to First Token) | 首个 Token 响应时间 | < 500ms |
| 延迟指标 | Total Latency | 完整响应时间 | < 2s |
| 延迟指标 | P95 Latency | 95分位延迟 | < 3s |
| 成本指标 | Cost per Request | 单次请求成本 | 最小化 |
| 成本指标 | Token Efficiency | Token 利用效率 | > 80% |
| 成本指标 | Cache Hit Rate | 缓存命中率 | > 60% |
| 并发指标 | Throughput | 吞吐量（QPS） | 最大化 |
| 并发指标 | Concurrent Users | 并发用户数 | 根据需求 |
| 并发指标 | Queue Length | 队列长度 | < 100 |

### 3. 优化策略矩阵

```
┌─────────────────────────────────────────────────────────────────┐
│                    优化策略决策矩阵                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   场景                │ 延迟优化         │ 成本优化             │
│   ─────────────────────────────────────────────────────────────│
│   实时交互            │ 流式输出         │ 缓存 + 轻量模型      │
│   批量处理            │ 并行推理         │ 批量折扣             │
│   长对话              │ KV Cache        │ 压缩历史             │
│   高并发              │ 连接池          │ 请求合并             │
│   低频查询            │ 预计算          │ 按需加载             │
│                                                                 │
│   场景                │ 缓存策略         │ 并发策略             │
│   ─────────────────────────────────────────────────────────────│
│   重复查询            │ 结果缓存         │ 异步处理             │
│   语义相似            │ 嵌入缓存         │ 批量处理             │
│   会话连续            │ 会话缓存         │ 连接复用             │
│   资源受限            │ 分层缓存         │ 限流队列             │
│   数据敏感            │ 不缓存/短过期    │ 优先级调度           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装依赖

```bash
# 核心依赖
pip install fastapi uvicorn redis
pip install aiohttp httpx

# 监控和统计
pip install prometheus-client
pip install statistics numpy

# 缓存相关
pip install cachetools
pip install diskcache

# 并发控制
pip install asyncio aiofiles
pip install semiphore
```

### 配置环境变量

```bash
# .env 文件
CACHE_ENABLED=true
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=100
COST_BUDGET_DAILY=100.0
LATENCY_TARGET_MS=2000
```

---

## 练习文件说明

### 1. `latency_optimization.py` - 延迟优化

推理延迟优化技术实现：

- **流式输出**：降低首 Token 延迟
- **KV Cache**：缓存注意力计算结果
- **模型预热**：减少冷启动延迟
- **批处理优化**：并行推理提高吞吐

### 2. `cost_control.py` - 成本控制

智能成本管理系统：

- **Token 计数**：精确统计 Token 使用
- **预算管理**：设置每日/每周预算上限
- **动态定价**：根据时间/负载调整策略
- **成本报告**：详细成本分析和预警

### 3. `caching_strategy.py` - 缓存策略

多级缓存系统实现：

- **结果缓存**：相同请求直接返回
- **嵌入缓存**：向量缓存避免重复计算
- **会话缓存**：上下文缓存减少重复
- **分层缓存**：内存 + Redis + 磁盘

### 4. `concurrent_processing.py` - 并发处理

并发控制和调度系统：

- **连接池**：复用连接减少开销
- **请求队列**：排队处理高并发
- **限流策略**：防止过载
- **异步处理**：提高吞吐量

---

## 运行示例

```bash
# 1. 延迟优化演示
python latency_optimization.py

# 2. 成本控制演示
python cost_control.py

# 3. 缓存策略演示
python caching_strategy.py

# 4. 并发处理演示
python concurrent_processing.py

# 5. 综合演示
python main.py
```

---

## 延迟优化详解

### 1. 流式输出优化

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

@app.post("/v1/agent/stream")
async def stream_chat(request: AgentRequest):
    """流式输出降低 TTFT"""
    
    async def generate():
        # 快速返回首 Token
        first_token = await agent.get_first_token(request.prompt)
        yield f"data: {first_token}\n\n"
        
        # 流式生成后续内容
        for token in agent.stream_generate(request.prompt):
            yield f"data: {token}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

### 2. KV Cache 优化

```python
class KVCacheManager:
    """KV Cache 管理 - 缓存注意力计算结果"""
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.max_size = max_cache_size
    
    def get_cached_attention(self, session_id: str, prompt_hash: str):
        """获取缓存的注意力状态"""
        key = f"{session_id}:{prompt_hash}"
        return self.cache.get(key)
    
    def cache_attention(self, session_id: str, prompt_hash: str, attention_state):
        """缓存注意力状态"""
        key = f"{session_id}:{prompt_hash}"
        
        # LRU 淘汰策略
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = attention_state
```

### 3. 模型预热策略

```python
class ModelPreheater:
    """模型预热 - 减少冷启动延迟"""
    
    async def preheat(self, warmup_prompts: list):
        """预热模型"""
        print("开始模型预热...")
        
        # 使用典型请求预热
        for prompt in warmup_prompts:
            await self.agent.generate(prompt, max_tokens=10)
        
        print("模型预热完成")
        
    @app.on_event("startup")
    async def startup_preheat():
        """服务启动时预热"""
        preheater = ModelPreheater(agent)
        warmup_prompts = [
            "你好",
            "请介绍一下自己",
            "帮我分析一下"
        ]
        await preheater.preheat(warmup_prompts)
```

### 4. 批处理优化

```python
class BatchProcessor:
    """批处理优化 - 并行推理提高吞吐"""
    
    def __init__(self, batch_size: int = 8, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = []
        self.lock = asyncio.Lock()
    
    async def add_request(self, prompt: str) -> str:
        """添加请求到批次"""
        future = asyncio.Future()
        
        async with self.lock:
            self.queue.append((prompt, future))
            
            # 达到批次大小或超时则处理
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
            else:
                # 设置超时触发
                asyncio.create_task(self._timeout_process())
        
        return await future
    
    async def _process_batch(self):
        """处理当前批次"""
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        # 批量推理
        prompts = [p for p, _ in batch]
        results = await self.agent.batch_generate(prompts)
        
        # 分发结果
        for (_, future), result in zip(batch, results):
            future.set_result(result)
```

---

## 成本控制详解

### 1. Token 精确计数

```python
import tiktoken

class TokenCounter:
    """Token 精确计数"""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """计算文本 Token 数量"""
        return len(self.encoder.encode(text))
    
    def count_messages(self, messages: list) -> dict:
        """计算消息列表的 Token 数"""
        input_tokens = 0
        output_tokens = 0
        
        for msg in messages:
            if msg["role"] in ["system", "user"]:
                input_tokens += self.count_tokens(msg["content"])
            elif msg["role"] == "assistant":
                output_tokens += self.count_tokens(msg["content"])
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
```

### 2. 预算管理系统

```python
class BudgetManager:
    """预算管理系统"""
    
    def __init__(self, daily_budget: float, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold
        self.current_usage = 0.0
        self.usage_history = []
    
    def check_budget(self, estimated_cost: float) -> dict:
        """检查预算是否允许"""
        remaining = self.daily_budget - self.current_usage
        
        if remaining < estimated_cost:
            return {
                "allowed": False,
                "reason": "预算不足",
                "remaining": remaining,
                "estimated": estimated_cost
            }
        
        # 预警检查
        projected_usage = self.current_usage + estimated_cost
        if projected_usage > self.daily_budget * self.alert_threshold:
            return {
                "allowed": True,
                "warning": f"预算使用已达 {self.alert_threshold*100}%",
                "remaining": remaining
            }
        
        return {"allowed": True}
    
    def record_usage(self, actual_cost: float):
        """记录实际使用"""
        self.current_usage += actual_cost
        self.usage_history.append({
            "cost": actual_cost,
            "timestamp": datetime.now(),
            "cumulative": self.current_usage
        })
```

### 3. 成本估算与报价

```python
class CostEstimator:
    """成本估算器"""
    
    # 各模型价格（每千 Token）
    MODEL_PRICES = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """估算请求成本"""
        prices = self.MODEL_PRICES.get(model)
        if not prices:
            return 0.0
        
        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]
        
        return input_cost + output_cost
    
    def optimize_model_selection(self, task_complexity: str, budget: float) -> str:
        """根据复杂度和预算选择最优模型"""
        if task_complexity == "simple" and budget < 0.01:
            return "gpt-3.5-turbo"
        elif task_complexity == "complex" and budget > 0.1:
            return "gpt-4"
        else:
            return "claude-3-sonnet"  # 平衡选择
```

### 4. 成本报告生成

```python
class CostReporter:
    """成本报告生成"""
    
    def generate_daily_report(self) -> dict:
        """生成每日成本报告"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_cost": self.budget_manager.current_usage,
            "budget_limit": self.budget_manager.daily_budget,
            "utilization_rate": self.budget_manager.current_usage / self.budget_manager.daily_budget,
            "request_count": len(self.budget_manager.usage_history),
            "avg_cost_per_request": self.budget_manager.current_usage / max(1, len(self.budget_manager.usage_history)),
            "model_breakdown": self._get_model_breakdown(),
            "hourly_distribution": self._get_hourly_distribution(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> list:
        """生成优化建议"""
        recommendations = []
        
        # 分析缓存命中率
        if self.cache_hit_rate < 0.5:
            recommendations.append({
                "type": "caching",
                "suggestion": "提高缓存命中率，预计可节省 30% 成本",
                "priority": "high"
            })
        
        # 分析模型使用
        if self.model_stats["gpt-4"]["ratio"] > 0.8:
            recommendations.append({
                "type": "model_selection",
                "suggestion": "部分简单任务可使用更便宜的模型",
                "potential_savings": "40%"
            })
        
        return recommendations
```

---

## 缓存策略详解

### 1. 多级缓存架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    多级缓存架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   请求                                                          │
│      │                                                          │
│      ▼                                                          │
│   ┌─────────────┐   TTL: 60s    响应时间: < 1ms                │
│   │  L1 内存缓存 │   ◄── 最快访问                              │
│   │ (In-Memory) │   Size: 100MB                                │
│   └─────────────┘                                               │
│      │ Miss                                                     │
│      ▼                                                          │
│   ┌─────────────┐   TTL: 1h     响应时间: < 5ms                │
│   │  L2 Redis   │   ◄── 分布式缓存                             │
│   │ (Distributed)│  Size: 1GB                                  │
│   └─────────────┘                                               │
│      │ Miss                                                     │
│      ▼                                                          │
│   ┌─────────────┐   TTL: 24h    响应时间: < 50ms               │
│   │  L3 磁盘缓存 │   ◄── 持久化缓存                            │
│   │ (Disk)      │   Size: 10GB                                 │
│   └─────────────┘                                               │
│      │ Miss                                                     │
│      ▼                                                          │
│   ┌─────────────┐                                               │
│   │  模型推理   │   响应时间: 100-2000ms                        │
│   │ (Compute)   │                                               │
│   └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 结果缓存实现

```python
from cachetools import TTLCache
import hashlib

class ResultCache:
    """结果缓存 - 相同请求直接返回"""
    
    def __init__(self, maxsize: int = 1000, ttl: float = 3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
    
    def _hash_request(self, prompt: str, params: dict) -> str:
        """生成请求唯一标识"""
        content = f"{prompt}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, params: dict) -> Optional[str]:
        """获取缓存结果"""
        key = self._hash_request(prompt, params)
        return self.cache.get(key)
    
    def set(self, prompt: str, params: dict, result: str):
        """设置缓存结果"""
        key = self._hash_request(prompt, params)
        self.cache[key] = result
    
    def get_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "size": len(self.cache),
            "maxsize": self.cache.maxsize,
            "hit_rate": self.hit_count / max(1, self.total_requests)
        }
```

### 3. 嵌入缓存实现

```python
class EmbeddingCache:
    """嵌入缓存 - 避免重复计算向量"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.embeddings = {}  # text_hash -> embedding
        self.similarity_threshold = similarity_threshold
    
    async def get_or_compute(self, text: str, embed_func) -> list:
        """获取或计算嵌入向量"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # 直接命中
        if text_hash in self.embeddings:
            return self.embeddings[text_hash]
        
        # 计算新嵌入
        embedding = await embed_func(text)
        self.embeddings[text_hash] = embedding
        
        return embedding
    
    async def find_similar_cached(self, embedding: list) -> Optional[str]:
        """查找相似缓存"""
        for cached_hash, cached_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(embedding, cached_embedding)
            if similarity > self.similarity_threshold:
                return cached_hash
        return None
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        """计算余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b)
```

### 4. 会话缓存实现

```python
class SessionCache:
    """会话缓存 - 上下文缓存减少重复"""
    
    def __init__(self, max_sessions: int = 1000, session_ttl: int = 3600):
        self.sessions = TTLCache(maxsize=max_sessions, ttl=session_ttl)
    
    def get_context(self, session_id: str) -> list:
        """获取会话上下文"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["context"]
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "context": [],
                "created_at": datetime.now()
            }
        
        self.sessions[session_id]["context"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    def compress_context(self, session_id: str, max_messages: int = 10):
        """压缩历史上下文"""
        context = self.sessions[session_id]["context"]
        
        if len(context) > max_messages:
            # 保留最近消息，压缩旧消息
            old_messages = context[:-max_messages]
            compressed = self._summarize_messages(old_messages)
            
            self.sessions[session_id]["context"] = [
                {"role": "system", "content": compressed, "compressed": True},
                *context[-max_messages:]
            ]
```

---

## 并发处理详解

### 1. 连接池管理

```python
import aiohttp

class ConnectionPool:
    """连接池管理 - 复用连接减少开销"""
    
    def __init__(self, max_connections: int = 100):
        self.pool = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            force_close=False,
            enable_cleanup_closed=True
        )
        self.session = None
    
    async def init(self):
        """初始化连接池"""
        self.session = aiohttp.ClientSession connector=self.pool)
    
    async def request(self, url: str, method: str = "GET", **kwargs):
        """发起请求"""
        async with self.session.request(method, url, **kwargs) as response:
            return await response.json()
    
    async def close(self):
        """关闭连接池"""
        await self.session.close()
```

### 2. 请求队列与调度

```python
import asyncio
from collections import deque

class RequestQueue:
    """请求队列 - 排队处理高并发"""
    
    def __init__(self, max_size: int = 1000, workers: int = 10):
        self.queue = deque(maxlen=max_size)
        self.workers = workers
        self.active_workers = 0
        self.semaphore = asyncio.Semaphore(workers)
    
    async def enqueue(self, request: dict) -> asyncio.Future:
        """将请求加入队列"""
        future = asyncio.Future()
        
        if len(self.queue) >= self.queue.maxlen:
            future.set_exception(Exception("队列已满"))
            return future
        
        self.queue.append((request, future))
        
        # 尝试立即处理
        asyncio.create_task(self._process_next())
        
        return future
    
    async def _process_next(self):
        """处理下一个请求"""
        async with self.semaphore:
            if self.queue:
                request, future = self.queue.popleft()
                try:
                    result = await self._handle_request(request)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
    
    async def _handle_request(self, request: dict) -> dict:
        """处理请求"""
        # 实际处理逻辑
        return await agent.process(request)
```

### 3. 限流策略

```python
class RateLimiter:
    """限流策略 - 防止过载"""
    
    def __init__(self, requests_per_second: int = 100):
        self.rps = requests_per_second
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def check(self) -> bool:
        """检查是否允许请求"""
        async with self.lock:
            now = time.time()
            
            # 清理过期记录
            self.requests = [t for t in self.requests if now - t < 1.0]
            
            # 检查当前速率
            if len(self.requests) >= self.rps:
                return False
            
            self.requests.append(now)
            return True
    
    async def wait_for_slot(self) -> None:
        """等待可用槽位"""
        while not await self.check():
            await asyncio.sleep(0.01)


# 多级限流
class MultiTierRateLimiter:
    """多级限流"""
    
    def __init__(self):
        self.global_limiter = RateLimiter(1000)  # 全局
        self.user_limiters = {}  # 每用户
        self.api_limiters = {}   # 每API
    
    async def check_all(self, user_id: str, api_key: str) -> bool:
        """多级检查"""
        # 全局检查
        if not await self.global_limiter.check():
            return False
        
        # 用户检查
        if user_id not in self.user_limiters:
            self.user_limiters[user_id] = RateLimiter(50)
        if not await self.user_limiters[user_id].check():
            return False
        
        return True
```

### 4. 异步处理模式

```python
class AsyncProcessor:
    """异步处理 - 提高吞吐量"""
    
    async def process_batch(self, requests: list) -> list:
        """批量异步处理"""
        tasks = [self._process_one(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_with_timeout(self, request: dict, timeout: float = 30.0) -> dict:
        """带超时的处理"""
        try:
            result = await asyncio.wait_for(
                self._process_one(request),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return {"error": "请求超时"}
    
    async def process_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """带重试的处理"""
        for attempt in range(max_retries):
            try:
                return await self._process_one(request)
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": str(e)}
                await asyncio.sleep(2 ** attempt)  # 指数退避
```

---

## 性能监控与分析

### 1. 性能指标收集

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# 定义指标
REQUEST_COUNT = Counter('agent_requests_total', 'Total Agent requests')
REQUEST_LATENCY = Histogram('agent_latency_seconds', 'Request latency')
ACTIVE_REQUESTS = Gauge('agent_active_requests', 'Currently active requests')
CACHE_HIT_RATE = Gauge('agent_cache_hit_rate', 'Cache hit rate')
TOKEN_USAGE = Counter('agent_tokens_total', 'Total tokens used', ['type'])

class PerformanceMonitor:
    """性能监控"""
    
    def __init__(self):
        self.metrics = {
            "latency": [],
            "throughput": [],
            "errors": [],
            "cache_stats": {"hits": 0, "misses": 0}
        }
    
    @contextmanager
    def track_latency(self):
        """延迟追踪"""
        start = time.time()
        yield
        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)
        self.metrics["latency"].append(latency)
    
    def record_request(self, success: bool, tokens_used: int):
        """记录请求"""
        REQUEST_COUNT.inc()
        if not success:
            self.metrics["errors"].append(time.time())
        TOKEN_USAGE.labels(type='total').inc(tokens_used)
```

### 2. 性能分析报告

```python
class PerformanceAnalyzer:
    """性能分析"""
    
    def analyze_latency(self) -> dict:
        """延迟分析"""
        latencies = self.metrics["latency"]
        
        return {
            "mean": np.mean(latencies),
            "median": np.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies)
        }
    
    def analyze_throughput(self, time_window: int = 60) -> dict:
        """吞吐量分析"""
        now = time.time()
        recent_requests = [
            t for t in self.metrics["requests"]
            if now - t < time_window
        ]
        
        return {
            "current_qps": len(recent_requests) / time_window,
            "peak_qps": self.metrics["peak_qps"],
            "avg_qps": self.metrics["total_requests"] / self.metrics["uptime"]
        }
    
    def generate_report(self) -> dict:
        """生成完整报告"""
        return {
            "timestamp": datetime.now(),
            "latency": self.analyze_latency(),
            "throughput": self.analyze_throughput(),
            "cache_performance": self.analyze_cache(),
            "cost_analysis": self.analyze_cost(),
            "recommendations": self.generate_recommendations()
        }
```

---

## 实战案例

### 案例1：实时客服系统优化
- **问题**：TTFT 过长（2秒+）
- **方案**：流式输出 + KV Cache + 模型预热
- **效果**：TTFT 降至 200ms，用户满意度提升

### 案例2：批量数据分析成本控制
- **问题**：每日成本超预算 50%
- **方案**：缓存策略 + 模型选择优化 + 批量折扣
- **效果**：成本降低 60%，预算达标

### 案例3：高并发知识库服务
- **问题**：并发超过 500 时服务崩溃
- **方案**：连接池 + 请求队列 + 多级限流
- **效果**：稳定支持 1000+ 并发

---

## 最佳实践

### 1. 延迟优化最佳实践

| 技术 | 适用场景 | 预期效果 |
|------|----------|----------|
| 流式输出 | 实时交互 | TTFT 降低 80% |
| KV Cache | 长对话 | 重复计算减少 70% |
| 模型预热 | 冷启动频繁 | 首次请求快 90% |
| 批处理 | 高吞吐场景 | 吞吐量提升 3x |

### 2. 成本控制最佳实践

| 策略 | 节省比例 | 实施难度 |
|------|----------|----------|
| 结果缓存 | 30-50% | 低 |
| 模型选择 | 40-60% | 中 |
| Token 优化 | 20-30% | 低 |
| 批量处理 | 10-20% | 中 |

### 3. 缓存策略最佳实践

| 缓存类型 | TTL 设置 | 更新策略 |
|----------|----------|----------|
| 结果缓存 | 根据数据变化频率 | 定时刷新 |
| 嵌入缓存 | 长期（24h+） | 版本更新时 |
| 会话缓存 | 会话时长 | 滚动更新 |

### 4. 并发处理最佳实践

| 配置参数 | 推荐值 | 说明 |
|----------|--------|------|
| 连接池大小 | CPU核心数 * 2 | 平衡资源 |
| 请求队列长度 | 100-500 | 防止内存溢出 |
| Worker数量 | 10-50 | 根据服务器能力 |
| 超时时间 | 30-60s | 避免资源浪费 |

---

## 常见问题与解决方案

### 问题1：延迟波动大
**解决方案**：
- 分析延迟分布找瓶颈
- 实施预热减少冷启动
- 优化网络连接稳定性
- 使用缓存减少计算

### 问题2：成本超预算
**解决方案**：
- 设置预算告警阈值
- 实施请求级成本检查
- 分析高成本请求特征
- 优化模型使用策略

### 问题3：缓存命中率低
**解决方案**：
- 分析请求重复率
- 调整缓存 TTL
- 实施语义相似缓存
- 扩大缓存容量

### 问题4：并发处理瓶颈
**解决方案**：
- 分析资源瓶颈点
- 增加处理 Worker
- 实施异步处理
- 优化队列策略

---

## 进阶主题

### 1. 自适应优化
- **动态调整**：根据负载自动调整参数
- **智能路由**：根据请求特征选择最优路径
- **预测性优化**：基于历史预测负载

### 2. 分布式性能优化
- **跨区域部署**：就近访问减少延迟
- **智能调度**：全局资源最优分配
- **一致性缓存**：分布式缓存同步

### 3. 绿色计算
- **能效优化**：降低计算能耗
- **资源回收**：合理释放资源
- **碳足迹追踪**：计算碳排放

---

## 工具和资源

### 推荐工具
- **Prometheus**：监控系统
- **Grafana**：可视化仪表板
- **Redis**：分布式缓存
- **tiktoken**：Token 计数
- **aiohttp**：异步 HTTP 客户端

### 学习资源
- **LLM 推理优化论文**：Flash Attention 等
- **缓存算法研究**：LRU、LFU 等
- **并发编程指南**：asyncio 最佳实践
- **成本优化案例**：各大厂实践经验

---

## 总结

性能优化是 Agent 生产服务的核心竞争力。通过本天的学习，你应该能够：

✅ 掌握多种延迟优化技术  
✅ 实现智能成本控制系统  
✅ 设计高效缓存策略  
✅ 处理高并发场景  
✅ 监控和分析性能指标  
✅ 持续优化 Agent 服务  

性能优化是一个持续迭代的过程，需要在实践中不断调整和改进。