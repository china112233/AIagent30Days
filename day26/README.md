# Day 26: Agent 部署

## 概述

Agent 部署是将开发完成的智能代理从原型转化为生产级服务的关键环节。通过合理的 API 设计、容器化和服务编排，可以实现高可用、可扩展的 Agent 服务。

### 学习目标

- 掌握 Agent API 设计的最佳实践
- 使用 FastAPI 构建高性能 Agent 服务
- 实现容器化部署（Docker）
- 学习服务编排和负载均衡策略

---

## 核心概念

### 1. Agent 部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent 生产部署架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   用户请求                                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────┐                                                │
│  │  负载均衡    │  ◄── Nginx / Kong / AWS ALB                   │
│  │ (Balancer)  │                                                │
│  └─────────────┘                                                │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Agent API 1 │    │ Agent API 2 │    │ Agent API 3 │        │
│  │ (FastAPI)   │    │ (FastAPI)   │    │ (FastAPI)   │        │
│  │  容器化     │    │  容器化     │    │  容器化     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│      │                   │                   │                │
│      └───────────────────┼───────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│              ┌─────────────────────┐                           │
│              │   Agent 核心服务    │                           │
│              │   • 模型推理        │                           │
│              │   • 工具执行        │                           │
│              │   • 状态管理        │                           │
│              └─────────────────────┘                           │
│                          │                                      │
│                          ▼                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│   │ Redis    │    │ Postgres │    │ 监控系统 │                │
│   │ (缓存)   │    │ (存储)   │    │ (监控)   │                │
│   └──────────┘    └──────────┘    └──────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. API 设计原则

| 原则 | 说明 | 实践方式 |
|------|------|----------|
| RESTful | 符合 REST 规范 | 使用标准 HTTP 方法 |
| 版本控制 | 支持 API 版本管理 | URL 或 Header 版本化 |
| 异步优先 | 处理长时间任务 | 异步任务 + 任务队列 |
| 错误处理 | 标准化错误响应 | 统一错误码和消息格式 |
| 安全认证 | 多种认证方式 | JWT、API Key、OAuth |
| 文档完善 | 自动生成文档 | Swagger/OpenAPI |

### 3. 容器化优势

- **一致性**：开发、测试、生产环境统一
- **可移植性**：跨平台部署无障碍
- **可扩展性**：轻松实现水平扩展
- **隔离性**：服务间资源隔离
- **版本管理**：镜像版本化管理

---

## 快速开始

### 安装依赖

```bash
# FastAPI 和相关组件
pip install fastapi uvicorn pydantic python-multipart

# 异步任务队列
pip install celery redis

# Docker 相关（需要安装 Docker Desktop）
# https://www.docker.com/products/docker-desktop

# 监控和日志
pip install prometheus-client loguru
```

### 配置环境变量

```bash
# .env 文件
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/db
```

---

## 练习文件说明

### 1. `api_design.py` - API 设计

Agent API 设计最佳实践：

- **请求模型**：Pydantic 数据验证
- **响应模型**：标准化响应格式
- **异步处理**：长时间任务处理
- **错误处理**：统一错误响应

### 2. `fastapi_agent.py` - FastAPI Agent

完整的 FastAPI Agent 服务：

- **Agent 封装**：将 Agent 包装为 API
- **路由设计**：API 路径规划
- **中间件**：认证、日志、限流
- **生命周期**：服务启动和关闭

### 3. `docker_config.py` - 容器化配置

Docker 容器化实践：

- **镜像构建**：多阶段构建优化
- **环境配置**：容器环境变量
- **健康检查**：容器健康状态
- **资源限制**：CPU、内存限制

### 4. `service_orchestration.py` - 服务编排

多服务编排管理：

- **服务发现**：服务注册与发现
- **负载均衡**：请求分发策略
- **配置管理**：集中配置管理
- **监控集成**：Prometheus 监控

### 5. `Dockerfile` - Docker 镜像配置

生产级 Docker 镜像构建。

### 6. `docker-compose.yml` - 服务编排配置

多服务容器编排配置。

---

## 运行示例

```bash
# 1. API 设计演示
python api_design.py

# 2. 启动 FastAPI Agent 服务
python fastapi_agent.py
# 或使用 uvicorn
uvicorn fastapi_agent:app --host 0.0.0.0 --port 8000

# 3. Docker 容器构建
docker build -t agent-api:latest .
docker run -p 8000:8000 agent-api:latest

# 4. Docker Compose 启动
docker-compose up -d

# 5. 服务编排演示
python service_orchestration.py
```

---

## API 设计详解

### 1. RESTful API 设计

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# 请求模型
class AgentRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    tools: Optional[list] = None
    max_tokens: int = 256

# 响应模型
class AgentResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int
    tool_calls: Optional[list] = None
    status: str = "success"

# API 端点
@app.post("/v1/agent/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """与 Agent 进行对话"""
    # 处理请求...
    return AgentResponse(...)
```

### 2. 异步任务处理

```python
from celery import Celery

celery_app = Celery('agent_tasks', broker='redis://localhost:6379')

@app.post("/v1/agent/task")
async def create_task(request: AgentRequest):
    """创建异步任务"""
    task = process_agent_task.delay(request.dict())
    return {"task_id": task.id, "status": "pending"}

@app.get("/v1/agent/task/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = process_agent_task.AsyncResult(task_id)
    return {"task_id": task_id, "status": task.status}
```

### 3. 错误处理标准

```python
class ErrorResponse(BaseModel):
    code: str          # 错误码: "E001", "E002"
    message: str       # 错误消息
    details: dict      # 详细信息
    timestamp: str     # 时间戳

# 错误码定义
ERROR_CODES = {
    "E001": "Invalid request parameters",
    "E002": "Agent processing error",
    "E003": "Session not found",
    "E004": "Rate limit exceeded",
    "E005": "Authentication failed",
}
```

---

## FastAPI Agent 详解

### 1. Agent 服务封装

```python
class AgentService:
    """Agent 服务封装"""
    
    def __init__(self, model_path: str, tools: list):
        self.agent = Agent(model_path, tools)
        self.sessions = {}  # 会话管理
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """处理 Agent 请求"""
        # 获取或创建会话
        session = self._get_session(request.session_id)
        
        # 执行 Agent
        result = await self.agent.run(
            prompt=request.prompt,
            session=session,
            tools=request.tools
        )
        
        return AgentResponse(
            response=result.response,
            session_id=session.id,
            tokens_used=result.tokens,
            tool_calls=result.tool_calls
        )
```

### 2. 生命周期管理

```python
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化"""
    # 初始化 Agent
    agent_service = AgentService(config.model_path, config.tools)
    # 连接数据库
    await database.connect()
    # 初始化缓存
    cache.init_redis(config.redis_url)

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时清理"""
    # 关闭数据库连接
    await database.disconnect()
    # 清理缓存
    await cache.close()
```

### 3. 中间件配置

```python
from fastapi.middleware.cors import CORSMiddleware

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求限流中间件
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # 检查请求频率
    if is_rate_limited(request):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )
    return await call_next(request)
```

---

## 容器化详解

### 1. Dockerfile 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 生产阶段
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1

# 资源限制
CMD ["uvicorn", "fastapi_agent:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose 服务编排

```yaml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - agent-api
```

### 3. 健康检查配置

```python
@app.get("/health")
async def health_check():
    """健康检查端点"""
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "agent": await check_agent(),
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }
```

---

## 服务编排详解

### 1. 负载均衡策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    负载均衡策略                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   策略           │ 说明                  │ 适用场景             │
│   ─────────────────────────────────────────────────────────────│
│   Round Robin    │ 按顺序轮询            │ 服务能力相近         │
│   Weighted       │ 按权重分配            │ 服务能力不同         │
│   Least Conn     │ 最少连接优先          │ 长连接场景           │
│   IP Hash        │ 按 IP 固定分配        │ 会话保持需求         │
│   Random         │ 随机选择              │ 简单场景             │
│                                                                 │
│   Agent 服务建议：                                               │
│   • 无状态 Agent：Round Robin                                   │
│   • 有状态 Agent：IP Hash 或 Least Conn                        │
│   • 差异化性能：Weighted                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 服务发现

```python
class ServiceRegistry:
    """服务注册中心"""
    
    def __init__(self):
        self.services = {}  # 服务列表
    
    def register(self, service_name: str, address: str, port: int):
        """注册服务"""
        self.services[service_name] = {
            "address": address,
            "port": port,
            "status": "healthy",
            "last_check": datetime.now()
        }
    
    def discover(self, service_name: str) -> dict:
        """发现服务"""
        return self.services.get(service_name)
    
    def heartbeat(self, service_name: str):
        """心跳检测"""
        if service_name in self.services:
            self.services[service_name]["last_check"] = datetime.now()
```

### 3. 配置管理

```python
class ConfigManager:
    """集中配置管理"""
    
    def __init__(self, config_source: str):
        self.source = config_source
        self.configs = {}
    
    def load_config(self) -> dict:
        """加载配置"""
        # 从环境变量、文件或远程加载
        return {
            "api": {
                "host": os.getenv("API_HOST", "0.0.0.0"),
                "port": int(os.getenv("API_PORT", "8000")),
            },
            "agent": {
                "model_path": os.getenv("MODEL_PATH"),
                "max_tokens": int(os.getenv("MAX_TOKENS", "256")),
            }
        }
    
    def watch_config(self):
        """监听配置变化"""
        # 实现配置热更新
        pass
```

---

## 监控集成

### 1. Prometheus 监控

```python
from prometheus_client import Counter, Histogram, generate_latest

# 定义指标
REQUEST_COUNT = Counter('agent_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('agent_request_latency_seconds', 'Request latency')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()
    
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return Response(content=generate_latest(), media_type="text/plain")
```

### 2. 日志管理

```python
from loguru import logger

# 配置日志
logger.add(
    "logs/agent_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)

@app.middleware("http")
async def logging_middleware(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response
```

---

## 实战案例

### 案例1：智能客服 Agent 服务
- **需求**：高并发、低延迟、会话保持
- **方案**：FastAPI + Redis + 多实例负载均衡
- **部署**：Docker Compose + Nginx

### 案例2：数据分析 Agent 服务
- **需求**：长时间任务、异步处理、进度跟踪
- **方案**：FastAPI + Celery + 任务队列
- **部署**：Docker + Redis + 多 Worker

### 案例3：多模型 Agent 服务
- **需求**：多模型支持、动态切换
- **方案**：服务发现 + 配置管理 + 负载均衡
- **部署**：Kubernetes + Helm

---

## 最佳实践

### 1. API 设计最佳实践

| 方面 | 建议 |
|------|------|
| URL 设计 | 使用复数名词，如 `/agents`, `/sessions` |
| 版本控制 | URL 版本化 `/v1/agent`, `/v2/agent` |
| 分页 | 支持 `page` 和 `limit` 参数 |
| 过滤 | 支持查询参数过滤 |
| 响应 | 包含状态码、消息、数据 |

### 2. 容器化最佳实践

| 方面 | 建议 |
|------|------|
| 镜像大小 | 多阶段构建，使用 slim 镜像 |
| 安全性 | 不使用 root 用户运行 |
| 健康检查 | 配置 HEALTHCHECK |
| 资源限制 | 设置 CPU 和内存限制 |
| 日志 | 日志输出到 stdout/stderr |

### 3. 服务编排最佳实践

| 方面 | 建议 |
|------|------|
| 服务发现 | 使用注册中心或 DNS |
| 配置分离 | 环境变量或配置中心 |
| 优雅关闭 | 处理 SIGTERM 信号 |
| 健康检查 | 应用层 + 容器层双重检查 |
| 监控 | Prometheus + Grafana |

---

## 常见问题与解决方案

### 问题1：API 响应慢
**解决方案**：
- 使用异步处理（asyncio）
- 添加缓存层（Redis）
- 实现请求批处理
- 优化模型推理

### 问题2：容器启动失败
**解决方案**：
- 检查环境变量配置
- 验证依赖安装
- 检查端口冲突
- 查看容器日志

### 问题3：服务不可用
**解决方案**：
- 配置健康检查
- 实现自动重启
- 设置服务备份
- 监控告警机制

### 问题4：资源不足
**解决方案**：
- 设置资源限制
- 实现请求限流
- 使用队列削峰
- 优化内存使用

---

## 进阶主题

### 1. Kubernetes 部署
- **Deployment**：声明式部署配置
- **Service**：服务发现和负载均衡
- **Ingress**：外部访问路由
- **ConfigMap/Secret**：配置和密钥管理
- **HPA**：自动扩缩容

### 2. 服务网格
- **Istio**：流量管理、安全、监控
- **Envoy**：高性能代理
- **mTLS**：服务间加密通信

### 3. CI/CD 集成
- **GitHub Actions**：自动化构建和部署
- **GitLab CI**：持续集成流水线
- **ArgoCD**：GitOps 部署工具

---

## 工具和资源

### 推荐工具
- **FastAPI**：高性能 Web 框架
- **Uvicorn**：ASGI 服务器
- **Docker**：容器化平台
- **Nginx**：负载均衡和反向代理
- **Prometheus**：监控系统
- **Grafana**：可视化仪表板

### 学习资源
- **FastAPI 官方文档**：https://fastapi.tiangolo.com/
- **Docker 官方文档**：https://docs.docker.com/
- **Kubernetes 官方文档**：https://kubernetes.io/docs/

---

## 总结

Agent 部署是将智能代理转化为生产服务的核心技术。通过本天的学习，你应该能够：

✅ 设计符合最佳实践的 Agent API  
✅ 使用 FastAPI 构建 Agent 服务  
✅ 实现容器化部署  
✅ 配置服务编排和负载均衡  
✅ 集成监控和日志系统  
✅ 解决常见部署问题  

部署技能是让 Agent 从原型走向生产的关键，是每个大模型工程师必须掌握的核心能力。