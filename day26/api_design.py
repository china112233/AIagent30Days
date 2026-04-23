# day26/api_design.py
"""
Day 26: Agent API 设计示例

本文件演示 Agent API 的最佳设计实践，包括：
1. 请求/响应模型 - Pydantic 数据验证
2. RESTful 设计 - 标准化 API 规范
3. 异步处理 - 长时间任务处理
4. 错误处理 - 统一错误响应格式
5. 版本控制 - API 版本管理

依赖安装：
pip install fastapi uvicorn pydantic python-multipart
"""

import os
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 数据模型设计
# ============================================================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Agent 状态枚举"""
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# ============================================================
# 请求模型
# ============================================================

class BaseRequest:
    """基础请求模型"""
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class AgentChatRequest:
    """
    Agent 对话请求模型
    
    属性：
        prompt: 用户输入的提示文本
        session_id: 会话ID（可选，用于多轮对话）
        tools: 工具列表（可选）
        max_tokens: 最大生成token数
        temperature: 生成温度
        stream: 是否流式输出
    """
    
    def __init__(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        tools: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.prompt = prompt
        self.session_id = session_id
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.metadata = metadata or {}
        
        # 验证
        self._validate()
    
    def _validate(self):
        """验证请求参数"""
        if not self.prompt or len(self.prompt.strip()) == 0:
            raise ValueError("Prompt 不能为空")
        
        if len(self.prompt) > 10000:
            raise ValueError("Prompt 长度不能超过 10000 字符")
        
        if self.max_tokens < 1 or self.max_tokens > 4096:
            raise ValueError("max_tokens 必须在 1-4096 范围内")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature 必须在 0-2 范围内")


class AgentTaskRequest:
    """
    Agent 任务请求模型
    用于创建异步执行的任务
    """
    
    def __init__(
        self,
        task_type: str,
        params: Dict[str, Any],
        priority: int = 0,
        timeout: int = 300,
        callback_url: Optional[str] = None
    ):
        self.task_type = task_type
        self.params = params
        self.priority = priority
        self.timeout = timeout
        self.callback_url = callback_url
        
        self._validate()
    
    def _validate(self):
        """验证任务请求"""
        valid_task_types = ["analysis", "generation", "translation", "summarization"]
        if self.task_type not in valid_task_types:
            raise ValueError(f"无效的任务类型: {self.task_type}")


# ============================================================
# 响应模型
# ============================================================

class AgentChatResponse:
    """
    Agent 对话响应模型
    
    属性：
        response: Agent 生成的响应文本
        session_id: 会话ID
        tokens_used: 使用的token数
        tool_calls: 工具调用记录
        status: 响应状态
        latency: 响应延迟时间
    """
    
    def __init__(
        self,
        response: str,
        session_id: str,
        tokens_used: int,
        tool_calls: Optional[List[Dict]] = None,
        status: str = "success",
        latency: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        self.response = response
        self.session_id = session_id
        self.tokens_used = tokens_used
        self.tool_calls = tool_calls or []
        self.status = status
        self.latency = latency
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "session_id": self.session_id,
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
            "status": self.status,
            "latency": self.latency,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class AgentTaskResponse:
    """
    Agent 任务响应模型
    """
    
    def __init__(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Any] = None,
        progress: float = 0.0,
        error: Optional[str] = None
    ):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.progress = progress
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "progress": self.progress,
            "error": self.error,
            "timestamp": self.timestamp
        }


# ============================================================
# 错误处理模型
# ============================================================

class ErrorCode(str, Enum):
    """错误码枚举"""
    # 请求错误 (E001-E010)
    INVALID_REQUEST = "E001"
    MISSING_PARAMETER = "E002"
    INVALID_PARAMETER = "E003"
    RATE_LIMIT_EXCEEDED = "E004"
    
    # 认证错误 (E011-E020)
    AUTHENTICATION_FAILED = "E011"
    TOKEN_EXPIRED = "E012"
    PERMISSION_DENIED = "E013"
    
    # Agent 错误 (E021-E030)
    AGENT_BUSY = "E021"
    AGENT_ERROR = "E022"
    SESSION_NOT_FOUND = "E023"
    TOOL_EXECUTION_FAILED = "E024"
    
    # 系统错误 (E031-E040)
    INTERNAL_ERROR = "E031"
    SERVICE_UNAVAILABLE = "E032"
    TIMEOUT = "E033"


ERROR_MESSAGES = {
    ErrorCode.INVALID_REQUEST: "请求参数无效",
    ErrorCode.MISSING_PARAMETER: "缺少必要参数",
    ErrorCode.INVALID_PARAMETER: "参数值无效",
    ErrorCode.RATE_LIMIT_EXCEEDED: "请求频率超限",
    ErrorCode.AUTHENTICATION_FAILED: "认证失败",
    ErrorCode.TOKEN_EXPIRED: "令牌已过期",
    ErrorCode.PERMISSION_DENIED: "权限不足",
    ErrorCode.AGENT_BUSY: "Agent 正在处理其他请求",
    ErrorCode.AGENT_ERROR: "Agent 处理出错",
    ErrorCode.SESSION_NOT_FOUND: "会话不存在",
    ErrorCode.TOOL_EXECUTION_FAILED: "工具执行失败",
    ErrorCode.INTERNAL_ERROR: "内部服务错误",
    ErrorCode.SERVICE_UNAVAILABLE: "服务不可用",
    ErrorCode.TIMEOUT: "请求超时",
}


class ErrorResponse:
    """
    统一错误响应模型
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, "未知错误")
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }


# ============================================================
# API 版本控制
# ============================================================

class APIVersion:
    """API 版本管理"""
    
    CURRENT_VERSION = "v1"
    SUPPORTED_VERSIONS = ["v1", "v2"]
    
    @staticmethod
    def get_version_prefix(version: str = None) -> str:
        """获取版本前缀"""
        if version is None:
            version = APIVersion.CURRENT_VERSION
        
        if version not in APIVersion.SUPPORTED_VERSIONS:
            raise ValueError(f"不支持的 API 版本: {version}")
        
        return f"/{version}"
    
    @staticmethod
    def get_endpoint(version: str, endpoint: str) -> str:
        """获取完整端点路径"""
        return f"{APIVersion.get_version_prefix(version)}{endpoint}"


# ============================================================
# API 路径设计
# ============================================================

class APIEndpoints:
    """API 端点定义"""
    
    # Agent 相关端点
    AGENT_CHAT = "/agent/chat"
    AGENT_TASK = "/agent/task"
    AGENT_TASK_STATUS = "/agent/task/{task_id}"
    AGENT_SESSION = "/agent/session/{session_id}"
    
    # 工具相关端点
    TOOLS_LIST = "/tools"
    TOOLS_EXECUTE = "/tools/{tool_name}/execute"
    
    # 系统端点
    HEALTH = "/health"
    METRICS = "/metrics"
    INFO = "/info"
    
    @staticmethod
    def get_all_endpoints(version: str = "v1") -> Dict[str, str]:
        """获取所有端点"""
        prefix = APIVersion.get_version_prefix(version)
        return {
            "chat": f"{prefix}{APIEndpoints.AGENT_CHAT}",
            "task": f"{prefix}{APIEndpoints.AGENT_TASK}",
            "task_status": f"{prefix}{APIEndpoints.AGENT_TASK_STATUS}",
            "session": f"{prefix}{APIEndpoints.AGENT_SESSION}",
            "tools": f"{prefix}{APIEndpoints.TOOLS_LIST}",
            "health": APIEndpoints.HEALTH,
            "metrics": APIEndpoints.METRICS,
        }


# ============================================================
# 请求限流设计
# ============================================================

class RateLimitConfig:
    """请求限流配置"""
    
    # 默认限流配置
    DEFAULT_LIMITS = {
        "chat": {"requests_per_minute": 60, "tokens_per_minute": 10000},
        "task": {"requests_per_minute": 30},
        "session": {"requests_per_minute": 120},
    }
    
    # 用户级别限流
    USER_LIMITS = {
        "free": {"requests_per_minute": 10, "tokens_per_minute": 1000},
        "basic": {"requests_per_minute": 30, "tokens_per_minute": 3000},
        "pro": {"requests_per_minute": 100, "tokens_per_minute": 10000},
        "enterprise": {"requests_per_minute": 500, "tokens_per_minute": 50000},
    }
    
    @staticmethod
    def check_limit(endpoint: str, user_tier: str = "free") -> Dict[str, int]:
        """获取限流配置"""
        base_limits = RateLimitConfig.DEFAULT_LIMITS.get(endpoint, {})
        user_limits = RateLimitConfig.USER_LIMITS.get(user_tier, {})
        
        return {
            "requests_per_minute": min(
                base_limits.get("requests_per_minute", 60),
                user_limits.get("requests_per_minute", 10)
            ),
            "tokens_per_minute": min(
                base_limits.get("tokens_per_minute", 10000),
                user_limits.get("tokens_per_minute", 1000)
            )
        }


# ============================================================
# 分页和过滤设计
# ============================================================

class PaginationParams:
    """分页参数"""
    
    def __init__(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: Optional[str] = None,
        sort_order: str = "desc"
    ):
        self.page = max(1, page)
        self.limit = min(max(1, limit), 100)  # 最大100
        self.sort_by = sort_by
        self.sort_order = sort_order if sort_order in ["asc", "desc"] else "desc"
    
    def get_offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.limit
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "limit": self.limit,
            "offset": self.get_offset(),
            "sort_by": self.sort_by,
            "sort_order": self.sort_order
        }


class PaginationResponse:
    """分页响应"""
    
    def __init__(
        self,
        items: List[Any],
        total: int,
        page: int,
        limit: int,
        has_next: bool,
        has_prev: bool
    ):
        self.items = items
        self.total = total
        self.page = page
        self.limit = limit
        self.has_next = has_next
        self.has_prev = has_prev
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": self.items,
            "total": self.total,
            "page": self.page,
            "limit": self.limit,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
            "total_pages": (self.total + self.limit - 1) // self.limit
        }


# ============================================================
# API 文档设计
# ============================================================

class APIDocumentation:
    """API 文档配置"""
    
    OPENAPI_INFO = {
        "title": "Agent API",
        "description": "智能 Agent 服务 API",
        "version": "1.0.0",
        "contact": {
            "name": "API Support",
            "email": "support@example.com"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    }
    
    TAGS = [
        {
            "name": "agent",
            "description": "Agent 相关操作"
        },
        {
            "name": "tools",
            "description": "工具执行相关"
        },
        {
            "name": "session",
            "description": "会话管理"
        },
        {
            "name": "system",
            "description": "系统状态和监控"
        }
    ]
    
    @staticmethod
    def get_endpoint_docs(endpoint: str) -> Dict[str, Any]:
        """获取端点文档"""
        docs = {
            "/agent/chat": {
                "summary": "与 Agent 进行对话",
                "description": "发送消息给 Agent，获取响应",
                "request_body": AgentChatRequest,
                "response": AgentChatResponse,
            },
            "/agent/task": {
                "summary": "创建异步任务",
                "description": "创建一个需要长时间处理的任务",
                "request_body": AgentTaskRequest,
                "response": AgentTaskResponse,
            },
            "/health": {
                "summary": "健康检查",
                "description": "检查服务健康状态",
                "response": Dict[str, Any],
            }
        }
        return docs.get(endpoint, {})


# ============================================================
# API 设计展示
# ============================================================

def show_api_design():
    """展示 API 设计规范"""
    
    print("=" * 60)
    print("Day 26: Agent API 设计演示")
    print("=" * 60)
    
    # 1. 数据模型示例
    print("\n1. 请求模型示例:")
    try:
        request = AgentChatRequest(
            prompt="你好，请帮我分析这段文本",
            session_id="session_123",
            max_tokens=512,
            temperature=0.7
        )
        print(f"   创建请求成功:")
        print(f"     prompt: {request.prompt}")
        print(f"     session_id: {request.session_id}")
        print(f"     max_tokens: {request.max_tokens}")
    except ValueError as e:
        print(f"   验证失败: {e}")
    
    # 验证失败示例
    print("\n   参数验证示例:")
    try:
        invalid_request = AgentChatRequest(prompt="")
    except ValueError as e:
        print(f"     空prompt验证: {e}")
    
    try:
        invalid_request = AgentChatRequest(prompt="test", max_tokens=5000)
    except ValueError as e:
        print(f"     max_tokens超限验证: {e}")
    
    # 2. 响应模型示例
    print("\n2. 响应模型示例:")
    response = AgentChatResponse(
        response="这是一段分析结果...",
        session_id="session_123",
        tokens_used=150,
        tool_calls=[{"tool": "analyzer", "result": "success"}],
        latency=0.5
    )
    print(f"   响应数据: {json.dumps(response.to_dict(), indent=2)}")
    
    # 3. 错误处理示例
    print("\n3. 错误处理示例:")
    error = ErrorResponse(
        code=ErrorCode.INVALID_REQUEST,
        details={"field": "prompt", "reason": "empty"}
    )
    print(f"   错误响应: {json.dumps(error.to_dict(), indent=2)}")
    
    # 4. API 版本控制
    print("\n4. API 版本控制:")
    print(f"   支持版本: {APIVersion.SUPPORTED_VERSIONS}")
    print(f"   当前版本: {APIVersion.CURRENT_VERSION}")
    print(f"   端点路径示例:")
    for name, path in APIEndpoints.get_all_endpoints().items():
        print(f"     {name}: {path}")
    
    # 5. 请求限流
    print("\n5. 请求限流配置:")
    for tier in ["free", "basic", "pro", "enterprise"]:
        limits = RateLimitConfig.check_limit("chat", tier)
        print(f"   {tier}: {limits['requests_per_minute']} req/min, {limits['tokens_per_minute']} tokens/min")
    
    # 6. 分页设计
    print("\n6. 分页设计:")
    pagination = PaginationParams(page=2, limit=20, sort_by="created_at")
    print(f"   分页参数: {pagination.to_dict()}")
    
    # 7. API 文档
    print("\n7. API 文档配置:")
    print(f"   OpenAPI 信息:")
    for key, value in APIDocumentation.OPENAPI_INFO.items():
        print(f"     {key}: {value}")
    
    # 8. RESTful 设计原则
    print("\n8. RESTful 设计原则:")
    principles = """
┌─────────────────────────────────────────────────────────────────┐
│                RESTful API 设计原则                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   方法    │ 用途              │ 示例                            │
│   ─────────────────────────────────────────────────────────────│
│   GET     │ 获取资源          │ GET /agents/{id}               │
│   POST    │ 创建资源          │ POST /agents                   │
│   PUT     │ 更新资源          │ PUT /agents/{id}               │
│   DELETE  │ 删除资源          │ DELETE /agents/{id}            │
│   PATCH   │ 部分更新          │ PATCH /agents/{id}/status      │
│                                                                 │
│   URL 设计原则:                                                 │
│   • 使用复数名词: /agents, /sessions                           │
│   • 层级清晰: /agents/{id}/sessions                            │
│   • 避免动词: 不用 /getAgent, 用 GET /agents                    │
│   • 版本控制: /v1/agents, /v2/agents                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(principles)
    
    print("\n" + "=" * 60)
    print("API 设计演示完成!")
    print("=" * 60)
    print("\n关键点:")
    print("  • 使用 Pydantic 进行数据验证")
    print("  • 统一的错误响应格式")
    print("  • API 版本控制支持")
    print("  • 合理的请求限流配置")
    print("  • RESTful 设计原则")


import json


if __name__ == "__main__":
    show_api_design()