# day26/fastapi_agent.py
"""
Day 26: FastAPI Agent 服务

完整的 FastAPI Agent 服务实现，包括：
1. Agent 封装 - 将 Agent 包装为 API
2. 路由设计 - API 路径规划
3. 中间件 - 认证、日志、限流
4. 生命周期 - 服务启动和关闭

依赖安装：
pip install fastapi uvicorn pydantic python-multipart redis
"""

import os
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 数据模型定义
# ============================================================

class AgentRequest(BaseModel):
    """Agent 请求模型"""
    prompt: str = Field(..., description="用户输入提示")
    session_id: Optional[str] = Field(None, description="会话ID")
    tools: Optional[List[str]] = Field(None, description="可用工具列表")
    max_tokens: int = Field(256, description="最大生成token数")
    temperature: float = Field(0.7, ge=0, le=2, description="温度参数")
    stream: bool = Field(False, description="是否流式输出")


class ToolCall(BaseModel):
    """工具调用模型"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent 响应模型"""
    response: str
    session_id: str
    tokens_used: int
    tool_calls: Optional[List[ToolCall]] = None
    status: str = "success"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """错误响应模型"""
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    checks: Dict[str, bool]
    timestamp: str


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str
    created_at: str
    message_count: int
    last_activity: str


# ============================================================
# Agent 服务核心
# ============================================================

class AgentService:
    """
    Agent 服务核心类
    封装 Agent 的核心功能
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.request_count = 0
        self.total_tokens = 0
        self.tool_registry = {
            "search": self._mock_search,
            "calculate": self._mock_calculate,
            "translate": self._mock_translate,
        }
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """处理 Agent 请求"""
        self.request_count += 1
        
        # 获取或创建会话
        session_id = request.session_id or self._create_session()
        session = self._get_session(session_id)
        
        # 模拟 Agent 处理
        response_text, tokens, tool_calls = await self._run_agent(
            prompt=request.prompt,
            session=session,
            tools=request.tools,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # 更新统计
        self.total_tokens += tokens
        
        return AgentResponse(
            response=response_text,
            session_id=session_id,
            tokens_used=tokens,
            tool_calls=tool_calls
        )
    
    def _create_session(self) -> str:
        """创建新会话"""
        import uuid
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(),
            "messages": [],
            "last_activity": datetime.now()
        }
        return session_id
    
    def _get_session(self, session_id: str) -> Dict:
        """获取会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.now(),
                "messages": [],
                "last_activity": datetime.now()
            }
        return self.sessions[session_id]
    
    async def _run_agent(
        self, 
        prompt: str, 
        session: Dict, 
        tools: List[str],
        max_tokens: int,
        temperature: float
    ) -> tuple:
        """
        运行 Agent
        
        Returns:
            (response_text, tokens_used, tool_calls)
        """
        # 模拟 Agent 思考过程
        await asyncio.sleep(0.1)  # 模拟处理延迟
        
        # 添加消息到会话历史
        session["messages"].append({"role": "user", "content": prompt})
        session["last_activity"] = datetime.now()
        
        # 模拟工具调用
        tool_calls = []
        if tools:
            for tool_name in tools[:2]:  # 最多调用2个工具
                if tool_name in self.tool_registry:
                    result = await self.tool_registry[tool_name](prompt)
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments={"query": prompt[:50]},
                        result=result
                    ))
        
        # 生成响应
        response_text = self._generate_response(prompt, tool_calls)
        
        # 模拟 token 计数
        tokens = len(response_text.split()) + len(prompt.split())
        
        # 添加响应到会话历史
        session["messages"].append({"role": "assistant", "content": response_text})
        
        return response_text, tokens, tool_calls
    
    def _generate_response(self, prompt: str, tool_calls: List[ToolCall]) -> str:
        """生成响应文本"""
        # 简单的响应生成模拟
        responses = {
            "你好": "你好！很高兴为您服务。我是Agent助手，可以帮助您完成各种任务。",
            "帮助": "我可以帮助您进行搜索、计算、翻译等操作。请告诉我您需要什么帮助。",
            "搜索": f"根据您的请求，我已完成搜索。找到相关信息：{tool_calls[0].result if tool_calls else '无结果'}",
            "计算": f"计算结果：{tool_calls[0].result if tool_calls else '请提供具体数值'}",
            "翻译": f"翻译结果：{tool_calls[0].result if tool_calls else '请提供原文'}",
        }
        
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        
        return f"收到您的请求：'{prompt[:30]}...'。我正在处理您的请求。如需使用工具，请指定需要的功能。"
    
    async def _mock_search(self, query: str) -> str:
        """模拟搜索工具"""
        await asyncio.sleep(0.05)
        return f"搜索结果：找到{len(query)}个相关内容"
    
    async def _mock_calculate(self, expression: str) -> str:
        """模拟计算工具"""
        await asyncio.sleep(0.05)
        try:
            # 简单计算模拟
            result = len(expression) * 2
            return f"计算结果：{result}"
        except:
            return "计算错误"
    
    async def _mock_translate(self, text: str) -> str:
        """模拟翻译工具"""
        await asyncio.sleep(0.05)
        return f"翻译：{text}"
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        session = self.sessions.get(session_id)
        if session:
            return SessionInfo(
                session_id=session_id,
                created_at=session["created_at"].isoformat(),
                message_count=len(session["messages"]),
                last_activity=session["last_activity"].isoformat()
            )
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "active_sessions": len(self.sessions),
            "avg_tokens_per_request": self.total_tokens / max(1, self.request_count)
        }
    
    async def check_health(self) -> Dict[str, bool]:
        """健康检查"""
        checks = {
            "service": True,
            "sessions": len(self.sessions) < 1000,  # 会话数量限制检查
            "tools": len(self.tool_registry) > 0,
        }
        return checks


# ============================================================
# 全局服务实例
# ============================================================

agent_service: AgentService = None


# ============================================================
# 生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("Agent 服务启动...")
    agent_service = AgentService()
    app.state.agent_service = agent_service
    print("Agent 服务初始化完成")
    
    yield
    
    # 关闭时清理
    print("Agent 服务关闭...")
    print(f"服务统计: {agent_service.get_stats()}")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Agent API",
    description="Agent 智能代理服务 API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# 中间件配置
# ============================================================

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    
    # 记录请求
    print(f"[{datetime.now()}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # 记录响应时间
    process_time = time.time() - start_time
    print(f"  响应时间: {process_time:.3f}s, 状态码: {response.status_code}")
    
    return response


# 请求限流中间件（简化版）
request_counts = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """请求限流中间件"""
    client_ip = request.client.host if request.client else "unknown"
    
    # 简化的限流检查
    current_count = request_counts.get(client_ip, 0)
    if current_count > 100:  # 每分钟限制100请求
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                code="E004",
                message="Rate limit exceeded",
                details={"limit": 100, "current": current_count}
            ).dict()
        )
    
    request_counts[client_ip] = current_count + 1
    
    response = await call_next(request)
    return response


# ============================================================
# API 路由
# ============================================================

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    service = app.state.agent_service
    checks = await service.check_health()
    
    return HealthResponse(
        status="healthy" if all(checks.values()) else "degraded",
        checks=checks,
        timestamp=datetime.now().isoformat()
    )


@app.post("/v1/agent/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """
    与 Agent 进行对话
    
    - **prompt**: 用户输入提示
    - **session_id**: 可选的会话ID
    - **tools**: 可用工具列表
    - **max_tokens**: 最大生成token数
    """
    service = app.state.agent_service
    
    try:
        response = await service.process(request)
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="E002",
                message="Agent processing error",
                details={"error": str(e)}
            ).dict()
        )


@app.get("/v1/agent/session/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """获取会话信息"""
    service = app.state.agent_service
    session_info = service.get_session_info(session_id)
    
    if not session_info:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code="E003",
                message="Session not found",
                details={"session_id": session_id}
            ).dict()
        )
    
    return session_info


@app.delete("/v1/agent/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    service = app.state.agent_service
    
    if session_id not in service.sessions:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code="E003",
                message="Session not found",
                details={"session_id": session_id}
            ).dict()
        )
    
    del service.sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/v1/agent/stats")
async def get_stats():
    """获取服务统计"""
    service = app.state.agent_service
    return service.get_stats()


@app.get("/v1/agent/tools")
async def list_tools():
    """列出可用工具"""
    return {
        "tools": [
            {"name": "search", "description": "搜索信息"},
            {"name": "calculate", "description": "执行计算"},
            {"name": "translate", "description": "翻译文本"},
        ]
    }


# ============================================================
# 错误处理
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=f"E{exc.status_code}",
            message=str(exc.detail),
            details={}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code="E500",
            message="Internal server error",
            details={"error": str(exc)}
        ).dict()
    )


# ============================================================
# 启动服务
# ============================================================

def run_server():
    """启动服务器"""
    import uvicorn
    
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"\n{'='*60}")
    print("Agent API 服务")
    print(f"{'='*60}")
    print(f"地址: http://{host}:{port}")
    print(f"文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()