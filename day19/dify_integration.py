"""
Day 19: Dify 平台集成示例

本文件演示如何通过 API 与 Dify 平台集成：
- API 调用基础
- Chat App 对话
- Completion 生成
- Workflow 执行
- Knowledge Base 检索

Dify 是一个开源的 LLM 应用开发平台，提供可视化构建界面。
本示例展示如何通过 Python 代码调用 Dify 的 API。
"""

import os
import json
import requests
from typing import Optional, Dict, Any, Generator
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ==========================================
# Dify API 客户端类
# ==========================================


class DifyClient:
    """Dify API 客户端"""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.dify.ai/v1"
    ):
        """
        初始化 Dify 客户端

        Args:
            api_key: Dify 应用 API Key
            api_url: Dify API 地址
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        stream: bool = False
    ) -> Any:
        """发送 API 请求"""
        url = f"{self.api_url}/{endpoint}"

        if stream:
            data["response_mode"] = "streaming"

        if method == "POST":
            response = requests.post(
                url,
                headers=self.headers,
                json=data,
                stream=stream
            )
        else:
            response = requests.get(url, headers=self.headers)

        return response

    def chat_messages(
        self,
        query: str,
        user: str = "default-user",
        conversation_id: Optional[str] = None,
        inputs: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict:
        """
        Chat App 对话接口

        Args:
            query: 用户问题
            user: 用户标识
            conversation_id: 会话 ID（继续对话时使用）
            inputs: 输入变量
            stream: 是否流式输出

        Returns:
            API 响应结果
        """
        data = {
            "query": query,
            "user": user,
            "inputs": inputs or {}
        }

        if conversation_id:
            data["conversation_id"] = conversation_id

        if stream:
            return self._stream_chat(data)

        # 阻塞模式
        data["response_mode"] = "blocking"
        response = self._make_request("chat-messages", "POST", data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def _stream_chat(self, data: Dict) -> Generator:
        """流式对话"""
        data["response_mode"] = "streaming"
        response = self._make_request("chat-messages", "POST", data, stream=True)

        for line in response.iter_lines():
            if line:
                yield json.loads(line.decode("utf-8"))

    def completion_messages(
        self,
        inputs: Dict,
        user: str = "default-user",
        stream: bool = False
    ) -> Dict:
        """
        Completion App 文本生成接口

        Args:
            inputs: 输入变量
            user: 用户标识
            stream: 是否流式输出

        Returns:
            API 响应结果
        """
        data = {
            "inputs": inputs,
            "user": user,
            "response_mode": "blocking" if not stream else "streaming"
        }

        response = self._make_request("completion-messages", "POST", data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def workflow_run(
        self,
        inputs: Dict,
        user: str = "default-user"
    ) -> Dict:
        """
        Workflow 执行接口

        Args:
            inputs: 输入变量
            user: 用户标识

        Returns:
            API 响应结果
        """
        data = {
            "inputs": inputs,
            "user": user
        }

        response = self._make_request("workflows/run", "POST", data)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def get_conversations(
        self,
        user: str = "default-user",
        last_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict:
        """获取对话列表"""
        params = {"user": user, "limit": limit}
        if last_id:
            params["last_id"] = last_id

        url = f"{self.api_url}/conversations?user={user}&limit={limit}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def get_messages(
        self,
        conversation_id: str,
        user: str = "default-user",
        first_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict:
        """获取对话消息历史"""
        url = f"{self.api_url}/messages?conversation_id={conversation_id}&user={user}&limit={limit}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")


# ==========================================
# Dify 模拟客户端（用于演示）
# ==========================================


class MockDifyClient:
    """模拟 Dify 客户端（用于无 API Key 时的演示）"""

    def __init__(self):
        """初始化模拟客户端"""
        print("⚠️ 使用模拟 Dify 客户端进行演示")
        print("要使用真实 API，请设置 DIFY_API_KEY 环境变量")

    def chat_messages(
        self,
        query: str,
        user: str = "default-user",
        conversation_id: Optional[str] = None,
        inputs: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict:
        """模拟对话"""
        return {
            "message_id": "mock-msg-id",
            "conversation_id": conversation_id or "mock-conv-id",
            "answer": f"[模拟响应] 这是一个针对 '{query}' 的示例回答。在实际使用中，这将是来自 Dify 平台的 AI 响应。",
            "created_at": 1704067200,
            "metadata": {
                "usage": {
                    "total_tokens": 100,
                    "prompt_tokens": 20,
                    "completion_tokens": 80
                }
            }
        }

    def completion_messages(
        self,
        inputs: Dict,
        user: str = "default-user",
        stream: bool = False
    ) -> Dict:
        """模拟文本生成"""
        input_text = inputs.get("text", "")
        return {
            "message_id": "mock-completion-id",
            "answer": f"[模拟生成] 基于 '{input_text}' 生成的示例内容。",
            "created_at": 1704067200
        }

    def workflow_run(
        self,
        inputs: Dict,
        user: str = "default-user"
    ) -> Dict:
        """模拟工作流执行"""
        return {
            "workflow_run_id": "mock-workflow-id",
            "outputs": {
                "result": "[模拟输出] 工作流执行完成的结果"
            },
            "status": "succeeded",
            "elapsed_time": 2.5
        }


# ==========================================
# 示例函数
# ==========================================


def get_dify_client() -> Any:
    """获取 Dify 客户端"""
    api_key = os.getenv("DIFY_API_KEY")

    if api_key:
        api_url = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1")
        return DifyClient(api_key, api_url)
    else:
        return MockDifyClient()


def example_dify_concept():
    """演示 Dify 平台的基本概念"""
    print("\n" + "=" * 50)
    print("示例 1: Dify 平台概念")
    print("=" * 50)

    print("""
    Dify 是一个开源的 LLM 应用开发平台，特点：

    1. 可视化构建
       - 拖拽式工作流设计
       - 无需编写代码即可构建应用
       - 支持多种节点类型

    2. 应用类型
       - Chat App: 对话式应用
       - Completion App: 文本补全应用
       - Agent App: Agent 应用
       - Workflow App: 工作流应用

    3. 核心功能
       - 知识库 RAG
       - 多模型支持
       - 工具集成
       - API 调用

    4. 使用方式
       - 云平台: https://dify.ai
       - 自部署: Docker/Kubernetes
       - API 调用: 本示例展示的方式

    5. API 调用流程
       ┌─────────────────────────────────────────────────┐
       │                                                 │
       │   Python App ──→ Dify API ──→ LLM Model        │
       │       │              │              │          │
       │       │              ↓              │          │
       │       │         Workflow          │          │
       │       │         / Knowledge       │          │
       │       │              │              │          │
       │       │              ↓              │          │
       │       └──────────── Response ──────┘          │
       │                                                 │
       └─────────────────────────────────────────────────┘
    """)


def example_chat_app():
    """演示 Chat App 对话"""
    print("\n" + "=" * 50)
    print("示例 2: Chat App 对话")
    print("=" * 50)

    client = get_dify_client()

    # 第一轮对话
    print("\n第一轮对话:")
    query1 = "什么是 Dify 平台？"

    result1 = client.chat_messages(query1)
    print(f"  用户: {query1}")
    print(f"  AI: {result1['answer'][:200]}...")
    print(f"  会话 ID: {result1['conversation_id']}")

    # 继续对话（使用相同的 conversation_id）
    print("\n第二轮对话（继续上下文）:")
    query2 = "它有哪些主要功能？"

    result2 = client.chat_messages(
        query2,
        conversation_id=result1['conversation_id']
    )
    print(f"  用户: {query2}")
    print(f"  AI: {result2['answer'][:200]}...")

    # Token 使用情况
    if 'metadata' in result2:
        usage = result2['metadata'].get('usage', {})
        print(f"\nToken 使用:")
        print(f"  输入: {usage.get('prompt_tokens', 0)}")
        print(f"  输出: {usage.get('completion_tokens', 0)}")
        print(f"  总计: {usage.get('total_tokens', 0)}")


def example_completion_app():
    """演示 Completion App 文本生成"""
    print("\n" + "=" * 50)
    print("示例 3: Completion App 文本生成")
    print("=" * 50)

    client = get_dify_client()

    # 使用 Completion App 生成内容
    inputs = {
        "text": "请介绍 Semantic Kernel 框架的特点",
        "style": "专业简洁"
    }

    print("\n输入变量:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")

    result = client.completion_messages(inputs)

    print(f"\n生成结果:")
    print(f"  {result['answer'][:200]}...")


def example_workflow_app():
    """演示 Workflow 执行"""
    print("\n" + "=" * 50)
    print("示例 4: Workflow 执行")
    print("=" * 50)

    client = get_dify_client()

    print("""
    Dify Workflow 是一个可视化的流程编排系统。

    工作流节点类型：
    - Start: 开始节点，定义输入变量
    - LLM: LLM 调用节点
    - Knowledge Retrieval: 知识库检索
    - Tool: 工具调用
    - Code: 代码执行
    - Template: 模板转换
    - Condition: 条件分支
    - Variable Assigner: 变量赋值
    - End: 结束节点

    示例工作流：
    ┌───────────────────────────────────────────────────┐
    │                                                   │
    │   Start → LLM → Knowledge → LLM → End           │
    │           │       │        │                     │
    │           ↓       ↓        ↓                     │
    │         模型   检索文档  综合回答                  │
    │                                                   │
    └───────────────────────────────────────────────────┘
    """)

    # 执行工作流
    inputs = {
        "topic": "LangGraph 框架",
        "target_audience": "开发者"
    }

    print("\n工作流输入:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")

    result = client.workflow_run(inputs)

    print(f"\n工作流执行状态: {result['status']}")
    print(f"执行时间: {result.get('elapsed_time', 0)} 秒")

    if 'outputs' in result:
        print("\n输出结果:")
        for key, value in result['outputs'].items():
            print(f"  {key}: {value[:150] if isinstance(value, str) else value}...")


def example_knowledge_retrieval():
    """演示知识库检索功能"""
    print("\n" + "=" * 50)
    print("示例 5: 知识库检索")
    print("=" * 50)

    client = get_dify_client()

    print("""
    Dify 知识库功能：

    1. 知识库类型
       - 文档导入: TXT, PDF, Markdown, HTML 等
       - 网页导入: 自动爬取和解析
       - API 导入: 通过 API 动态添加

    2. 检索方式
       - 向量检索: 语义相似度匹配
       - 关键词检索: 精确关键词匹配
       - 混合检索: 结合向量和关键词

    3. 知识库配置
       - 嵌入模型: 选择嵌入模型
       - 分块策略: 配置文档分块
       - 检索参数: top_k, 相似度阈值

    示例知识库检索流程：
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  用户问题 ──→ 向量化 ──→ 知识库检索 ──→ 相关文档    │
    │                        │                           │
    │                        ↓                           │
    │                      排序过滤                       │
    │                        │                           │
    │                        ↓                           │
    │               相关文档 + 问题 ──→ LLM ──→ 回答     │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)

    # 模拟知识库检索对话
    query = "Semantic Kernel 的核心组件有哪些？"

    print(f"\n用户问题: {query}")
    print("(模拟知识库检索后生成回答)")

    result = client.chat_messages(query)

    print(f"\nAI 回答 (包含知识库内容):")
    print(f"  {result['answer'][:200]}...")


def example_stream_response():
    """演示流式响应处理"""
    print("\n" + "=" * 50)
    print("示例 6: 流式响应")
    print("=" * 50)

    print("""
    Dify 支持两种响应模式：

    1. Blocking (阻塞模式)
       - 等待完整响应返回
       - 适合后台处理、批量调用
       - 响应格式: 完整 JSON

    2. Streaming (流式模式)
       - 实时返回部分响应
       - 适合实时对话展示
       - 响应格式: SSE (Server-Sent Events)

    流式响应事件类型：
    - workflow_started: 工作流开始
    - node_finished: 节点完成
    - message: 消息片段
    - message_end: 消息结束
    - workflow_finished: 工作流完成

    示例 SSE 数据格式：
    data: {"event": "message", "answer": "这"}
    data: {"event": "message", "answer": "是"}
    data: {"event": "message", "answer": "回答"}
    data: {"event": "message_end", "metadata": {...}}
    """)

    # 演示流式处理逻辑
    print("\n流式响应处理示例代码:")
    code = """
    def handle_stream_response(client, query):
        for event in client.chat_messages(query, stream=True):
            if event["event"] == "message":
                print(event["answer"], end="", flush=True)
            elif event["event"] == "message_end":
                print("\\n完成!")

    # 使用
    handle_stream_response(client, "你好")
    """
    print(code)


def example_api_integration():
    """演示 API 集成的实际应用"""
    print("\n" + "=" * 50)
    print("示例 7: API 集成实际应用")
    print("=" * 50)

    print("""
    将 Dify API 集成到实际应用中：

    1. Web 应用集成
       - FastAPI/Flask 后端调用 Dify
       - 前端通过 WebSocket 接收流式响应

    2. 移动应用集成
       - 通过 API 调用 Dify
       - 缓存对话历史本地存储

    3. 企业系统集成
       - CRM 系统集成客服功能
       - OA 系统集成文档助手

    示例架构：
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │   ┌─────────────┐         ┌─────────────┐           │
    │   │ Web Frontend│ ←───→   │ FastAPI     │           │
    │   │             │         │ Backend     │           │
    │   └─────────────┘         └──────┬──────┘           │
    │                                  │                   │
    │                                  ↓                   │
    │                           ┌─────────────┐           │
    │                           │ Dify API    │           │
    │                           │ Client      │           │
    │                           └──────┬──────┘           │
    │                                  │                   │
    │                                  ↓                   │
    │                           ┌─────────────┐           │
    │                           │ Dify Cloud  │           │
    │                           │ / Self-host │           │
    │                           └─────────────┘           │
    │                                                       │
    └───────────────────────────────────────────────────────┘
    """)

    # 模拟 FastAPI 路由示例
    print("\nFastAPI 集成示例代码:")
    code = """
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    dify_client = DifyClient(api_key="your_key")

    @app.post("/chat")
    async def chat(query: str):
        result = dify_client.chat_messages(query)
        return {"answer": result["answer"]}

    @app.post("/chat/stream")
    async def chat_stream(query: str):
        def generate():
            for event in dify_client.chat_messages(query, stream=True):
                yield json.dumps(event) + "\\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    """
    print(code)


def example_app_management():
    """演示应用管理功能"""
    print("\n" + "=" * 50)
    print("示例 8: 应用管理")
    print("=" * 50)

    client = get_dify_client()

    print("""
    Dify 应用管理功能：

    1. 应用配置
       - 模型选择
       - 提示词模板
       - 变量定义
       - 工具集成

    2. 监控与分析
       - Token 使用统计
       - 响应时间分析
       - 用户行为追踪
       - 成本分析

    3. 版本管理
       - 应用版本发布
       - A/B 测试
       - 灰度发布
    """)

    # 获取对话列表（模拟）
    print("\n对话历史查询:")
    try:
        conversations = client.get_conversations()
        print(f"  总对话数: {len(conversations.get('data', []))}")

        for conv in conversations.get('data', [])[:3]:
            print(f"  - 会话 {conv.get('id', 'unknown')}")
    except Exception as e:
        print(f"  (演示模式，无需查询历史)")

    # 应用配置建议
    print("\n应用配置最佳实践:")
    print("  1. 清晰定义提示词模板")
    print("  2. 合理设置温度参数")
    print("  3. 配置适当的上下文窗口")
    print("  4. 监控 Token 使用和成本")
    print("  5. 定期优化知识库内容")


def example_dify_without_api():
    """演示无 API Key 时的学习方式"""
    print("\n" + "=" * 50)
    print("示例 9: 无 API Key 学习方式")
    print("=" * 50)

    print("""
    如果没有 Dify API Key，可以通过以下方式学习：

    1. 在线体验
       - 访问 https://dify.ai 注册账号
       - 使用免费额度创建应用
       - 在可视化界面构建工作流

    2. 本地部署
       - Docker 部署 Dify
       - 完整控制数据和配置
       - 支持私有模型接入

    Docker 部署命令：
    ```bash
    git clone https://github.com/langgenius/dify.git
    cd dify/docker
    docker compose up -d
    ```

    3. 学习资源
       - 官方文档: https://docs.dify.ai
       - GitHub: https://github.com/langgenius/dify
       - 社区教程: Discord、论坛

    4. 核心概念学习
       - 理解工作流编排
       - 了解知识库 RAG
       - 掌握提示词设计
       - 熟悉 API 调用方式

    本示例代码展示了完整的 API 调用流程，
    即使暂时无法调用真实 API，也可以理解工作原理。
    """)


def example_comparison_with_other_frameworks():
    """对比 Dify 与其他框架"""
    print("\n" + "=" * 50)
    print("示例 10: Dify vs 其他框架")
    print("=" * 50)

    print("""
    Dify 与其他框架对比：

    | 特性           | Dify         | LangChain    | Semantic Kernel |
    |----------------|--------------|--------------|-----------------|
    | 开发方式       | 低代码可视化 | 代码开发     | 代码开发        |
    | 学习曲线       | 低           | 中高         | 中              |
    | 灵活性         | 中           | 高           | 高              |
    | 企业部署       | 内置支持     | 需自建       | Azure集成       |
    | 知识库         | 内置强大     | 需集成       | 基础支持        |
    | Agent支持      | 有           | 强大         | 有              |
    | 成本           | 云服务付费   | 自建免费     | Azure付费       |

    选择建议：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  快速原型/非技术团队 ──→ Dify                           │
    │                                                         │
    │  深度定制/技术团队 ──→ LangChain/LangGraph              │
    │                                                         │
    │  企业微软生态 ──→ Semantic Kernel                       │
    │                                                         │
    │  研究实验 ──→ AutoGen                                   │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    组合使用：
    - Dify (前端) + LangChain (后端扩展)
    - Dify (原型验证) + 代码框架 (生产部署)
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 19: Dify 平台集成示例")
    print("=" * 60)

    # 运行各示例
    example_dify_concept()
    example_chat_app()
    example_completion_app()
    example_workflow_app()
    example_knowledge_retrieval()
    example_stream_response()
    example_api_integration()
    example_app_management()
    example_dify_without_api()
    example_comparison_with_other_frameworks()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
    print("\n提示:")
    print("  - 要使用真实 API，请在 .env 中设置 DIFY_API_KEY")
    print("  - API Key 可在 Dify 应用设置页面获取")
    print("  - 本示例展示了完整的 API 调用流程")


if __name__ == "__main__":
    main()