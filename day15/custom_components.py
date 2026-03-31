"""
Day 15: 自定义组件示例

本文件演示如何开发自定义组件：
- 自定义 LLM 包装器
- 自定义工具类
- 自定义检索器
- 自定义输出解析器
"""

import os
import json
import re
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_core.tools import BaseTool, tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.callbacks import CallbackManagerForLLMRun, CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# 加载环境变量
load_dotenv()


# ==========================================
# 示例 1: 自定义 LLM 包装器
# ==========================================


class MockLLM(LLM):
    """模拟 LLM - 用于测试和开发"""

    # LLM 特性参数
    n: int = 1
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        return "mock_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数"""
        return {"n": self.n, "temperature": self.temperature}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 LLM 调用"""
        # 模拟响应
        if "你好" in prompt or "hello" in prompt.lower():
            return "你好！我是一个模拟的 LLM，用于测试目的。"
        elif "介绍" in prompt:
            return "这是一个模拟的介绍响应。在实际应用中，这里会调用真实的 LLM API。"
        else:
            return f"[MockLLM] 收到提示词: {prompt[:50]}..."


class SimpleAPILLM(LLM):
    """简单的 API LLM 包装器"""

    api_url: str = "https://api.example.com/v1/chat"
    api_key: str = ""
    model_name: str = "default"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "simple_api_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用 API"""
        # 这里应该实现真实的 API 调用
        # 示例代码仅展示结构
        import requests

        try:
            # 实际实现：
            # response = requests.post(
            #     self.api_url,
            #     headers={"Authorization": f"Bearer {self.api_key}"},
            #     json={
            #         "model": self.model_name,
            #         "prompt": prompt,
            #         "temperature": self.temperature,
            #         "stop": stop,
            #     },
            # )
            # return response.json()["text"]

            # 模拟响应用于演示
            return f"[SimpleAPILLM] 模拟响应: {prompt[:30]}..."

        except Exception as e:
            return f"API 调用失败: {str(e)}"


def example_custom_llm():
    """演示自定义 LLM"""
    print("\n" + "=" * 50)
    print("示例 1: 自定义 LLM 包装器")
    print("=" * 50)

    # 使用 MockLLM
    mock_llm = MockLLM(temperature=0.7)

    print("\n测试 MockLLM:")
    response = mock_llm.invoke("你好")
    print(f"响应: {response}")

    response = mock_llm.invoke("请介绍一下 LangChain")
    print(f"响应: {response}")

    # 使用 MockLLM 构建链
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template("请回答：{question}")
    chain = prompt | mock_llm | StrOutputParser()

    print("\n使用自定义 LLM 构建链:")
    result = chain.invoke({"question": "你好"})
    print(f"链输出: {result}")

    # 使用 SimpleAPILLM
    print("\n测试 SimpleAPILLM (模拟):")
    api_llm = SimpleAPILLM(
        api_url="https://api.example.com",
        model_name="test-model",
    )
    response = api_llm.invoke("测试 API")
    print(f"响应: {response}")


# ==========================================
# 示例 2: 自定义工具
# ==========================================


class CalculatorInput(BaseModel):
    """计算器工具输入参数"""
    expression: str = Field(description="数学表达式，如 '2+3*4' 或 'sqrt(16)'")


class CalculatorTool(BaseTool):
    """自定义计算器工具"""

    name: str = "calculator"
    description: str = "计算数学表达式。输入应为数学表达式字符串，支持基本运算和 sqrt 函数。"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """执行计算"""
        try:
            # 安全地评估表达式
            # 注意：生产环境应使用更安全的方式
            allowed_chars = set("0123456789+-*/().sqrt ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"

            # 替换 sqrt 为 math.sqrt
            safe_expr = expression.replace("sqrt", "math.sqrt")

            import math
            result = eval(safe_expr, {"math": math})
            return f"计算结果: {result}"

        except Exception as e:
            return f"计算错误: {str(e)}"


class WebSearchInput(BaseModel):
    """Web 搜索工具输入"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")


class WebSearchTool(BaseTool):
    """模拟的 Web 搜索工具"""

    name: str = "web_search"
    description: str = "在网络上搜索信息。返回与查询相关的搜索结果列表。"
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """执行搜索（模拟）"""
        # 模拟搜索结果
        mock_results = [
            {"title": f"关于 {query} 的介绍", "url": f"https://example.com/1", "snippet": f"这是关于 {query} 的详细信息..."},
            {"title": f"{query} 应用案例", "url": f"https://example.com/2", "snippet": f"{query} 在实际中的应用..."},
            {"title": f"{query} 最新动态", "url": f"https://example.com/3", "snippet": f"{query} 的最新发展..."},
        ]

        results = mock_results[:max_results]
        formatted = "\n".join([
            f"{i+1}. {r['title']}\n   URL: {r['url']}\n   摘要: {r['snippet']}"
            for i, r in enumerate(results)
        ])
        return formatted


# 使用 @tool 装饰器创建简单工具
@tool
def get_current_time() -> str:
    """获取当前时间和日期。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def format_json(text: str) -> str:
    """格式化 JSON 字符串，使其更易读。

    Args:
        text: 要格式化的 JSON 字符串

    Returns:
        格式化后的 JSON 字符串
    """
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {str(e)}"


def example_custom_tools():
    """演示自定义工具"""
    print("\n" + "=" * 50)
    print("示例 2: 自定义工具")
    print("=" * 50)

    # 测试计算器工具
    print("\n测试 CalculatorTool:")
    calc = CalculatorTool()
    result = calc._run("2 + 3 * 4")
    print(f"表达式 '2 + 3 * 4': {result}")

    result = calc._run("sqrt(16)")
    print(f"表达式 'sqrt(16)': {result}")

    # 测试搜索工具
    print("\n测试 WebSearchTool:")
    search = WebSearchTool()
    result = search._run("LangChain", max_results=3)
    print(f"搜索 'LangChain':\n{result}")

    # 测试装饰器工具
    print("\n测试 @tool 装饰器工具:")
    print(f"当前时间: {get_current_time.invoke({})}")

    json_text = '{"name":"LangChain","type":"framework"}'
    print(f"格式化 JSON:\n{format_json.invoke({'text': json_text})}")

    # 组合工具使用
    print("\n--- 工具与 Agent 集成示例 ---")
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    try:
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
        )

        tools = [get_current_time, format_json]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手，可以使用工具。"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        print("\n使用 Agent 执行任务...")
        result = agent_executor.invoke({"input": "现在是什么时间？"})
        print(f"结果: {result['output']}")
    except Exception as e:
        print(f"Agent 执行失败（可能不支持 function calling）: {e}")


# ==========================================
# 示例 3: 自定义检索器
# ==========================================


class SimpleRetriever(BaseRetriever):
    """简单的自定义检索器"""

    documents: List[Document] = Field(default_factory=list)
    k: int = Field(default=3, description="返回文档数量")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """检索相关文档"""
        # 简单的关键词匹配检索
        relevant_docs = []
        for doc in self.documents:
            if any(keyword.lower() in doc.page_content.lower() for keyword in query.split()):
                relevant_docs.append(doc)

        return relevant_docs[:self.k]


class KeywordRetriever(BaseRetriever):
    """基于关键词的检索器"""

    documents: List[Document] = Field(default_factory=list)
    top_k: int = Field(default=5)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """关键词检索"""
        keywords = query.lower().split()

        # 计算每个文档的匹配分数
        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scored_docs.append((score, doc))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored_docs[:self.top_k]]


def example_custom_retriever():
    """演示自定义检索器"""
    print("\n" + "=" * 50)
    print("示例 3: 自定义检索器")
    print("=" * 50)

    # 创建示例文档
    docs = [
        Document(
            page_content="LangChain 是一个用于开发大语言模型应用的框架。",
            metadata={"source": "doc1", "topic": "LangChain"}
        ),
        Document(
            page_content="RAG（检索增强生成）是一种结合检索和生成的技术。",
            metadata={"source": "doc2", "topic": "RAG"}
        ),
        Document(
            page_content="向量数据库用于存储和检索嵌入向量。",
            metadata={"source": "doc3", "topic": "Vector Database"}
        ),
        Document(
            page_content="LangChain 支持多种 LLM 后端，包括 OpenAI、DeepSeek 等。",
            metadata={"source": "doc4", "topic": "LangChain"}
        ),
        Document(
            page_content="Agent 是能够自主决策和执行任务的 AI 系统。",
            metadata={"source": "doc5", "topic": "Agent"}
        ),
    ]

    # 测试 SimpleRetriever
    print("\n测试 SimpleRetriever:")
    retriever = SimpleRetriever(documents=docs, k=3)

    query = "LangChain 框架"
    results = retriever.invoke(query)
    print(f"查询: '{query}'")
    print(f"找到 {len(results)} 个文档:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content[:50]}... (来源: {doc.metadata['source']})")

    # 测试 KeywordRetriever
    print("\n测试 KeywordRetriever:")
    kw_retriever = KeywordRetriever(documents=docs, top_k=5)

    query = "LangChain RAG"
    results = kw_retriever.invoke(query)
    print(f"查询: '{query}'")
    print(f"找到 {len(results)} 个文档:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content[:50]}...")

    # 集成到 RAG 链
    print("\n--- 集成到 RAG 链 ---")
    mock_llm = MockLLM()
    prompt = ChatPromptTemplate.from_template(
        """基于以下信息回答问题：
{context}

问题：{question}"""
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        RunnablePassthrough.assign(context=retriever | format_docs)
        | prompt
        | mock_llm
    )

    result = rag_chain.invoke({"question": "什么是 LangChain？"})
    print(f"RAG 结果: {result}")


# ==========================================
# 示例 4: 自定义输出解析器
# ==========================================


class JSONOutputParser(BaseOutputParser[Dict]):
    """JSON 输出解析器"""

    def parse(self, text: str) -> Dict:
        """解析 JSON 输出"""
        try:
            # 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            json_match = re.search(r'\{[^{}]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("无法从文本中解析 JSON")

    @property
    def _type(self) -> str:
        return "json_parser"


class ListOutputParser(BaseOutputParser[List[str]]):
    """列表输出解析器"""

    def parse(self, text: str) -> List[str]:
        """解析列表输出"""
        # 支持多种格式
        # 1. 编号列表: 1. xxx 2. xxx
        numbered = re.findall(r'^\d+\.\s*(.+)$', text, re.MULTILINE)
        if numbered:
            return numbered

        # 2. 无序列表: - xxx 或 * xxx
        bullet = re.findall(r'^[-*]\s*(.+)$', text, re.MULTILINE)
        if bullet:
            return bullet

        # 3. 换行分隔
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines

    @property
    def _type(self) -> str:
        return "list_parser"


class StructuredOutputParser(BaseOutputParser[Dict]):
    """结构化输出解析器"""

    fields: Dict[str, str] = Field(default_factory=dict)

    def parse(self, text: str) -> Dict:
        """解析结构化输出"""
        result = {}
        for field, pattern in self.fields.items():
            match = re.search(pattern, text)
            if match:
                result[field] = match.group(1).strip()
        return result

    @property
    def _type(self) -> str:
        return "structured_parser"


def example_custom_output_parser():
    """演示自定义输出解析器"""
    print("\n" + "=" * 50)
    print("示例 4: 自定义输出解析器")
    print("=" * 50)

    # 测试 JSON 解析器
    print("\n测试 JSONOutputParser:")
    json_parser = JSONOutputParser()

    json_text = '{"name": "LangChain", "version": "0.1.0"}'
    result = json_parser.parse(json_text)
    print(f"输入: {json_text}")
    print(f"解析结果: {result}")

    # 嵌入文本中的 JSON
    text_with_json = "这是一个 JSON 示例：{\"key\": \"value\"} 结束。"
    result = json_parser.parse(text_with_json)
    print(f"\n输入: {text_with_json}")
    print(f"解析结果: {result}")

    # 测试列表解析器
    print("\n测试 ListOutputParser:")
    list_parser = ListOutputParser()

    numbered_list = """
1. 第一项
2. 第二项
3. 第三项
"""
    result = list_parser.parse(numbered_list)
    print(f"编号列表解析: {result}")

    bullet_list = """
- 项目 A
- 项目 B
- 项目 C
"""
    result = list_parser.parse(bullet_list)
    print(f"无序列表解析: {result}")

    # 测试结构化解析器
    print("\n测试 StructuredOutputParser:")
    structured_parser = StructuredOutputParser(
        fields={
            "name": r"名称[:：]\s*(.+)",
            "version": r"版本[:：]\s*(.+)",
            "description": r"描述[:：]\s*(.+)",
        }
    )

    text = """
名称：LangChain
版本：v0.1.0
描述：一个强大的 LLM 应用开发框架
"""
    result = structured_parser.parse(text)
    print(f"结构化解析: {result}")

    # 集成到链中
    print("\n--- 集成到链 ---")
    mock_llm = MockLLM()
    prompt = ChatPromptTemplate.from_template(
        "请以 JSON 格式返回关于 {topic} 的信息，包含 name 和 description 字段。"
    )

    chain = prompt | mock_llm | json_parser
    try:
        result = chain.invoke({"topic": "Python"})
        print(f"链输出: {result}")
    except Exception as e:
        print(f"解析失败（模拟 LLM 不返回标准 JSON）: {e}")


# ==========================================
# 示例 5: 完整自定义组件集成
# ==========================================


def example_full_integration():
    """演示完整自定义组件集成"""
    print("\n" + "=" * 50)
    print("示例 5: 完整自定义组件集成")
    print("=" * 50)

    # 创建所有自定义组件
    custom_llm = MockLLM(temperature=0.7)

    docs = [
        Document(page_content="LangChain 是 LLM 应用开发框架。"),
        Document(page_content="RAG 结合检索和生成技术。"),
        Document(page_content="Agent 可以自主执行任务。"),
    ]
    custom_retriever = SimpleRetriever(documents=docs)

    json_parser = JSONOutputParser()

    # 构建完整链
    prompt = ChatPromptTemplate.from_template(
        """基于以下信息，以 JSON 格式返回答案：
{context}

问题：{question}
请返回包含 answer 字段的 JSON。"""
    )

    def format_docs(docs):
        return "\n".join([d.page_content for d in docs])

    # RAG + 自定义 LLM + 自定义解析器
    full_chain = (
        RunnablePassthrough.assign(context=custom_retriever | format_docs)
        | prompt
        | custom_llm
        # | json_parser  # 模拟 LLM 输出可能不是标准 JSON
    )

    print("\n执行完整自定义链...")
    result = full_chain.invoke({"question": "什么是 LangChain？"})
    print(f"结果: {result}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 15: 自定义组件示例")
    print("=" * 60)

    example_custom_llm()
    example_custom_tools()
    example_custom_retriever()
    example_custom_output_parser()
    example_full_integration()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()