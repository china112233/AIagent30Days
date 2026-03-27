"""
Day 14: LangChain 基础 - Agent 智能代理示例

本示例演示 LangChain 的 Agent 组件
包括 Agent 创建、工具绑定和执行流程
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
    create_react_agent
)
from langchain import hub

# 加载环境变量
load_dotenv()


# ============================================================
# 定义工具
# ============================================================

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"、"广州"
    
    Returns:
        天气信息字符串
    """
    weather_data = {
        "北京": "晴天，温度 25°C，空气质量良好",
        "上海": "多云，温度 28°C，有轻微雾霾",
        "广州": "小雨，温度 30°C，湿度较高",
        "深圳": "晴天，温度 32°C，紫外线较强"
    }
    return weather_data.get(city, f"未找到 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """执行数学计算
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    
    Returns:
        计算结果
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库
    
    Args:
        query: 搜索关键词
    
    Returns:
        相关知识内容
    """
    knowledge_base = {
        "Python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。以简洁、易读著称。",
        "AI": "AI（人工智能）是计算机科学的一个分支，致力于创建能够执行需要人类智能的任务的系统。",
        "LangChain": "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。"
    }
    
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value
    return f"知识库中未找到关于 '{query}' 的信息"


def basic_agent_example():
    """
    示例1：基础 Agent 创建和使用
    使用 create_tool_calling_agent
    """
    print("=" * 60)
    print("示例1：基础 Agent（Tool Calling Agent）")
    print("=" * 60)

    # 创建 LLM
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    # 定义工具列表
    tools = [get_weather, calculate, search_knowledge]

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手，可以使用工具来回答问题。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程
    ])

    # 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 创建 AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示详细执行过程
        handle_parsing_errors=True
    )

    # 测试问题
    questions = [
        "北京今天天气怎么样？",
        "帮我计算 123 * 456",
        "什么是 LangChain？"
    ]

    for question in questions:
        print(f"\n问题: {question}")
        result = agent_executor.invoke({"input": question})
        print(f"回答: {result['output']}")
        print("-" * 40)

    print()


def agent_with_memory():
    """
    示例2：带记忆的 Agent
    """
    print("=" * 60)
    print("示例2：带记忆的 Agent")
    print("=" * 60)

    from langchain.memory import ConversationBufferMemory

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    tools = [get_weather, search_knowledge]

    # 创建记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手，记住之前的对话内容。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

    # 多轮对话
    conversations = [
        "北京天气怎么样？",
        "那上海呢？",
        "你刚才说北京天气怎么样来着？"
    ]

    for user_input in conversations:
        print(f"\n用户: {user_input}")
        result = agent_executor.invoke({"input": user_input})
        print(f"助手: {result['output']}")
        print("-" * 40)

    print()


def react_agent_example():
    """
    示例3：ReAct Agent
    使用 ReAct 推理模式
    """
    print("=" * 60)
    print("示例3：ReAct Agent")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    tools = [get_weather, calculate, search_knowledge]

    # 使用 LangChain Hub 的 ReAct Prompt
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # 如果无法从 Hub 获取，使用本地 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个可以使用工具的智能助手。

你可以使用以下工具：
{tool_names}

工具详情：
{tools}

请使用以下格式回答：

Question: 用户的问题
Thought: 你应该思考做什么
Action: 要使用的工具名称（必须是 [{tool_names}] 中的一个）
Action Input: 工具的输入参数
Observation: 工具的输出结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

开始！"""),
            ("user", "Question: {input}\n{agent_scratchpad}")
        ])

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5  # 限制迭代次数
    )

    # 测试复杂问题
    question = "北京的天气如何？如果温度是 25 度，那么 25 * 2 是多少？"
    print(f"\n问题: {question}")
    
    result = agent_executor.invoke({"input": question})
    print(f"回答: {result['output']}")

    print()


def manual_agent_loop():
    """
    示例4：手动实现 Agent 循环
    理解 Agent 的执行原理
    """
    print("=" * 60)
    print("示例4：手动实现 Agent 循环")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    tools = [get_weather, calculate]
    tools_map = {t.name: t for t in tools}

    def run_agent(question: str, max_steps: int = 5):
        """手动运行 Agent 循环"""
        print(f"问题: {question}")
        
        messages = [{"role": "user", "content": question}]
        
        for step in range(max_steps):
            print(f"\n--- 步骤 {step + 1} ---")
            
            # 让 LLM 决定下一步
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke(messages)
            
            # 检查是否需要调用工具
            if not response.tool_calls:
                print(f"最终回答: {response.content}")
                return response.content
            
            # 执行工具调用
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"调用工具: {tool_name}({tool_args})")
                
                # 执行工具
                tool = tools_map[tool_name]
                tool_result = tool.invoke(tool_args)
                print(f"工具结果: {tool_result}")
                
                # 添加结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
        
        return "达到最大步数限制"

    # 测试
    run_agent("北京和上海的天气分别怎么样？")
    print()

    print("=" * 60)


def agent_error_handling():
    """
    示例5：Agent 错误处理
    """
    print("=" * 60)
    print("示例5：Agent 错误处理")
    print("=" * 60)

    @tool
    def problematic_tool(x: str) -> str:
        """一个可能会出错的工具"""
        if "error" in x.lower():
            raise ValueError("这是一个故意抛出的错误")
        return f"处理成功: {x}"

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    tools = [problematic_tool, get_weather]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。如果工具调用失败，请尝试其他方法。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    # 配置错误处理
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,  # 处理解析错误
        max_iterations=3,  # 最大迭代次数
        early_stopping_method="generate"  # 超时后生成回答
    )

    # 测试正常情况
    print("\n正常情况:")
    result = agent_executor.invoke({"input": "北京天气怎么样？"})
    print(f"结果: {result['output']}")

    # 测试错误情况
    print("\n错误情况:")
    result = agent_executor.invoke({"input": "请使用 problematic_tool 处理 'error' 这个词"})
    print(f"结果: {result['output']}")

    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 14: LangChain 基础 - Agent 智能代理示例")
    print("=" * 60 + "\n")

    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    try:
        basic_agent_example()
        agent_with_memory()
        react_agent_example()
        manual_agent_loop()
        agent_error_handling()

        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()