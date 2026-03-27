"""
Day 14: LangChain 基础 - Tools 工具使用示例

本示例演示 LangChain 的 Tools 组件
包括工具定义、工具调用和工具绑定
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, Tool
from typing import Optional

# 加载环境变量
load_dotenv()


# ============================================================
# 定义各种工具
# ============================================================

@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """获取当前时间
    
    Args:
        timezone: 时区（可选），如 'Asia/Shanghai'
    
    Returns:
        当前时间字符串
    """
    now = datetime.now()
    if timezone:
        # 简化处理，实际应用中应使用 pytz
        return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone})"
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculate(expression: str) -> str:
    """执行数学计算
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    
    Returns:
        计算结果
    """
    try:
        # 安全计算（仅允许数学运算）
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def search_weather(city: str) -> str:
    """查询城市天气
    
    Args:
        city: 城市名称，如"北京"、"上海"、"广州"
    
    Returns:
        天气信息
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"weather": "晴", "temperature": "25°C", "humidity": "40%"},
        "上海": {"weather": "多云", "temperature": "28°C", "humidity": "60%"},
        "广州": {"weather": "小雨", "temperature": "30°C", "humidity": "80%"},
        "深圳": {"weather": "晴", "temperature": "32°C", "humidity": "70%"},
    }
    
    if city in weather_data:
        data = weather_data[city]
        return f"{city}天气: {data['weather']}, 温度 {data['temperature']}, 湿度 {data['humidity']}"
    return f"未找到 {city} 的天气信息"


@tool
def search_product(product_name: str) -> str:
    """搜索产品信息
    
    Args:
        product_name: 产品名称或关键词
    
    Returns:
        产品信息
    """
    # 模拟产品数据库
    products = {
        "iPhone": {"name": "iPhone 15", "price": 5999, "stock": 100},
        "MacBook": {"name": "MacBook Pro", "price": 12999, "stock": 50},
        "AirPods": {"name": "AirPods Pro", "price": 1799, "stock": 200},
    }
    
    for key, info in products.items():
        if key.lower() in product_name.lower():
            return json.dumps(info, ensure_ascii=False)
    
    return f"未找到产品: {product_name}"


def basic_tool_usage():
    """
    示例1：基础工具定义和使用
    """
    print("=" * 60)
    print("示例1：基础工具定义和使用")
    print("=" * 60)

    # 查看工具信息
    print(f"工具名称: {get_current_time.name}")
    print(f"工具描述: {get_current_time.description}")
    print(f"工具参数: {get_current_time.args}")
    print()

    # 直接调用工具
    result = get_current_time.invoke({"timezone": "Asia/Shanghai"})
    print(f"调用结果: {result}")
    print()


def tool_with_llm():
    """
    示例2：LLM 与工具结合
    让 LLM 决定何时调用工具
    """
    print("=" * 60)
    print("示例2：LLM 与工具结合")
    print("=" * 60)

    # 创建 LLM 并绑定工具
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    # 绑定工具
    tools = [get_current_time, calculate, search_weather]
    llm_with_tools = llm.bind_tools(tools)

    # 测试不同问题
    questions = [
        "现在几点了？",
        "帮我计算 123 * 456",
        "北京今天天气怎么样？",
        "你好，你是谁？"
    ]

    for question in questions:
        print(f"问题: {question}")
        response = llm_with_tools.invoke(question)
        
        # 检查是否需要调用工具
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"  需要调用工具: {tool_call['name']}")
                print(f"  参数: {tool_call['args']}")
        else:
            print(f"  直接回复: {response.content[:50]}...")
        print()

    print()


def manual_tool_execution():
    """
    示例3：手动执行工具调用
    """
    print("=" * 60)
    print("示例3：手动执行工具调用")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    tools = [get_current_time, calculate, search_weather, search_product]
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def run_with_tools(question: str) -> str:
        """运行问题并执行工具调用"""
        print(f"问题: {question}")
        
        # 第一次调用：LLM 决定是否使用工具
        response = llm_with_tools.invoke(question)
        
        if not response.tool_calls:
            return response.content
        
        # 执行工具调用
        messages = [{"role": "user", "content": question}]
        messages.append(response)
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  调用工具: {tool_name}({tool_args})")
            
            # 执行工具
            tool = tools_map[tool_name]
            tool_result = tool.invoke(tool_args)
            print(f"  工具结果: {tool_result}")
            
            # 添加工具结果到消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })
        
        # 第二次调用：LLM 基于工具结果生成回答
        final_response = llm.invoke(messages)
        return final_response.content

    # 测试
    questions = [
        "北京今天天气怎么样？温度是多少？",
        "帮我计算 (100 + 200) * 3"
    ]

    for q in questions:
        result = run_with_tools(q)
        print(f"回答: {result}")
        print("-" * 40)

    print()


def tool_with_context():
    """
    示例4：带上下文的工具调用
    """
    print("=" * 60)
    print("示例4：带上下文的工具调用")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    tools = [search_weather, search_product]
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    # 多轮对话
    conversation = [
        "北京天气怎么样？",
        "那上海呢？",
        "有没有 iPhone 的信息？",
        "MacBook 呢？"
    ]

    messages = []

    for user_input in conversation:
        print(f"用户: {user_input}")
        messages.append({"role": "user", "content": user_input})
        
        # 调用 LLM
        response = llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            # 执行工具
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_result = tools_map[tool_call["name"]].invoke(tool_call["args"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
            
            # 生成最终回答
            response = llm.invoke(messages)
        
        messages.append({"role": "assistant", "content": response.content})
        print(f"助手: {response.content}")
        print("-" * 40)

    print()


def custom_tool_class():
    """
    示例5：使用 Tool 类创建自定义工具
    """
    print("=" * 60)
    print("示例5：使用 Tool 类创建自定义工具")
    print("=" * 60)

    # 方式1：使用函数创建
    def translate_text(text: str, target_lang: str = "English") -> str:
        """翻译文本（模拟）"""
        return f"[翻译结果-{target_lang}]: {text}"

    translate_tool = Tool(
        name="translate",
        description="将文本翻译成指定语言。输入格式: 'text|target_lang'",
        func=lambda x: translate_text(*x.split("|") if "|" in x else (x, "English"))
    )

    # 方式2：使用 @tool 装饰器（推荐）
    @tool
    def word_count(text: str) -> str:
        """计算文本的字数和词数
        
        Args:
            text: 要计算的文本
        
        Returns:
            字数和词数统计
        """
        chars = len(text)
        words = len(text.split())
        return f"字数: {chars}, 词数: {words}"

    # 测试工具
    print(f"工具: {translate_tool.name}")
    print(f"结果: {translate_tool.run('你好世界|English')}")
    print()
    
    print(f"工具: {word_count.name}")
    print(f"结果: {word_count.invoke({'text': 'Hello World 你好世界'})}")

    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 14: LangChain 基础 - Tools 工具使用示例")
    print("=" * 60 + "\n")

    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    try:
        basic_tool_usage()
        tool_with_llm()
        manual_tool_execution()
        tool_with_context()
        custom_tool_class()

        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()