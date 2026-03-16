"""
Function Calling 基础示例
学习如何让 AI 调用自定义函数
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 定义工具函数 ====================

def get_current_time():
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    计算数学表达式
    支持基本运算：+、-、*、/
    """
    try:
        # 安全计算（只允许数学运算）
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def get_weather(city: str) -> str:
    """
    获取城市天气（模拟数据）
    实际应用中应调用真实天气 API
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": "15°C", "weather": "晴", "humidity": "45%"},
        "上海": {"temp": "18°C", "weather": "多云", "humidity": "65%"},
        "广州": {"temp": "25°C", "weather": "晴", "humidity": "70%"},
        "深圳": {"temp": "26°C", "weather": "晴", "humidity": "75%"},
        "杭州": {"temp": "16°C", "weather": "小雨", "humidity": "80%"},
    }
    
    city = city.replace("市", "")  # 处理"北京市"这种输入
    
    if city in weather_data:
        data = weather_data[city]
        return f"{city}天气: {data['weather']}, 温度: {data['temp']}, 湿度: {data['humidity']}"
    else:
        return f"未找到 {city} 的天气信息，支持的城市：北京、上海、广州、深圳、杭州"


# ==================== 定义工具列表 ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如：2+3*4"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 函数映射表
available_functions = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather
}


# ==================== 核心处理函数 ====================

def run_conversation(user_message: str, messages: list = None) -> str:
    """
    执行对话，自动处理函数调用
    
    Args:
        user_message: 用户消息
        messages: 对话历史（可选）
    
    Returns:
        助手的最终回复
    """
    if messages is None:
        messages = []
    
    # 添加用户消息
    messages.append({"role": "user", "content": user_message})
    
    # 第一次调用：让模型决定是否使用工具
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 自动决定
    )
    
    message = response.choices[0].message
    
    # 检查是否需要调用函数
    if message.tool_calls:
        print(f"\n[调试] 模型决定调用 {len(message.tool_calls)} 个函数")
        
        # 将助手的消息添加到历史（包含工具调用请求）
        messages.append(message)
        
        # 处理每个工具调用
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"[调试] 调用函数: {function_name}({function_args})")
            
            # 执行函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_result = function_to_call(**function_args)
            else:
                function_result = f"错误：未知函数 {function_name}"
            
            print(f"[调试] 函数结果: {function_result}")
            
            # 将函数结果添加到消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_result
            })
        
        # 第二次调用：让模型根据函数结果生成最终回复
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    else:
        # 模型直接回复，没有调用函数
        return message.content


# ==================== 示例演示 ====================

def demo_basic():
    """基础示例：单个函数调用"""
    print("\n" + "=" * 50)
    print("示例1：基础函数调用")
    print("=" * 50)
    
    questions = [
        "现在几点了？",
        "帮我算一下 123 * 456",
        "北京今天天气怎么样？"
    ]
    
    for question in questions:
        print(f"\n用户: {question}")
        answer = run_conversation(question)
        print(f"助手: {answer}")


def demo_multi_tool():
    """多工具示例：复杂请求"""
    print("\n" + "=" * 50)
    print("示例2：复杂请求处理")
    print("=" * 50)
    
    question = "帮我查一下上海和北京的天气，然后算一下18*7等于多少"
    print(f"\n用户: {question}")
    answer = run_conversation(question)
    print(f"助手: {answer}")


def demo_interactive():
    """交互式对话"""
    print("\n" + "=" * 50)
    print("交互式对话（输入 'quit' 退出）")
    print("=" * 50)
    
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具帮助用户。"}
    ]
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        response = run_conversation(user_input, messages)
        print(f"助手: {response}")
        
        # 更新消息历史（run_conversation 已经添加了消息）


if __name__ == "__main__":
    print("=" * 50)
    print("Function Calling 基础示例")
    print("=" * 50)
    
    # 示例1：基础函数调用
    demo_basic()
    
    # 示例2：复杂请求
    demo_multi_tool()
    
    # 示例3：交互式对话（可选）
    print("\n是否启动交互式对话？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()