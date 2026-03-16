"""
计算器代理
支持复杂数学计算和单位转换
"""

import os
import json
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    支持：+、-、*、/、**(幂)、sqrt(平方根)、sin、cos、tan等
    """
    try:
        # 允许的数学函数
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
        }
        
        # 安全检查
        allowed = set("0123456789+-*/.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # 格式化结果
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)
        
        return f"✅ 计算结果: {expression} = {result}"
    
    except ZeroDivisionError:
        return "❌ 错误：除数不能为零"
    except Exception as e:
        return f"❌ 计算错误: {e}"


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    温度单位转换
    
    Args:
        value: 数值
        from_unit: 原单位 (celsius/fahrenheit/kelvin)
        to_unit: 目标单位
    """
    from_unit = from_unit.lower()[0]  # c, f, k
    to_unit = to_unit.lower()[0]
    
    # 先转换为摄氏度
    if from_unit == 'c':
        celsius = value
    elif from_unit == 'f':
        celsius = (value - 32) * 5 / 9
    elif from_unit == 'k':
        celsius = value - 273.15
    else:
        return f"未知单位: {from_unit}"
    
    # 从摄氏度转换到目标单位
    if to_unit == 'c':
        result = celsius
        unit = "°C"
    elif to_unit == 'f':
        result = celsius * 9 / 5 + 32
        unit = "°F"
    elif to_unit == 'k':
        result = celsius + 273.15
        unit = "K"
    else:
        return f"未知单位: {to_unit}"
    
    return f"🌡️  温度转换: {value}°{from_unit.upper()} = {round(result, 2)}{unit}"


def convert_length(value: float, from_unit: str, to_unit: str) -> str:
    """
    长度单位转换
    
    支持：m(米)、km(千米)、cm(厘米)、mm(毫米)、mile(英里)、ft(英尺)、inch(英寸)
    """
    conversions = {
        'm': 1, 'meter': 1, '米': 1,
        'km': 1000, 'kilometer': 1000, '千米': 1000,
        'cm': 0.01, 'centimeter': 0.01, '厘米': 0.01,
        'mm': 0.001, 'millimeter': 0.001, '毫米': 0.001,
        'mile': 1609.344, '英里': 1609.344,
        'ft': 0.3048, 'foot': 0.3048, '英尺': 0.3048,
        'inch': 0.0254, '英寸': 0.0254,
    }
    
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit not in conversions:
        return f"未知单位: {from_unit}"
    if to_unit not in conversions:
        return f"未知单位: {to_unit}"
    
    # 转换为米，再转换为目标单位
    meters = value * conversions[from_unit]
    result = meters / conversions[to_unit]
    
    return f"📏 长度转换: {value} {from_unit} = {round(result, 6)} {to_unit}"


def convert_weight(value: float, from_unit: str, to_unit: str) -> str:
    """
    重量单位转换
    
    支持：kg(千克)、g(克)、mg(毫克)、lb(磅)、oz(盎司)、斤
    """
    conversions = {
        'kg': 1, 'kilogram': 1, '千克': 1,
        'g': 0.001, 'gram': 0.001, '克': 0.001,
        'mg': 0.000001, 'milligram': 0.000001, '毫克': 0.000001,
        'lb': 0.453592, 'pound': 0.453592, '磅': 0.453592,
        'oz': 0.0283495, 'ounce': 0.0283495, '盎司': 0.0283495,
        '斤': 0.5,
    }
    
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit not in conversions:
        return f"未知单位: {from_unit}"
    if to_unit not in conversions:
        return f"未知单位: {to_unit}"
    
    kg = value * conversions[from_unit]
    result = kg / conversions[to_unit]
    
    return f"⚖️ 重量转换: {value} {from_unit} = {round(result, 6)} {to_unit}"


def solve_equation(equation: str) -> str:
    """
    解一元方程（简单版，仅支持线性方程）
    """
    try:
        # 简单的线性方程求解 ax + b = c
        # 这是一个简化版本，实际需要更复杂的解析
        if 'x' not in equation:
            return "目前只支持含 x 的一元方程"
        
        # 示例：解析 2x + 3 = 7
        # 这里需要更复杂的解析逻辑，简化处理
        return f"🧮 方程求解功能开发中...\n方程: {equation}"
    
    except Exception as e:
        return f"方程解析错误: {e}"


# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持基础运算和数学函数（sin, cos, sqrt等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如: 2+3*4, sqrt(16), sin(pi/2)"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "温度单位转换（摄氏度、华氏度、开尔文）",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "数值"},
                    "from_unit": {"type": "string", "description": "原单位: celsius/fahrenheit/kelvin"},
                    "to_unit": {"type": "string", "description": "目标单位"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_length",
            "description": "长度单位转换（米、千米、厘米、毫米、英里、英尺、英寸）",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "数值"},
                    "from_unit": {"type": "string", "description": "原单位"},
                    "to_unit": {"type": "string", "description": "目标单位"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_weight",
            "description": "重量单位转换（千克、克、毫克、磅、盎司、斤）",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "数值"},
                    "from_unit": {"type": "string", "description": "原单位"},
                    "to_unit": {"type": "string", "description": "目标单位"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]

available_functions = {
    "calculate": calculate,
    "convert_temperature": convert_temperature,
    "convert_length": convert_length,
    "convert_weight": convert_weight,
}


class CalculatorAgent:
    """计算器代理"""
    
    def __init__(self):
        self.messages = [{
            "role": "system",
            "content": """你是一个计算器助手，可以帮助用户：
- 计算数学表达式
- 温度单位转换（摄氏度、华氏度、开尔文）
- 长度单位转换（米、千米、厘米、毫米、英里、英尺、英寸）
- 重量单位转换（千克、克、毫克、磅、盎司、斤）

选择合适的工具帮助用户计算或转换单位。"""
        }]
    
    def chat(self, user_message: str) -> str:
        """处理用户消息"""
        self.messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        while message.tool_calls:
            self.messages.append(message)
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                result = available_functions[func_name](**func_args)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages
            )
            message = response.choices[0].message
        
        self.messages.append(message)
        return message.content


def demo():
    """演示计算器代理"""
    print("=" * 50)
    print("🔢 计算器代理")
    print("=" * 50)
    
    agent = CalculatorAgent()
    
    queries = [
        "帮我算一下 sqrt(144) + 5 * 3",
        "100华氏度等于多少摄氏度？",
        "5公里等于多少英里？",
        "我体重150斤，相当于多少千克？",
        "计算 sin(pi/2) 的值"
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        response = agent.chat(query)
        print(f"助手: {response}")


def interactive():
    """交互模式"""
    print("\n" + "=" * 50)
    print("🔢 计算器助手 - 交互模式")
    print("支持：数学计算、温度/长度/重量单位转换")
    print("输入 'quit' 退出")
    print("=" * 50)
    
    agent = CalculatorAgent()
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\n助手: {response}")


if __name__ == "__main__":
    demo()
    
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        interactive()