"""
多函数智能代理
实现一个能使用多种工具的智能助手
"""

import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 工具函数定义 ====================

def get_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


def get_weather(city: str) -> str:
    """查询天气（模拟）"""
    weathers = {
        "北京": "晴，温度 12°C，空气质量良好",
        "上海": "多云，温度 16°C，有轻微雾霾",
        "广州": "晴，温度 24°C，湿度较高",
        "深圳": "晴，温度 25°C，适合户外活动",
        "成都": "阴，温度 14°C，可能有小雨",
    }
    return weathers.get(city.replace("市", ""), f"抱歉，暂无{city}的天气数据")


def search_knowledge(query: str) -> str:
    """搜索知识库（模拟）"""
    knowledge = {
        "Python": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。特点是语法简洁、易学易用。",
        "AI": "人工智能（AI）是计算机科学的一个分支，致力于创建能模拟人类智能的系统。",
        "机器学习": "机器学习是AI的子领域，通过数据训练模型，使计算机能够从经验中学习。",
        "深度学习": "深度学习使用多层神经网络，在图像识别、自然语言处理等领域表现出色。",
    }
    
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    
    return f"知识库中未找到关于'{query}'的信息，建议查阅官方文档。"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def translate(text: str, target_lang: str = "英语") -> str:
    """翻译文本（模拟）"""
    translations = {
        ("你好", "英语"): "Hello",
        ("谢谢", "英语"): "Thank you",
        ("再见", "英语"): "Goodbye",
        ("你好", "日语"): "こんにちは",
        ("谢谢", "日语"): "ありがとう",
    }
    
    key = (text, target_lang)
    if key in translations:
        return f"翻译结果: {text} -> {translations[key]} ({target_lang})"
    
    return f"[模拟翻译] {text} -> [需要实际翻译API支持] ({target_lang})"


def create_reminder(task: str, time: str) -> str:
    """创建提醒"""
    return f"✅ 已创建提醒：{time} - {task}"


def get_stock_price(symbol: str) -> str:
    """获取股票价格（模拟）"""
    stocks = {
        "AAPL": "苹果公司 (AAPL) 股价: $178.50, 涨幅: +1.2%",
        "GOOGL": "谷歌 (GOOGL) 股价: $141.80, 涨幅: -0.5%",
        "TSLA": "特斯拉 (TSLA) 股价: $248.50, 涨幅: +2.8%",
        "BABA": "阿里巴巴 (BABA) 股价: $85.20, 涨幅: +0.3%",
    }
    return stocks.get(symbol.upper(), f"未找到股票代码 {symbol}")


# ==================== 工具定义 ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
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
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "翻译文本到指定语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要翻译的文本"},
                    "target_lang": {"type": "string", "description": "目标语言，如：英语、日语"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "创建提醒事项",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "提醒内容"},
                    "time": {"type": "string", "description": "提醒时间"}
                },
                "required": ["task", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取股票价格信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 AAPL、TSLA"}
                },
                "required": ["symbol"]
            }
        }
    }
]

# 函数映射
available_functions = {
    "get_time": get_time,
    "get_weather": get_weather,
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "translate": translate,
    "create_reminder": create_reminder,
    "get_stock_price": get_stock_price
}


# ==================== 智能代理核心 ====================

class MultiToolAgent:
    """多工具智能代理"""
    
    def __init__(self, system_prompt: str = None):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        else:
            self.messages.append({
                "role": "system",
                "content": """你是一个智能助手，可以使用多种工具帮助用户。
                
可用工具：
- 时间查询：获取当前时间
- 天气查询：查询城市天气
- 知识搜索：搜索知识库
- 计算器：计算数学表达式
- 翻译：翻译文本
- 提醒：创建提醒事项
- 股票：查询股票价格

根据用户需求选择合适的工具。如果不需要工具，直接回答用户。"""
            })
    
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具函数"""
        if tool_name not in available_functions:
            return f"错误：未知工具 {tool_name}"
        
        try:
            func = available_functions[tool_name]
            return func(**arguments)
        except Exception as e:
            return f"执行错误: {e}"
    
    def chat(self, user_message: str, verbose: bool = True) -> str:
        """
        处理用户消息
        
        Args:
            user_message: 用户输入
            verbose: 是否打印调试信息
        
        Returns:
            助手回复
        """
        self.messages.append({"role": "user", "content": user_message})
        
        # 调用模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # 处理工具调用
        while message.tool_calls:
            self.messages.append(message)
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"  [工具调用] {func_name}({func_args})")
                
                result = self.execute_tool(func_name, func_args)
                
                if verbose:
                    print(f"  [工具结果] {result}")
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # 继续调用模型处理结果
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages
            )
            message = response.choices[0].message
        
        # 保存最终回复
        self.messages.append(message)
        
        return message.content
    
    def reset(self):
        """重置对话历史"""
        system_prompt = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        self.messages = [system_prompt] if system_prompt else []


# ==================== 演示 ====================

def demo_multi_step():
    """多步骤任务演示"""
    print("\n" + "=" * 50)
    print("示例：多步骤任务")
    print("=" * 50)
    
    agent = MultiToolAgent()
    
    tasks = [
        "现在几点了？",
        "北京和上海天气怎么样？",
        "帮我算一下 (100 + 50) * 2",
        "苹果公司的股票价格是多少？"
    ]
    
    for task in tasks:
        print(f"\n用户: {task}")
        response = agent.chat(task)
        print(f"助手: {response}")


def demo_complex_request():
    """复杂请求演示"""
    print("\n" + "=" * 50)
    print("示例：复杂请求")
    print("=" * 50)
    
    agent = MultiToolAgent()
    
    request = """
    帮我完成以下任务：
    1. 查一下现在几点
    2. 查一下深圳的天气
    3. 算一下 256 / 8
    4. 翻译"你好"成英语
    """
    
    print(f"用户: {request}")
    response = agent.chat(request)
    print(f"助手: {response}")


def demo_interactive():
    """交互式对话"""
    print("\n" + "=" * 50)
    print("智能助手 - 交互模式")
    print("可用工具：时间、天气、知识搜索、计算、翻译、提醒、股票")
    print("输入 'quit' 退出，'reset' 重置对话")
    print("=" * 50)
    
    agent = MultiToolAgent()
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if user_input.lower() == 'reset':
            agent.reset()
            print("对话已重置")
            continue
        
        if not user_input:
            continue
        
        response = agent.chat(user_input, verbose=True)
        print(f"\n助手: {response}")


if __name__ == "__main__":
    print("=" * 50)
    print("多函数智能代理")
    print("=" * 50)
    
    # 多步骤任务
    demo_multi_step()
    
    # 复杂请求
    demo_complex_request()
    
    # 交互模式
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()