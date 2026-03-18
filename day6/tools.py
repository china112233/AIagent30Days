"""
工具定义和管理
实现 Tool 基类、工具注册表、内置工具
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
from datetime import datetime
import json


# ==================== Tool 基类 ====================

@dataclass
class Tool:
    """
    工具定义类

    Attributes:
        name: 工具名称（唯一标识）
        description: 工具描述（LLM 用于决策）
        parameters: 参数 Schema
        function: 执行函数
        examples: 使用示例
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    examples: List[str] = field(default_factory=list)

    def run(self, **kwargs) -> str:
        """
        执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            执行结果
        """
        try:
            # 验证参数
            self._validate_parameters(kwargs)
            # 执行函数
            result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def _validate_parameters(self, params: Dict[str, Any]):
        """验证参数"""
        required = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        # 检查必需参数
        for param in required:
            if param not in params:
                raise ValueError(f"缺少必需参数: {param}")

        # 检查参数类型（简化版）
        for param, value in params.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    params[param] = str(value)

    def get_schema(self) -> Dict[str, Any]:
        """
        获取 OpenAI Function Calling 格式的工具定义

        Returns:
            工具 Schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# ==================== 工具注册表 ====================

class ToolRegistry:
    """
    工具注册表
    管理所有可用工具
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """
        注册工具

        Args:
            tool: Tool 对象
        """
        self._tools[tool.name] = tool
        print(f"[工具注册] {tool.name}: {tool.description}")

    def unregister(self, name: str):
        """
        注销工具

        Args:
            name: 工具名称
        """
        if name in self._tools:
            del self._tools[name]
            print(f"[工具注销] {name}")

    def get(self, name: str) -> Optional[Tool]:
        """
        获取工具

        Args:
            name: 工具名称

        Returns:
            Tool 对象或 None
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())

    def get_descriptions(self) -> str:
        """
        获取所有工具描述（用于 prompt）

        Returns:
            格式化的工具描述
        """
        descriptions = []
        for name, tool in self._tools.items():
            desc = f"- {name}: {tool.description}"
            if tool.examples:
                desc += f"\n  示例: {tool.examples[0]}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的 OpenAI Schema

        Returns:
            工具 Schema 列表
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        """
        执行指定工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            执行结果
        """
        tool = self.get(name)
        if tool is None:
            return f"错误: 工具 '{name}' 不存在"
        return tool.run(**kwargs)


# ==================== 内置工具 ====================

def calculator_tool(expression: str) -> str:
    """
    计算器工具
    支持基本数学运算

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    # 允许的数学函数和常量
    allowed_names = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'pi': math.pi,
        'e': math.e,
        'pow': pow,
        'max': max,
        'min': min
    }

    try:
        # 安全执行数学表达式
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


def datetime_tool(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前日期时间

    Args:
        format_str: 日期格式字符串

    Returns:
        格式化的日期时间
    """
    try:
        return datetime.now().strftime(format_str)
    except Exception as e:
        return f"日期格式错误: {str(e)}"


def string_tool(operation: str, text: str, **kwargs) -> str:
    """
    字符串处理工具

    Args:
        operation: 操作类型 (upper, lower, reverse, length, split, replace)
        text: 输入文本

    Returns:
        处理结果
    """
    operations = {
        'upper': lambda t: t.upper(),
        'lower': lambda t: t.lower(),
        'reverse': lambda t: t[::-1],
        'length': lambda t: str(len(t)),
        'split': lambda t: json.dumps(t.split(kwargs.get('delimiter', ' '))),
        'replace': lambda t: t.replace(kwargs.get('old', ''), kwargs.get('new', ''))
    }

    if operation not in operations:
        return f"不支持的操作: {operation}。支持的操作: {list(operations.keys())}"

    try:
        return operations[operation](text)
    except Exception as e:
        return f"操作错误: {str(e)}"


def search_tool(query: str) -> str:
    """
    模拟搜索工具
    实际项目中可接入真实搜索 API

    Args:
        query: 搜索关键词

    Returns:
        搜索结果
    """
    # 模拟数据
    mock_database = {
        "天气": "今天天气晴朗，气温 25°C，湿度 40%，空气质量良好。",
        "北京": "北京是中国的首都，人口约 2100 万，面积 16410 平方公里。",
        "Python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        "AI": "人工智能（AI）是计算机科学的一个分支，致力于创建智能机器。",
        "机器学习": "机器学习是 AI 的子领域，让计算机从数据中学习模式。",
        "新闻": "最新科技新闻：AI 技术持续发展，大模型应用越来越广泛。",
    }

    # 简单关键词匹配
    for key, value in mock_database.items():
        if key in query:
            return f"搜索结果: {value}"

    return f"未找到与 '{query}' 相关的信息。"


def json_tool(operation: str, data: str, **kwargs) -> str:
    """
    JSON 处理工具

    Args:
        operation: 操作类型 (parse, stringify, get)
        data: JSON 数据或字符串

    Returns:
        处理结果
    """
    try:
        if operation == "parse":
            return json.dumps(json.loads(data), indent=2, ensure_ascii=False)
        elif operation == "stringify":
            return json.dumps(data, ensure_ascii=False)
        elif operation == "get":
            obj = json.loads(data)
            key = kwargs.get("key", "")
            keys = key.split(".")
            result = obj
            for k in keys:
                if isinstance(result, dict):
                    result = result.get(k)
                elif isinstance(result, list) and k.isdigit():
                    result = result[int(k)]
                else:
                    return f"无法获取键: {key}"
            return json.dumps(result, ensure_ascii=False)
        else:
            return f"不支持的操作: {operation}"
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {str(e)}"
    except Exception as e:
        return f"处理错误: {str(e)}"


# ==================== 创建默认工具注册表 ====================

def create_default_registry() -> ToolRegistry:
    """
    创建包含默认工具的注册表

    Returns:
        配置好的 ToolRegistry
    """
    registry = ToolRegistry()

    # 注册计算器工具
    registry.register(Tool(
        name="calculator",
        description="计算数学表达式，支持加减乘除、平方根、三角函数等。例如: 2+2, sqrt(16), sin(pi/2)",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如: 2+2, sqrt(16), sin(pi/2)"
                }
            },
            "required": ["expression"]
        },
        function=calculator_tool,
        examples=["calculator(expression='2+2')", "calculator(expression='sqrt(16)')"]
    ))

    # 注册日期时间工具
    registry.register(Tool(
        name="datetime",
        description="获取当前日期和时间。可选参数 format 指定日期格式。",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "日期格式，如 '%Y-%m-%d' 表示年-月-日"
                }
            },
            "required": []
        },
        function=datetime_tool,
        examples=["datetime()", "datetime(format='%Y年%m月%d日')"]
    ))

    # 注册字符串工具
    registry.register(Tool(
        name="string",
        description="字符串处理工具，支持大小写转换、反转、长度计算、分割、替换等操作。",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "操作类型: upper(大写), lower(小写), reverse(反转), length(长度), split(分割), replace(替换)"
                },
                "text": {
                    "type": "string",
                    "description": "要处理的文本"
                },
                "delimiter": {
                    "type": "string",
                    "description": "分割符（split操作时使用）"
                },
                "old": {
                    "type": "string",
                    "description": "要替换的文本（replace操作时使用）"
                },
                "new": {
                    "type": "string",
                    "description": "替换后的文本（replace操作时使用）"
                }
            },
            "required": ["operation", "text"]
        },
        function=lambda **kwargs: string_tool(
            kwargs.get('operation', ''),
            kwargs.get('text', ''),
            **{k: v for k, v in kwargs.items() if k not in ['operation', 'text']}
        ),
        examples=["string(operation='upper', text='hello')", "string(operation='length', text='hello world')"]
    ))

    # 注册搜索工具
    registry.register(Tool(
        name="search",
        description="搜索信息。输入关键词，返回相关信息。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        },
        function=search_tool,
        examples=["search(query='Python')", "search(query='天气')"]
    ))

    # 注册 JSON 工具
    registry.register(Tool(
        name="json",
        description="JSON 处理工具，支持解析、序列化、提取字段等操作。",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "操作类型: parse(格式化), stringify(序列化), get(提取字段)"
                },
                "data": {
                    "type": "string",
                    "description": "JSON 字符串"
                },
                "key": {
                    "type": "string",
                    "description": "要提取的键（get操作时使用），支持点号分隔的嵌套键"
                }
            },
            "required": ["operation", "data"]
        },
        function=lambda **kwargs: json_tool(
            kwargs.get('operation', ''),
            kwargs.get('data', ''),
            **{k: v for k, v in kwargs.items() if k not in ['operation', 'data']}
        ),
        examples=["json(operation='parse', data='{\"a\":1}')"]
    ))

    return registry


# ==================== 演示函数 ====================

def demo_tools():
    """演示工具使用"""
    print("=" * 50)
    print("工具系统演示")
    print("=" * 50)

    # 创建工具注册表
    registry = create_default_registry()

    print("\n[1] 可用工具列表:")
    for name in registry.list_tools():
        print(f"  - {name}")

    print("\n[2] 测试计算器工具:")
    test_expressions = ["2 + 2", "sqrt(16)", "sin(pi/2)", "2 ** 10"]
    for expr in test_expressions:
        result = registry.execute("calculator", expression=expr)
        print(f"  {expr} = {result}")

    print("\n[3] 测试日期时间工具:")
    print(f"  默认格式: {registry.execute('datetime')}")
    print(f"  自定义格式: {registry.execute('datetime', format='%Y年%m月%d日')}")

    print("\n[4] 测试字符串工具:")
    print(f"  upper('hello'): {registry.execute('string', operation='upper', text='hello')}")
    print(f"  length('hello world'): {registry.execute('string', operation='length', text='hello world')}")
    print(f"  reverse('Python'): {registry.execute('string', operation='reverse', text='Python')}")

    print("\n[5] 测试搜索工具:")
    queries = ["天气", "Python", "AI"]
    for q in queries:
        result = registry.execute("search", query=q)
        print(f"  搜索'{q}': {result[:50]}...")

    print("\n[6] 获取工具 Schema (OpenAI 格式):")
    schemas = registry.get_all_schemas()
    print(f"  共 {len(schemas)} 个工具 Schema")
    print(f"  示例 (calculator): {schemas[0]}")


if __name__ == "__main__":
    demo_tools()