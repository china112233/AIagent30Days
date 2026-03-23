"""
工具定义模块

定义工具的数据结构和装饰器，用于创建标准化的工具。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
import inspect
import json


class ParameterType(Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, integer, number, boolean, array, object
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # 枚举值列表
    min_value: Optional[Union[int, float]] = None  # 最小值（数字类型）
    max_value: Optional[Union[int, float]] = None  # 最大值（数字类型）
    min_length: Optional[int] = None  # 最小长度（字符串/数组）
    max_length: Optional[int] = None  # 最大长度（字符串/数组）
    
    def to_json_schema(self) -> Dict[str, Any]:
        """转换为 JSON Schema 格式"""
        schema = {
            "type": self.type,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.default is not None:
            schema["default"] = self.default
            
        return schema


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def ok(cls, data: Any, **metadata) -> "ToolResult":
        """创建成功结果"""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error(cls, error: str, **metadata) -> "ToolResult":
        """创建错误结果"""
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    returns: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout: int = 30  # 超时时间（秒）
    
    def to_json_schema(self) -> Dict[str, Any]:
        """转换为 JSON Schema 格式（OpenAI Function Calling 格式）"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_openai_function(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": self.to_json_schema()
        }
    
    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        验证参数
        
        Returns:
            错误信息，如果验证通过则返回 None
        """
        param_dict = {p.name: p for p in self.parameters}
        
        # 检查必需参数
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return f"缺少必需参数: {param.name}"
        
        # 检查参数类型和约束
        for name, value in kwargs.items():
            if name not in param_dict:
                return f"未知参数: {name}"
            
            param = param_dict[name]
            
            # 类型检查
            if param.type == "string" and not isinstance(value, str):
                return f"参数 {name} 应该是字符串类型"
            elif param.type == "integer" and not isinstance(value, int):
                return f"参数 {name} 应该是整数类型"
            elif param.type == "number" and not isinstance(value, (int, float)):
                return f"参数 {name} 应该是数字类型"
            elif param.type == "boolean" and not isinstance(value, bool):
                return f"参数 {name} 应该是布尔类型"
            elif param.type == "array" and not isinstance(value, list):
                return f"参数 {name} 应该是数组类型"
            
            # 枚举值检查
            if param.enum and value not in param.enum:
                return f"参数 {name} 的值必须在 {param.enum} 中"
            
            # 数值范围检查
            if param.type in ("integer", "number"):
                if param.min_value is not None and value < param.min_value:
                    return f"参数 {name} 的值不能小于 {param.min_value}"
                if param.max_value is not None and value > param.max_value:
                    return f"参数 {name} 的值不能大于 {param.max_value}"
            
            # 长度检查
            if param.type in ("string", "array"):
                length = len(value)
                if param.min_length is not None and length < param.min_length:
                    return f"参数 {name} 的长度不能小于 {param.min_length}"
                if param.max_length is not None and length > param.max_length:
                    return f"参数 {name} 的长度不能大于 {param.max_length}"
        
        return None
    
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        # 验证参数
        error = self.validate_parameters(**kwargs)
        if error:
            return ToolResult.error(error)
        
        # 填充默认值
        param_dict = {p.name: p for p in self.parameters}
        for param in self.parameters:
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default
        
        try:
            # 执行函数
            result = self.function(**kwargs)
            
            # 如果返回值已经是 ToolResult，直接返回
            if isinstance(result, ToolResult):
                return result
            
            # 否则包装为成功结果
            return ToolResult.ok(result)
            
        except Exception as e:
            return ToolResult.error(str(e))
    
    def get_usage_example(self) -> str:
        """获取使用示例"""
        if self.examples:
            example = self.examples[0]
            args = ", ".join(f"{k}={repr(v)}" for k, v in example.items())
            return f"{self.name}({args})"
        else:
            args = ", ".join(p.name for p in self.parameters if p.required)
            return f"{self.name}({args})"


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[ToolParameter]] = None,
    returns: Optional[Dict[str, Any]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    tags: Optional[List[str]] = None,
    timeout: int = 30
):
    """
    工具装饰器
    
    用法:
        @tool(
            name="get_weather",
            description="获取天气信息",
            parameters=[
                ToolParameter(name="city", type="string", description="城市名称", required=True)
            ]
        )
        def get_weather(city: str) -> dict:
            return {"city": city, "temperature": 25}
    """
    def decorator(func: Callable) -> Tool:
        # 从函数签名自动提取参数信息
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""
        func_parameters = parameters or []
        
        # 如果没有提供参数，从函数签名推断
        if not func_parameters:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                param_type = "string"  # 默认类型
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"
                
                func_parameters.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    required=param.default == inspect.Parameter.empty,
                    default=param.default if param.default != inspect.Parameter.empty else None
                ))
        
        return Tool(
            name=func_name,
            description=func_description,
            parameters=func_parameters,
            function=func,
            returns=returns,
            examples=examples or [],
            tags=tags or [],
            timeout=timeout
        )
    
    return decorator


# ============ 示例工具定义 ============

@tool(
    name="get_weather",
    description="获取指定城市的天气信息",
    parameters=[
        ToolParameter(
            name="city",
            type="string",
            description="城市名称，如：北京、上海、广州",
            required=True
        ),
        ToolParameter(
            name="unit",
            type="string",
            description="温度单位",
            enum=["celsius", "fahrenheit"],
            default="celsius"
        )
    ],
    examples=[
        {"city": "北京", "unit": "celsius"},
        {"city": "上海", "unit": "fahrenheit"}
    ],
    tags=["weather", "api"]
)
def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    获取天气信息（模拟）
    
    实际应用中会调用真实的天气 API
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴"},
        "上海": {"temperature": 28, "condition": "多云"},
        "广州": {"temperature": 32, "condition": "阴"},
    }
    
    if city not in weather_data:
        return {"error": f"未找到城市 {city} 的天气信息"}
    
    data = weather_data[city].copy()
    data["city"] = city
    
    # 温度转换
    if unit == "fahrenheit":
        data["temperature"] = data["temperature"] * 9 / 5 + 32
        data["unit"] = "°F"
    else:
        data["unit"] = "°C"
    
    return data


@tool(
    name="calculate",
    description="执行数学计算表达式",
    parameters=[
        ToolParameter(
            name="expression",
            type="string",
            description="数学表达式，如：2+3*4, sqrt(16), sin(30)",
            required=True
        )
    ],
    examples=[
        {"expression": "2 + 3 * 4"},
        {"expression": "sqrt(16) + 10"}
    ],
    tags=["math", "calculator"]
)
def calculate(expression: str) -> dict:
    """
    执行数学计算（安全版本）
    """
    import math
    
    # 安全的数学函数白名单
    safe_functions = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'pow': pow,
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
        'tan': math.tan, 'log': math.log, 'log10': math.log10,
        'exp': math.exp, 'pi': math.pi, 'e': math.e
    }
    
    try:
        # 只允许数字、运算符和安全的函数名
        result = eval(expression, {"__builtins__": {}}, safe_functions)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"计算错误: {str(e)}"}


@tool(
    name="search_web",
    description="搜索网络信息",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="搜索关键词",
            required=True,
            min_length=1,
            max_length=200
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="返回结果数量",
            default=5,
            min_value=1,
            max_value=20
        )
    ],
    examples=[
        {"query": "Python 教程", "num_results": 5}
    ],
    tags=["search", "web"]
)
def search_web(query: str, num_results: int = 5) -> dict:
    """
    搜索网络信息（模拟）
    """
    # 模拟搜索结果
    mock_results = [
        {"title": f"搜索结果 {i+1}: {query}", "url": f"https://example.com/{i+1}", "snippet": f"这是关于 {query} 的第 {i+1} 个结果..."}
        for i in range(num_results)
    ]
    
    return {
        "query": query,
        "results": mock_results,
        "total": num_results
    }


@tool(
    name="read_file",
    description="读取文件内容",
    parameters=[
        ToolParameter(
            name="file_path",
            type="string",
            description="文件路径",
            required=True
        )
    ],
    tags=["file", "io"]
)
def read_file(file_path: str) -> dict:
    """
    读取文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"file_path": file_path, "content": content, "success": True}
    except FileNotFoundError:
        return {"error": f"文件不存在: {file_path}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# ============ 演示 ============

def demo():
    """演示工具定义和使用"""
    print("=" * 60)
    print("工具定义演示")
    print("=" * 60)
    
    # 显示工具信息
    tools = [get_weather, calculate, search_web, read_file]
    
    for tool in tools:
        print(f"\n工具名称: {tool.name}")
        print(f"描述: {tool.description}")
        print(f"参数:")
        for param in tool.parameters:
            required = "必需" if param.required else "可选"
            default = f", 默认: {param.default}" if param.default is not None else ""
            print(f"  - {param.name} ({param.type}, {required}{default})")
        
        # 显示 JSON Schema
        print(f"\nJSON Schema:")
        import json
        print(json.dumps(tool.to_json_schema(), indent=2, ensure_ascii=False))
    
    # 执行工具
    print("\n" + "=" * 60)
    print("工具执行演示")
    print("=" * 60)
    
    # 天气查询
    print("\n1. 查询北京天气:")
    result = get_weather.execute(city="北京")
    print(f"结果: {result.to_dict()}")
    
    # 数学计算
    print("\n2. 数学计算:")
    result = calculate.execute(expression="2 + 3 * 4")
    print(f"结果: {result.to_dict()}")
    
    # 搜索
    print("\n3. 搜索:")
    result = search_web.execute(query="Python 教程", num_results=3)
    print(f"结果: {result.to_dict()}")
    
    # 参数验证
    print("\n" + "=" * 60)
    print("参数验证演示")
    print("=" * 60)
    
    print("\n缺少必需参数:")
    result = get_weather.execute()
    print(f"结果: {result.to_dict()}")
    
    print("\n参数值不在枚举中:")
    result = get_weather.execute(city="北京", unit="kelvin")
    print(f"结果: {result.to_dict()}")


if __name__ == "__main__":
    demo()