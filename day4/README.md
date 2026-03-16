# Day 4: Function Calling（函数调用）

## 概述

第四天学习 Function Calling，让 AI 能够调用外部工具和 API，这是构建智能代理（Agent）的核心能力。

## 学习目标

- 理解 Function Calling 的工作原理
- 学会定义和注册函数工具
- 实现自动参数解析与函数执行
- 构建实用的工具调用代理

## 核心概念

### 1. 什么是 Function Calling？

Function Calling 让大语言模型能够"调用"预定义的函数：

```
用户请求 → 模型分析 → 决定调用哪个函数 → 提取参数 → 执行函数 → 返回结果
```

**核心价值**：
- 获取实时信息（天气、股价、新闻等）
- 执行实际操作（发送邮件、创建文件等）
- 访问外部系统（数据库、API等）

### 2. 函数定义结构

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",        # 函数名称
            "description": "获取指定城市的天气信息",  # 功能描述（重要！）
            "parameters": {               # 参数定义（JSON Schema）
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]      # 必需参数
            }
        }
    }
]
```

### 3. 调用流程

```python
# 1. 发送用户消息 + 工具定义
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,  # 注册工具
    tool_choice="auto"  # 自动决定是否调用
)

# 2. 检查模型是否想调用函数
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # 3. 执行函数
    result = get_weather(**arguments)
    
    # 4. 将结果返回给模型
    messages.append(response.choices[0].message)  # 助手的工具调用消息
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })
    
    # 5. 获取最终响应
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
```

## 练习文件

### `basic_function_call.py`
基础函数调用示例：
- 定义单个工具函数
- 模型自动选择和调用
- 参数自动解析
- 结果处理和返回

### `multi_function_agent.py`
多函数智能代理：
- 注册多个工具函数
- 智能选择合适的工具
- 处理复杂的多步骤任务
- 错误处理和重试机制

### `weather_agent.py`
天气查询代理：
- 实用的天气查询工具
- 多城市天气对比
- 天气预警功能
- 完整的交互式体验

### `calculator_agent.py`
计算器代理：
- 数学计算工具
- 复杂表达式解析
- 单位转换功能

## 工具定义最佳实践

### 1. 清晰的描述
```python
# ❌ 不好
"description": "查天气"

# ✅ 好
"description": "获取指定城市的实时天气信息，包括温度、湿度、天气状况等"
```

### 2. 完整的参数说明
```python
# ❌ 不好
"properties": {
    "city": {"type": "string"}
}

# ✅ 好
"properties": {
    "city": {
        "type": "string",
        "description": "城市名称，支持中英文，如：北京、上海、Beijing",
        "examples": ["北京", "上海", "广州"]
    }
}
```

### 3. 合理的约束
```python
"properties": {
    "count": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "返回结果数量（1-100）"
    }
}
```

## 常见应用场景

### 1. 信息获取
- 天气查询
- 股票行情
- 新闻检索
- 知识问答

### 2. 外部系统集成
- 数据库查询
- API 调用
- 文件操作
- 邮件发送

### 3. 智能决策
- 条件判断
- 多步骤任务
- 自动化流程

## 高级技巧

### 并行调用
```python
# 模型可以一次请求调用多个函数
tool_calls = response.choices[0].message.tool_calls

# 并行执行所有调用
results = []
for tool_call in tool_calls:
    result = execute_function(tool_call)
    results.append(result)
```

### 强制调用
```python
# 强制模型调用特定函数
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

### 不调用任何工具
```python
# 强制模型不使用工具
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    tool_choice="none"
)
```

## 错误处理

```python
def safe_execute_function(tool_call):
    """安全的函数执行，包含错误处理"""
    try:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name not in available_functions:
            return f"错误：未知函数 {function_name}"
        
        func = available_functions[function_name]
        result = func(**arguments)
        return result
        
    except json.JSONDecodeError:
        return "错误：参数解析失败"
    except TypeError as e:
        return f"错误：参数类型不匹配 - {e}"
    except Exception as e:
        return f"错误：执行失败 - {e}"
```

## 成本优化

Function Calling 会消耗更多 Token：
- 工具定义会增加 prompt tokens
- 多轮调用会增加总 tokens

**优化建议**：
- 只注册必要的工具
- 简化工具描述
- 使用 `tool_choice` 控制行为

## 学习成果

完成本天学习后，你将能够：
- ✅ 理解 Function Calling 的工作原理
- ✅ 定义和注册自定义工具函数
- ✅ 实现自动化的函数调用流程
- ✅ 构建实用的工具调用代理

## 下一步

第五天将学习 RAG（检索增强生成），结合向量数据库让 AI 拥有长期记忆和知识库检索能力。