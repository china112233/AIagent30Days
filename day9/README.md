# Day 9: Agent 工具使用 (Tool Use)

## 概述

第九天学习 Agent 的工具使用能力，这是现代 AI Agent 最核心的能力之一。通过工具使用，Agent 可以突破大模型的局限，调用外部 API、执行代码、访问数据库、操作文件系统等，极大地扩展了 Agent 的能力边界。

## 学习目标

- 理解 Agent 工具使用的核心概念和架构
- 掌握工具定义、注册和调用的标准方法
- 学会实现 ReAct (Reasoning + Acting) 模式
- 掌握 Function Calling 和工具选择策略
- 实现完整的工具使用 Agent

## 核心概念

### 1. 为什么 Agent 需要工具？

大语言模型虽然强大，但存在固有的局限：

```
┌─────────────────────────────────────────────────────────────┐
│                    大模型的局限性                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   知识截止      │  │   无法执行      │                  │
│  │   训练数据      │  │   实际操作      │                  │
│  │   有时间限制    │  │   (如发邮件)    │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   无法访问      │  │   计算能力      │                  │
│  │   实时数据      │  │   有限          │                  │
│  │   (如股价)      │  │   (数学运算)    │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   可能产生      │  │   无法访问      │                  │
│  │   幻觉          │  │   外部系统      │                  │
│  │   (编造事实)    │  │   (数据库/API)  │                  │
│  └─────────────────┘  └─────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                    解决方案：工具使用

┌─────────────────────────────────────────────────────────────┐
│                    工具扩展的能力                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔍 搜索引擎     → 获取实时信息                              │
│  💻 代码执行     → 精确计算、数据处理                        │
│  🗄️ 数据库       → 存取结构化数据                           │
│  📧 通讯工具     → 发送邮件、消息                            │
│  📁 文件系统     → 读写文件                                  │
│  🌐 API调用      → 接入外部服务                              │
│  📊 数据分析     → 可视化、统计分析                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 工具使用的架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent Tool Use Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         用户请求                                   │ │
│  │                   "帮我查今天的天气"                                │ │
│  └─────────────────────────────┬─────────────────────────────────────┘ │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Agent 核心引擎                                │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │ │
│  │   │  推理引擎   │ ←→ │  决策模块   │ ←→ │  执行器     │          │ │
│  │   │ (Reasoning) │    │ (Decision)  │    │ (Executor)  │          │ │
│  │   └─────────────┘    └─────────────┘    └──────┬──────┘          │ │
│  │                                                  │                  │ │
│  └──────────────────────────────────────────────────┼────────────────┘ │
│                                                     │                  │
│                                ┌────────────────────┴───────────────┐  │
│                                │           工具注册表                │  │
│                                │         (Tool Registry)            │  │
│                                ├────────────────────────────────────┤  │
│                                │                                    │  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│  │
│  │ 搜索工具 │  │ 计算器   │  │ 天气API  │  │ 数据库   │  │ 文件系统 ││  │
│  │ search() │  │ calc()   │  │ weather()│  │ query()  │  │ read()   ││  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘│  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. 工具定义规范

一个完整的工具定义应包含以下要素：

```python
Tool = {
    "name": "get_weather",           # 工具名称（唯一标识）
    "description": "获取指定城市的天气信息",  # 工具描述
    "parameters": {                   # 参数定义
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "温度单位"
            }
        },
        "required": ["city"]
    },
    "returns": {                      # 返回值定义
        "type": "object",
        "properties": {
            "temperature": {"type": "number"},
            "condition": {"type": "string"}
        }
    }
}
```

### 4. ReAct 模式

ReAct (Reasoning + Acting) 是一种经典的工具使用模式：

```
┌─────────────────────────────────────────────────────────────┐
│                     ReAct 循环                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────────────────────────────────────────┐     │
│   │                                                  │     │
│   │    ┌─────────┐                                   │     │
│   │    │  Thought │  "我需要查询天气信息"            │     │
│   │    └────┬────┘                                   │     │
│   │         │                                        │     │
│   │         ▼                                        │     │
│   │    ┌─────────┐                                   │     │
│   │    │  Action  │  get_weather(city="北京")        │     │
│   │    └────┬────┘                                   │     │
│   │         │                                        │     │
│   │         ▼                                        │     │
│   │    ┌─────────┐                                   │     │
│   │    │Observation│ {"temp": 25, "condition":"晴"}  │     │
│   │    └────┬────┘                                   │     │
│   │         │                                        │     │
│   │         ▼                                        │     │
│   │    ┌─────────┐                                   │     │
│   │    │  Thought │  "我已经获得了天气信息，可以回答" │     │
│   │    └────┬────┘                                   │     │
│   │         │                                        │     │
│   │         ▼                                        │     │
│   │    ┌─────────┐                                   │     │
│   │    │  Answer  │  "北京今天天气晴朗，温度25°C"    │     │
│   │    └─────────┘                                   │     │
│   │                                                  │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

ReAct 循环的核心组件：

| 组件 | 说明 | 示例 |
|------|------|------|
| Thought | 推理思考，决定下一步行动 | "用户想知道天气，我需要调用天气API" |
| Action | 执行工具调用 | `get_weather(city="北京")` |
| Observation | 观察工具返回结果 | `{"temp": 25, "condition": "晴"}` |
| Answer | 最终回答用户 | "北京今天天气晴朗，温度25°C" |

### 5. 工具选择策略

Agent 如何决定使用哪个工具？

```
┌─────────────────────────────────────────────────────────────┐
│                     工具选择策略                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 基于语义匹配 (Semantic Matching)                         │
│     ┌─────────────────────────────────────────────┐         │
│     │  用户请求 → 向量化 → 与工具描述向量匹配      │         │
│     │  选择相似度最高的工具                        │         │
│     └─────────────────────────────────────────────┘         │
│                                                             │
│  2. 基于 LLM 决策 (LLM-based Selection)                      │
│     ┌─────────────────────────────────────────────┐         │
│     │  将用户请求和工具列表作为 Prompt 输入 LLM    │         │
│     │  LLM 输出要调用的工具名称和参数              │         │
│     └─────────────────────────────────────────────┘         │
│                                                             │
│  3. Function Calling (OpenAI 格式)                          │
│     ┌─────────────────────────────────────────────┐         │
│     │  将工具定义为 function schema               │         │
│     │  LLM 返回结构化的 function call             │         │
│     │  {                                          │         │
│     │    "name": "get_weather",                   │         │
│     │    "arguments": "{\"city\": \"北京\"}"      │         │
│     │  }                                          │         │
│     └─────────────────────────────────────────────┘         │
│                                                             │
│  4. 规则匹配 (Rule-based)                                    │
│     ┌─────────────────────────────────────────────┐         │
│     │  根据关键词或意图匹配工具                    │         │
│     │  "天气" → weather_tool                      │         │
│     │  "计算" → calculator_tool                   │         │
│     └─────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6. 工具链与组合

复杂任务可能需要组合多个工具：

```
┌─────────────────────────────────────────────────────────────┐
│                     工具链示例                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  任务: "分析某公司股价并生成报告"                            │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ 股价API │ →  │ 数据分析 │ →  │ 图表生成 │ →  │ 文件保存 │ │
│  │获取数据 │    │ 计算指标 │    │ 可视化  │    │ 写入文件 │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                             │
│  Tool Chain:                                                │
│  get_stock_price → analyze_data → create_chart → save_file │
│                                                             │
└─────────────────────────────────────────────────────────────┘

工具组合模式：

1. 顺序模式 (Sequential)
   A → B → C → D
   
2. 并行模式 (Parallel)
   A → B
   A → C
   B + C → D
   
3. 条件模式 (Conditional)
   A → if condition then B else C → D
   
4. 循环模式 (Loop)
   A → while condition: B → C
```

### 7. 错误处理与重试

工具调用可能失败，需要健壮的错误处理：

```python
class ToolExecutionError(Exception):
    """工具执行错误"""
    def __init__(self, tool_name: str, error: str, retry_count: int = 0):
        self.tool_name = tool_name
        self.error = error
        self.retry_count = retry_count

# 错误处理策略
error_handling_strategies = {
    "retry": "重试执行（带退避策略）",
    "fallback": "使用备用工具",
    "ask_user": "询问用户提供更多信息",
    "skip": "跳过此步骤，继续执行",
    "abort": "中止整个任务"
}

# 重试策略
def execute_with_retry(tool, args, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            return tool.execute(**args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise ToolExecutionError(tool.name, str(e), attempt)
            time.sleep(backoff_factor ** attempt)
```

## 文件说明

### 1. tool_definition.py - 工具定义

实现工具的定义和描述：
- `Tool`: 工具基类
- `ToolParameter`: 参数定义
- `ToolResult`: 执行结果
- 装饰器 `@tool` 快速定义工具

### 2. tool_registry.py - 工具注册表

实现工具的注册和管理：
- `ToolRegistry`: 工具注册表
- 工具的注册、查找、列出
- 工具验证和冲突处理

### 3. tool_agent.py - 工具使用 Agent

实现工具调用 Agent：
- `ToolAgent`: 工具使用 Agent
- 工具选择和调用
- 结果处理和响应生成

### 4. react_agent.py - ReAct Agent

实现 ReAct 模式的 Agent：
- `ReActAgent`: ReAct Agent
- Thought-Action-Observation 循环
- 多轮工具调用

## 使用示例

### 基本工具定义

```python
from tool_definition import tool, ToolParameter

@tool(
    name="get_weather",
    description="获取指定城市的天气信息",
    parameters=[
        ToolParameter(name="city", type="string", description="城市名称", required=True),
        ToolParameter(name="unit", type="string", description="温度单位", default="celsius")
    ]
)
def get_weather(city: str, unit: str = "celsius") -> dict:
    """获取天气信息"""
    # 实际实现会调用天气 API
    return {
        "city": city,
        "temperature": 25,
        "unit": unit,
        "condition": "晴天"
    }
```

### 使用工具注册表

```python
from tool_registry import ToolRegistry

# 创建注册表
registry = ToolRegistry()

# 注册工具
registry.register(get_weather)
registry.register(calculate)
registry.register(search_web)

# 列出所有工具
tools = registry.list_tools()

# 查找工具
weather_tool = registry.get_tool("get_weather")
```

### 使用 ReAct Agent

```python
from react_agent import ReActAgent
from tool_registry import ToolRegistry

# 创建工具注册表
registry = ToolRegistry()
registry.register(get_weather)
registry.register(calculate)
registry.register(search_web)

# 创建 Agent
agent = ReActAgent(
    tools=registry,
    model="gpt-4",
    max_iterations=5
)

# 执行任务
result = agent.run("北京今天的天气怎么样？温度是多少华氏度？")
print(result)
```

### 执行流程示例

```
任务: "北京今天的天气怎么样？温度是多少华氏度？"

Thought 1: 用户想知道北京的天气，还需要转换为华氏度
Action 1: get_weather(city="北京", unit="celsius")
Observation 1: {"city": "北京", "temperature": 25, "condition": "晴天"}

Thought 2: 获得了摄氏度温度25，需要转换为华氏度
Action 2: calculate(expression="(25 * 9/5) + 32")
Observation 2: {"result": 77}

Thought 3: 已经获得了所有需要的信息，可以回答用户
Answer: 北京今天天气晴朗，温度是25°C，相当于77°F。
```

## 运行示例

```bash
# 运行工具定义示例
python tool_definition.py

# 运行工具注册表示例
python tool_registry.py

# 运行工具 Agent 示例
python tool_agent.py

# 运行 ReAct Agent 示例
python react_agent.py
```

## 最佳实践

### 1. 工具设计原则

```
┌─────────────────────────────────────────────────────────────┐
│                     工具设计原则                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ 单一职责：每个工具只做一件事                             │
│  ✅ 清晰描述：让 LLM 能准确理解工具用途                       │
│  ✅ 完整文档：参数说明、返回值、异常情况                      │
│  ✅ 输入验证：检查参数类型和范围                             │
│  ✅ 错误处理：优雅地处理异常情况                             │
│  ✅ 幂等性：相同输入产生相同输出（如果适用）                  │
│  ✅ 超时控制：避免长时间阻塞                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 工具命名规范

```python
# ✅ 好的命名
get_weather      # 动词+名词，清晰明确
search_web       # 描述性强
calculate_math   # 功能明确

# ❌ 不好的命名
weather          # 缺少动词
do_something     # 描述不清
tool1            # 无意义命名
```

### 3. 安全考虑

```
┌─────────────────────────────────────────────────────────────┐
│                     工具安全考虑                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔒 权限控制：限制工具的访问权限                             │
│  🔒 输入过滤：防止注入攻击                                   │
│  🔒 输出检查：敏感信息过滤                                   │
│  🔒 资源限制：防止资源滥用                                   │
│  🔒 审计日志：记录工具调用                                   │
│  🔒 人工确认：敏感操作需要确认                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 扩展阅读

### Function Calling 标准

- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Tool Use: https://docs.anthropic.com/claude/docs/tool-use
- Google Gemini Function Calling: https://ai.google.dev/docs/function_calling

### 相关框架

- LangChain Tools: https://python.langchain.com/docs/modules/tools/
- Semantic Kernel: https://learn.microsoft.com/en-us/semantic-kernel/
- AutoGPT: https://github.com/Significant-Gravitas/Auto-GPT

### 学术论文

- ReAct: Synergizing Reasoning and Acting in Language Models (2022)
- Toolformer: Language Models Can Teach Themselves to Use Tools (2023)
- Gorilla: Large Language Model Connected with Massive APIs (2023)

## 练习

### 1. 基础练习

- 定义一个简单的计算器工具
- 实现工具注册和调用

### 2. 进阶练习

- 实现 ReAct Agent
- 添加多工具组合调用

### 3. 挑战练习

- 实现带错误处理和重试的工具执行
- 添加工具使用的历史记录和学习能力

## 小结

工具使用是 AI Agent 的核心能力：

- **工具定义**：清晰描述工具的能力和参数
- **工具注册**：统一管理和查找工具
- **工具选择**：让 LLM 智能选择合适的工具
- **工具执行**：安全、可靠地调用工具
- **结果处理**：将工具结果融入 Agent 回答

通过工具使用，Agent 可以突破大模型的局限，实现真正的"行动力"。

## 下一步

Day 10 将学习多 Agent 协作，让多个 Agent 协同完成复杂任务。