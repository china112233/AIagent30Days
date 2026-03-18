# Day 6: AI Agent（智能代理）

## 概述

第六天学习 AI Agent 开发，让 AI 能够自主规划和执行复杂任务。Agent 是能够感知环境、做出决策、执行动作的智能系统。

## 学习目标

- 理解 AI Agent 的核心概念和架构
- 掌握 ReAct（推理+行动）模式
- 学会定义和调用工具（Tools）
- 实现 Agent 执行循环
- 构建能自主规划的多步任务系统

## 核心概念

### 1. 什么是 AI Agent？

AI Agent 是一个能够自主完成任务的智能系统：

```
用户目标 → Agent 思考 → 选择工具 → 执行动作 → 观察结果 → 继续思考 → ... → 完成任务
```

**与传统 LLM 的区别**：

| 特性 | 传统 LLM | AI Agent |
|------|----------|----------|
| 交互模式 | 单轮问答 | 多轮自主执行 |
| 能力范围 | 仅文本生成 | 可调用工具、执行操作 |
| 决策能力 | 被响应 | 主动规划、自我修正 |
| 环境感知 | 无 | 可获取实时信息 |

### 2. Agent 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Agent 架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Agent 核心                        │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────┐   │   │
│  │  │  规划器   │  │  推理器   │  │   记忆管理    │   │   │
│  │  │ (Planner) │  │ (Reasoner)│  │   (Memory)    │   │   │
│  │  └───────────┘  └───────────┘  └───────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    工具层                            │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  搜索工具  │  计算器  │  文件操作  │  API调用  │ ... │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    执行循环                          │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │   while not done:                                   │   │
│  │       thought = think(current_state)               │   │
│  │       action = decide_action(thought)              │   │
│  │       result = execute(action)                     │   │
│  │       observe(result)                              │   │
│  │       update_state()                               │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. ReAct 模式

ReAct = Reasoning（推理）+ Acting（行动）

```
用户问题: "北京现在的天气怎么样？"

Thought 1: 我需要查询北京的实时天气
Action 1: search_weather("北京")
Observation 1: 北京今天晴，气温 25°C，湿度 40%

Thought 2: 我已经获取了天气信息，可以回答用户了
Action 2: finish("北京今天天气晴朗，气温25°C，湿度40%")
```

**ReAct 循环**：
1. **Thought**：分析当前状态，思考下一步
2. **Action**：选择并执行一个工具
3. **Observation**：观察执行结果
4. 重复直到任务完成

### 4. 工具（Tools）定义

工具是 Agent 与外部世界交互的接口：

```python
@dataclass
class Tool:
    name: str           # 工具名称
    description: str    # 工具描述（LLM用于决策）
    parameters: dict    # 参数schema
    function: Callable  # 执行函数
```

常用工具类型：
- **搜索工具**：网络搜索、知识库检索
- **计算工具**：数学计算、数据分析
- **文件工具**：读写文件、处理文档
- **API工具**：调用外部服务

### 5. Agent 记忆

Agent 需要记忆来保持上下文：

```
┌─────────────────────────────────────────┐
│              记忆类型                    │
├─────────────────────────────────────────┤
│                                         │
│  短期记忆（工作记忆）                    │
│  ├── 当前对话上下文                     │
│  ├── 最近执行的动作                     │
│  └── 临时推理状态                       │
│                                         │
│  长期记忆                               │
│  ├── 历史交互记录                       │
│  ├── 学到的知识/规则                    │
│  └── 用户偏好设置                       │
│                                         │
└─────────────────────────────────────────┘
```

## 快速开始

### 安装依赖

```bash
pip install openai python-dotenv
```

### 基础示例

```python
from tools import Tool, ToolRegistry
from basic_agent import BasicAgent

# 1. 定义工具
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

# 2. 注册工具
registry = ToolRegistry()
registry.register(Tool(
    name="calculator",
    description="计算数学表达式，如: 2+2, 3*4, sqrt(16)",
    function=calculator
))

# 3. 创建 Agent
agent = BasicAgent(tools=registry)

# 4. 运行 Agent
result = agent.run("请帮我计算 23 * 45 + 67")
print(result)
```

## 练习文件

### `tools.py`
工具定义和管理：
- Tool 基类定义
- 工具注册表
- 内置工具实现（搜索、计算器等）
- 工具调用和验证

### `basic_agent.py`
基础 Agent 实现：
- Agent 核心循环
- ReAct 模式实现
- 消息历史管理
- 简单任务执行

### `react_agent.py`
完整 ReAct Agent：
- 结构化思维链
- 多步推理
- 自我修正能力
- 详细执行日志

### `planning_agent.py`
任务规划 Agent：
- 目标分解
- 计划生成
- 步骤执行
- 进度跟踪

## Agent 执行流程

### 1. 任务接收

```python
def run(self, task: str) -> str:
    """
    执行用户任务

    Args:
        task: 用户任务描述

    Returns:
        最终结果
    """
    self.task = task
    self.history = []
    self.step_count = 0

    return self.execute_loop()
```

### 2. 执行循环

```python
def execute_loop(self) -> str:
    """Agent 执行主循环"""
    while self.step_count < self.max_steps:
        # 1. 思考
        thought = self.think()

        # 2. 决策动作
        action = self.decide_action(thought)

        # 3. 检查是否完成
        if action.is_finish():
            return action.result

        # 4. 执行动作
        observation = self.execute(action)

        # 5. 记录历史
        self.record(thought, action, observation)

        self.step_count += 1

    return "达到最大步数限制"
```

### 3. 思考与决策

```python
def think(self) -> str:
    """
    Agent 思考过程

    Returns:
        思考结果
    """
    # 构建 prompt
    prompt = self.build_thought_prompt()

    # 调用 LLM
    response = self.llm.chat(prompt)

    # 解析思考内容
    return self.parse_thought(response)
```

## 工具定义详解

### 1. 基础工具类

```python
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class Tool:
    """工具定义"""
    name: str                      # 工具名称
    description: str               # 功能描述
    parameters: Dict[str, Any]     # 参数 Schema
    function: Callable             # 执行函数
    examples: list = None          # 使用示例

    def run(self, **kwargs) -> str:
        """执行工具"""
        return self.function(**kwargs)
```

### 2. 工具注册表

```python
class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """获取工具"""
        return self._tools.get(name)

    def get_descriptions(self) -> str:
        """获取所有工具描述（用于 prompt）"""
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
```

### 3. 内置工具示例

```python
# 计算器工具
def calculator(expression: str) -> str:
    """计算数学表达式"""
    import math
    allowed_names = {
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
        'log': math.log, 'pi': math.pi, 'e': math.e
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"错误: {e}"

# 搜索工具
def search(query: str) -> str:
    """模拟搜索"""
    # 实际项目中可接入真实搜索 API
    mock_data = {
        "天气": "今天晴天，气温25°C",
        "新闻": "AI技术持续发展...",
    }
    for key, value in mock_data.items():
        if key in query:
            return value
    return "未找到相关信息"
```

## ReAct Prompt 设计

### 系统提示词

```python
REACT_SYSTEM_PROMPT = """你是一个智能代理，能够使用工具来完成任务。

你可以使用以下工具:
{tool_descriptions}

请使用以下格式进行思考和行动:

Thought: 思考当前情况，分析需要做什么
Action: 工具名称
Action Input: 工具输入参数（JSON 格式）

当你完成任务时，使用:
Thought: 我已经完成了任务
Final Answer: 最终答案

开始！
"""
```

### 交互示例

```
用户: 帮我计算 (23 + 45) * 2 的结果

Thought: 用户需要计算一个数学表达式，我应该使用计算器工具
Action: calculator
Action Input: {"expression": "(23 + 45) * 2"}

Observation: 136

Thought: 我已经得到计算结果，可以回答用户了
Final Answer: (23 + 45) * 2 = 136
```

## 任务规划

### 1. 目标分解

```python
def decompose_goal(self, goal: str) -> List[str]:
    """
    将目标分解为子任务

    Args:
        goal: 总体目标

    Returns:
        子任务列表
    """
    prompt = f"""请将以下目标分解为具体的执行步骤：

目标: {goal}

要求:
1. 每个步骤应该是一个独立的可执行任务
2. 步骤之间有明确的依赖关系
3. 按执行顺序排列

请输出步骤列表（每行一个步骤）:
"""
    response = self.llm.chat(prompt)
    return self.parse_steps(response)
```

### 2. 计划执行

```python
class PlanningAgent:
    """规划型 Agent"""

    def run(self, goal: str) -> str:
        # 1. 生成计划
        plan = self.generate_plan(goal)

        # 2. 执行计划
        results = []
        for i, step in enumerate(plan.steps):
            # 检查前置条件
            if not self.check_preconditions(step):
                return f"步骤 {i+1} 前置条件不满足"

            # 执行步骤
            result = self.execute_step(step)
            results.append(result)

            # 检查是否需要调整计划
            if self.need_replan(result):
                plan = self.update_plan(plan, i, result)

        return self.synthesize_results(results)
```

## 最佳实践

### 1. 工具设计原则

```python
# 好的工具设计
Tool(
    name="search_web",
    description="搜索互联网获取实时信息。输入搜索关键词，返回相关结果。",
    parameters={
        "query": {"type": "string", "description": "搜索关键词"}
    }
)

# 避免过度复杂的工具
# 一个工具只做一件事
```

### 2. Prompt 优化

```python
# 提供清晰的工具使用示例
TOOL_EXAMPLES = """
示例1:
Question: 北京现在几点？
Thought: 我需要查询当前时间
Action: get_time
Action Input: {"location": "北京"}

示例2:
Question: 150的平方根是多少？
Thought: 需要进行数学计算
Action: calculator
Action Input: {"expression": "sqrt(150)"}
"""
```

### 3. 错误处理

```python
def execute_with_retry(self, action, max_retries=3):
    """带重试的执行"""
    for i in range(max_retries):
        try:
            result = self.execute(action)
            if "错误" not in result:
                return result

            # 反思并修正
            thought = self.reflect_on_error(action, result)
            action = self.correct_action(thought)

        except Exception as e:
            if i == max_retries - 1:
                return f"执行失败: {e}"
            time.sleep(1)

    return "重试次数耗尽"
```

### 4. 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent 陷入循环 | 工具返回结果不明确 | 优化工具输出，添加终止条件 |
| 选择错误工具 | 工具描述不清晰 | 完善工具描述，添加示例 |
| 执行超时 | 步骤太多或卡住 | 设置最大步数，添加超时机制 |
| 结果不准确 | 推理过程有误 | 添加验证步骤，引入反思机制 |

## 进阶主题

### 1. 多 Agent 协作

```python
class MultiAgentSystem:
    """多 Agent 协作系统"""

    def __init__(self):
        self.planner = PlanningAgent()
        self.executor = ExecutionAgent()
        self.critic = CriticAgent()

    def run(self, task):
        # 规划
        plan = self.planner.create_plan(task)

        # 执行
        result = self.executor.execute_plan(plan)

        # 审核
        feedback = self.critic.review(result)

        # 根据反馈调整
        if feedback.needs_revision:
            result = self.executor.revise(result, feedback)

        return result
```

### 2. 反思机制

```python
def reflect(self, history: List[dict]) -> str:
    """
    反思执行过程

    Args:
        history: 执行历史

    Returns:
        反思结果
    """
    prompt = f"""请回顾以下执行历史，分析:

1. 哪些步骤执行得很好？
2. 哪些步骤可以改进？
3. 是否有更好的方法？

执行历史:
{self.format_history(history)}
"""
    return self.llm.chat(prompt)
```

### 3. 自我修正

```python
def self_correct(self, error: str, history: List[dict]):
    """自我修正"""
    correction_prompt = f"""在执行过程中遇到错误:

错误: {error}

历史: {history}

请分析错误原因并提供修正方案。
"""
    correction = self.llm.chat(correction_prompt)
    return self.parse_correction(correction)
```

## 学习成果

完成本天学习后，你将能够：
- 理解 AI Agent 的核心原理和架构
- 实现基于 ReAct 模式的智能代理
- 定义和管理工具（Tools）
- 构建能自主规划和执行任务的 Agent
- 处理 Agent 执行中的错误和异常

## 下一步

第七天将学习 Multi-Agent 系统，让多个 Agent 协作完成复杂任务。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)