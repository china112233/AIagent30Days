# Day 16: LangGraph

## 概述

LangGraph 是 LangChain 生态中的图编排框架，专为构建有状态、多角色的 LLM 应用而设计。相比 LangChain 的链式调用，LangGraph 支持循环、分支和持久化状态，是实现复杂 Agent 工作流的理想工具。

## 学习目标

- 理解 LangGraph 的核心概念：节点、边、状态图
- 掌握状态定义与状态管理机制
- 学会构建循环与条件分支工作流
- 实现多 Agent 协作的复杂系统
- 了解人机交互和持久化机制

## 核心概念

### 1. LangGraph 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 架构图                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                    StateGraph                        │  │
│   │                                                     │  │
│   │    ┌────────┐      ┌────────┐      ┌────────┐     │  │
│   │    │ Node A │ ───→ │ Node B │ ───→ │ Node C │     │  │
│   │    └────────┘      └────────┘      └────────┘     │  │
│   │         ↑                               │          │  │
│   │         └───────────────────────────────┘          │  │
│   │                    (循环边)                         │  │
│   │                                                     │  │
│   │    State: { messages: [], count: 0, ... }          │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   核心组件：                                                │
│   - State: 图的状态对象，在节点间传递                      │
│   - Node: 执行单元，接收状态、返回状态更新                  │
│   - Edge: 连接节点，支持条件和循环                         │
│   - Graph: 状态图，编排整体流程                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. StateGraph vs Chain

| 特性 | LangChain Chain | LangGraph StateGraph |
|------|-----------------|---------------------|
| 执行模式 | 线性、DAG | 支持循环、分支 |
| 状态管理 | 无状态或简单内存 | 完整状态管理 |
| 控制流 | 固定流程 | 动态条件路由 |
| 持久化 | 无内置支持 | 支持检查点 |
| 人机交互 | 无内置支持 | 原生支持中断/恢复 |
| 适用场景 | 简单管道 | 复杂 Agent 系统 |

### 3. 状态定义

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

# 定义状态结构
class AgentState(TypedDict):
    messages: list[dict]      # 对话历史
    current_step: str          # 当前步骤
    iterations: int            # 迭代次数
    result: str | None        # 最终结果
```

### 4. 节点类型

```
┌─────────────────────────────────────────────────────────────┐
│                      节点类型                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 函数节点                                                │
│     ┌─────────────────────────────────────────────────┐   │
│     │ def my_node(state: AgentState) -> AgentState:   │   │
│     │     # 处理逻辑                                    │   │
│     │     return { "field": "new_value" }             │   │
│     └─────────────────────────────────────────────────┘   │
│                                                             │
│  2. LLM 节点                                                │
│     ┌─────────────────────────────────────────────────┐   │
│     │ # 将 LLM 调用封装为节点                          │   │
│     │ llm_node = create_llm_node(llm, prompt_template)│   │
│     └─────────────────────────────────────────────────┘   │
│                                                             │
│  3. Tool 节点                                               │
│     ┌─────────────────────────────────────────────────┐   │
│     │ # 工具执行节点                                    │   │
│     │ tool_node = ToolNode(tools=[...])               │   │
│     └─────────────────────────────────────────────────┘   │
│                                                             │
│  4. 条件节点                                                │
│     ┌─────────────────────────────────────────────────┐   │
│     │ # 根据状态决定下一步                              │   │
│     │ def router(state) -> str:                       │   │
│     │     if state["need_tool"]:                       │   │
│     │         return "tool_node"                       │   │
│     │     return "final"                               │   │
│     └─────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. 边的类型

```python
# 1. 普通边：固定连接
graph.add_edge("node_a", "node_b")

# 2. 条件边：动态路由
graph.add_conditional_edges(
    "router_node",
    lambda state: "tool" if state["need_tool"] else "final",
    {
        "tool": "tool_node",
        "final": "final_node"
    }
)

# 3. 入口边：指定起点
graph.set_entry_point("start_node")

# 4. 结束边：指定终点
graph.set_finish_point("end_node")
```

### 6. 多 Agent 工作流模式

```
┌─────────────────────────────────────────────────────────────┐
│                   多 Agent 工作流模式                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式 1: 顺序协作                                           │
│  ┌────────┐    ┌────────┐    ┌────────┐                   │
│  │ Agent 1│ → │ Agent 2│ → │ Agent 3│                   │
│  │ 研究员  │    │ 分析师  │    │ 撰稿人  │                   │
│  └────────┘    └────────┘    └────────┘                   │
│                                                             │
│  模式 2: 层级管理                                           │
│              ┌────────┐                                     │
│              │ 主控Agent│                                    │
│              │ (Supervisor)│                               │
│              └────────┘                                     │
│               ↙    ↘                                        │
│        ┌────────┐  ┌────────┐                              │
│        │ Worker1│  │ Worker2│                              │
│        └────────┘  └────────┘                              │
│                                                             │
│  模式 3: 专家团队                                           │
│                    ┌────────┐                               │
│              ┌────→│ 专家 A │←────┐                        │
│              │     └────────┘     │                        │
│        ┌────────┐           ┌────────┐                     │
│        │ 路由器  │──────────→│ 专家 B │                     │
│        └────────┘           └────────┘                     │
│              │     ┌────────┐     │                        │
│              └────→│ 专家 C │←────┘                        │
│                    └────────┘                               │
│                                                             │
│  模式 4: 循环迭代                                           │
│        ┌──────────────────────────────┐                    │
│        ↓                              │                    │
│  ┌────────┐    ┌────────┐    ┌────────┐                   │
│  │ 生成器  │ → │ 评估器  │ → │ 优化器  │                   │
│  └────────┘    └────────┘    └────────┘                   │
│        ↑                              │                    │
│        └────────(不满意则循环)──────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装依赖

```bash
pip install langgraph langchain-openai langchain-core
pip install langchain-community  # 可选，用于工具集成
```

### 配置环境

参考 `.env.example` 创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

## 练习文件说明

### `langgraph_basics.py` - 基础入门

演示 LangGraph 的核心概念：
- StateGraph 创建与配置
- 节点定义和添加
- 边的连接方式
- 图的编译和执行

### `state_management.py` - 状态管理

深入状态管理机制：
- 状态定义与类型注解
- 状态更新与合并策略
- 消息历史管理
- 检查点持久化

### `control_flow.py` - 循环与分支

展示控制流设计：
- 条件边与动态路由
- 循环与迭代执行
- 提前终止条件
- 错误处理与重试

### `multi_agent.py` - 多 Agent 工作流

实现多 Agent 协作：
- Agent 间通信
- 层级管理架构
- 专家团队路由
- 复杂任务分解

## 运行示例

```bash
# 运行基础示例
python day16/langgraph_basics.py

# 运行状态管理示例
python day16/state_management.py

# 运行控制流示例
python day16/control_flow.py

# 运行多 Agent 示例
python day16/multi_agent.py
```

## 最佳实践

### 1. 状态设计原则

```python
# ✅ 好的状态设计：清晰、有明确用途
class GoodState(TypedDict):
    messages: Annotated[list, "对话历史"]
    current_task: str
    completed_steps: list[str]
    max_iterations: int

# ❌ 避免：过于复杂或缺少类型注解
class BadState(TypedDict):
    data: dict  # 太模糊
    temp: Any   # 类型不明确
```

### 2. 节点职责单一

```python
# ✅ 每个节点专注一件事
def analyze_node(state: State) -> dict:
    """分析任务，不做执行"""
    return {"analysis": "..."}

def execute_node(state: State) -> dict:
    """执行任务，不做决策"""
    return {"result": "..."}

# ❌ 避免：一个节点做太多事
def do_everything_node(state: State) -> dict:
    # 分析 + 决策 + 执行 + 验证
    pass
```

### 3. 条件边清晰

```python
# ✅ 清晰的路由函数
def route_next_step(state: State) -> str:
    if state["need_tool"]:
        return "tool_node"
    elif state["is_complete"]:
        return END
    else:
        return "agent_node"

# ❌ 避免复杂的内联逻辑
graph.add_conditional_edges(
    "node",
    lambda s: "a" if (s["x"] > 0 and s["y"] < 10) or s["z"] else "b"
)
```

### 4. 循环限制

```python
# ✅ 始终设置循环上限
class SafeState(TypedDict):
    messages: list
    iterations: int  # 追踪迭代次数

def check_iterations(state: State) -> str:
    if state["iterations"] >= 10:  # 防止无限循环
        return END
    return "continue"
```

### 5. 持久化检查点

```python
from langgraph.checkpoint.memory import MemorySaver

# 使用检查点支持中断/恢复
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 执行时传入 thread_id
result = graph.invoke(
    {"input": "..."},
    config={"configurable": {"thread_id": "user-123"}}
)
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无限循环 | 缺少终止条件 | 添加迭代计数器和上限检查 |
| 状态丢失 | 状态更新不正确 | 确保节点返回正确的状态更新 |
| 类型错误 | 状态类型不匹配 | 使用 TypedDict 明确定义类型 |
| 图编译失败 | 节点未连接完整 | 确保所有节点都有入边或出边 |
| 条件路由错误 | 条件函数返回值不匹配 | 检查条件边映射是否完整 |
| 检查点不工作 | 未传入 thread_id | 使用 config.thread_id |

## 架构对比

| 框架 | 适用场景 | 复杂度 | 学习曲线 |
|------|----------|--------|----------|
| LangChain Chain | 简单线性流程 | 低 | 低 |
| LangGraph | 复杂有状态 Agent | 中 | 中 |
| AutoGen | 多 Agent 对话 | 高 | 高 |
| CrewAI | 团队协作场景 | 中 | 低 |

## 学习成果

完成本天学习后，你将能够：
- ✅ 理解 LangGraph 的核心架构和设计理念
- ✅ 构建有状态的 LLM 应用工作流
- ✅ 设计循环迭代和条件分支的 Agent 系统
- ✅ 实现多 Agent 协作的复杂应用
- ✅ 使用检查点实现人机交互和断点恢复

## 下一步

Day 17 将学习 LlamaIndex 基础，内容包括：数据索引、查询引擎、RAG 管道、文档管理。

## 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph 示例](https://langchain-ai.github.io/langgraph/tutorials/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)