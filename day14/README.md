# Day 14: LangChain 基础

## 概述

LangChain 是目前最流行的 LLM 应用开发框架之一，它提供了一套完整的工具链，帮助开发者快速构建复杂的 LLM 应用。本日学习 LangChain 的核心概念和基础使用方法。

## 学习目标

- 理解 LangChain 的整体架构和设计理念
- 掌握 Chain（链）的概念和使用方法
- 学会使用 Memory（记忆）管理对话状态
- 理解 Tools（工具）的定义和使用
- 实现基础的 Agent（智能代理）

## 核心概念

### 1. LangChain 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LangChain 架构图                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         应用层 (Applications)                       │ │
│  │     聊天机器人 │ RAG 系统 │ Agent │ 数据分析 │ 内容生成           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         Chain 层 (链编排)                          │ │
│  │                                                                     │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │ │
│  │   │ LLMChain │    │ Router  │    │ Transform│    │ Sequential│     │ │
│  │   │         │    │  Chain  │    │  Chain   │    │   Chain   │     │ │
│  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                       核心组件层 (Core Components)                  │ │
│  │                                                                     │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │ │
│  │   │   LLM   │  │  Prompt │  │  Memory │  │  Tools  │            │ │
│  │   │  Model  │  │ Template│  │         │  │         │            │ │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘            │ │
│  │                                                                     │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │ │
│  │   │  Output │  │ Document│  │ Retriever│  │  Agent  │            │ │
│  │   │  Parser │  │ Loader  │  │         │  │         │            │ │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         集成层 (Integrations)                       │ │
│  │                                                                     │ │
│  │   OpenAI │ Anthropic │ DeepSeek │ Chroma │ Pinecone │ Serper     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. 核心组件详解

#### 2.1 Model I/O（模型输入输出）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 模型 (LLM/Chat Model)
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 2. 提示词模板 (Prompt Template)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}。"),
    ("user", "{input}")
])

# 3. 输出解析器 (Output Parser)
parser = StrOutputParser()

# 组合成链
chain = prompt | llm | parser
```

#### 2.2 Chain（链）

Chain 是 LangChain 的核心概念，用于将多个组件串联起来：

```
┌─────────────────────────────────────────────────────────────┐
│                     Chain 执行流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入 ──→ [Prompt] ──→ [LLM] ──→ [Parser] ──→ 输出        │
│                                                             │
│   示例：                                                    │
│   {"role": "翻译", "input": "hello"}                       │
│       ↓                                                    │
│   System: 你是一个专业的翻译。                               │
│   User: hello                                              │
│       ↓                                                    │
│   LLM 处理...                                               │
│       ↓                                                    │
│   "你好"                                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**常用 Chain 类型**：

| Chain 类型 | 用途 | 示例场景 |
|-----------|------|----------|
| LLMChain | 最基础的链，组合 Prompt + LLM | 简单问答、文本生成 |
| SequentialChain | 顺序执行多个链 | 多步骤处理流程 |
| RouterChain | 根据输入路由到不同链 | 多领域问答系统 |
| TransformChain | 自定义转换逻辑 | 数据预处理 |

#### 2.3 Memory（记忆）

Memory 让应用能够记住之前的对话：

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory 类型                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ConversationBufferMemory                                │
│     ┌─────────────────────────────────────────────────┐     │
│     │  存储完整对话历史                                │     │
│     │  优点：信息完整                                  │     │
│     │  缺点：Token 消耗大                              │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  2. ConversationBufferWindowMemory                          │
│     ┌─────────────────────────────────────────────────┐     │
│     │  只保留最近 N 轮对话                             │     │
│     │  优点：控制 Token 消耗                           │     │
│     │  缺点：可能丢失重要信息                          │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  3. ConversationSummaryMemory                               │
│     ┌─────────────────────────────────────────────────┐     │
│     │  将历史对话压缩为摘要                            │     │
│     │  优点：节省 Token                                │     │
│     │  缺点：细节可能丢失                              │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
│  4. VectorStoreMemory                                       │
│     ┌─────────────────────────────────────────────────┐     │
│     │  使用向量存储和检索相关记忆                      │     │
│     │  优点：语义检索，可扩展                          │     │
│     │  缺点：需要额外存储                              │     │
│     └─────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4 Tools（工具）

Tools 让 LLM 能够调用外部功能：

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称，如"北京"、"上海"
    
    Returns:
        天气信息字符串
    """
    # 实际实现会调用天气 API
    weather_data = {
        "北京": "晴天，温度 25°C",
        "上海": "多云，温度 28°C",
        "广州": "小雨，温度 30°C"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

# 工具定义包含：
# - name: 工具名称
# - description: 工具描述（LLM 用它来决定何时调用）
# - args_schema: 参数 Schema
# - function: 实际执行函数
```

#### 2.5 Agent（智能代理）

Agent 能够自主决定使用哪些工具来完成任务：

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent 执行流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户输入: "北京今天天气怎么样？"                          │
│       │                                                    │
│       ▼                                                    │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Agent 思考过程                       │     │
│   │  Thought: 用户想知道天气，我应该使用天气工具     │     │
│   │  Action: get_weather                             │     │
│   │  Action Input: {"city": "北京"}                  │     │
│   └──────────────────────────────────────────────────┘     │
│       │                                                    │
│       ▼                                                    │
│   工具执行: get_weather("北京")                            │
│       │                                                    │
│       ▼                                                    │
│   Observation: "晴天，温度 25°C"                           │
│       │                                                    │
│       ▼                                                    │
│   ┌──────────────────────────────────────────────────┐     │
│   │              Agent 最终回答                       │     │
│   │  "北京今天天气晴朗，温度 25 摄氏度。"            │     │
│   └──────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Agent 类型**：

| Agent 类型 | 特点 | 适用场景 |
|-----------|------|----------|
| Zero-shot Agent | 不需要示例，直接执行 | 简单任务 |
| Structured Tool Agent | 支持复杂参数结构 | 需要多参数的工具 |
| OpenAI Functions Agent | 使用 OpenAI Function Calling | OpenAI 模型 |
| ReAct Agent | 推理+行动模式 | 需要多步推理的任务 |

### 3. LCEL（LangChain Expression Language）

LCEL 是 LangChain 的声明式语法，让链的组合更加简洁：

```python
# 传统方式
chain = LLMChain(llm=llm, prompt=prompt)

# LCEL 方式（推荐）
chain = prompt | llm | output_parser

# LCEL 优势：
# 1. 简洁直观
# 2. 支持流式输出
# 3. 支持异步
# 4. 自动处理错误
```

## 快速开始

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### 配置环境

```bash
# 创建 .env 文件
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### 基础示例

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 创建模型
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7
)

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("user", "{input}")
])

# 创建输出解析器
parser = StrOutputParser()

# 创建链
chain = prompt | llm | parser

# 执行
response = chain.invoke({"input": "你好，请介绍一下你自己"})
print(response)
```

## 练习文件说明

### `basic_chain.py` - 基础 Chain 使用

演示 LangChain 最核心的链操作：
- Prompt Template 的创建和使用
- LLM 模型配置
- Output Parser 输出解析
- LCEL 语法实战
- 多步骤链的组合

### `memory_example.py` - Memory 记忆系统

演示对话记忆管理：
- ConversationBufferMemory 完整记忆
- ConversationBufferWindowMemory 滑动窗口
- ConversationSummaryMemory 摘要记忆
- 记忆与链的集成

### `tools_example.py` - Tools 工具使用

演示工具的定义和使用：
- 自定义工具创建
- 工具描述最佳实践
- 多工具组合
- 工具调用流程

### `agent_example.py` - Agent 智能代理

演示 Agent 的创建和使用：
- ReAct Agent 实现
- 工具绑定
- Agent 执行过程
- 多轮对话 Agent

## 运行示例

```bash
# 运行基础链示例
python day14/basic_chain.py

# 运行记忆系统示例
python day14/memory_example.py

# 运行工具示例
python day14/tools_example.py

# 运行 Agent 示例
python day14/agent_example.py
```

## 最佳实践

### 1. 使用 LCEL 语法

```python
# ✅ 推荐：使用 LCEL
chain = prompt | llm | parser

# ❌ 不推荐：使用旧的 LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

### 2. 合理选择 Memory 类型

```python
# 短对话：完整记忆
memory = ConversationBufferMemory()

# 长对话：窗口记忆
memory = ConversationBufferWindowMemory(k=5)

# 超长对话：摘要记忆
memory = ConversationSummaryMemory(llm=llm)
```

### 3. 工具描述要清晰

```python
@tool
def search_database(query: str) -> str:
    """在产品数据库中搜索相关信息。
    
    当用户询问产品信息、价格、库存时使用此工具。
    
    Args:
        query: 搜索关键词，如产品名称或型号
    
    Returns:
        匹配的产品信息，包括名称、价格、库存状态
    """
    pass
```

### 4. 处理错误和异常

```python
from langchain_core.runnables import RunnableLambda

def safe_invoke(chain, input_data):
    try:
        return chain.invoke(input_data)
    except Exception as e:
        return f"执行出错: {str(e)}"

# 或者使用 LCEL 的错误处理
chain = prompt | llm | parser
chain = chain.with_retry(stop_after_attempt=3)
```

### 5. 使用流式输出提升体验

```python
# 流式输出
for chunk in chain.stream({"input": "讲一个故事"}):
    print(chunk, end="", flush=True)
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| API 调用失败 | API Key 错误或网络问题 | 检查 .env 配置，确认网络连接 |
| Token 超限 | 输入或记忆过长 | 使用窗口记忆或摘要记忆 |
| Agent 选错工具 | 工具描述不清晰 | 优化工具的 description |
| 输出格式不对 | 缺少输出解析器 | 添加 OutputParser 或格式要求 |
| 响应慢 | 链太长或模型慢 | 优化链结构，考虑并行执行 |

## LangChain vs 原生开发

| 特性 | 原生开发 | LangChain |
|------|----------|-----------|
| 代码量 | 较多 | 较少 |
| 学习曲线 | 低 | 中等 |
| 灵活性 | 高 | 中等 |
| 可维护性 | 一般 | 较好 |
| 生态系统 | 无 | 丰富 |
| 调试难度 | 低 | 较高 |

**选择建议**：
- 简单应用：原生开发足够
- 复杂应用：LangChain 提高效率
- 团队协作：LangChain 统一规范
- 快速原型：LangChain 加速开发

## 学习成果

完成本天学习后，你将能够：
- ✅ 理解 LangChain 的核心架构和设计理念
- ✅ 使用 LCEL 构建处理链
- ✅ 实现对话记忆管理
- ✅ 定义和使用自定义工具
- ✅ 创建基础的 Agent 应用

## 下一步

Day 15 将深入学习 LangChain 进阶，包括 LCEL 高级用法、回调系统、自定义组件和 RAG 实现。

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain 教程](https://python.langchain.com/docs/tutorials/)
- [LCEL 文档](https://python.langchain.com/docs/expression_language/)