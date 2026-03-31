# Day 15: LangChain 进阶

## 概述

在 Day 14 掌握了 LangChain 基础概念后，本日深入学习 LangChain 的进阶功能，包括 LCEL 高级用法、回调系统、自定义组件开发以及完整的 RAG 实现。这些技能将帮助你构建更复杂、更可控的 LLM 应用。

## 学习目标

- 深入理解 LCEL 表达式语言的高级特性
- 掌握回调系统实现追踪和监控
- 学会开发自定义 LLM、工具和检索器组件
- 实现完整的 RAG 应用管道
- 理解生产级应用的最佳实践

## 核心概念

### 1. LCEL 高级特性

#### 1.1 Runnable 接口详解

LCEL 的核心是 Runnable 接口，所有组件都实现了这个接口：

```
┌─────────────────────────────────────────────────────────────┐
│                  Runnable 接口方法                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  invoke()       - 同步执行，返回完整结果                     │
│  stream()       - 流式执行，逐块返回结果                     │
│  batch()        - 批量执行，处理多个输入                     │
│  ainvoke()      - 异步执行                                   │
│  astream()      - 异步流式执行                               │
│  abatch()       - 异步批量执行                               │
│                                                             │
│  compose()      - 组合另一个 Runnable                       │
│  with_retry()   - 添加重试机制                               │
│  with_fallbacks() - 添加备用方案                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2 管道操作符 `|`

管道操作符实现了 Runnable 的自动组合：

```python
# | 操作符等价于 .compose()
chain = prompt | llm | parser
# 等价于
chain = prompt.compose(llm).compose(parser)

# 类型流转：
# PromptTemplate (dict -> PromptValue)
#   → ChatModel (PromptValue -> AIMessage)
#   → StrOutputParser (AIMessage -> str)
```

#### 1.3 RunnableParallel 并行执行

```python
from langchain_core.runnables import RunnableParallel

# 并行执行多个分支
parallel_chain = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    analysis=analysis_chain
)

# 执行结果是一个字典：
# {
#     "summary": "...",
#     "translation": "...",
#     "analysis": "..."
# }
```

#### 1.4 RunnablePassthrough 数据透传

```python
from langchain_core.runnables import RunnablePassthrough

# 透传输入，同时添加新字段
chain = RunnablePassthrough.assign(
    context=retriever,  # 添加检索结果
    timestamp=lambda x: datetime.now()  # 添加时间戳
) | prompt | llm | parser

# 输入: {"question": "..."}
# 中间: {"question": "...", "context": [...], "timestamp: "..."}
```

#### 1.5 动态链 (RunnableLambda)

```python
from langchain_core.runnables import RunnableLambda

# 自定义处理逻辑
def process_input(x):
    if "translate" in x["question"].lower():
        return {"mode": "translation", "text": x["question"]}
    elif "summarize" in x["question"].lower():
        return {"mode": "summary", "text": x["question"]}
    else:
        return {"mode": "general", "text": x["question"]}

dynamic_chain = RunnableLambda(process_input) | router_chain
```

#### 1.6 条件分支 (RunnableBranch)

```python
from langchain_core.runnables import RunnableBranch

# 根据条件选择不同分支
branch = RunnableBranch(
    (lambda x: x["mode"] == "translation", translation_chain),
    (lambda x: x["mode"] == "summary", summary_chain),
    default_chain  # 默认分支
)
```

### 2. 回调系统

#### 2.1 回调处理器架构

```
┌─────────────────────────────────────────────────────────────┐
│                    回调系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │               BaseCallbackHandler                    │  │
│   │                                                     │  │
│   │   生命周期回调：                                     │  │
│   │   - on_llm_start         LLM 开始调用               │  │
│   │   - on_llm_end           LLM 调用结束               │  │
│   │   - on_llm_error         LLM 调用出错               │  │
│   │   - on_llm_new_token     新 Token 生成              │  │
│   │                                                     │  │
│   │   - on_chain_start       Chain 开始执行             │  │
│   │   - on_chain_end         Chain 执行结束             │  │
│   │   - on_chain_error       Chain 执行出错             │  │
│   │                                                     │  │
│   │   - on_tool_start        Tool 开始执行              │  │
│   │   - on_tool_end          Tool 执行结束              │  │
│   │   - on_tool_error        Tool 执行出错              │  │
│   │                                                     │  │
│   │   - on_retriever_start   Retriever 开始检索         │  │
│   │   - on_retriever_end     Retriever 检索结束         │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │           AsyncCallbackHandler (异步版本)           │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2 自定义回调处理器

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 开始调用，提示词数量: {len(prompts)}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM 调用结束，Token 使用: {response.llm_output}")
    
    def on_llm_new_token(self, token, **kwargs):
        print(f"新 Token: {token}", end="", flush=True)
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain 开始执行，输入: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        print(f"Chain 执行结束，输出: {outputs}")

# 使用回调
chain.invoke(input_data, config={"callbacks": [MyCallbackHandler()]})
```

#### 2.3 追踪集成

LangChain 支持多种追踪工具：

| 工具 | 特点 | 用途 |
|------|------|------|
| LangSmith | 官方追踪平台 | 调试、评估、监控 |
| LangFuse | 开源替代 | 自托管追踪 |
| Arize Phoenix | 可视化追踪 | 本地开发调试 |
| WandB | 实验追踪 | 模型实验记录 |

### 3. 自定义组件

#### 3.1 自定义 LLM 包装器

```python
from langchain_core.language_models.llms import LLM

class CustomLLM(LLM):
    """自定义 LLM 实现"""
    
    @property
    def _llm_type(self) -> str:
        return "custom_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # 实现自定义调用逻辑
        # 可以调用本地模型、其他 API 等
        response = self._custom_api_call(prompt)
        return response
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": "custom"}
```

#### 3.2 自定义工具

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    """计算器工具输入"""
    expression: str = Field(description="数学表达式，如 '2+3*4'")

class CalculatorTool(BaseTool):
    """自定义计算器工具"""
    
    name = "calculator"
    description = "计算数学表达式。输入应为数学表达式字符串。"
    args_schema = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)  # 注意：生产环境应使用更安全的方式
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
```

#### 3.3 自定义检索器

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class CustomRetriever(BaseRetriever):
    """自定义检索器"""
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 实现自定义检索逻辑
        # 可以是数据库查询、API 调用等
        docs = self._search(query)
        return docs
```

### 4. RAG 完整实现

#### 4.1 RAG 管道架构

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 管道架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Query  │ → │Retriever│ → │ Context │ → │ Prompt  │   │
│  │         │    │         │    │ Builder │    │Template│   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                             │
│                    ↓                                        │
│                                                             │
│              ┌─────────┐    ┌─────────┐                     │
│              │   LLM   │ → │ Output  │ → 最终回答           │
│              │         │    │ Parser  │                     │
│              └─────────┘    └─────────┘                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  知识库构建                           │   │
│  │                                                     │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐         │   │
│  │  │ Loader  │ → │ Splitter│ → │Embedding│ → Store   │   │
│  │  │         │    │         │    │         │         │   │
│  │  └─────────┘    └─────────┘    └─────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2 文档处理流程

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. 分割文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 每块最大字符数
    chunk_overlap=200,     # 块之间重叠
    length_function=len,   # 计算长度函数
    separators=["\n\n", "\n", " ", ""]  # 分隔符优先级
)
chunks = splitter.split_documents(docs)

# 3. 嵌入向量化
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 4. 存入向量数据库
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### 4.3 检索策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| Similarity | 纺粹相似度检索 | 基础 RAG |
| MMR | 最大边际相关性，减少重复 | 多样性要求 |
| Similarity Score Threshold | 带阈值过滤 | 高质量要求 |
| Multi-Query | 多查询扩展 | 提高召回率 |

#### 4.4 完整 RAG 链

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# RAG 链
rag_chain = (
    RunnablePassthrough.assign(
        context=retriever | format_docs  # 检索并格式化
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 执行
response = rag_chain.invoke({"question": "什么是 RAG？"})
```

## 快速开始

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community langchain-core
pip install chromadb pypdf tiktoken
```

### 配置环境

参考 `.env.example` 创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

## 练习文件说明

### `lcel_advanced.py` - LCEL 高级用法

演示 LCEL 表达式语言的高级特性：
- RunnableParallel 并行执行
- RunnablePassthrough 数据透传
- RunnableLambda 自定义逻辑
- RunnableBranch 条件分支
- 链的组合与嵌套

### `callback_system.py` - 回调系统实战

演示回调系统的使用：
- 自定义回调处理器
- Token 流追踪
- 执行日志记录
- 错误监控

### `custom_components.py` - 自定义组件开发

演示如何开发自定义组件：
- 自定义 LLM 包装器
- 自定义工具类
- 自定义检索器
- 组件集成测试

### `rag_pipeline.py` - 完整 RAG 实现

演示完整 RAG 应用：
- 文档加载与分割
- 向量嵌入与存储
- 多种检索策略
- RAG 链构建
- 问答系统实现

## 运行示例

```bash
# 运行 LCEL 高级示例
python day15/lcel_advanced.py

# 运行回调系统示例
python day15/callback_system.py

# 运行自定义组件示例
python day15/custom_components.py

# 运行 RAG 实现示例
python day15/rag_pipeline.py
```

## 最佳实践

### 1. 使用 RunnableParallel 优化性能

```python
# 并行执行不依赖的步骤
chain = RunnableParallel(
    context=retriever,
    history=memory_loader,
    metadata=metadata_fetcher
) | prompt | llm | parser
```

### 2. 合理使用回调和追踪

```python
# 开发阶段：详细追踪
config = {"callbacks": [DetailedCallbackHandler()]}

# 生产阶段：轻量追踪
config = {"callbacks": [ProductionCallbackHandler()]}
```

### 3. 文档分割的最佳参数

```python
# 根据内容类型调整
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 小块：精确检索
    chunk_overlap=100, # 适度重叠：保持上下文
)

# 或者按语义分割
splitter = SemanticSplitter()  # 更智能的分割
```

### 4. 错误处理和备用方案

```python
# 主链失败时使用备用
chain = primary_chain.with_fallbacks([
    fallback_chain_1,
    fallback_chain_2
])

# 重试机制
chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1000
)
```

### 5. 流式输出提升用户体验

```python
# RAG 流式输出
for chunk in rag_chain.stream({"question": "..."}):
    print(chunk, end="", flush=True)
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 并行执行变慢 | 子任务本身慢 | 优化子任务或减少并行 |
| 回调内存泄漏 | 未正确清理回调 | 使用回调上下文管理器 |
| 自定义组件不工作 | 未正确实现接口 | 检查 _call/_run 方法 |
| RAG 答案不相关 | 检索质量差 | 优化分割或使用 MMR |
| Token 超限 | 上下文太长 | 减少 chunk 数量或使用摘要 |

## 进阶对比

| 特性 | Day 14 基础 | Day 15 进阶 |
|------|-------------|-------------|
| LCEL | 基本管道操作 | 并行、分支、透传 |
| 回调 | 未涉及 | 完整追踪系统 |
| 组件 | 使用内置组件 | 开发自定义组件 |
| RAG | 概念介绍 | 完整实现 |
| 错误处理 | 基本 | 重试、备用方案 |

## 学习成果

完成本天学习后，你将能够：
- ✅ 灵活使用 LCEL 构建复杂处理链
- ✅ 实现生产级的追踪和监控
- ✅ 开发符合业务需求的自定义组件
- ✅ 构建完整的 RAG 应用系统
- ✅ 处理复杂场景的异常情况

## 下一步

Day 16 将学习 LangGraph，内容包括：图编排、状态管理、循环与分支、多 Agent 工作流。

## 参考资料

- [LCEL 文档](https://python.langchain.com/docs/expression_language/)
- [回调系统](https://python.langchain.com/docs/modules/callbacks/)
- [自定义组件](https://python.langchain.com/docs/modules/model_io/)
- [RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [LangSmith](https://www.langchain.com/langsmith)