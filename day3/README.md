# Day 3: 多轮对话管理与流式输出

## 概述

第三天学习多轮对话管理、流式输出和高级 API 功能，这些是构建实用 AI 应用的核心技能。

## 学习目标

- 掌握流式输出（Streaming）的实现方法
- 理解上下文窗口管理与对话历史维护
- 学会 Token 统计与成本控制
- 实现高级的对话管理系统

## 核心概念

### 1. 流式输出（Streaming）

流式输出允许逐字逐句地接收模型响应，而不是等待完整响应：

```python
# 非流式：等待完整响应
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "讲一个故事"}],
    stream=False  # 默认值
)
print(response.choices[0].message.content)  # 一次性打印

# 流式：实时接收
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "讲一个故事"}],
    stream=True  # 开启流式
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**流式输出的优势**：
- 更好的用户体验：用户能立即看到响应开始
- 降低首字延迟：无需等待完整响应
- 适合长文本生成：故事、文章、代码等

### 2. 上下文窗口管理

大语言模型有 Token 数量限制，需要合理管理对话历史：

```python
class ConversationManager:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim_history()  # 超出限制时裁剪历史

    def _trim_history(self):
        # 保留系统提示 + 最近的对话
        while self._estimate_tokens() > self.max_tokens:
            if len(self.messages) > 2:
                self.messages.pop(1)  # 移除最早的对话（保留系统提示）
```

### 3. Token 统计

Token 是计费的基本单位，了解 Token 消耗很重要：

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...]
)

# 获取 Token 使用情况
print(f"输入 Token: {response.usage.prompt_tokens}")
print(f"输出 Token: {response.usage.completion_tokens}")
print(f"总 Token: {response.usage.total_tokens}")
```

### 4. 高级参数

| 参数 | 说明 | 使用场景 |
|------|------|----------|
| `stream` | 流式输出开关 | 实时响应场景 |
| `stop` | 停止序列 | 控制输出终止点 |
| `presence_penalty` | 存在惩罚 (-2.0 to 2.0) | 减少重复话题 |
| `frequency_penalty` | 频率惩罚 (-2.0 to 2.0) | 减少重复词语 |
| `logprobs` | 返回 token 概率 | 调试和分析 |
| `n` | 生成多个响应 | 多样性需求 |

## 练习文件

### `streaming_chat.py`
流式输出基础示例：
- 基本流式输出实现
- 流式 vs 非流式对比
- 处理流式响应的注意事项

### `conversation_manager.py`
对话管理器实现：
- 对话历史维护
- Token 估算与限制
- 上下文窗口裁剪策略
- 对话持久化保存

### `token_counter.py`
Token 统计工具：
- 实时 Token 统计
- 成本估算
- 对话效率分析

### `advanced_chatbot.py`
高级聊天机器人：
- 整合流式输出
- 智能对话管理
- 多轮对话优化
- 用户交互优化

## 流式输出详解

### 流式响应数据结构
```python
for chunk in stream:
    # chunk 是一个 ChatCompletionChunk 对象
    chunk.id              # 响应 ID
    chunk.choices[0].delta.content    # 当前内容片段
    chunk.choices[0].finish_reason    # 结束原因（完成时）
```

### 流式输出最佳实践
```python
def stream_chat(messages):
    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            temperature=0.7
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()  # 换行
        return full_response

    except Exception as e:
        print(f"\n流式输出出错: {e}")
        return None
```

## 对话管理策略

### 策略1：固定消息数量
```python
def trim_by_count(messages, max_count=10):
    """保留最近的 N 条消息"""
    if len(messages) > max_count:
        return [messages[0]] + messages[-(max_count-1):]
    return messages
```

### 策略2：Token 估算裁剪
```python
def trim_by_tokens(messages, max_tokens=4000):
    """根据 Token 估算裁剪"""
    # 粗略估算：中文约 0.5 token/字，英文约 0.25 token/word
    while estimate_tokens(messages) > max_tokens:
        if len(messages) > 2:
            messages.pop(1)
    return messages
```

### 策略3：摘要压缩
```python
def summarize_history(messages):
    """将历史对话压缩为摘要"""
    summary = call_llm("请用一句话总结以下对话: " + str(messages))
    return [{"role": "system", "content": f"历史对话摘要: {summary}"}]
```

## 高级功能示例

### 停止序列（Stop Sequences）
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "列出5种水果"}],
    stop=["\n\n", "结束"]  # 遇到这些序列时停止
)
```

### 多响应生成
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "给公司起个名字"}],
    n=3  # 生成3个不同的响应
)

for i, choice in enumerate(response.choices):
    print(f"方案{i+1}: {choice.message.content}")
```

### 惩罚参数调优
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一首诗"}],
    presence_penalty=0.6,   # 鼓励谈论新话题
    frequency_penalty=0.3   # 减少重复用词
)
```

## 最佳实践

### 1. 流式输出注意事项
- 使用 `flush=True` 确保即时显示
- 处理网络中断和错误
- 收集完整响应用于对话历史

### 2. 对话历史管理
- 始终保留系统提示词
- 合理设置上下文窗口大小
- 监控 Token 消耗

### 3. 成本优化
- 选择合适的 max_tokens
- 使用更短的系统提示词
- 缓存常见问题的响应

### 4. 用户体验
- 流式输出提升交互感
- 添加加载提示
- 优雅处理错误和超时

## 常见问题

### 1. 流式输出中断怎么办？
捕获异常并提示用户重试，记录未完成的响应。

### 2. 如何估算中文 Token？
中文约 0.5-1 token/字，建议使用 tiktoken 库精确计算。

### 3. 对话历史太长怎么办？
使用摘要压缩、滑动窗口或向量检索等策略。

### 4. 流式和非流式可以混用吗？
可以，但建议统一使用一种方式便于代码维护。

## 学习成果

完成本天学习后，你将能够：
- 实现流式输出提升用户体验
- 管理多轮对话的上下文窗口
- 控制 Token 消耗和成本
- 构建生产级别的聊天应用

## 下一步

第四天将学习 Function Calling（函数调用），让 AI 能够调用外部工具和 API，实现真正的智能代理能力。