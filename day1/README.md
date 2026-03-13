# Day 1: OpenAI API 基础入门

## 概述

第一天学习 OpenAI API 的基础使用，掌握如何通过代码调用大语言模型。

## 学习目标

- 理解 OpenAI API 的基本工作原理
- 掌握 API 认证和配置方法
- 学会发起简单的对话请求
- 理解响应数据的结构

## 核心概念

### 1. API 客户端初始化
使用 `openai` 库创建客户端实例：
```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
```

### 2. 聊天完成（Chat Completions）
调用大语言模型的核心方法：
```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "用户问题"}
    ],
    temperature=0.7,
    max_tokens=100
)
```

### 3. 响应数据结构
```python
response.id              # 响应唯一标识
response.model           # 使用的模型名称
response.choices[0].message.content  # 模型回复内容
response.usage           # Token 使用统计
```

## 练习文件

### `first_call.py`
第一次 API 调用示例，包含：
- 客户端初始化
- 简单对话请求
- 响应数据解析

### `playground.py`
实验场文件，用于：
- 测试不同的系统提示词
- 尝试不同的 temperature 参数
- 探索各种参数组合

### `chatbot.py`
简单聊天机器人实现：
- 多轮对话支持
- 对话历史管理
- 用户交互界面

## 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model` | 使用的模型 | `deepseek-chat` |
| `temperature` | 随机性控制 (0-1) | `0.7` (中等随机性) |
| `max_tokens` | 最大输出 Token 数 | 根据需求设定 |
| `top_p` | 核心采样参数 | `1.0` |

## 常见问题

### 1. API 密钥如何保存？
使用 `.env` 文件存储敏感信息：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 2. 余额不足怎么办？
充值 DeepSeek 账户余额后重试。

### 3. 如何控制输出长度？
使用 `max_tokens` 参数限制响应长度。

## 学习成果

完成本天学习后，你将能够：
- ✅ 正确配置和使用 OpenAI API
- ✅ 发起简单的对话请求
- ✅ 解析和理解 API 响应
- ✅ 构建基础的聊天机器人

## 下一步

第二天将学习如何通过参数控制模型输出行为，包括温度、Top-p 等高级参数的使用。
