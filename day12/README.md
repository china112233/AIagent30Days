# Day 12: Prompt 工程

## 概述

第十二天深入学习 Prompt 工程（提示词工程），这是与大语言模型有效交互的核心技能。好的提示词能够显著提升模型输出质量，是 AI 应用开发的关键技术。

## 学习目标

- 掌握提示词设计的基本结构和最佳实践
- 学会使用 Few-shot Learning 提升模型表现
- 理解 Chain of Thought 思维链技术
- 实现结构化输出和输出验证

## 核心概念

### 1. 提示词设计基础

提示词（Prompt）是与大语言模型交互的输入文本，好的提示词应该包含以下要素：

```
┌─────────────────────────────────────────────────────────────┐
│                     提示词结构框架                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【角色设定】Role                                            │
│   "你是一个专业的XX助手..."                                  │
│                                                             │
│  【任务描述】Task                                           │
│   "请完成以下任务..."                                        │
│                                                             │
│  【上下文信息】Context                                       │
│   "背景信息如下..."                                          │
│                                                             │
│  【输出要求】Output Format                                  │
│   "请按以下格式输出..."                                      │
│                                                             │
│  【示例】Examples (Few-shot)                                │
│   "例如：输入... 输出..."                                    │
│                                                             │
│  【约束条件】Constraints                                    │
│   "注意：不要... 必须..."                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Few-shot Learning

Few-shot Learning 通过提供少量示例来引导模型理解任务：

```python
# Zero-shot（无示例）
prompt = "将以下文本翻译成英语：你好世界"

# One-shot（一个示例）
prompt = """
将以下文本翻译成英语：
例子：早上好 -> Good morning
翻译：你好世界
"""

# Few-shot（多个示例）
prompt = """
将以下文本翻译成英语：
例子1：早上好 -> Good morning
例子2：晚上好 -> Good evening
例子3：再见 -> Goodbye
翻译：你好世界
"""
```

**Few-shot 设计原则**：
- 示例要与任务相关
- 示例要有多样性
- 示例数量通常 3-5 个最佳
- 示例格式要一致

### 3. Chain of Thought (CoT)

思维链是一种让模型"逐步思考"的技术：

```
┌─────────────────────────────────────────────────────────────┐
│                 Chain of Thought 示例                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题：小明有5个苹果，给了小红2个，又买了3个，还剩多少？      │
│                                                             │
│  【普通回答】                                                │
│  答案：6个苹果                                              │
│                                                             │
│  【CoT 回答】                                                │
│  让我们一步步思考：                                          │
│  1. 小明最初有 5 个苹果                                      │
│  2. 给了小红 2 个，所以剩下 5 - 2 = 3 个                     │
│  3. 又买了 3 个，所以现在有 3 + 3 = 6 个                     │
│  4. 答案是 6 个苹果                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**CoT 类型**：

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| Zero-shot CoT | 只添加"让我们一步步思考" | 简单问题 |
| Few-shot CoT | 提供带推理步骤的示例 | 复杂问题 |
| Auto-CoT | 自动生成推理链 | 批量处理 |

### 4. 结构化输出

让模型输出结构化数据（如 JSON），便于程序处理：

```python
# 结构化输出示例
response_schema = {
    "name": "商品名称",
    "price": "价格（数字）",
    "category": "分类",
    "features": ["特性1", "特性2"]
}

prompt = f"""
请分析以下商品信息，并以 JSON 格式输出：
{product_description}

输出格式：
{json.dumps(response_schema, ensure_ascii=False, indent=2)}
"""
```

**结构化输出技巧**：
- 使用 JSON Schema 定义格式
- 使用 Pydantic 进行验证
- 添加输出修正逻辑
- 处理格式错误

## 快速开始

### 安装依赖

```bash
pip install openai python-dotenv pydantic
```

### 配置环境

```bash
# 复制 .env.example 为 .env
cp day12/.env.example day12/.env

# 编辑 .env 文件，填入你的 API Key
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

## 练习文件

### `prompt_design.py`
提示词设计基础：
- 角色设定技巧
- 上下文设计
- 输出格式控制
- 约束条件设置

### `few_shot.py`
Few-shot Learning：
- Zero-shot vs Few-shot 对比
- 示例设计最佳实践
- 示例数量和多样性
- 负面示例的使用

### `cot.py`
Chain of Thought：
- Zero-shot CoT 实现
- Few-shot CoT 实现
- 复杂问题分解
- 推理链验证

### `structured_output.py`
结构化输出：
- JSON 格式输出
- Pydantic 模型约束
- 输出验证和修正
- 错误处理策略

## 提示词设计最佳实践

### 1. 使用清晰的角色设定

```python
# 好的角色设定
system_prompt = """你是一位资深的前端开发工程师，擅长 React 和 TypeScript。
你的回答应该：
1. 提供可运行的代码示例
2. 解释关键概念
3. 指出潜在的问题和注意事项
4. 遵循最佳实践"""
```

### 2. 提供足够的上下文

```python
# 好的上下文
prompt = f"""
项目背景：这是一个电商网站的商品推荐系统
技术栈：Python + FastAPI + PostgreSQL
当前问题：推荐算法响应时间过长

请分析可能的原因并提供优化方案。
"""
```

### 3. 明确输出格式

```python
# 好的格式要求
format_instruction = """
请按以下格式输出：

## 问题分析
[问题原因分析]

## 解决方案
1. [方案一]
2. [方案二]

## 代码示例
```python
[相关代码]
```

## 注意事项
- [注意事项]
"""
```

### 4. 使用分隔符

```python
prompt = """
请分析以下文本的情感：

'''text
{user_input}
'''

请返回：正面/负面/中性
"""
```

### 5. 添加约束条件

```python
constraints = """
约束条件：
- 不要编造信息
- 如果不确定，请明确说明
- 回答不超过200字
- 使用简体中文
"""
```

## Few-shot Learning 技巧

### 示例选择原则

```python
# 好的示例设计
examples = [
    # 简单示例
    {"input": "开心", "output": "正面"},
    # 中等示例
    {"input": "这个产品还行，但价格有点贵", "output": "中性"},
    # 复杂示例
    {"input": "虽然服务态度不好，但产品质量确实不错", "output": "中性"},
    # 边界示例
    {"input": "不推荐，非常失望", "output": "负面"},
]
```

### 示例模板

```python
def format_few_shot_prompt(task_description: str, examples: list, query: str) -> str:
    """构建 Few-shot 提示词"""
    prompt = f"{task_description}\n\n"

    for i, ex in enumerate(examples):
        prompt += f"示例 {i+1}:\n"
        prompt += f"输入: {ex['input']}\n"
        prompt += f"输出: {ex['output']}\n\n"

    prompt += f"现在请处理:\n输入: {query}\n输出:"

    return prompt
```

## Chain of Thought 技术

### Zero-shot CoT

```python
# 简单添加"让我们一步步思考"
prompt = f"""
{question}

让我们一步步思考：
"""
```

### Few-shot CoT

```python
cot_examples = [
    {
        "question": "如果一个数的两倍加3等于11，这个数是多少？",
        "reasoning": """
1. 设这个数为 x
2. 根据题意：2x + 3 = 11
3. 两边减 3：2x = 8
4. 两边除以 2：x = 4
5. 验证：2 × 4 + 3 = 11 ✓
""",
        "answer": "4"
    }
]
```

### 复杂问题分解

```python
def decompose_complex_question(question: str) -> list:
    """将复杂问题分解为子问题"""
    prompt = f"""
请将以下复杂问题分解为多个简单的子问题：

问题：{question}

要求：
1. 每个子问题应该可以独立回答
2. 子问题的答案组合起来能解决原问题
3. 按解决顺序列出

子问题列表：
"""
    # 调用 LLM 分解...
```

## 结构化输出实现

### 使用 JSON Schema

```python
from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    name: str
    price: float
    category: str
    features: List[str]
    rating: Optional[float] = None

# 生成提示词
prompt = f"""
分析以下商品描述，提取信息：

{product_text}

输出 JSON 格式：
{Product.model_json_schema()}
"""
```

### 输出验证和修正

```python
def get_structured_output(prompt: str, model: BaseModel) -> dict:
    """获取并验证结构化输出"""
    response = call_llm(prompt)

    try:
        # 尝试解析 JSON
        data = json.loads(response)
        # 使用 Pydantic 验证
        validated = model.model_validate(data)
        return validated.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        # 尝试修正
        corrected = fix_and_retry(prompt, response, e)
        return corrected
```

## 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 输出格式不一致 | 提示词不够明确 | 使用更严格的格式要求 |
| 回答偏离主题 | 缺少约束条件 | 添加明确的边界限制 |
| 推理错误 | 缺少中间步骤 | 使用 CoT 引导推理 |
| 示例效果不好 | 示例不相关或太少 | 优化示例选择和数量 |
| JSON 解析失败 | 模型输出不规范 | 添加修正逻辑 |

## 学习成果

完成本天学习后，你将能够：
- 设计高质量的提示词
- 使用 Few-shot Learning 提升模型表现
- 应用 Chain of Thought 解决复杂问题
- 实现可靠的结构化输出

## 下一步

第十三天将学习 Agent 工具集成，让 AI 能够使用外部工具完成任务。

## 参考资料

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
- [Pydantic Documentation](https://docs.pydantic.dev/)