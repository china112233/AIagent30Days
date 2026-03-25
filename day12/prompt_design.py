"""
提示词设计基础
演示提示词结构和最佳实践、角色设定、上下文设计、输出格式控制

核心要素：
1. 角色设定 (Role) - 赋予模型一个明确的身份
2. 任务描述 (Task) - 清晰说明要完成什么
3. 上下文信息 (Context) - 提供必要的背景
4. 输出要求 (Output Format) - 指定输出格式
5. 示例 (Examples) - Few-shot 示例
6. 约束条件 (Constraints) - 限制和边界
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


# ==================== 提示词模板 ====================

class PromptTemplate:
    """提示词模板类，用于构建结构化提示词"""

    def __init__(self):
        self.role = ""
        self.task = ""
        self.context = ""
        self.output_format = ""
        self.examples = []
        self.constraints = []

    def set_role(self, role: str):
        """设置角色设定"""
        self.role = role
        return self

    def set_task(self, task: str):
        """设置任务描述"""
        self.task = task
        return self

    def set_context(self, context: str):
        """设置上下文信息"""
        self.context = context
        return self

    def set_output_format(self, format_desc: str):
        """设置输出格式要求"""
        self.output_format = format_desc
        return self

    def add_example(self, input_text: str, output_text: str):
        """添加示例"""
        self.examples.append({"input": input_text, "output": output_text})
        return self

    def add_constraint(self, constraint: str):
        """添加约束条件"""
        self.constraints.append(constraint)
        return self

    def build(self) -> str:
        """构建完整的提示词"""
        parts = []

        # 角色设定
        if self.role:
            parts.append(f"【角色设定】\n{self.role}\n")

        # 任务描述
        if self.task:
            parts.append(f"【任务描述】\n{self.task}\n")

        # 上下文信息
        if self.context:
            parts.append(f"【上下文信息】\n{self.context}\n")

        # 示例
        if self.examples:
            parts.append("【示例】")
            for i, ex in enumerate(self.examples, 1):
                parts.append(f"示例{i}:")
                parts.append(f"  输入: {ex['input']}")
                parts.append(f"  输出: {ex['output']}")
            parts.append("")

        # 输出格式
        if self.output_format:
            parts.append(f"【输出格式】\n{self.output_format}\n")

        # 约束条件
        if self.constraints:
            parts.append("【约束条件】")
            for c in self.constraints:
                parts.append(f"- {c}")
            parts.append("")

        return "\n".join(parts)


# ==================== 角色设定技巧 ====================

def demo_role_design():
    """演示不同角色设定的效果"""
    print("\n" + "=" * 60)
    print("示例1: 角色设定技巧")
    print("=" * 60)

    questions = ["什么是递归？请解释一下。"]

    # 角色1：普通助手
    role1 = "你是一个有帮助的AI助手。"

    # 角色2：资深程序员
    role2 = """你是一位资深的高级软件工程师，拥有15年的编程经验。
你的特点：
- 擅长用通俗易懂的方式解释复杂概念
- 总是提供代码示例来辅助说明
- 会指出常见的陷阱和最佳实践
- 使用专业但不过于晦涩的语言"""

    # 角色3：编程教师
    role3 = """你是一位计算机科学教授，正在给大一新生上课。
你的教学风格：
- 使用生活中的类比帮助理解
- 循序渐进，从简单到复杂
- 会提出思考问题引导学生
- 鼓励学生动手实践"""

    roles = [
        ("普通助手", role1),
        ("资深程序员", role2),
        ("编程教师", role3)
    ]

    for role_name, role_prompt in roles:
        print(f"\n--- 角色设定: {role_name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": role_prompt},
                {"role": "user", "content": questions[0]}
            ],
            max_tokens=500
        )

        print(f"回答:\n{response.choices[0].message.content}")


# ==================== 上下文设计 ====================

def demo_context_design():
    """演示上下文信息的重要性"""
    print("\n" + "=" * 60)
    print("示例2: 上下文设计")
    print("=" * 60)

    # 问题相同，上下文不同
    question = "请推荐一个解决方案"

    # 无上下文
    no_context = question

    # 有上下文
    with_context = f"""
问题背景：
- 项目类型：电商网站后端API
- 技术栈：Python + FastAPI + PostgreSQL
- 当前问题：用户登录接口响应时间超过3秒
- 并发用户：约5000人同时在线
- 服务器：AWS EC2 t3.medium

{question}
"""

    prompts = [
        ("无上下文", no_context),
        ("有完整上下文", with_context)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")
        print(f"提示词:\n{prompt[:200]}..." if len(prompt) > 200 else f"提示词:\n{prompt}")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        print(f"\n回答:\n{response.choices[0].message.content}")


# ==================== 输出格式控制 ====================

def demo_output_format():
    """演示不同的输出格式控制"""
    print("\n" + "=" * 60)
    print("示例3: 输出格式控制")
    print("=" * 60)

    task = "分析文本'这款手机拍照效果很好，但是电池续航一般'的情感"

    # 格式1：简单格式
    format1 = f"{task}\n请返回：正面/负面/中性"

    # 格式2：结构化格式
    format2 = f"""
{task}

请按以下格式输出：
## 情感分类
[分类结果]

## 关键词
- 正面关键词：[列出]
- 负面关键词：[列出]

## 分析说明
[简要说明判断依据]
"""

    # 格式3：JSON格式
    format3 = f"""
{task}

请以JSON格式输出，格式如下：
{{
    "sentiment": "正面/负面/中性",
    "positive_keywords": ["关键词1", "关键词2"],
    "negative_keywords": ["关键词1", "关键词2"],
    "confidence": 0.0-1.0,
    "explanation": "判断依据"
}}
"""

    formats = [
        ("简单格式", format1),
        ("结构化格式", format2),
        ("JSON格式", format3)
    ]

    for name, prompt in formats:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        print(f"输出:\n{response.choices[0].message.content}")


# ==================== 约束条件设置 ====================

def demo_constraints():
    """演示约束条件的作用"""
    print("\n" + "=" * 60)
    print("示例4: 约束条件设置")
    print("=" * 60)

    question = "什么是机器学习？"

    # 无约束
    no_constraint = question

    # 有约束
    with_constraints = f"""
{question}

约束条件：
- 回答不超过100字
- 使用简体中文
- 不使用专业术语
- 适合小学生理解
"""

    # 更严格的约束
    strict_constraints = f"""
{question}

约束条件：
- 必须使用"小明"作为主角讲一个故事来解释
- 故事不超过150字
- 不能出现任何英文单词
- 不能使用"算法"、"数据"、"模型"这些词
"""

    prompts = [
        ("无约束", no_constraint),
        ("有约束", with_constraints),
        ("严格约束", strict_constraints)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        print(f"回答:\n{response.choices[0].message.content}")


# ==================== 使用分隔符 ====================

def demo_delimiters():
    """演示分隔符的使用"""
    print("\n" + "=" * 60)
    print("示例5: 分隔符的使用")
    print("=" * 60)

    # 需要处理的文本
    user_text = """
Python是一门流行的编程语言。
它的设计哲学是"优雅"、"明确"、"简单"。
Python拥有丰富的第三方库。
"""

    # 使用不同的分隔符
    delimiters = [
        ("三引号", '"""'),
        ("三反引号", "```"),
        ("XML标签", "<text>"),
        ("特殊标记", "---")
    ]

    for name, delim in delimiters:
        if name == "XML标签":
            prompt = f"""
请总结以下文本的主要内容：

<text>
{user_text}
</text>

用一句话概括。
"""
        elif name == "特殊标记":
            prompt = f"""
请总结以下文本的主要内容：

---文本开始---
{user_text}
---文本结束---

用一句话概括。
"""
        else:
            prompt = f"""
请总结以下文本的主要内容：

{delim}
{user_text}
{delim}

用一句话概括。
"""

        print(f"\n--- 分隔符: {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        print(f"回答: {response.choices[0].message.content}")


# ==================== 使用模板类 ====================

def demo_template_class():
    """演示使用 PromptTemplate 类构建提示词"""
    print("\n" + "=" * 60)
    print("示例6: 使用 PromptTemplate 类")
    print("=" * 60)

    # 构建一个代码审查提示词
    template = PromptTemplate()

    template.set_role("""
你是一位资深代码审查专家，擅长发现代码中的问题和改进机会。
你的审查风格严谨但不刻薄，总是提供建设性的意见。
""")

    template.set_task("审查以下 Python 代码，指出潜在问题并提供改进建议。")

    template.set_context("""
代码来源：用户认证模块
重要程度：高（涉及安全性）
""")

    template.add_example(
        input_text="def check_password(p): return p == 'admin'",
        output_text="""
问题：
1. 硬编码密码 - 安全隐患严重
2. 无长度验证 - 可能导致DoS攻击
3. 无错误处理 - 缺少防御性编程

建议改进：
- 使用环境变量存储密码
- 添加密码复杂度验证
- 使用密码哈希比对
"""
    )

    template.set_output_format("""
## 发现的问题
1. [问题描述]
2. [问题描述]

## 改进建议
- [建议1]
- [建议2]

## 改进后的代码
```python
[代码]
```
""")

    template.add_constraint("使用中文回答")
    template.add_constraint("每个问题都要说明风险等级（高/中/低）")
    template.add_constraint("提供可运行的改进代码")

    # 构建提示词
    full_prompt = template.build()
    print("构建的提示词:")
    print("-" * 40)
    print(full_prompt)
    print("-" * 40)

    # 使用提示词
    code_to_review = """
def login(username, password):
    if username == "admin" and password == "123456":
        return True
    return False
"""

    final_prompt = full_prompt + f"\n\n待审查代码：\n```python\n{code_to_review}\n```"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=800
    )

    print("\n审查结果:")
    print(response.choices[0].message.content)


# ==================== 完整示例：技术文档生成器 ====================

def create_tech_doc_generator():
    """创建一个技术文档生成器"""

    system_prompt = """你是一位技术文档专家，擅长撰写清晰、专业的技术文档。

你的文档风格：
- 结构清晰，层次分明
- 包含代码示例
- 有详细的参数说明
- 包含使用注意事项

输出格式：
1. 概述
2. 使用方法
3. 参数说明
4. 示例代码
5. 注意事项
"""

    return system_prompt


def demo_tech_doc():
    """演示生成技术文档"""
    print("\n" + "=" * 60)
    print("示例7: 技术文档生成")
    print("=" * 60)

    system_prompt = create_tech_doc_generator()

    # 模拟一个函数
    function_code = """
def process_data(data: list, operation: str = 'sum', ignore_errors: bool = False) -> dict:
    '''处理数据列表'''
    result = {'success': True, 'data': None, 'errors': []}

    try:
        if operation == 'sum':
            result['data'] = sum(data)
        elif operation == 'avg':
            result['data'] = sum(data) / len(data)
        elif operation == 'max':
            result['data'] = max(data)
    except Exception as e:
        if not ignore_errors:
            raise
        result['errors'].append(str(e))
        result['success'] = False

    return result
"""

    user_prompt = f"""
请为以下 Python 函数生成技术文档：

```python
{function_code}
```

要求：
- 使用 Markdown 格式
- 包含完整的使用示例
- 说明每个参数的类型和默认值
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1000
    )

    print("生成的文档:")
    print(response.choices[0].message.content)


# ==================== 交互式演示 ====================

def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("提示词设计 - 交互模式")
    print("=" * 60)
    print("输入你的问题，我将展示不同提示词设计的回答差异")
    print("输入 'quit' 退出")
    print("=" * 60)

    while True:
        user_input = input("\n请输入问题: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input:
            continue

        # 使用不同的角色设定
        roles = [
            ("普通助手", "你是一个有帮助的AI助手。"),
            ("专家模式", "你是一位该领域的资深专家，请给出专业、深入的分析。"),
            ("简化模式", "请用最简单的语言解释，让小学生也能理解。")
        ]

        for role_name, role_prompt in roles:
            print(f"\n--- {role_name} ---")

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300
            )

            print(response.choices[0].message.content)


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - 提示词设计基础")
    print("=" * 60)

    # 运行所有演示
    demo_role_design()
    demo_context_design()
    demo_output_format()
    demo_constraints()
    demo_delimiters()
    demo_template_class()
    demo_tech_doc()

    # 交互模式
    print("\n" + "=" * 60)
    print("是否进入交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        interactive_demo()