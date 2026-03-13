"""
Token 统计工具
用于分析和估算 API 调用的 Token 消耗
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


class TokenCounter:
    """Token 计数和成本估算工具"""

    # DeepSeek 价格（示例，实际价格请查询官网）
    # 价格单位：元 / 千 Token
    PRICING = {
        "deepseek-chat": {
            "input": 0.001,   # 输入价格
            "output": 0.002   # 输出价格
        },
        "deepseek-reasoner": {
            "input": 0.002,
            "output": 0.004
        }
    }

    def __init__(self, model="deepseek-chat"):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def estimate_tokens(self, text):
        """
        估算文本的 Token 数量
        使用启发式方法估算

        规则：
        - 中文字符：约 0.5-1 token
        - 英文单词：约 0.25 token
        - 标点和空白：约 0.25 token
        """
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

        # 统计英文单词
        english_words = len(re.findall(r'[a-zA-Z]+', text))

        # 统计数字
        numbers = len(re.findall(r'\d+', text))

        # 其他字符（标点、空白等）
        other_chars = len(text) - chinese_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))

        # 估算 Token
        # 中文约 0.5 token/字，英文约 0.25 token/word
        estimated = int(
            chinese_chars * 0.5 +    # 中文
            english_words * 0.25 +    # 英文单词
            numbers * 0.25 +          # 数字
            other_chars * 0.1         # 其他
        )

        return max(estimated, 1)  # 至少返回 1

    def estimate_messages_tokens(self, messages):
        """估算消息列表的 Token 数量"""
        total = 0
        for msg in messages:
            # 每条消息有额外开销
            total += 4  # role 和 content 字段开销
            total += self.estimate_tokens(msg["content"])
        total += 2  # 对话开始和结束标记
        return total

    def count_response_tokens(self, response):
        """从 API 响应中获取实际 Token 数量"""
        usage = response.usage
        return {
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "total": usage.total_tokens
        }

    def track_usage(self, input_tokens, output_tokens):
        """记录 Token 使用量"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

    def calculate_cost(self, input_tokens, output_tokens):
        """计算成本（单位：元）"""
        pricing = self.PRICING.get(self.model, self.PRICING["deepseek-chat"])
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def get_stats(self):
        """获取统计信息"""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        total_cost = self.calculate_cost(
            self.total_input_tokens,
            self.total_output_tokens
        )

        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost_yuan": round(total_cost, 6),
            "average_tokens_per_call": total_tokens // max(self.call_count, 1)
        }

    def reset(self):
        """重置统计"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0


def analyze_text_complexity(text):
    """分析文本复杂度"""
    counter = TokenCounter()

    # 基本信息
    char_count = len(text)
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_count = len(re.findall(r'[a-zA-Z]+', text))
    number_count = len(re.findall(r'\d+', text))

    # Token 估算
    estimated_tokens = counter.estimate_tokens(text)

    print("=" * 50)
    print("文本分析结果")
    print("=" * 50)
    print(f"总字符数: {char_count}")
    print(f"中文字符: {chinese_count}")
    print(f"英文单词: {english_count}")
    print(f"数字个数: {number_count}")
    print(f"估算 Token: {estimated_tokens}")
    print(f"预估成本: {counter.calculate_cost(estimated_tokens, 0):.6f} 元")

    return estimated_tokens


def demo_token_tracking():
    """演示 Token 追踪"""
    print("=" * 50)
    print("Token 追踪演示")
    print("=" * 50)

    counter = TokenCounter()

    messages = [
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": "你好！"}
    ]

    # 估算输入 Token
    estimated_input = counter.estimate_messages_tokens(messages)
    print(f"估算输入 Token: {estimated_input}")

    # 实际调用
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    # 获取实际 Token
    actual = counter.count_response_tokens(response)
    print(f"实际输入 Token: {actual['input']}")
    print(f"实际输出 Token: {actual['output']}")
    print(f"总 Token: {actual['total']}")

    # 记录使用
    counter.track_usage(actual['input'], actual['output'])

    # 显示成本
    cost = counter.calculate_cost(actual['input'], actual['output'])
    print(f"本次成本: {cost:.6f} 元")


def demo_cost_comparison():
    """演示不同长度文本的成本对比"""
    print("\n" + "=" * 50)
    print("成本对比演示")
    print("=" * 50)

    counter = TokenCounter()

    test_cases = [
        ("短文本", "你好"),
        ("中等文本", "请介绍一下人工智能的发展历史和主要应用领域"),
        ("长文本", """
请详细介绍一下机器学习和深度学习的区别，包括：
1. 基本概念和原理
2. 主要算法和技术
3. 应用场景和优缺点
4. 未来发展趋势

请用通俗易懂的语言解释，并举出具体例子。
        """)
    ]

    print(f"{'文本类型':<10} {'估算Token':<12} {'预估成本(元)':<15}")
    print("-" * 40)

    for name, text in test_cases:
        tokens = counter.estimate_tokens(text)
        cost = counter.calculate_cost(tokens, 0)
        print(f"{name:<10} {tokens:<12} {cost:<15.6f}")


def demo_session_tracking():
    """演示会话级别的 Token 追踪"""
    print("\n" + "=" * 50)
    print("会话 Token 追踪演示")
    print("=" * 50)

    counter = TokenCounter()

    # 模拟一个对话会话
    conversation = [
        "你好，请问有什么可以帮助你的？",
        "我想了解一下 Python 编程",
        "Python 有哪些主要特点？",
        "可以给我推荐一些学习资源吗？",
        "非常感谢！"
    ]

    messages = [{"role": "system", "content": "你是一个编程助手。"}]

    for i, user_input in enumerate(conversation):
        messages.append({"role": "user", "content": user_input})

        # 估算
        estimated = counter.estimate_messages_tokens(messages)
        print(f"\n第{i+1}轮对话 - 估算输入Token: {estimated}")

        # 实际调用（这里仅做演示，实际使用时取消注释）
        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=messages
        # )
        # actual = counter.count_response_tokens(response)
        # counter.track_usage(actual['input'], actual['output'])
        # messages.append({"role": "assistant", "content": response.choices[0].message.content})

    print("\n" + "=" * 50)
    print("会话统计（模拟数据）")
    print("=" * 50)
    stats = counter.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def interactive_token_analyzer():
    """交互式 Token 分析器"""
    print("\n" + "=" * 50)
    print("交互式 Token 分析器")
    print("输入文本进行分析，输入 'quit' 退出")
    print("=" * 50)

    counter = TokenCounter()

    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == 'quit':
            break

        if not text:
            continue

        tokens = counter.estimate_tokens(text)
        cost = counter.calculate_cost(tokens, 0)

        print(f"字符数: {len(text)}")
        print(f"估算 Token: {tokens}")
        print(f"预估成本: {cost:.6f} 元")


def batch_cost_estimator():
    """批量成本估算器"""
    print("\n" + "=" * 50)
    print("批量成本估算器")
    print("=" * 50)

    counter = TokenCounter()

    # 模拟批量任务
    tasks = [
        {"name": "文章摘要", "input": "这是一篇很长的文章..." * 100, "expected_output": 100},
        {"name": "翻译任务", "input": "Translate this text to Chinese.", "expected_output": 50},
        {"name": "代码生成", "input": "Write a Python function to sort a list", "expected_output": 200},
    ]

    total_cost = 0

    print(f"\n{'任务名称':<15} {'输入Token':<12} {'输出Token':<12} {'成本(元)':<12}")
    print("-" * 55)

    for task in tasks:
        input_tokens = counter.estimate_tokens(task["input"])
        output_tokens = task["expected_output"]
        cost = counter.calculate_cost(input_tokens, output_tokens)
        total_cost += cost

        print(f"{task['name']:<15} {input_tokens:<12} {output_tokens:<12} {cost:<12.6f}")

    print("-" * 55)
    print(f"{'总计':<15} {'':<12} {'':<12} {total_cost:<12.6f}")


if __name__ == "__main__":
    # 文本分析
    analyze_text_complexity("人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。")

    # Token 追踪
    demo_token_tracking()

    # 成本对比
    demo_cost_comparison()

    # 会话追踪
    demo_session_tracking()

    # 批量估算
    batch_cost_estimator()

    # 交互式分析（可选）
    print("\n是否启动交互式分析器？(y/n): ", end="")
    if input().lower() == 'y':
        interactive_token_analyzer()