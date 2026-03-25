"""
Chain of Thought (CoT) 思维链示例
演示 Zero-shot CoT、Few-shot CoT、复杂问题分解

核心概念：
1. Zero-shot CoT - 只添加"让我们一步步思考"引导模型推理
2. Few-shot CoT - 提供带推理步骤的示例
3. 问题分解 - 将复杂问题拆分为简单子问题

CoT 的优势：
- 提高复杂推理任务的准确率
- 使推理过程可解释
- 便于发现和纠正错误
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


# ==================== Zero-shot CoT ====================

def demo_zero_shot_cot():
    """演示 Zero-shot CoT 效果"""
    print("\n" + "=" * 60)
    print("示例1: Zero-shot CoT 效果对比")
    print("=" * 60)

    # 数学问题
    math_question = """
小明有5个苹果，给了小红2个，又从妈妈那里得到了3个，
然后吃了1个，请问小明现在有多少个苹果？
"""

    # 普通提问（无 CoT）
    normal_prompt = math_question + "\n请直接给出答案。"

    # Zero-shot CoT
    cot_prompt = math_question + "\n让我们一步步思考："

    # 另一种 Zero-shot CoT 格式
    cot_prompt_v2 = math_question + """
请按以下格式回答：
思考过程：
1. ...
2. ...
...

最终答案：...
"""

    prompts = [
        ("普通提问（无 CoT）", normal_prompt),
        ("Zero-shot CoT（让我们一步步思考）", cot_prompt),
        ("Zero-shot CoT（结构化格式）", cot_prompt_v2)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        print(f"回答:\n{response.choices[0].message.content}")


# ==================== Few-shot CoT ====================

def demo_few_shot_cot():
    """演示 Few-shot CoT"""
    print("\n" + "=" * 60)
    print("示例2: Few-shot CoT")
    print("=" * 60)

    # 数学推理示例
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
        },
        {
            "question": "张三的年龄是李四的2倍，5年后张三比李四大15岁，请问张三现在多大？",
            "reasoning": """
1. 设李四现在年龄为 x 岁
2. 那么张三现在年龄为 2x 岁
3. 5年后：
   - 李四年龄：x + 5
   - 张三年龄：2x + 5
4. 根据题意：5年后张三比李四大15岁
   - (2x + 5) - (x + 5) = 15
   - 2x + 5 - x - 5 = 15
   - x = 15
5. 所以李四现在15岁，张三现在30岁
6. 验证：5年后，李四20岁，张三35岁，差15岁 ✓
""",
            "answer": "30岁"
        }
    ]

    # 构建 Few-shot CoT 提示词
    prompt = "请解答以下数学问题，并展示完整的推理过程。\n\n"

    for i, ex in enumerate(cot_examples, 1):
        prompt += f"问题{i}: {ex['question']}\n"
        prompt += f"推理过程:{ex['reasoning']}\n"
        prompt += f"答案: {ex['answer']}\n\n"

    # 新问题
    new_question = """
学校买了足球和篮球共20个，足球每个30元，篮球每个25元，总共花了550元。
请问足球和篮球各买了多少个？
"""

    prompt += f"问题3: {new_question}\n推理过程:"

    print("Few-shot CoT 提示词:")
    print("-" * 40)
    print(prompt[:500] + "...")
    print("-" * 40)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )

    print(f"\n回答:\n{response.choices[0].message.content}")


# ==================== 复杂问题分解 ====================

def demo_problem_decomposition():
    """演示复杂问题分解"""
    print("\n" + "=" * 60)
    print("示例3: 复杂问题分解")
    print("=" * 60)

    # 复杂问题
    complex_question = """
一家公司要开发一个新产品，需要考虑以下因素：
1. 研发成本约100万元
2. 预计销量为第一年5000件，每年增长20%
3. 每件产品定价200元，成本120元
4. 市场推广费用第一年30万元，之后每年递减10%
5. 公司要求3年内收回成本

请问这个产品是否值得开发？请给出详细分析。
"""

    # 方法1：直接提问
    direct_prompt = complex_question

    # 方法2：引导分解
    decompose_prompt = complex_question + """

请按以下步骤分析：

步骤1：计算第一年的收入和支出
- 收入 = ?
- 支出 = ?
- 净利润 = ?

步骤2：计算第二年的收入和支出
- 收入 = ?
- 支出 = ?
- 净利润 = ?

步骤3：计算第三年的收入和支出
- 收入 = ?
- 支出 = ?
- 净利润 = ?

步骤4：汇总三年净利润，判断是否收回成本

步骤5：给出最终建议
"""

    # 方法3：自动分解
    auto_decompose_prompt = f"""
复杂问题：{complex_question}

首先，请将这个复杂问题分解为多个简单的子问题：
"""

    prompts = [
        ("直接提问", direct_prompt),
        ("引导分解", decompose_prompt),
        ("自动分解", auto_decompose_prompt)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        print(f"回答:\n{response.choices[0].message.content}")


# ==================== 推理验证 ====================

def demo_reasoning_verification():
    """演示推理过程验证"""
    print("\n" + "=" * 60)
    print("示例4: 推理过程验证")
    print("=" * 60)

    # 包含错误的推理
    question = """
下面的推理过程是否正确？如有错误，请指出。

问题：小明有10个苹果，给了小红3个，又买了5个，请问小明现在有多少个苹果？

给出的推理：
1. 小明最初有10个苹果
2. 给了小红3个，剩下 10 - 3 = 8 个
3. 又买了5个，所以现在有 8 + 5 = 12 个
4. 但是买苹果花了一些钱，所以实际只有10个
答案：10个

请检查这个推理是否正确。
"""

    # 使用 CoT 进行验证
    verification_prompt = question + """

请一步步验证：
1. 检查每个推理步骤是否正确
2. 找出任何逻辑错误或不当假设
3. 给出正确的推理和答案
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": verification_prompt}],
        max_tokens=500
    )

    print(f"验证结果:\n{response.choices[0].message.content}")


# ==================== CoT 辅助工具 ====================

class CoTReasoner:
    """思维链推理辅助类"""

    def __init__(self):
        self.examples = []

    def add_example(self, question: str, reasoning: str, answer: str):
        """添加 CoT 示例"""
        self.examples.append({
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        })

    def solve(self, question: str, use_cot: bool = True, show_examples: bool = True) -> str:
        """
        解答问题

        Args:
            question: 问题
            use_cot: 是否使用 CoT
            show_examples: 是否展示示例

        Returns:
            模型的回答
        """
        if not use_cot:
            prompt = question
        elif show_examples and self.examples:
            prompt = "请解答以下问题，展示完整的推理过程：\n\n"
            for i, ex in enumerate(self.examples, 1):
                prompt += f"示例{i}:\n问题: {ex['question']}\n"
                prompt += f"推理过程:\n{ex['reasoning']}\n"
                prompt += f"答案: {ex['answer']}\n\n"
            prompt += f"问题: {question}\n推理过程:"
        else:
            prompt = question + "\n\n让我们一步步思考："

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        return response.choices[0].message.content

    def verify(self, question: str, solution: str) -> str:
        """
        验证解答的正确性

        Args:
            question: 原问题
            solution: 给出的解答

        Returns:
            验证结果
        """
        prompt = f"""
请验证以下解答的正确性：

问题：{question}

给出的解答：
{solution}

请：
1. 检查每一步推理是否正确
2. 指出任何错误或遗漏
3. 如有错误，给出正确的解答
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        return response.choices[0].message.content


def demo_cot_reasoner():
    """演示 CoTReasoner 类的使用"""
    print("\n" + "=" * 60)
    print("示例5: 使用 CoTReasoner 类")
    print("=" * 60)

    reasoner = CoTReasoner()

    # 添加示例
    reasoner.add_example(
        question="一本书有300页，小明第一天读了1/6，第二天读了余下的1/5，两天共读了多少页？",
        reasoning="""
1. 第一天读的页数：300 × 1/6 = 50页
2. 第一天后剩余：300 - 50 = 250页
3. 第二天读的页数：250 × 1/5 = 50页
4. 两天共读：50 + 50 = 100页
""",
        answer="100页"
    )

    # 新问题
    new_question = """
学校图书馆有故事书和科技书共1200本，故事书是科技书的3倍。
后来又买进科技书200本，现在故事书是科技书的几倍？
"""

    print("问题:", new_question)
    print("\n使用 Few-shot CoT 求解...")
    print("-" * 40)

    result = reasoner.solve(new_question, use_cot=True, show_examples=True)
    print(result)


# ==================== 多步骤任务规划 ====================

def demo_task_planning():
    """演示多步骤任务规划"""
    print("\n" + "=" * 60)
    print("示例6: 多步骤任务规划")
    print("=" * 60)

    # 复杂任务
    task = """
我计划下周去日本旅游5天，需要帮我规划行程。

我的需求：
- 预算约10000元人民币
- 想去东京和京都
- 对历史文化和美食感兴趣
- 不喜欢太赶的行程

请帮我：
1. 规划具体的行程安排
2. 估算各项费用
3. 给出实用的建议
"""

    # 使用 CoT 进行规划
    planning_prompt = task + """

请按以下步骤进行规划：

步骤1：了解基本信息
- 机票价格
- 住宿费用
- 交通费用
- 餐饮费用

步骤2：制定行程框架
- 东京几天，京都几天
- 主要景点安排

步骤3：详细行程规划
- 每天的行程安排
- 预计费用

步骤4：费用汇总
- 总费用估算
- 是否在预算内

步骤5：实用建议
- 注意事项
- 省钱技巧
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": planning_prompt}],
        max_tokens=2000
    )

    print(f"规划结果:\n{response.choices[0].message.content}")


# ==================== CoT 与不同问题类型 ====================

def demo_cot_for_different_tasks():
    """演示 CoT 在不同类型任务中的应用"""
    print("\n" + "=" * 60)
    print("示例7: CoT 在不同任务中的应用")
    print("=" * 60)

    tasks = [
        {
            "type": "逻辑推理",
            "question": """
所有的猫都是动物。
所有的动物都需要食物。
小花是一只猫。
请问：小花需要食物吗？
""",
            "cot_prompt": "让我们一步步推理："
        },
        {
            "type": "常识推理",
            "question": """
早上小明出门时天气晴朗，所以没带伞。
下午突然下起了大雨。
小明在公司，离家有5公里。
请问小明可能面临什么情况？他可以怎么解决？
""",
            "cot_prompt": "让我们分析可能的情况和解决方案："
        },
        {
            "type": "阅读理解",
            "question": """
文章：电动汽车是未来交通的发展方向。相比传统燃油车，电动汽车有诸多优势：
首先是环保，电动汽车零排放，不会产生尾气污染；其次是经济，电费比油费便宜得多；
第三是维护成本低，电动机结构简单，故障率低。当然，电动汽车也面临一些挑战，
比如充电时间长、续航里程有限、充电设施不够完善等。但随着技术进步，这些问题
正在逐步解决。

问题：根据文章，电动汽车的优势和挑战分别是什么？
""",
            "cot_prompt": "让我们先找出文章中的关键信息："
        }
    ]

    for task in tasks:
        print(f"\n--- {task['type']} ---")
        print(f"问题:{task['question']}")

        prompt = task['question'] + "\n\n" + task['cot_prompt']

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        print(f"\n推理过程:\n{response.choices[0].message.content}")
        print("-" * 40)


# ==================== 自我一致性 CoT ====================

def demo_self_consistency():
    """演示自一致性 CoT（多次采样取多数）"""
    print("\n" + "=" * 60)
    print("示例8: 自一致性 CoT (Self-Consistency)")
    print("=" * 60)

    question = """
一个袋子里有红球、蓝球和绿球共30个。
红球比蓝球多5个，蓝球比绿球多3个。
请问红球有多少个？
"""

    print(f"问题:{question}")
    print("\n使用自一致性方法：生成多个推理路径，选择最一致的答案\n")

    # 设置随机种子以获得不同的回答
    answers = []
    reasonings = []

    for i in range(3):  # 生成3个推理路径
        prompt = question + "\n让我们一步步思考："

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7  # 增加随机性
        )

        result = response.choices[0].message.content
        reasonings.append(result)
        print(f"推理路径 {i+1}:\n{result}\n")
        print("-" * 40)

        # 提取答案（简单示例，实际需要更复杂的解析）
        # 这里我们让模型来总结最终答案

    # 让模型进行一致性检查并给出最终答案
    final_prompt = f"""
问题：{question}

我得到了以下推理结果：
{'-' * 40}
推理1：{reasonings[0]}
{'-' * 40}
推理2：{reasonings[1]}
{'-' * 40}
推理3：{reasonings[2]}
{'-' * 40}

请检查以上推理是否一致，如果一致请给出最终答案；如果不一致，请指出哪个推理是正确的，并给出正确答案。
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=500,
        temperature=0  # 更确定性的回答
    )

    print(f"一致性分析:\n{response.choices[0].message.content}")


# ==================== 交互式演示 ====================

def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("Chain of Thought - 交互模式")
    print("=" * 60)
    print("输入问题，体验 CoT 推理过程")
    print("输入 'quit' 退出")
    print("=" * 60)

    while True:
        print("\n选择模式：")
        print("1. 普通（无 CoT）")
        print("2. Zero-shot CoT")
        print("3. Few-shot CoT（需要先添加示例）")
        print("4. 退出")

        choice = input("\n请选择 (1-4): ").strip()

        if choice == '4' or choice.lower() in ['quit', 'exit', 'q']:
            break

        question = input("请输入问题: ").strip()
        if not question:
            continue

        if choice == '1':
            # 普通
            prompt = question
        elif choice == '2':
            # Zero-shot CoT
            prompt = question + "\n\n让我们一步步思考："
        elif choice == '3':
            # Few-shot CoT
            print("Few-shot CoT 需要示例。这里使用预设示例。")
            prompt = """
示例问题：小明有5个苹果，给了小红2个，又买了3个，还剩多少？
推理过程：
1. 小明最初有5个苹果
2. 给了小红2个，剩余5 - 2 = 3个
3. 又买了3个，现在有3 + 3 = 6个
答案：6个苹果

""" + question + "\n推理过程:"
        else:
            print("无效选择")
            continue

        print("\n生成中...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        print(f"\n回答:\n{response.choices[0].message.content}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - Chain of Thought (CoT)")
    print("=" * 60)

    # 运行所有演示
    demo_zero_shot_cot()
    demo_few_shot_cot()
    demo_problem_decomposition()
    demo_reasoning_verification()
    demo_cot_reasoner()
    demo_task_planning()
    demo_cot_for_different_tasks()
    demo_self_consistency()

    # 交互模式
    print("\n" + "=" * 60)
    print("是否进入交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        interactive_demo()