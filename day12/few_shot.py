"""
Few-shot Learning 示例
演示示例设计技巧、示例数量和多样性、负面示例的使用

核心概念：
1. Zero-shot - 无示例，直接让模型完成任务
2. One-shot - 提供一个示例
3. Few-shot - 提供多个示例，引导模型理解任务模式

Few-shot 设计原则：
- 示例要代表性：覆盖各种情况
- 示例要多样性：简单、复杂、边界情况
- 格式要一致性：相同的输入输出格式
- 数量要适中：通常 3-5 个最佳
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


# ==================== Few-shot 提示词构建器 ====================

class FewShotPromptBuilder:
    """Few-shot 提示词构建器"""

    def __init__(self, task_description: str):
        """
        初始化

        Args:
            task_description: 任务描述
        """
        self.task_description = task_description
        self.examples = []

    def add_example(self, input_text: str, output_text: str, explanation: str = None):
        """
        添加示例

        Args:
            input_text: 输入文本
            output_text: 输出文本
            explanation: 可选的解释说明
        """
        example = {
            "input": input_text,
            "output": output_text
        }
        if explanation:
            example["explanation"] = explanation
        self.examples.append(example)

    def add_negative_example(self, input_text: str, wrong_output: str, correct_output: str, reason: str):
        """
        添加负面示例（展示错误和正确做法）

        Args:
            input_text: 输入文本
            wrong_output: 错误的输出
            correct_output: 正确的输出
            reason: 错误原因
        """
        self.examples.append({
            "type": "negative",
            "input": input_text,
            "wrong_output": wrong_output,
            "correct_output": correct_output,
            "reason": reason
        })

    def build(self, query: str, format_type: str = "standard") -> str:
        """
        构建完整提示词

        Args:
            query: 待处理的查询
            format_type: 格式类型 ("standard", "conversational", "structured")

        Returns:
            完整的提示词
        """
        if format_type == "conversational":
            return self._build_conversational(query)
        elif format_type == "structured":
            return self._build_structured(query)
        else:
            return self._build_standard(query)

    def _build_standard(self, query: str) -> str:
        """标准格式"""
        prompt = f"{self.task_description}\n\n"

        for i, ex in enumerate(self.examples, 1):
            if ex.get("type") == "negative":
                prompt += f"负面示例 {i}:\n"
                prompt += f"输入: {ex['input']}\n"
                prompt += f"❌ 错误输出: {ex['wrong_output']}\n"
                prompt += f"✅ 正确输出: {ex['correct_output']}\n"
                prompt += f"错误原因: {ex['reason']}\n\n"
            else:
                prompt += f"示例 {i}:\n"
                prompt += f"输入: {ex['input']}\n"
                prompt += f"输出: {ex['output']}"
                if ex.get("explanation"):
                    prompt += f" ({ex['explanation']})"
                prompt += "\n\n"

        prompt += f"现在请处理:\n输入: {query}\n输出:"
        return prompt

    def _build_conversational(self, query: str) -> str:
        """对话格式"""
        prompt = f"{self.task_description}\n\n"
        prompt += "以下是几个对话示例：\n\n"

        for ex in self.examples:
            if ex.get("type") == "negative":
                continue  # 对话格式不适合负面示例
            prompt += f"用户: {ex['input']}\n"
            prompt += f"助手: {ex['output']}\n\n"

        prompt += f"用户: {query}\n助手:"
        return prompt

    def _build_structured(self, query: str) -> str:
        """结构化格式（使用 JSON）"""
        prompt = f"{self.task_description}\n\n"
        prompt += "参考以下示例：\n\n"

        for i, ex in enumerate(self.examples, 1):
            if ex.get("type") == "negative":
                prompt += f"示例 {i} (负面示例):\n"
                prompt += f"输入: {ex['input']}\n"
                prompt += f"错误输出: {ex['wrong_output']}\n"
                prompt += f"正确输出: {ex['correct_output']}\n"
                prompt += f"原因: {ex['reason']}\n\n"
            else:
                prompt += f"示例 {i}:\n"
                prompt += json.dumps({
                    "input": ex['input'],
                    "output": ex['output']
                }, ensure_ascii=False, indent=2)
                prompt += "\n\n"

        prompt += f"请处理以下输入:\n{query}"
        return prompt


# ==================== 示例1: Zero-shot vs One-shot vs Few-shot 对比 ====================

def demo_shot_comparison():
    """对比 Zero-shot、One-shot、Few-shot 的效果"""
    print("\n" + "=" * 60)
    print("示例1: Zero-shot vs One-shot vs Few-shot 对比")
    print("=" * 60)

    # 任务：文本分类
    text_to_classify = "这家餐厅的服务太差了，等了一个小时才上菜，再也不来了！"

    # Zero-shot
    zero_shot_prompt = f"请判断以下文本的情感（正面/负面/中性）：\n{text_toClassify}"

    # One-shot
    one_shot_prompt = f"""
请判断文本的情感（正面/负面/中性）。

示例：
文本：这个产品很好用，推荐购买！
情感：正面

请判断以下文本的情感：
文本：{text_to_classify}
情感：
"""

    # Few-shot (3个示例)
    few_shot_prompt = f"""
请判断文本的情感（正面/负面/中性）。

示例1：
文本：这个产品很好用，推荐购买！
情感：正面

示例2：
文本：服务态度一般，价格偏贵，不太满意。
情感：负面

示例3：
文本：还可以吧，没什么特别的感觉。
情感：中性

请判断以下文本的情感：
文本：{text_to_classify}
情感：
"""

    # Few-shot (5个示例，展示多样性)
    few_shot_diverse_prompt = f"""
请判断文本的情感（正面/负面/中性）。

示例1（简单正面）：
文本：很好！
情感：正面

示例2（简单负面）：
文本：太差了。
情感：负面

示例3（混合情感）：
文本：虽然质量不错，但价格太贵了，性价比不高。
情感：负面（整体不满意）

示例4（隐含情感）：
文本：我等了三个小时才收到货。
情感：负面（暗示不满）

示例5（中性）：
文本：产品已收到，包装完好。
情感：中性

请判断以下文本的情感：
文本：{text_to_classify}
情感：
"""

    prompts = [
        ("Zero-shot", zero_shot_prompt),
        ("One-shot", one_shot_prompt),
        ("Few-shot (3个示例)", few_shot_prompt),
        ("Few-shot (5个多样化示例)", few_shot_diverse_prompt)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        print(f"输出:\n{response.choices[0].message.content}")


# ==================== 示例2: 示例设计技巧 ====================

def demo_example_design():
    """演示示例设计技巧"""
    print("\n" + "=" * 60)
    print("示例2: 示例设计技巧")
    print("=" * 60)

    # 任务：命名实体识别
    task = "从文本中识别人名、地点和组织"

    # 差的示例设计（缺乏多样性）
    bad_examples = [
        {"input": "张三在北京工作", "output": "人名：张三，地点：北京"},
        {"input": "李四在上海工作", "output": "人名：李四，地点：上海"},
        {"input": "王五在广州工作", "output": "人名：王五，地点：广州"},
    ]

    # 好的示例设计（多样性）
    good_examples = [
        # 简单示例
        {"input": "张三在北京工作", "output": "人名：张三，地点：北京"},
        # 复杂示例
        {"input": "阿里巴巴集团和腾讯公司在北京和深圳都有办公室",
         "output": "组织：阿里巴巴集团、腾讯公司，地点：北京、深圳"},
        # 边界情况
        {"input": "李小明想去美国斯坦福大学读书",
         "output": "人名：李小明，地点：美国、斯坦福大学，组织：斯坦福大学"},
        # 无实体情况
        {"input": "今天天气真好", "output": "无命名实体"},
    ]

    query = "华为公司在深圳和东莞都有研发中心，任正非是创始人"

    # 构建差的示例提示词
    bad_prompt = f"{task}\n\n"
    for i, ex in enumerate(bad_examples, 1):
        bad_prompt += f"示例{i}: {ex['input']} -> {ex['output']}\n"
    bad_prompt += f"\n处理: {query}"

    # 构建好的示例提示词
    good_prompt = f"{task}\n\n"
    for i, ex in enumerate(good_examples, 1):
        good_prompt += f"示例{i}:\n输入: {ex['input']}\n输出: {ex['output']}\n\n"
    good_prompt += f"处理: {query}"

    prompts = [
        ("差的设计（缺乏多样性）", bad_prompt),
        ("好的设计（多样性）", good_prompt)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        print(f"输出:\n{response.choices[0].message.content}")


# ==================== 示例3: 负面示例的使用 ====================

def demo_negative_examples():
    """演示负面示例的使用"""
    print("\n" + "=" * 60)
    print("示例3: 负面示例的使用")
    print("=" * 60)

    # 任务：文本摘要
    # 负面示例可以帮助模型理解"不要做什么"

    # 无负面示例
    without_negative = """
任务：生成文本摘要，不超过50字。

示例1：
原文：人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
摘要：AI是制造智能机器的计算机科学分支。

示例2：
原文：机器学习是人工智能的核心，是使计算机具有智能的根本途径。
摘要：机器学习是实现AI智能的核心方法。

请生成以下文本的摘要：
原文：深度学习是机器学习领域中一个新的研究方向，其目的是让机器能够像人一样具有分析学习能力，能够识别文字、图像和声音等数据。
摘要：
"""

    # 有负面示例
    with_negative = """
任务：生成文本摘要，不超过50字。

正面示例1：
原文：人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
摘要：AI是制造智能机器的计算机科学分支。

正面示例2：
原文：机器学习是人工智能的核心，是使计算机具有智能的根本途径。
摘要：机器学习是实现AI智能的核心方法。

负面示例1（错误示范）：
原文：深度学习是机器学习领域中一个新的研究方向。
❌ 错误摘要：深度学习是机器学习领域中的一个新研究方向，其目的是让机器能够像人一样具有分析学习能力，能够识别文字、图像和声音等数据。（太长，直接复制原文）
✅ 正确摘要：深度学习是让机器具备分析学习能力的新方向。
错误原因：摘要应精炼概括，不能直接复制原文，字数应控制在50字以内。

负面示例2（错误示范）：
原文：区块链是一种去中心化的分布式账本技术。
❌ 错误摘要：一种技术。（过于简略）
✅ 正确摘要：区块链是去中心化的分布式账本技术。
错误原因：过于简略，丢失了关键信息。

请生成以下文本的摘要：
原文：深度学习是机器学习领域中一个新的研究方向，其目的是让机器能够像人一样具有分析学习能力，能够识别文字、图像和声音等数据。
摘要：
"""

    prompts = [
        ("无负面示例", without_negative),
        ("有负面示例", with_negative)
    ]

    for name, prompt in prompts:
        print(f"\n--- {name} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        print(f"输出:\n{response.choices[0].message.content}")


# ==================== 示例4: 示例数量对比 ====================

def demo_example_count():
    """对比不同示例数量的效果"""
    print("\n" + "=" * 60)
    print("示例4: 示例数量对比")
    print("=" * 60)

    # 任务：情感分析（带情感词提取）
    task_description = "分析文本情感，并提取关键词"

    examples = [
        {"input": "这电影太精彩了！", "output": "情感：正面\n关键词：精彩"},
        {"input": "服务很差，很失望", "output": "情感：负面\n关键词：差、失望"},
        {"input": "还行吧，一般般", "output": "情感：中性\n关键词：还行、一般"},
        {"input": "虽然有点贵，但质量确实好", "output": "情感：正面\n关键词：贵、质量好"},
        {"input": "等了半小时没人理", "output": "情感：负面\n关键词：等、没人理"},
    ]

    query = "这个产品功能很强大，但客服态度不好"

    # 测试不同数量
    test_counts = [0, 1, 2, 3, 5]

    for count in test_counts:
        if count == 0:
            prompt = f"{task_description}\n\n处理: {query}"
            label = "Zero-shot (0个示例)"
        else:
            selected = examples[:count]
            prompt = f"{task_description}\n\n"
            for i, ex in enumerate(selected, 1):
                prompt += f"示例{i}:\n输入: {ex['input']}\n输出: {ex['output']}\n\n"
            prompt += f"处理: {query}"
            label = f"Few-shot ({count}个示例)"

        print(f"\n--- {label} ---")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        print(f"输出:\n{response.choices[0].message.content}")


# ==================== 示例5: 使用 FewShotPromptBuilder ====================

def demo_prompt_builder():
    """演示使用 FewShotPromptBuilder 类"""
    print("\n" + "=" * 60)
    print("示例5: 使用 FewShotPromptBuilder 类")
    print("=" * 60)

    # 创建一个翻译任务
    builder = FewShotPromptBuilder(
        task_description="将中文翻译成英文，保持原意，用词自然"
    )

    # 添加正面示例
    builder.add_example(
        input_text="早上好",
        output_text="Good morning",
        explanation="常用问候语"
    )

    builder.add_example(
        input_text="请稍等一下",
        output_text="Please wait a moment",
        explanation="礼貌用语"
    )

    builder.add_example(
        input_text="这个项目的进展如何？",
        output_text="How is the project going?",
        explanation="商务询问"
    )

    # 添加负面示例
    builder.add_negative_example(
        input_text="他是个好人",
        wrong_output="He is a good person",
        correct_output="He is a good man",
        reason="'good person' 虽然语法正确，但 'good man' 更自然常用"
    )

    # 构建提示词
    query = "我很期待这次的合作"
    prompt = builder.build(query, format_type="standard")

    print("构建的提示词:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    # 调用模型
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    print(f"\n翻译结果:\n{response.choices[0].message.content}")


# ==================== 示例6: 实战案例 - 意图识别 ====================

def demo_intent_recognition():
    """实战案例：意图识别"""
    print("\n" + "=" * 60)
    print("示例6: 实战案例 - 用户意图识别")
    print("=" * 60)

    # 意图分类任务
    intent_examples = [
        # 查询意图
        {"input": "今天天气怎么样", "intent": "query_weather", "slots": {"city": "默认城市"}},
        {"input": "北京明天会下雨吗", "intent": "query_weather", "slots": {"city": "北京", "date": "明天"}},

        # 播放音乐
        {"input": "播放周杰伦的歌", "intent": "play_music", "slots": {"artist": "周杰伦"}},
        {"input": "我想听安静的音乐", "intent": "play_music", "slots": {"mood": "安静"}},

        # 设置闹钟
        {"input": "明天早上7点叫我", "intent": "set_alarm", "slots": {"time": "明天早上7点"}},
        {"input": "提醒我下午3点开会", "intent": "set_reminder", "slots": {"time": "下午3点", "content": "开会"}},

        # 闲聊
        {"input": "你好", "intent": "greeting", "slots": {}},
        {"input": "讲个笑话", "intent": "chat", "slots": {"topic": "笑话"}},
    ]

    # 构建 Few-shot 提示词
    prompt = """任务：识别用户意图并提取槽位信息

支持的意图类型：
- query_weather: 查询天气
- play_music: 播放音乐
- set_alarm: 设置闹钟
- set_reminder: 设置提醒
- greeting: 问候
- chat: 闲聊

示例：
"""

    for ex in intent_examples:
        prompt += f"\n用户：{ex['input']}\n"
        prompt += f"意图：{ex['intent']}\n"
        if ex['slots']:
            prompt += f"槽位：{json.dumps(ex['slots'], ensure_ascii=False)}\n"

    # 测试查询
    queries = [
        "上海后天天气如何",
        "放一首轻快的歌",
        "下周一早上9点提醒我有会议",
    ]

    for query in queries:
        print(f"\n用户输入: {query}")
        print("-" * 40)

        full_prompt = prompt + f"\n用户：{query}\n意图："

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=100
        )

        print(f"识别结果:\n{response.choices[0].message.content}")


# ==================== 示例7: 实战案例 - 文本风格转换 ====================

def demo_style_transfer():
    """实战案例：文本风格转换"""
    print("\n" + "=" * 60)
    print("示例7: 实战案例 - 文本风格转换")
    print("=" * 60)

    # 风格转换任务
    formal_examples = [
        {"informal": "咱们明天见啊", "formal": "我们明天再见"},
        {"informal": "这事挺麻烦的", "formal": "此事较为复杂"},
        {"informal": "你看着办吧", "formal": "请您自行决定"},
    ]

    prompt = """任务：将口语化的文本转换为正式书面语

示例：
"""

    for ex in formal_examples:
        prompt += f"口语：{ex['informal']}\n"
        prompt += f"正式：{ex['formal']}\n\n"

    # 测试
    test_cases = [
        "这个东西挺好的",
        "我明天不去上班了",
        "那啥，你帮我看看这个",
    ]

    for test in test_cases:
        print(f"\n口语: {test}")

        full_prompt = prompt + f"口语：{test}\n正式："

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=50
        )

        print(f"正式: {response.choices[0].message.content.strip()}")


# ==================== 交互式演示 ====================

def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("Few-shot Learning - 交互模式")
    print("=" * 60)
    print("输入一个任务描述和一些示例，然后测试新查询")
    print("输入 'quit' 退出")
    print("=" * 60)

    # 获取任务描述
    task = input("\n请输入任务描述（如：将中文翻译成英文）: ").strip()
    if task.lower() in ['quit', 'exit', 'q']:
        return

    # 获取示例
    builder = FewShotPromptBuilder(task)
    print("\n请输入示例（输入空行结束）：")

    i = 1
    while True:
        print(f"\n示例 {i}:")
        example_input = input("  输入: ").strip()
        if not example_input:
            break
        example_output = input("  输出: ").strip()
        builder.add_example(example_input, example_output)
        i += 1

    # 测试循环
    print("\n现在可以输入查询进行测试（输入 'quit' 退出）：")

    while True:
        query = input("\n查询: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        prompt = builder.build(query)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        print(f"结果: {response.choices[0].message.content}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - Few-shot Learning")
    print("=" * 60)

    # 运行所有演示
    demo_shot_comparison()
    demo_example_design()
    demo_negative_examples()
    demo_example_count()
    demo_prompt_builder()
    demo_intent_recognition()
    demo_style_transfer()

    # 交互模式
    print("\n" + "=" * 60)
    print("是否进入交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        interactive_demo()