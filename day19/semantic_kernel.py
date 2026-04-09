"""
Day 19: Semantic Kernel 示例

本文件演示 Semantic Kernel 的核心概念：
- Kernel 初始化与配置（使用 DeepSeek API）
- Semantic Function 创建
- Native Function 定义
- Skills 组合使用
- Planner 自动规划

注意：Semantic Kernel Python 版本需要特定配置以使用 OpenAI 兼容 API。
"""

import os
import asyncio
from typing import Annotated
from dotenv import load_dotenv

# Semantic Kernel 导入
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import (
        OpenAIChatCompletion,
        OpenAITextCompletion,
    )
    from semantic_kernel.core_skills import TextSkill
    HAS_SEMANTIC_KERNEL = True
except ImportError:
    HAS_SEMANTIC_KERNEL = False
    print("⚠️ Semantic Kernel 未安装，请运行: pip install semantic-kernel")


# 加载环境变量
load_dotenv()


def create_kernel_with_deepseek():
    """创建配置 DeepSeek API 的 Kernel"""
    if not HAS_SEMANTIC_KERNEL:
        print("请先安装 Semantic Kernel")
        return None

    # 创建 Kernel
    kernel = sk.Kernel()

    # 配置 OpenAI 兼容的 Chat 服务
    # DeepSeek API 兼容 OpenAI 接口
    chat_service = OpenAIChatCompletion(
        ai_model_id=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        endpoint=os.getenv("DEEPSEEK_BASE_URL"),
    )

    # 添加服务到 Kernel
    kernel.add_chat_service("deepseek_chat", chat_service)

    print("✅ Kernel 已配置 DeepSeek API")
    return kernel


# ==========================================
# 示例 1: 基础 Semantic Function
# ==========================================


async def example_semantic_function():
    """演示 Semantic Function（语义函数）的创建和使用"""
    print("\n" + "=" * 50)
    print("示例 1: Semantic Function 基础")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    # 定义语义函数模板
    # Semantic Function 使用提示模板定义功能
    summarize_template = """
    请将以下文本进行摘要，保持核心信息：

    {{$input}}

    摘要：
    """

    # 创建语义函数
    summarize_function = kernel.create_semantic_function(
        prompt_template=summarize_template,
        function_name="summarize",
        skill_name="text_skills",
        max_tokens=200,
        temperature=0.3,
    )

    print("✅ Semantic Function 'summarize' 已创建")

    # 使用函数
    input_text = """
    Semantic Kernel 是微软开发的一个轻量级 SDK，
    它允许开发者将大语言模型与传统编程语言结合起来。
    通过 Semantic Kernel，开发者可以定义"技能"——包括语义技能
    和原生技能，然后使用规划器自动编排这些技能来完成复杂任务。
    """

    # 调用函数
    context = kernel.create_new_context()
    context["input"] = input_text

    result = await kernel.run_async(summarize_function, input_vars=context)

    print(f"\n原文摘要:")
    print(f"  {result.result[:150]}...")


# ==========================================
# 示例 2: 多参数 Semantic Function
# ==========================================


async def example_multi_param_function():
    """演示多参数的 Semantic Function"""
    print("\n" + "=" * 50)
    print("示例 2: 多参数 Semantic Function")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    # 定义翻译函数模板
    translate_template = """
    请将以下文本翻译为 {{$target_language}}：

    {{$input}}

    翻译结果：
    """

    # 创建翻译函数
    translate_function = kernel.create_semantic_function(
        prompt_template=translate_template,
        function_name="translate",
        skill_name="language_skills",
        max_tokens=500,
        temperature=0.3,
    )

    print("✅ Semantic Function 'translate' 已创建")

    # 使用函数
    context = kernel.create_new_context()
    context["input"] = "Hello, how are you today?"
    context["target_language"] = "中文"

    result = await kernel.run_async(translate_function, input_vars=context)

    print(f"\n原文: Hello, how are you today?")
    print(f"翻译: {result.result}")


# ==========================================
# 示例 3: Native Function（原生函数）
# ==========================================


def example_native_function():
    """演示 Native Function（原生函数）的定义和使用"""
    print("\n" + "=" * 50)
    print("示例 3: Native Function")
    print("=" * 50)

    if not HAS_SEMANTIC_KERNEL:
        print("请先安装 Semantic Kernel")
        return

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    # 定义原生函数（Python 代码实现的功能）
    # 使用装饰器将普通函数转换为 Semantic Kernel 的技能

    from semantic_kernel import sk_function

    class MathSkills:
        """数学技能类"""

        @sk_function(
            description="计算两个数的和",
            name="add"
        )
        def add_numbers(
            self,
            a: Annotated[float, "第一个数字"],
            b: Annotated[float, "第二个数字"]
        ) -> Annotated[float, "计算结果"]:
            """加法运算"""
            return a + b

        @sk_function(
            description="计算两个数的乘积",
            name="multiply"
        )
        def multiply_numbers(
            self,
            a: Annotated[float, "第一个数字"],
            b: Annotated[float, "第二个数字"]
        ) -> Annotated[float, "计算结果"]:
            """乘法运算"""
            return a * b

    # 将原生技能导入 Kernel
    math_skills = MathSkills()
    kernel.import_skill(math_skills, "math")

    print("✅ Native Skills 'math' 已导入")

    # 获取函数并执行
    add_func = kernel.skills.get_function("math", "add")
    multiply_func = kernel.skills.get_function("math", "multiply")

    # 创建上下文并设置参数
    context = kernel.create_new_context()
    context["a"] = "10"
    context["b"] = "5"

    # 执行原生函数
    add_result = await add_func.invoke_async(context)
    print(f"\n加法: 10 + 5 = {add_result.result}")

    multiply_result = await multiply_func.invoke_async(context)
    print(f"乘法: 10 * 5 = {multiply_result.result}")


# ==========================================
# 示例 4: Skills 组合使用
# ==========================================


async def example_skills_combination():
    """演示多种技能的组合使用"""
    print("\n" + "=" * 50)
    print("示例 4: Skills 组合使用")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    from semantic_kernel import sk_function

    # 定义文本处理技能
    class TextProcessingSkills:
        """文本处理技能"""

        @sk_function(
            description="计算文本的单词数量",
            name="count_words"
        )
        def count_words(
            self,
            text: Annotated[str, "输入文本"]
        ) -> Annotated[int, "单词数量"]:
            """计算单词数"""
            return len(text.split())

        @sk_function(
            description="截取文本的前N个字符",
            name="truncate"
        )
        def truncate_text(
            self,
            text: Annotated[str, "输入文本"],
            length: Annotated[int, "截取长度"]
        ) -> Annotated[str, "截取后的文本"]:
            """截取文本"""
            return text[:length] + "..." if len(text) > length else text

    # 定义语义技能
    sentiment_template = """
    分析以下文本的情感倾向（正面/负面/中性）：

    {{$input}}

    情感分析结果：
    """

    # 导入技能
    text_skills = TextProcessingSkills()
    kernel.import_skill(text_skills, "text_utils")

    sentiment_func = kernel.create_semantic_function(
        prompt_template=sentiment_template,
        function_name="analyze_sentiment",
        skill_name="analysis_skills",
        max_tokens=100,
        temperature=0.3,
    )

    print("✅ Skills 组合配置完成")

    # 测试文本
    test_text = "Semantic Kernel 是一个非常有用的工具，让我轻松构建 AI 应用！"

    # 1. 计算单词数
    context = kernel.create_new_context()
    context["text"] = test_text
    context["length"] = "50"

    count_func = kernel.skills.get_function("text_utils", "count_words")
    count_result = await count_func.invoke_async(context)
    print(f"\n单词数量: {count_result.result}")

    # 2. 截取文本
    truncate_func = kernel.skills.get_function("text_utils", "truncate")
    truncate_result = await truncate_func.invoke_async(context)
    print(f"截取文本: {truncate_result.result}")

    # 3. 情感分析
    context["input"] = test_text
    sentiment_result = await kernel.run_async(sentiment_func, input_vars=context)
    print(f"情感分析: {sentiment_result.result}")


# ==========================================
# 示例 5: Planner 自动规划（概念演示）
# ==========================================


async def example_planner_concept():
    """演示 Planner 自动规划的概念"""
    print("\n" + "=" * 50)
    print("示例 5: Planner 自动规划概念")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    print("""
    Planner 是 Semantic Kernel 的核心功能之一，
    它可以根据用户的目标自动规划执行顺序。

    工作流程：
    1. 用户提出目标："分析这段文本并生成报告"
    2. Planner 分析可用的 Skills
    3. Planner 生成执行计划：
       - Step 1: 使用 sentiment 函数分析情感
       - Step 2: 使用 summarize 函数生成摘要
       - Step 3: 使用 report 函数生成报告
    4. Kernel 按计划执行

    Semantic Kernel 支持多种 Planner：
    - SequentialPlanner: 顺序执行计划
    - ActionPlanner: 单步行动计划
    - StepwisePlanner: 分步执行计划

    注意：完整的 Planner 功能需要更多的 Skills 定义，
    这里展示概念和基本配置。
    """)

    # 创建一些简单的语义技能
    extract_template = """
    从以下文本中提取关键信息：

    {{$input}}

    关键信息：
    """

    report_template = """
    基于以下信息生成简短报告：

    {{$input}}

    报告：
    """

    extract_func = kernel.create_semantic_function(
        prompt_template=extract_template,
        function_name="extract_key_info",
        skill_name="analysis_skills",
        max_tokens=200,
    )

    report_func = kernel.create_semantic_function(
        prompt_template=report_template,
        function_name="generate_report",
        skill_name="output_skills",
        max_tokens=300,
    )

    print("✅ 已创建技能用于演示")

    # 模拟手动编排（实际中 Planner 会自动完成）
    print("\n模拟 Planner 执行过程:")

    text = """
    Semantic Kernel 提供了一种全新的 AI 应用开发方式。
    它让开发者能够轻松地将大语言模型的能力集成到应用中。
    通过语义技能和原生技能的组合，可以实现复杂的功能。
    """

    # Step 1: 提取关键信息
    print("\n  Step 1: 提取关键信息")
    context = kernel.create_new_context()
    context["input"] = text
    extract_result = await kernel.run_async(extract_func, input_vars=context)
    print(f"    结果: {extract_result.result[:100]}...")

    # Step 2: 生成报告
    print("\n  Step 2: 生成报告")
    context["input"] = extract_result.result
    report_result = await kernel.run_async(report_func, input_vars=context)
    print(f"    结果: {report_result.result}")


# ==========================================
# 示例 6: 使用内置技能
# ==========================================


async def example_builtin_skills():
    """演示 Semantic Kernel 的内置技能"""
    print("\n" + "=" * 50)
    print("示例 6: 内置技能")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    # Semantic Kernel 提供了一些内置的核心技能
    print("""
    Semantic Kernel 内置技能包括：

    1. TextSkill - 文本处理
       - trim: 去除空格
       - uppercase: 转大写
       - lowercase: 转小写

    2. FileSkill - 文件操作（需要配置）

    3. TimeSkill - 时间处理

    4. MathSkill - 数学运算

    这里演示使用内置 TextSkill 的概念。
    """)

    # 导入内置技能（如果可用）
    try:
        from semantic_kernel.core_skills import TextSkill

        text_skill = TextSkill()
        kernel.import_skill(text_skill, "text_core")

        print("✅ TextSkill 已导入")

        # 测试内置技能
        context = kernel.create_new_context()
        context["input"] = "Hello Semantic Kernel"

        # 使用 uppercase
        uppercase_func = kernel.skills.get_function("text_core", "uppercase")
        if uppercase_func:
            result = await uppercase_func.invoke_async(context)
            print(f"\nuppercase: {result.result}")

    except Exception as e:
        print(f"内置技能演示需要完整 Semantic Kernel 安装: {e}")


# ==========================================
# 示例 7: 语义函数链式调用
# ==========================================


async def example_function_chain():
    """演示语义函数的链式调用"""
    print("\n" + "=" * 50)
    print("示例 7: 语义函数链式调用")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    # 创建一系列语义函数形成处理链
    # 1. 输入预处理
    preprocess_template = """
    优化以下用户输入，使其更加清晰明确：

    {{$input}}

    优化后的输入：
    """

    # 2. 内容分析
    analyze_template = """
    分析以下内容的主要主题和要点：

    {{$input}}

    分析结果：
    """

    # 3. 结果输出
    output_template = """
    将以下分析结果以结构化的方式呈现：

    {{$input}}

    结构化输出：
    """

    # 创建函数
    preprocess_func = kernel.create_semantic_function(
        prompt_template=preprocess_template,
        function_name="preprocess",
        skill_name="pipeline",
        max_tokens=100,
    )

    analyze_func = kernel.create_semantic_function(
        prompt_template=analyze_template,
        function_name="analyze",
        skill_name="pipeline",
        max_tokens=200,
    )

    output_func = kernel.create_semantic_function(
        prompt_template=output_template,
        function_name="format_output",
        skill_name="pipeline",
        max_tokens=300,
    )

    print("✅ 处理链函数已创建")

    # 链式调用
    user_input = "我想了解 Semantic Kernel 是什么"

    print(f"\n原始输入: {user_input}")

    context = kernel.create_new_context()

    # Step 1: 预处理
    print("\n处理链执行:")
    context["input"] = user_input
    preprocess_result = await kernel.run_async(preprocess_func, input_vars=context)
    print(f"  Step 1 (预处理): {preprocess_result.result[:80]}...")

    # Step 2: 分析
    context["input"] = preprocess_result.result
    analyze_result = await kernel.run_async(analyze_func, input_vars=context)
    print(f"  Step 2 (分析): {analyze_result.result[:80]}...")

    # Step 3: 输出格式化
    context["input"] = analyze_result.result
    output_result = await kernel.run_async(output_func, input_vars=context)
    print(f"  Step 3 (输出): {output_result.result[:150]}...")


# ==========================================
# 示例 8: 上下文变量传递
# ==========================================


async def example_context_variables():
    """演示上下文变量的传递和使用"""
    print("\n" + "=" * 50)
    print("示例 8: 上下文变量")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    # 定义使用多个变量的语义函数
    custom_template = """
    根据以下信息生成一个简短介绍：

    主题：{{$topic}}
    目标受众：{{$audience}}
    语言风格：{{$style}}

    请生成介绍：
    """

    custom_func = kernel.create_semantic_function(
        prompt_template=custom_template,
        function_name="custom_intro",
        skill_name="content_skills",
        max_tokens=300,
        temperature=0.7,
    )

    print("✅ 自定义变量函数已创建")

    # 使用多个上下文变量
    context = kernel.create_new_context()
    context["topic"] = "Semantic Kernel 框架"
    context["audience"] = "软件开发者"
    context["style"] = "专业简洁"

    print("\n变量设置:")
    print(f"  topic: {context['topic']}")
    print(f"  audience: {context['audience']}")
    print(f"  style: {context['style']}")

    result = await kernel.run_async(custom_func, input_vars=context)

    print(f"\n生成结果:")
    print(f"  {result.result}")


# ==========================================
# 示例 9: 模拟复杂任务编排
# ==========================================


async def example_complex_orchestration():
    """演示复杂任务的编排"""
    print("\n" + "=" * 50)
    print("示例 9: 复杂任务编排")
    print("=" * 50)

    kernel = create_kernel_with_deepseek()
    if kernel is None:
        return

    print("""
    在实际应用中，Semantic Kernel 可以编排复杂任务。

    示例场景：构建一个内容创作系统

    可用技能：
    1. brainstorm: 生成创意想法
    2. draft: 根据想法撰写草稿
    3. review: 审阅草稿质量
    4. refine: 根据审阅优化内容
    5. polish: 最终润色

    Planner 可以根据目标自动选择和排序这些技能。
    """)

    # 创建模拟的技能
    skills_templates = {
        "brainstorm": """
        为以下主题生成3个创意角度：

        {{$topic}}

        创意角度：
        """,

        "draft": """
        根据以下创意撰写简短内容：

        {{$input}}

        内容草稿：
        """,

        "refine": """
        优化以下内容，使其更加专业：

        {{$input}}

        优化后的内容：
        """
    }

    # 创建函数
    for name, template in skills_templates.items():
        kernel.create_semantic_function(
            prompt_template=template,
            function_name=name,
            skill_name="content_creation",
            max_tokens=200,
        )

    print("✅ 内容创作技能已创建")

    # 手动编排执行
    topic = "AI Agent 的工作原理"

    print(f"\n任务: 创作关于 '{topic}' 的内容")

    context = kernel.create_new_context()
    context["topic"] = topic

    # 执行流程
    print("\n执行编排流程:")

    brainstorm_func = kernel.skills.get_function("content_creation", "brainstorm")
    draft_func = kernel.skills.get_function("content_creation", "draft")
    refine_func = kernel.skills.get_function("content_creation", "refine")

    # Step 1: 创意生成
    print("  Step 1: 创意生成")
    brainstorm_result = await kernel.run_async(brainstorm_func, input_vars=context)
    print(f"    结果: {brainstorm_result.result[:100]}...")

    # Step 2: 撰写草稿
    print("  Step 2: 撰写草稿")
    context["input"] = brainstorm_result.result
    draft_result = await kernel.run_async(draft_func, input_vars=context)
    print(f"    结果: {draft_result.result[:100]}...")

    # Step 3: 优化内容
    print("  Step 3: 优化内容")
    context["input"] = draft_result.result
    refine_result = await kernel.run_async(refine_func, input_vars=context)
    print(f"    结果: {refine_result.result[:150]}...")


# ==========================================
# 主程序
# ==========================================


async def main_async():
    """运行所有示例（异步版本）"""
    print("\n" + "=" * 60)
    print("Day 19: Semantic Kernel 示例")
    print("=" * 60)

    if not HAS_SEMANTIC_KERNEL:
        print("""
        ⚠️ Semantic Kernel 未安装

        请运行以下命令安装：
        pip install semantic-kernel

        安装后可以运行完整的示例代码。

        Semantic Kernel 简介：
        - Microsoft 开发的 LLM 应用开发 SDK
        - 支持语义函数（Semantic Functions）和原生函数（Native Functions）
        - 提供 Planner 自动编排技能执行
        - 适合企业级 AI 应用集成

        核心概念：
        1. Kernel: 核心引擎，管理 AI 服务和技能
        2. Semantic Function: 使用提示模板定义的 AI 功能
        3. Native Function: 用代码实现的本地功能
        4. Skill: 多个函数的集合
        5. Planner: 自动规划执行顺序的组件

        示例代码展示了：
        - Kernel 初始化和配置
        - Semantic Function 创建和使用
        - Native Function 定义和执行
        - Skills 组合和编排
        - 链式调用和上下文变量
        """)
        return

    # 运行各示例
    await example_semantic_function()
    await example_multi_param_function()
    await example_native_function()
    await example_skills_combination()
    await example_planner_concept()
    await example_builtin_skills()
    await example_function_chain()
    await example_context_variables()
    await example_complex_orchestration()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


def main():
    """主程序入口"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()