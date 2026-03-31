"""
Day 15: LCEL (LangChain Expression Language) 高级用法示例

本文件演示 LCEL 的高级特性：
- RunnableParallel 并行执行
- RunnablePassthrough 数据透传
- RunnableLambda 自定义处理逻辑
- RunnableBranch 条件分支
- 链的组合与嵌套
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    RunnableSequence,
)


# 加载环境变量
load_dotenv()


def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7,
    )


# ==========================================
# 示例 1: RunnableParallel 并行执行
# ==========================================


def example_parallel():
    """演示并行执行多个独立任务"""
    print("\n" + "=" * 50)
    print("示例 1: RunnableParallel 并行执行")
    print("=" * 50)

    llm = create_llm()

    # 定义多个并行任务
    summary_prompt = ChatPromptTemplate.from_template(
        "请用一句话总结以下内容：\n{content}"
    )
    translation_prompt = ChatPromptTemplate.from_template(
        "请将以下内容翻译成英文：\n{content}"
    )
    analysis_prompt = ChatPromptTemplate.from_template(
        "请分析以下内容的情感倾向（正面/负面/中性）：\n{content}"
    )

    # 创建并行链
    parallel_chain = RunnableParallel(
        summary=summary_prompt | llm | StrOutputParser(),
        translation=translation_prompt | llm | StrOutputParser(),
        analysis=analysis_prompt | llm | StrOutputParser(),
    )

    # 执行并行链
    content = "人工智能正在改变我们的生活，从智能家居到自动驾驶，AI 技术的应用越来越广泛。"

    print(f"输入内容: {content}")
    print("\n并行执行三个任务...")

    result = parallel_chain.invoke({"content": content})

    print("\n--- 结果 ---")
    print(f"摘要: {result['summary']}")
    print(f"翻译: {result['translation']}")
    print(f"情感分析: {result['analysis']}")


# ==========================================
# 示例 2: RunnablePassthrough 数据透传
# ==========================================


def example_passthrough():
    """演示数据透传，保留原始输入"""
    print("\n" + "=" * 50)
    print("示例 2: RunnablePassthrough 数据透传")
    print("=" * 50)

    llm = create_llm()

    # 简单透传示例
    passthrough_chain = RunnablePassthrough()

    print("透传测试:")
    input_data = {"question": "什么是人工智能？"}
    result = passthrough_chain.invoke(input_data)
    print(f"输入: {input_data}")
    print(f"输出: {result}")

    # 使用 assign 添加新字段
    print("\n--- 使用 assign 添加字段 ---")

    def get_timestamp(x):
        from datetime import datetime
        return datetime.now().isoformat()

    def process_question(x):
        return f"[处理后的问题] {x['question']}"

    # 构建透传 + assign 链
    chain_with_assign = RunnablePassthrough.assign(
        processed_question=RunnableLambda(process_question),
        timestamp=RunnableLambda(get_timestamp),
    )

    result = chain_with_assign.invoke(input_data)
    print(f"添加字段后的结果: {result}")

    # 实际应用：RAG 场景
    print("\n--- 实际应用：RAG 风格链 ---")

    # 模拟检索函数
    def mock_retriever(x):
        """模拟检索器"""
        question = x["question"]
        # 返回模拟的文档
        docs = [
            f"文档1: AI 是计算机科学的一个分支...",
            f"文档2: 机器学习是 AI 的核心技术...",
        ]
        return docs

    def format_docs(docs):
        """格式化检索结果"""
        return "\n\n".join(docs)

    rag_prompt = ChatPromptTemplate.from_template(
        """基于以下参考信息回答问题：

参考信息：
{context}

问题：{question}

请提供详细的回答："""
    )

    # 构建 RAG 风格链（透传原始问题，添加检索结果）
    rag_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(mock_retriever) | RunnableLambda(format_docs))
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("执行 RAG 风格链...")
    result = rag_chain.invoke({"question": "请介绍人工智能"})
    print(f"回答: {result[:200]}...")  # 截断显示


# ==========================================
# 示例 3: RunnableLambda 自定义逻辑
# ==========================================


def example_lambda():
    """演示使用 RunnableLambda 插入自定义处理逻辑"""
    print("\n" + "=" * 50)
    print("示例 3: RunnableLambda 自定义逻辑")
    print("=" * 50)

    llm = create_llm()

    # 自定义预处理函数
    def preprocess_input(x):
        """预处理输入"""
        print(f"  [预处理] 原始输入: {x}")

        # 提取关键词
        keywords = []
        important_words = ["人工智能", "机器学习", "深度学习", "AI", "ML"]
        for word in important_words:
            if word in x.get("question", ""):
                keywords.append(word)

        result = {
            "original_question": x["question"],
            "keywords": keywords,
            "processed_question": x["question"],
        }
        print(f"  [预处理] 处理后: {result}")
        return result

    # 自定义后处理函数
    def postprocess_output(x):
        """后处理输出"""
        print(f"  [后处理] 原始输出类型: {type(x)}")

        # 添加标记
        if isinstance(x, str):
            result = f"[AI 回答] {x}"
        else:
            result = x

        print(f"  [后处理] 处理后: {result[:100]}...")
        return result

    # 创建提示词模板（使用预处理后的字段）
    prompt = ChatPromptTemplate.from_template(
        """关键词提示：{keywords}

请回答以下问题：{processed_question}"""
    )

    # 构建包含自定义逻辑的链
    custom_chain = (
        RunnableLambda(preprocess_input)  # 预处理
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(postprocess_output)  # 后处理
    )

    print("执行包含自定义逻辑的链...")
    result = custom_chain.invoke({"question": "请介绍一下人工智能和机器学习的关系"})
    print(f"\n最终结果: {result[:200]}...")

    # 实际应用：验证和清洗
    print("\n--- 实际应用：输入验证 ---")

    def validate_input(x):
        """验证输入"""
        question = x.get("question", "")
        if len(question) < 5:
            raise ValueError("问题太短，至少需要 5 个字符")
        if len(question) > 1000:
            question = question[:1000]  # 截断
        return {"question": question, "length": len(question)}

    validation_chain = RunnableLambda(validate_input)

    try:
        result = validation_chain.invoke({"question": "你好"})
        print(f"验证成功: {result}")
    except ValueError as e:
        print(f"验证失败: {e}")


# ==========================================
# 示例 4: RunnableBranch 条件分支
# ==========================================


def example_branch():
    """演示根据条件选择不同执行路径"""
    print("\n" + "=" * 50)
    print("示例 4: RunnableBranch 条件分支")
    print("=" * 50)

    llm = create_llm()

    # 定义不同场景的处理链
    translation_chain = (
        ChatPromptTemplate.from_template("请将以下内容翻译成英文：{text}")
        | llm
        | StrOutputParser()
    )

    summary_chain = (
        ChatPromptTemplate.from_template("请用一句话总结以下内容：{text}")
        | llm
        | StrOutputParser()
    )

    explanation_chain = (
        ChatPromptTemplate.from_template("请详细解释以下概念：{text}")
        | llm
        | StrOutputParser()
    )

    default_chain = (
        ChatPromptTemplate.from_template("请回答以下问题：{text}")
        | llm
        | StrOutputParser()
    )

    # 定义条件判断函数
    def is_translation_request(x):
        return any(word in x["text"].lower() for word in ["翻译", "translate"])

    def is_summary_request(x):
        return any(word in x["text"].lower() for word in ["总结", "summarize", "摘要"])

    def is_explanation_request(x):
        return any(word in x["text"].lower() for word in ["解释", "explain", "是什么"])

    # 创建分支链
    branch_chain = RunnableBranch(
        (is_translation_request, translation_chain),
        (is_summary_request, summary_chain),
        (is_explanation_request, explanation_chain),
        default_chain,  # 默认分支
    )

    # 测试不同场景
    test_cases = [
        "请翻译这段话：人工智能是未来",
        "请总结一下机器学习的发展历程",
        "请解释什么是深度学习",
        "请介绍一下自然语言处理",
    ]

    for test in test_cases:
        print(f"\n输入: {test}")
        result = branch_chain.invoke({"text": test})
        print(f"输出: {result[:100]}...")


# ==========================================
# 示例 5: 链的组合与嵌套
# ==========================================


def example_composition():
    """演示复杂链的组合与嵌套"""
    print("\n" + "=" * 50)
    print("示例 5: 链的组合与嵌套")
    print("=" * 50)

    llm = create_llm()

    # 子链 1：问题理解
    understanding_chain = (
        ChatPromptTemplate.from_template(
            """分析以下问题的意图和关键词：
问题：{question}

请输出：
1. 问题意图（查询/解释/操作）
2. 关键词列表"""
        )
        | llm
        | StrOutputParser()
    )

    # 子链 2：回答生成
    answer_chain = (
        ChatPromptTemplate.from_template(
            """基于以下分析结果回答问题：
分析结果：{analysis}
原始问题：{question}

请提供详细回答："""
        )
        | llm
        | StrOutputParser()
    )

    # 组合成完整链
    # 步骤 1: 分析问题
    # 步骤 2: 透传原始问题，同时添加分析结果
    # 步骤 3: 生成回答
    full_chain = (
        RunnablePassthrough.assign(analysis=understanding_chain)
        | answer_chain
    )

    print("执行组合链...")
    result = full_chain.invoke({"question": "什么是 RAG 技术？"})
    print(f"回答: {result[:300]}...")

    # 更复杂的组合：并行分析 + 汇总
    print("\n--- 并行分析 + 汇总 ---")

    # 多角度分析
    intent_analysis = (
        ChatPromptTemplate.from_template("分析问题意图：{question}")
        | llm
        | StrOutputParser()
    )

    keyword_analysis = (
        ChatPromptTemplate.from_template("提取关键词：{question}")
        | llm
        | StrOutputParser()
    )

    complexity_analysis = (
        ChatPromptTemplate.from_template("评估问题复杂度（简单/中等/复杂）：{question}")
        | llm
        | StrOutputParser()
    )

    # 并行分析
    parallel_analysis = RunnableParallel(
        intent=intent_analysis,
        keywords=keyword_analysis,
        complexity=complexity_analysis,
    )

    # 汇总回答
    final_prompt = ChatPromptTemplate.from_template(
        """综合分析结果：
- 意图：{intent}
- 关键词：{keywords}
- 复杂度：{complexity}

原始问题：{question}

请提供针对性回答："""
    )

    final_chain = (
        RunnablePassthrough.assign(
            intent=parallel_analysis["intent"],
            keywords=parallel_analysis["keywords"],
            complexity=parallel_analysis["complexity"],
        )
        | final_prompt
        | llm
        | StrOutputParser()
    )

    print("执行复杂组合链...")
    result = final_chain.invoke({"question": "请解释 LangChain 的 LCEL 表达式语言"})
    print(f"回答: {result[:200]}...")


# ==========================================
# 示例 6: 流式处理
# ==========================================


def example_stream():
    """演示 LCEL 链的流式处理"""
    print("\n" + "=" * 50)
    print("示例 6: 流式处理")
    print("=" * 50)

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template("请详细介绍：{topic}")

    chain = prompt | llm | StrOutputParser()

    print("流式输出 LangChain LCEL:")
    print("-" * 30)

    for chunk in chain.stream({"topic": "LangChain Expression Language"}):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 30)


# ==========================================
# 示例 7: 批量处理
# ==========================================


def example_batch():
    """演示批量处理多个输入"""
    print("\n" + "=" * 50)
    print("示例 7: 批量处理")
    print("=" * 50)

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template("请用一句话介绍：{topic}")

    chain = prompt | llm | StrOutputParser()

    # 批量输入
    topics = [
        {"topic": "Python"},
        {"topic": "LangChain"},
        {"topic": "RAG"},
        {"topic": "Vector Database"},
    ]

    print("批量处理多个主题...")
    results = chain.batch(topics)

    for i, (topic, result) in enumerate(zip(topics, results)):
        print(f"\n{topic['topic']}: {result}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 15: LCEL 高级用法示例")
    print("=" * 60)

    # 运行各示例
    example_parallel()
    example_passthrough()
    example_lambda()
    example_branch()
    example_composition()
    example_stream()
    example_batch()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()