"""
Day 14: LangChain 基础 - Chain 使用示例

本示例演示 LangChain 的核心概念：Chain（链）
包括 Prompt Template、LLM、Output Parser 的使用
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()


def basic_chain_example():
    """
    示例1：最基础的 Chain 使用
    Prompt -> LLM -> Parser
    """
    print("=" * 60)
    print("示例1：基础 Chain")
    print("=" * 60)

    # 1. 创建 LLM
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 2. 创建 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}，请用简洁的语言回答问题。"),
        ("user", "{question}")
    ])

    # 3. 创建输出解析器
    parser = StrOutputParser()

    # 4. 使用 LCEL 语法组合成链
    chain = prompt | llm | parser

    # 5. 执行链
    response = chain.invoke({
        "role": "Python 开发工程师",
        "question": "什么是装饰器？请举个例子。"
    })

    print(f"回答：{response}")
    print()


def template_variations():
    """
    示例2：不同的 Prompt Template 类型
    """
    print("=" * 60)
    print("示例2：Prompt Template 变体")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 方式1：from_messages（推荐用于聊天模型）
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        ("user", "{input}")
    ])

    # 方式2：from_template（简单模板）
    simple_prompt = ChatPromptTemplate.from_template(
        "请用一句话解释：{concept}"
    )

    # 方式3：PromptTemplate（传统方式）
    traditional_prompt = PromptTemplate(
        input_variables=["topic", "style"],
        template="请以{style}的风格介绍{topic}。"
    )

    # 测试不同模板
    chain1 = chat_prompt | llm | StrOutputParser()
    print("方式1:", chain1.invoke({"input": "什么是 AI？"})[:100], "...")

    chain2 = simple_prompt | llm | StrOutputParser()
    print("方式2:", chain2.invoke({"concept": "机器学习"})[:100], "...")

    chain3 = traditional_prompt | llm | StrOutputParser()
    print("方式3:", chain3.invoke({"topic": "Python", "style": "幽默"})[:100], "...")

    print()


def structured_output():
    """
    示例3：结构化输出
    使用 Pydantic 模型定义输出格式
    """
    print("=" * 60)
    print("示例3：结构化输出")
    print("=" * 60)

    # 定义输出模型
    class MovieReview(BaseModel):
        """电影评价模型"""
        movie_name: str = Field(description="电影名称")
        rating: float = Field(description="评分，1-10分")
        summary: str = Field(description="简短评价")
        pros: list[str] = Field(description="优点列表")
        cons: list[str] = Field(description="缺点列表")

    # 创建 JSON 解析器
    parser = JsonOutputParser(pydantic_object=MovieReview)

    # 创建 LLM
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 创建 Prompt（包含格式说明）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的电影评论家。"),
        ("user", "{format_instructions}\n\n请评价电影：{movie}")
    ])

    # 部分填充格式说明
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # 组合链
    chain = prompt | llm | parser

    # 执行
    try:
        result = chain.invoke({"movie": "肖申克的救赎"})
        print(f"电影：{result['movie_name']}")
        print(f"评分：{result['rating']}/10")
        print(f"评价：{result['summary']}")
        print(f"优点：{', '.join(result['pros'])}")
        print(f"缺点：{', '.join(result['cons'])}")
    except Exception as e:
        print(f"解析失败：{e}")

    print()


def parallel_chain():
    """
    示例4：并行执行多条链
    """
    print("=" * 60)
    print("示例4：并行执行")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 定义多个处理链
    joke_chain = (
        ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
        | llm
        | StrOutputParser()
    )

    fact_chain = (
        ChatPromptTemplate.from_template("告诉我一个关于{topic}的有趣事实")
        | llm
        | StrOutputParser()
    )

    # 并行执行
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        fact=fact_chain
    )

    # 执行
    result = parallel_chain.invoke({"topic": "Python编程"})

    print("笑话：")
    print(result["joke"])
    print()
    print("有趣事实：")
    print(result["fact"])

    print()


def sequential_chain():
    """
    示例5：顺序执行多步骤链
    """
    print("=" * 60)
    print("示例5：顺序执行多步骤")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 第一步：生成主题
    topic_prompt = ChatPromptTemplate.from_template(
        "给我一个关于{domain}的主题建议，只要一个短句。"
    )

    # 第二步：基于主题写文章
    article_prompt = ChatPromptTemplate.from_template(
        "请写一篇关于'{topic}'的短文，100字左右。"
    )

    # 组合成顺序链
    chain = (
        {"topic": topic_prompt | llm | StrOutputParser()}
        | article_prompt
        | llm
        | StrOutputParser()
    )

    # 执行
    result = chain.invoke({"domain": "人工智能"})
    print("生成的文章：")
    print(result)

    print()


def stream_example():
    """
    示例6：流式输出
    """
    print("=" * 60)
    print("示例6：流式输出")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template(
        "请写一首关于{topic}的诗，四句。"
    )

    chain = prompt | llm | StrOutputParser()

    print("流式输出：")
    for chunk in chain.stream({"topic": "春天"}):
        print(chunk, end="", flush=True)

    print("\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 14: LangChain 基础 - Chain 使用示例")
    print("=" * 60 + "\n")

    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        print("创建 .env 文件并添加：DEEPSEEK_API_KEY=your_key")
        return

    try:
        basic_chain_example()
        template_variations()
        structured_output()
        parallel_chain()
        sequential_chain()
        stream_example()

        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()