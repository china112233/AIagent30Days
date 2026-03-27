"""
Day 14: LangChain 基础 - Memory 记忆系统示例

本示例演示 LangChain 的 Memory 组件
包括不同类型的记忆管理方式
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory
)
from langchain_community.chat_message_histories import ChatMessageHistory

# 加载环境变量
load_dotenv()


def conversation_buffer_memory_example():
    """
    示例1：ConversationBufferMemory - 完整记忆
    保存所有对话历史
    """
    print("=" * 60)
    print("示例1：ConversationBufferMemory（完整记忆）")
    print("=" * 60)

    # 创建 LLM
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 创建完整记忆
    memory = ConversationBufferMemory(
        return_messages=True,  # 返回消息对象而非字符串
        memory_key="chat_history"
    )

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手，请记住之前的对话内容。"),
        ("placeholder", "{chat_history}"),  # 历史消息占位符
        ("user", "{input}")
    ])

    # 创建链
    chain = prompt | llm | StrOutputParser()

    # 模拟多轮对话
    conversations = [
        "你好，我叫小明",
        "我喜欢打篮球",
        "你还记得我的名字吗？",
        "我有什么爱好？"
    ]

    for user_input in conversations:
        # 获取历史记忆
        history = memory.load_memory_variables({})["chat_history"]

        # 执行对话
        response = chain.invoke({
            "chat_history": history,
            "input": user_input
        })

        # 保存到记忆
        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        print(f"用户：{user_input}")
        print(f"助手：{response}")
        print("-" * 40)

    print()


def conversation_buffer_window_memory_example():
    """
    示例2：ConversationBufferWindowMemory - 滑动窗口记忆
    只保留最近 N 轮对话
    """
    print("=" * 60)
    print("示例2：ConversationBufferWindowMemory（滑动窗口）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 创建滑动窗口记忆，只保留最近 2 轮
    memory = ConversationBufferWindowMemory(
        k=2,  # 保留最近 2 轮对话
        return_messages=True,
        memory_key="chat_history"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    # 模拟多轮对话
    conversations = [
        "第一轮：我喜欢吃苹果",
        "第二轮：我喜欢吃香蕉",
        "第三轮：我喜欢吃橙子",
        "第四轮：我刚才说我喜欢吃什么水果？"
    ]

    for user_input in conversations:
        history = memory.load_memory_variables({})["chat_history"]

        response = chain.invoke({
            "chat_history": history,
            "input": user_input
        })

        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        print(f"用户：{user_input}")
        print(f"助手：{response}")

        # 显示当前记忆内容
        current_memory = memory.load_memory_variables({})["chat_history"]
        print(f"当前记忆轮数：{len(current_memory) // 2}")
        print("-" * 40)

    print()


def token_buffer_memory_example():
    """
    示例3：ConversationTokenBufferMemory - Token 限制记忆
    根据Token数量限制记忆
    """
    print("=" * 60)
    print("示例3：ConversationTokenBufferMemory（Token限制）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 创建 Token 限制记忆
    memory = ConversationTokenBufferMemory(
        llm=llm,
        max_token_limit=100,  # 限制 100 个 Token
        return_messages=True,
        memory_key="chat_history"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手，回答要简洁。"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    conversations = [
        "你好",
        "介绍一下 Python",
        "什么是机器学习？",
        "深度学习和机器学习有什么区别？"
    ]

    for user_input in conversations:
        history = memory.load_memory_variables({})["chat_history"]

        response = chain.invoke({
            "chat_history": history,
            "input": user_input
        })

        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        print(f"用户：{user_input}")
        print(f"助手：{response[:50]}...")
        print("-" * 40)

    print()


def manual_memory_management():
    """
    示例4：手动管理对话历史
    使用 ChatMessageHistory 直接操作
    """
    print("=" * 60)
    print("示例4：手动管理对话历史")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 创建消息历史
    history = ChatMessageHistory()

    # 手动添加系统消息
    history.add_message(
        HumanMessage(content="你是一个专业的 Python 助手")
    )
    history.add_message(
        AIMessage(content="好的，我明白了。我会以专业 Python 助手的身份回答问题。")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的 Python 助手。"),
        ("placeholder", "{history}"),
        ("user", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    def chat(user_input: str) -> str:
        """带记忆的聊天函数"""
        # 获取历史消息
        messages = history.messages

        # 调用模型
        response = chain.invoke({
            "history": messages,
            "input": user_input
        })

        # 保存对话
        history.add_user_message(user_input)
        history.add_ai_message(response)

        return response

    # 测试
    print("用户：什么是装饰器？")
    print(f"助手：{chat('什么是装饰器？')[:100]}...")
    print()

    print("用户：能举个例子吗？")
    print(f"助手：{chat('能举个例子吗？')[:100]}...")
    print()

    print("对话历史：")
    for msg in history.messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        print(f"  [{role}]: {msg.content[:50]}...")

    print()


def memory_with_summary():
    """
    示例5：带摘要的记忆（简化实现）
    定期对历史对话进行总结
    """
    print("=" * 60)
    print("示例5：带摘要的记忆")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7
    )

    # 对话历史和摘要
    history = []
    summary = ""

    def summarize_history(history: list) -> str:
        """将对话历史压缩为摘要"""
        if not history:
            return ""

        history_text = "\n".join([
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
            for h in history
        ])

        summarize_prompt = ChatPromptTemplate.from_template(
            "请用一句话总结以下对话的关键信息：\n\n{history}"
        )

        chain = summarize_prompt | llm | StrOutputParser()
        return chain.invoke({"history": history_text})

    def chat_with_summary(user_input: str) -> str:
        """带摘要的聊天"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有帮助的助手。\n\n历史摘要：{summary}"),
            ("placeholder", "{history}"),
            ("user", "{input}")
        ])

        chain = prompt | llm | StrOutputParser()

        # 构建历史消息
        history_messages = [
            HumanMessage(content=h["content"]) if h["role"] == "user"
            else AIMessage(content=h["content"])
            for h in history[-4:]  # 只保留最近 4 条
        ]

        response = chain.invoke({
            "summary": summary,
            "history": history_messages,
            "input": user_input
        })

        # 保存对话
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # 每 6 轮更新摘要
        if len(history) >= 6 and len(history) % 6 == 0:
            summary = summarize_history(history)
            print(f"[摘要更新]: {summary}")

        return response

    # 测试
    conversations = [
        "我叫小红",
        "我喜欢画画",
        "我住在上海",
        "你记得我的名字吗？",
        "我有什么爱好？",
        "我住在哪里？"
    ]

    for user_input in conversations:
        response = chat_with_summary(user_input)
        print(f"用户：{user_input}")
        print(f"助手：{response[:80]}...")
        print("-" * 40)

    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 14: LangChain 基础 - Memory 记忆系统示例")
    print("=" * 60 + "\n")

    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    try:
        conversation_buffer_memory_example()
        conversation_buffer_window_memory_example()
        token_buffer_memory_example()
        manual_memory_management()
        memory_with_summary()

        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()