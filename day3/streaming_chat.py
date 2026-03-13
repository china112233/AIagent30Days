"""
流式输出（Streaming）基础示例
学习如何使用流式输出获得实时响应
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def non_streaming_chat(prompt):
    """非流式输出：等待完整响应后一次性返回"""
    print("\n[非流式输出] 等待响应...")
    start_time = time.time()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False  # 非流式（默认）
    )

    elapsed = time.time() - start_time
    content = response.choices[0].message.content

    print(f"[非流式输出] 耗时: {elapsed:.2f}秒")
    print(f"[非流式输出] 内容:\n{content}")

    return content


def streaming_chat(prompt):
    """流式输出：实时逐字显示响应"""
    print("\n[流式输出] 开始接收...")

    start_time = time.time()
    first_chunk_time = None
    full_content = ""

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # 开启流式
    )

    print("[流式输出] 内容: ", end="", flush=True)

    for chunk in stream:
        # 记录首字延迟
        if first_chunk_time is None:
            first_chunk_time = time.time()
            ttft = first_chunk_time - start_time  # Time to First Token
            print(f"\n[首字延迟: {ttft:.2f}秒] ", end="", flush=True)

        # 获取内容片段
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content

    elapsed = time.time() - start_time
    print(f"\n[流式输出] 总耗时: {elapsed:.2f}秒")

    return full_content


def streaming_with_indicator(prompt):
    """带加载指示器的流式输出"""
    print("\n[智能助手] 思考中", end="", flush=True)

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # 用于检测是否开始输出
    started = False

    for chunk in stream:
        if chunk.choices[0].delta.content:
            if not started:
                # 清除"思考中"提示，开始输出
                print("\r" + " " * 20 + "\r", end="")  # 清除上一行
                print("[智能助手] ", end="", flush=True)
                started = True

            print(chunk.choices[0].delta.content, end="", flush=True)

    print()  # 换行


def streaming_with_stats(prompt):
    """带 Token 统计的流式输出"""
    print("\n[流式输出 + 统计]")

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}  # 包含使用统计
    )

    full_content = ""
    token_count = 0

    for chunk in stream:
        # 内容片段
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content

        # 使用统计（在最后一个 chunk 中）
        if hasattr(chunk, 'usage') and chunk.usage:
            print(f"\n\n--- Token 统计 ---")
            print(f"输入 Token: {chunk.usage.prompt_tokens}")
            print(f"输出 Token: {chunk.usage.completion_tokens}")
            print(f"总 Token: {chunk.usage.total_tokens}")

    return full_content


def compare_stream_modes():
    """对比流式和非流式输出"""
    prompt = "用简短的话介绍一下人工智能的发展历程（100字以内）"

    print("=" * 50)
    print("对比流式和非流式输出")
    print("=" * 50)

    # 非流式
    print("\n>>> 非流式输出测试 <<<")
    non_streaming_chat(prompt)

    # 流式
    print("\n>>> 流式输出测试 <<<")
    streaming_chat(prompt)


def interactive_streaming_chat():
    """交互式流式聊天"""
    print("=" * 50)
    print("交互式流式聊天（输入 'quit' 退出）")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "你是一个友好的助手，回答简洁有趣。"}
    ]

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == 'quit':
            print("再见！")
            break

        if not user_input:
            continue

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        # 流式响应
        print("助手: ", end="", flush=True)

        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            temperature=0.7
        )

        assistant_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                assistant_content += content

        print()  # 换行

        # 添加助手响应到历史
        messages.append({"role": "assistant", "content": assistant_content})


if __name__ == "__main__":
    # 示例1：对比流式和非流式
    print("示例1：对比流式和非流式输出")
    compare_stream_modes()

    print("\n" + "=" * 50)

    # 示例2：带加载指示器
    print("\n示例2：带加载指示器的流式输出")
    streaming_with_indicator("什么是机器学习？用一句话解释。")

    print("\n" + "=" * 50)

    # 示例3：带统计信息
    print("\n示例3：带 Token 统计的流式输出")
    streaming_with_stats("列出3种常见的编程语言。")

    print("\n" + "=" * 50)

    # 示例4：交互式聊天（可选运行）
    print("\n是否启动交互式聊天？(y/n): ", end="")
    if input().lower() == 'y':
        interactive_streaming_chat()