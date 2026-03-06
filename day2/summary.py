import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def summarize(text, length="短"):
    """生成摘要，支持长度控制"""
    # 提示模板
    prompt_template = """请对以下文本进行{length}摘要，要求保留关键信息，语言简洁。

文本：{text}

{length}摘要："""
    
    prompt = prompt_template.format(length=length, text=text)
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400
    )
    return response.choices[0].message.content

# 测试文本
long_text = """
人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。近年来，深度学习技术的突破使得AI在图像识别、自然语言处理等领域取得了显著进展。例如，GPT系列模型能够生成连贯的文本，AlphaGo击败了世界围棋冠军。然而，AI的发展也引发了伦理和社会问题的讨论，如就业影响、隐私保护和算法偏见。未来，AI有望在医疗、教育、交通等领域发挥更大作用，但需要谨慎管理其风险。
"""

# 测试不同长度
print("短摘要：", summarize(long_text, "短"))
print("\n详细摘要：", summarize(long_text, "详细"))