import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def generate_idea(temperature):
    """生成一个关于源源不断赚钱的创意"""
    prompt = "请用3句话表述清楚你的方案。"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100
    )
    return response.choices[0].message.content

# 测试不同温度
temps = [0.2, 0.7, 1.2]
for t in temps:
    print(f"\n温度 = {t}:")
    for i in range(2):  # 每个温度试两次
        idea = generate_idea(t)
        print(f"  尝试{i+1}: {idea}")