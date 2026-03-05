import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), 
                base_url="https://api.deepseek.com")

def test_parameters(temp, top_p=1.0):
    """测试不同参数的效果"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "给我一个关于'AI未来'的创意想法，50字以内。"}
        ],
        temperature=temp,
        top_p=top_p,
        max_tokens=100
    )
    return response.choices[0].message.content

# 实验1：不同温度的对比
print("🔥 温度实验（temperature）")
print("-" * 50)
print("温度=0.0（保守）：", test_parameters(0.0))
print("\n温度=0.7（平衡）：", test_parameters(0.7))
print("\n温度=1.5（随机）：", test_parameters(1.5))

# 实验2：固定温度，改变Top_p（可选）
print("\n\n🎯 Top_p实验（temperature=0.7）")
print("-" * 50)
print("Top_p=0.1（严格）：", test_parameters(0.7, 0.1))
print("\nTop_p=0.9（宽松）：", test_parameters(0.7, 0.9))