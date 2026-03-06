import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def classify_sentiment(text):
    """使用思维链进行情感分类"""
    prompt = f"""请分析以下评论的情感倾向（积极、消极、中性），并一步步解释原因。

评论：{text}

请一步步思考：
1. 找出评论中的关键词和语气
2. 判断整体情感倾向
3. 给出结论

最终输出格式：情感：[积极/消极/中性] 理由：..."""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

# 测试不同评论
reviews = [
    "这个产品太棒了，效果超出预期！",
    "质量一般，但价格还算合理。",
    "垃圾产品，再也不会买了！"
]

for review in reviews:
    print(f"\n评论：{review}")
    print("分析结果：", classify_sentiment(review))
    print("-" * 50)