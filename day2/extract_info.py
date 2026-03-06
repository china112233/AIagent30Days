import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def extract_info(text):
    """从文本中提取人名、地点、日期"""
    prompt = f"""请从以下文本中提取出所有人名、地名和日期，以JSON格式返回。
示例：
文本：张三昨天在北京参加了会议。
输出：{{"人名": ["张三"], "地名": ["北京"], "日期": ["昨天"]}}

文本：李华计划2024年5月1日去上海旅行。
输出：{{"人名": ["李华"], "地名": ["上海"], "日期": ["2024年5月1日"]}}

现在请处理：
文本：{text}
输出："""
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # 低温度让输出更稳定
        max_tokens=200
    )
    return response.choices[0].message.content

# 测试
text = "慕容龙车通知我们，（3.15）在尼泊尔开会。"
result = extract_info(text)
print("提取结果：", result)