import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件中的密钥
load_dotenv()

# 初始化客户端（DeepSeek和OpenAI的API格式兼容）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"  # DeepSeek的地址
)

# 第一次调用
try:
    response = client.chat.completions.create(
        model="deepseek-chat",  # 指定模型
        messages=[
            {"role": "system", "content": "你是一个友好的助手。"},
            {"role": "user", "content": "你好！请用一句话介绍你自己。"}
        ],
        temperature=0.7,  # 中等随机性
        max_tokens=100    # 限制回答长度
    )
    
    # 打印回复内容
    print("模型回复：", response.choices[0].message.content)

    # 查看完整响应结构（了解API返回了什么）
    print("\n完整响应结构：")
    print(f"响应ID: {response.id}")
    print(f"模型: {response.model}")
    print(f"Token使用: {response.usage}")

except Exception as e:
    print(f"出错了：{e}")