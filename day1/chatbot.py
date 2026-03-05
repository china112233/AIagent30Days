import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")

def chat_with_bot():
    print("🤖 聊天机器人启动（输入'quit'退出）")
    print("-" * 40)
    
    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手，回答简洁友好。"}
    ]
    
    while True:
        # 获取用户输入
        user_input = input("\n你: ")
        if user_input.lower() == 'quit':
            print("👋 再见！")
            break
        
        # 添加用户消息到历史
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 调用API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            # 获取助手回复
            assistant_reply = response.choices[0].message.content
            print(f"\n🤖 助手: {assistant_reply}")
            
            # 将助手回复添加到历史（保持对话上下文）
            messages.append({"role": "assistant", "content": assistant_reply})
            
        except Exception as e:
            print(f"❌ 出错了：{e}")

if __name__ == "__main__":
    chat_with_bot()