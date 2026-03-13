"""
对话管理器（Conversation Manager）
实现智能的多轮对话历史管理
"""

import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


class ConversationManager:
    """
    对话管理器：管理多轮对话的历史和上下文
    """

    def __init__(self, system_prompt="你是一个乐于助人的助手。", max_messages=20, max_tokens=4000):
        """
        初始化对话管理器

        Args:
            system_prompt: 系统提示词
            max_messages: 最大消息数量限制
            max_tokens: 最大 Token 估算限制
        """
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.total_tokens_used = 0

    def add_message(self, role, content):
        """添加一条消息到对话历史"""
        self.messages.append({"role": role, "content": content})
        self._trim_history()

    def _estimate_tokens(self, text):
        """
        估算文本的 Token 数量
        粗略估算：中文约 0.5 token/字，英文约 0.25 token/word
        """
        # 简单估算：按字符计算
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.5 + other_chars * 0.25)

    def _estimate_messages_tokens(self):
        """估算当前所有消息的 Token 数量"""
        total = 0
        for msg in self.messages:
            total += self._estimate_tokens(msg["content"])
            total += 4  # role 和 content 字段的开销
        return total

    def _trim_history(self):
        """裁剪对话历史以保持在限制内"""
        # 策略1：按消息数量裁剪
        while len(self.messages) > self.max_messages:
            # 保留系统提示，删除最早的对话
            if len(self.messages) > 2:
                self.messages.pop(1)  # 删除索引1（第一条用户消息）

        # 策略2：按 Token 估算裁剪
        while self._estimate_messages_tokens() > self.max_tokens:
            if len(self.messages) > 2:
                self.messages.pop(1)
            else:
                break

    def chat(self, user_input, stream=False, temperature=0.7):
        """
        发送用户消息并获取响应

        Args:
            user_input: 用户输入
            stream: 是否使用流式输出
            temperature: 温度参数

        Returns:
            助手的响应内容
        """
        self.add_message("user", user_input)

        if stream:
            return self._stream_chat(temperature)
        else:
            return self._normal_chat(temperature)

    def _normal_chat(self, temperature):
        """非流式聊天"""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            temperature=temperature
        )

        # 记录 Token 使用
        self.total_tokens_used += response.usage.total_tokens

        # 获取响应并添加到历史
        assistant_content = response.choices[0].message.content
        self.add_message("assistant", assistant_content)

        return assistant_content, response.usage

    def _stream_chat(self, temperature):
        """流式聊天"""
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            temperature=temperature,
            stream=True
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content

        print()  # 换行

        # 添加到历史
        self.add_message("assistant", full_content)

        return full_content, None

    def get_history(self):
        """获取对话历史"""
        return self.messages.copy()

    def get_history_summary(self):
        """获取对话历史摘要"""
        summary = []
        for i, msg in enumerate(self.messages):
            role = msg["role"]
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            summary.append(f"{i}. [{role}]: {content}")
        return "\n".join(summary)

    def clear_history(self, keep_system=True):
        """清空对话历史"""
        if keep_system and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def set_system_prompt(self, prompt):
        """设置/更新系统提示词"""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def save_to_file(self, filename):
        """保存对话历史到文件"""
        data = {
            "messages": self.messages,
            "total_tokens": self.total_tokens_used,
            "saved_at": datetime.now().isoformat()
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到 {filename}")

    def load_from_file(self, filename):
        """从文件加载对话历史"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.messages = data["messages"]
            self.total_tokens_used = data.get("total_tokens", 0)
            print(f"已从 {filename} 加载对话")
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
        except json.JSONDecodeError:
            print(f"文件 {filename} 格式错误")


class SmartConversationManager(ConversationManager):
    """
    智能对话管理器：带摘要压缩功能
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_threshold = 10  # 触发摘要的消息数量阈值

    def summarize_old_messages(self):
        """将旧消息压缩为摘要"""
        if len(self.messages) <= self.summary_threshold:
            return

        # 保留系统提示和最近的几条消息
        system_msg = self.messages[0]
        recent_msgs = self.messages[-(self.summary_threshold - 1):]

        # 将中间的消息压缩为摘要
        old_msgs = self.messages[1:-self.summary_threshold + 1]
        if old_msgs:
            summary_text = self._create_summary(old_msgs)
            summary_msg = {
                "role": "system",
                "content": f"[历史对话摘要]\n{summary_text}"
            }
            self.messages = [system_msg, summary_msg] + recent_msgs
            print("已压缩对话历史")

    def _create_summary(self, messages):
        """使用 AI 创建对话摘要"""
        history_text = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])

        prompt = f"""请用简短的几句话总结以下对话的主要内容：

{history_text}

摘要："""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )

        return response.choices[0].message.content


def demo_basic_conversation():
    """演示基础对话管理"""
    print("=" * 50)
    print("演示1：基础对话管理")
    print("=" * 50)

    conv = ConversationManager(
        system_prompt="你是一个专业的Python编程助手。",
        max_messages=5  # 最多保留5条消息
    )

    # 进行几轮对话
    questions = [
        "什么是列表推导式？",
        "给我一个例子。",
        "可以嵌套使用吗？",
        "有没有性能优势？"
    ]

    for q in questions:
        print(f"\n用户: {q}")
        response, usage = conv.chat(q)

        print(f"助手: {response}")
        if usage:
            print(f"[Token: {usage.total_tokens}]")

    print("\n--- 对话历史 ---")
    print(conv.get_history_summary())


def demo_token_management():
    """演示 Token 管理"""
    print("\n" + "=" * 50)
    print("演示2：Token 管理")
    print("=" * 50)

    conv = ConversationManager(
        system_prompt="你是一个简洁的助手，回答尽量简短。",
        max_tokens=500  # 限制 Token 数量
    )

    # 模拟长对话
    for i in range(5):
        print(f"\n用户: 第{i+1}个问题 - 请简短介绍Python的特点")
        response, usage = conv.chat("请简短介绍Python的一个特点")

        print(f"助手: {response}")
        print(f"[当前Token: {usage.total_tokens}, 累计: {conv.total_tokens_used}]")

    print(f"\n总 Token 消耗: {conv.total_tokens_used}")


def demo_persistence():
    """演示对话持久化"""
    print("\n" + "=" * 50)
    print("演示3：对话持久化")
    print("=" * 50)

    # 创建对话
    conv = ConversationManager(system_prompt="你是一个友好的助手。")

    # 进行对话
    response, _ = conv.chat("你好！我是小明。")
    print(f"助手: {response}")

    # 保存对话
    filename = "conversation_backup.json"
    conv.save_to_file(filename)

    # 创建新管理器并加载
    print("\n--- 加载保存的对话 ---")
    new_conv = ConversationManager()
    new_conv.load_from_file(filename)

    print("对话历史:")
    print(new_conv.get_history_summary())


def demo_streaming_conversation():
    """演示流式对话"""
    print("\n" + "=" * 50)
    print("演示4：流式对话")
    print("=" * 50)

    conv = ConversationManager(
        system_prompt="你是一个讲故事的高手，语言生动有趣。"
    )

    print("用户: 讲一个小猫钓鱼的故事")
    print("助手: ", end="", flush=True)
    conv.chat("讲一个小猫钓鱼的故事，简短一点", stream=True)


def demo_smart_conversation():
    """演示智能对话管理（带摘要）"""
    print("\n" + "=" * 50)
    print("演示5：智能对话管理（摘要压缩）")
    print("=" * 50)

    conv = SmartConversationManager(
        system_prompt="你是一个助手。",
        summary_threshold=4
    )

    # 进行多轮对话触发摘要
    for i in range(6):
        print(f"\n用户: 问题{i+1}")
        response, _ = conv.chat(f"这是第{i+1}个问题，请简短回答")

    print("\n--- 压缩后的对话历史 ---")
    print(conv.get_history_summary())


if __name__ == "__main__":
    demo_basic_conversation()
    demo_token_management()
    demo_persistence()
    demo_streaming_conversation()
    demo_smart_conversation()