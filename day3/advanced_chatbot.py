"""
高级聊天机器人
整合流式输出、对话管理、Token 统计等功能
"""

import os
import sys
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


class AdvancedChatBot:
    """
    高级聊天机器人
    功能：
    - 流式输出
    - 对话历史管理
    - Token 统计
    - 多种交互模式
    - 会话持久化
    """

    def __init__(self, system_prompt="你是一个智能、友好且乐于助人的AI助手。"):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0
        self.session_start = datetime.now()
        self.mode = "normal"  # normal, stream, debug

    def chat(self, user_input, stream=True, temperature=0.7):
        """
        发送消息并获取响应

        Args:
            user_input: 用户输入
            stream: 是否流式输出
            temperature: 温度参数

        Returns:
            助手响应内容
        """
        self.messages.append({"role": "user", "content": user_input})

        try:
            if stream:
                response_content = self._stream_chat(temperature)
                return response_content
            else:
                return self._normal_chat(temperature)

        except Exception as e:
            print(f"\n[错误] {e}")
            # 移除失败的用户消息
            self.messages.pop()
            return None

    def _normal_chat(self, temperature):
        """非流式聊天"""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            temperature=temperature
        )

        # 统计
        self.total_tokens += response.usage.total_tokens

        # 获取响应
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})

        # 调试模式显示统计
        if self.mode == "debug":
            self._show_usage(response.usage)

        return content

    def _stream_chat(self, temperature):
        """流式聊天"""
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            temperature=temperature,
            stream=True
        )

        content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                content += text

        print()  # 换行

        # 添加到历史
        self.messages.append({"role": "assistant", "content": content})

        return content

    def _show_usage(self, usage):
        """显示使用统计"""
        print(f"\n[Token统计] 输入: {usage.prompt_tokens}, 输出: {usage.completion_tokens}, 总计: {usage.total_tokens}")

    def set_mode(self, mode):
        """设置交互模式"""
        if mode in ["normal", "stream", "debug"]:
            self.mode = mode
            print(f"已切换到 {mode} 模式")
        else:
            print("无效模式，可选: normal, stream, debug")

    def clear_history(self):
        """清空对话历史"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        print("对话历史已清空")

    def show_history(self, limit=10):
        """显示对话历史"""
        print("\n" + "=" * 50)
        print("对话历史")
        print("=" * 50)

        # 显示最近的对话
        recent = self.messages[-limit:] if len(self.messages) > limit else self.messages

        for msg in recent:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 100:
                content = content[:100] + "..."

            if role == "system":
                print(f"[系统]: {content}")
            elif role == "user":
                print(f"[用户]: {content}")
            elif role == "assistant":
                print(f"[助手]: {content}")

        print("=" * 50)

    def show_stats(self):
        """显示会话统计"""
        duration = datetime.now() - self.session_start

        print("\n" + "=" * 50)
        print("会话统计")
        print("=" * 50)
        print(f"会话时长: {duration}")
        print(f"消息数量: {len(self.messages)}")
        print(f"累计 Token: {self.total_tokens}")
        print(f"预估成本: {self.total_tokens * 0.001:.4f} 元")  # 粗略估算
        print("=" * 50)

    def save_session(self, filename="chat_session.json"):
        """保存会话"""
        data = {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "total_tokens": self.total_tokens,
            "session_start": self.session_start.isoformat(),
            "saved_at": datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"会话已保存到 {filename}")

    def load_session(self, filename="chat_session.json"):
        """加载会话"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.system_prompt = data["system_prompt"]
            self.messages = data["messages"]
            self.total_tokens = data.get("total_tokens", 0)
            self.session_start = datetime.fromisoformat(data["session_start"])

            print(f"已从 {filename} 加载会话")
            print(f"会话开始于: {self.session_start}")
            print(f"消息数量: {len(self.messages)}")

        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
        except Exception as e:
            print(f"加载失败: {e}")

    def run_interactive(self):
        """运行交互式聊天"""
        print("=" * 60)
        print("高级聊天机器人 v1.0")
        print("=" * 60)
        print("命令:")
        print("  /help     - 显示帮助")
        print("  /mode     - 切换模式 (normal/stream/debug)")
        print("  /history  - 显示对话历史")
        print("  /stats    - 显示会话统计")
        print("  /save     - 保存会话")
        print("  /load     - 加载会话")
        print("  /clear    - 清空对话历史")
        print("  /system   - 修改系统提示词")
        print("  /quit     - 退出程序")
        print("=" * 60)
        print("开始聊天吧！（输入消息或命令）")
        print()

        while True:
            try:
                user_input = input("你: ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                # 处理普通消息
                print("\n助手: ", end="", flush=True)
                self.chat(user_input, stream=(self.mode != "normal"))
                print()

            except KeyboardInterrupt:
                print("\n\n检测到中断，是否退出？(y/n): ", end="")
                if input().lower() == 'y':
                    self._exit()
                    break
            except EOFError:
                self._exit()
                break

    def _handle_command(self, command):
        """处理命令"""
        cmd = command.lower().strip()

        if cmd == '/help':
            print("可用命令: /help, /mode, /history, /stats, /save, /load, /clear, /system, /quit")

        elif cmd == '/mode':
            print("选择模式: 1.normal(非流式) 2.stream(流式) 3.debug(调试)")
            choice = input("请输入选项(1/2/3): ").strip()
            modes = {'1': 'normal', '2': 'stream', '3': 'debug'}
            if choice in modes:
                self.set_mode(modes[choice])

        elif cmd == '/history':
            self.show_history()

        elif cmd == '/stats':
            self.show_stats()

        elif cmd == '/save':
            filename = input("输入文件名(默认 chat_session.json): ").strip()
            self.save_session(filename or "chat_session.json")

        elif cmd == '/load':
            filename = input("输入文件名(默认 chat_session.json): ").strip()
            self.load_session(filename or "chat_session.json")

        elif cmd == '/clear':
            confirm = input("确定清空对话历史？(y/n): ").strip().lower()
            if confirm == 'y':
                self.clear_history()

        elif cmd == '/system':
            new_prompt = input("输入新的系统提示词: ").strip()
            if new_prompt:
                self.system_prompt = new_prompt
                self.messages[0]["content"] = new_prompt
                print("系统提示词已更新")

        elif cmd == '/quit':
            self._exit()
            sys.exit(0)

        else:
            print(f"未知命令: {command}")

    def _exit(self):
        """退出处理"""
        print("\n再见！")
        self.show_stats()

        # 询问是否保存
        save = input("是否保存会话？(y/n): ").strip().lower()
        if save == 'y':
            self.save_session()


class RolePlayBot(AdvancedChatBot):
    """角色扮演机器人"""

    ROLES = {
        "teacher": "你是一位耐心、知识渊博的老师，善于用简单易懂的方式解释复杂概念。",
        "coder": "你是一位资深程序员，擅长解决各种编程问题，代码风格优雅。",
        "writer": "你是一位专业作家，文字优美，善于讲故事。",
        "translator": "你是一位精通多国语言的翻译专家，翻译准确流畅。",
        "analyst": "你是一位数据分析师，善于从数据中发现洞察。"
    }

    def __init__(self, role="teacher"):
        system_prompt = self.ROLES.get(role, self.ROLES["teacher"])
        super().__init__(system_prompt=system_prompt)
        self.role = role

    def change_role(self, role):
        """切换角色"""
        if role in self.ROLES:
            self.role = role
            self.system_prompt = self.ROLES[role]
            self.messages[0]["content"] = self.system_prompt
            print(f"已切换为 {role} 角色")
        else:
            print(f"无效角色，可选: {', '.join(self.ROLES.keys())}")


def demo_advanced_bot():
    """演示高级聊天机器人"""
    print("=" * 60)
    print("高级聊天机器人演示")
    print("=" * 60)

    bot = AdvancedChatBot()

    # 非流式测试
    print("\n--- 非流式模式测试 ---")
    bot.set_mode("normal")
    print("你: 你好！请介绍一下自己。")
    print("助手: ", end="")
    response = bot.chat("你好！请介绍一下自己。", stream=False)
    print(response)

    # 流式测试
    print("\n--- 流式模式测试 ---")
    bot.set_mode("stream")
    print("你: 请用一句话概括人工智能")
    print("助手: ", end="")
    bot.chat("请用一句话概括人工智能", stream=True)

    # 显示统计
    bot.show_stats()


def demo_role_play():
    """演示角色扮演机器人"""
    print("\n" + "=" * 60)
    print("角色扮演机器人演示")
    print("=" * 60)

    # 创建老师角色
    teacher = RolePlayBot(role="teacher")
    print("\n[老师角色]")
    print("你: 什么是量子计算？")
    print("老师: ", end="", flush=True)
    teacher.chat("什么是量子计算？用简单的话解释")
    print()

    # 创建程序员角色
    coder = RolePlayBot(role="coder")
    print("\n[程序员角色]")
    print("你: 如何写一个冒泡排序？")
    print("程序员: ", end="", flush=True)
    coder.chat("如何写一个冒泡排序？请给出Python代码")
    print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="高级聊天机器人")
    parser.add_argument("--mode", choices=["interactive", "demo", "role"], default="interactive",
                        help="运行模式: interactive(交互), demo(演示), role(角色扮演)")
    parser.add_argument("--role", default="teacher",
                        help="角色扮演模式下的角色")
    args = parser.parse_args()

    if args.mode == "demo":
        demo_advanced_bot()
        demo_role_play()
    elif args.mode == "role":
        bot = RolePlayBot(role=args.role)
        print(f"当前角色: {args.role}")
        bot.run_interactive()
    else:
        bot = AdvancedChatBot()
        bot.run_interactive()


if __name__ == "__main__":
    main()