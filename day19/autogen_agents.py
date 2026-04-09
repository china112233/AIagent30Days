"""
Day 19: AutoGen 多 Agent 示例

本文件演示 AutoGen 的多 Agent 功能：
- Two-Agent 对话模式
- Group Chat 群聊模式
- 代码执行沙箱
- 人机交互模式
- 工具集成示例

AutoGen 是微软开发的多 Agent 对话框架，支持 Agent 之间的自主对话和协作。
"""

import os
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# AutoGen 导入检查
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    HAS_AUTOGEN = True
except ImportError:
    HAS_AUTOGEN = False
    print("⚠️ AutoGen 未安装，请运行: pip install pyautogen")


# 加载环境变量
load_dotenv()


# ==========================================
# LLM 配置
# ==========================================


def get_llm_config() -> Dict:
    """获取 LLM 配置（使用 DeepSeek API）"""
    return {
        "config_list": [
            {
                "model": os.getenv("MODEL_NAME", "deepseek-chat"),
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL"),
            }
        ],
        "temperature": 0.7,
        "timeout": 120,
    }


def get_llm_config_for_code() -> Dict:
    """获取用于代码执行的 LLM 配置"""
    return {
        "config_list": [
            {
                "model": os.getenv("MODEL_NAME", "deepseek-chat"),
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL"),
            }
        ],
        "temperature": 0.0,  # 代码生成使用低温度
        "timeout": 120,
    }


# ==========================================
# 示例 1: AutoGen 概念介绍
# ==========================================


def example_autogen_concept():
    """演示 AutoGen 的基本概念"""
    print("\n" + "=" * 50)
    print("示例 1: AutoGen 核心概念")
    print("=" * 50)

    print("""
    AutoGen 是微软开发的多 Agent 对话框架。

    核心概念：

    1. ConversableAgent (可对话 Agent)
       - 基础 Agent 类
       - 支持发送和接收消息
       - 可配置 LLM、代码执行、人机交互

    2. AssistantAgent (助手 Agent)
       - 继承自 ConversableAgent
       - 自动生成回复
       - 通常配置 LLM 能力

    3. UserProxyAgent (用户代理 Agent)
       - 模拟用户行为
       - 可以执行代码
       - 可以请求人类输入

    4. GroupChat (群聊)
       - 管理多个 Agent
       - 自动选择下一个发言者
       - 支持自定义选择策略

    消息传递流程：
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   UserProxyAgent ──→ 消息 ──→ AssistantAgent       │
    │        ↑                            │               │
    │        │                            ↓               │
    │        └────── 回复/代码执行 ←──────┘               │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    特点：
    - 自动对话：Agent 自主完成多轮对话
    - 代码执行：内置沙箱执行生成的代码
    - 人机协作：支持人类参与决策
    - 灵活配置：可自定义 Agent 行为
    """)


# ==========================================
# 示例 2: Two-Agent 对话模式
# ==========================================


def example_two_agent_chat():
    """演示两 Agent 对话模式"""
    print("\n" + "=" * 50)
    print("示例 2: Two-Agent 对话模式")
    print("=" * 50)

    if not HAS_AUTOGEN:
        print("请先安装 AutoGen: pip install pyautogen")
        return

    llm_config = get_llm_config()

    # 创建助手 Agent
    assistant = AssistantAgent(
        name="assistant",
        system_message="""你是一个有帮助的 AI 助手。
        你可以帮助用户回答问题、提供信息。
        请用简洁专业的语言回答。""",
        llm_config=llm_config,
    )

    # 创建用户代理 Agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # 自动模式，不请求人类输入
        max_consecutive_auto_reply=3,  # 最大自动回复次数
        code_execution_config=False,  # 禁用代码执行
    )

    print("✅ Two-Agent 配置完成")
    print(f"   - Assistant: {assistant.name}")
    print(f"   - UserProxy: {user_proxy.name}")

    # 发起对话
    print("\n发起对话...")
    user_proxy.initiate_chat(
        assistant,
        message="请简单介绍一下 AutoGen 框架的主要特点。",
    )


# ==========================================
# 示例 3: 代码执行模式
# ==========================================


def example_code_execution():
    """演示代码执行功能"""
    print("\n" + "=" * 50)
    print("示例 3: 代码执行模式")
    print("=" * 50)

    if not HAS_AUTOGEN:
        print("请先安装 AutoGen: pip install pyautogen")
        return

    llm_config = get_llm_config_for_code()

    # 创建助手 Agent
    assistant = AssistantAgent(
        name="code_assistant",
        system_message="""你是一个 Python 编程助手。
        当用户需要代码时，请提供可执行的 Python 代码。
        代码应该用 ```python``` 代码块包裹。""",
        llm_config=llm_config,
    )

    # 创建用户代理 Agent（启用代码执行）
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        # 配置代码执行
        code_execution_config={
            "work_dir": "day19/coding",  # 工作目录
            "use_docker": False,  # 是否使用 Docker（生产环境建议 True）
            "timeout": 60,
        },
    )

    print("✅ 代码执行配置完成")
    print(f"   工作目录: day19/coding")

    # 发起代码生成和执行请求
    print("\n发起代码生成请求...")
    user_proxy.initiate_chat(
        assistant,
        message="请写一个 Python 函数计算斐波那契数列的第 N 项，并测试打印前 10 项。",
    )


# ==========================================
# 示例 4: Group Chat 群聊模式
# ==========================================


def example_group_chat():
    """演示 Group Chat 群聊模式"""
    print("\n" + "=" * 50)
    print("示例 4: Group Chat 群聊模式")
    print("=" * 50)

    if not HAS_AUTOGEN:
        print("请先安装 AutoGen: pip install pyautogen")
        return

    llm_config = get_llm_config()

    # 创建多个专业 Agent
    # 1. 研究员
    researcher = AssistantAgent(
        name="researcher",
        system_message="""你是一个研究员。
        你的任务是收集和分析信息。
        提供清晰的事实和数据。""",
        llm_config=llm_config,
    )

    # 2. 分析师
    analyst = AssistantAgent(
        name="analyst",
        system_message="""你是一个分析师。
        你的任务是分析研究员提供的信息。
        给出深入的见解和结论。""",
        llm_config=llm_config,
    )

    # 3. 撰稿人
    writer = AssistantAgent(
        name="writer",
        system_message="""你是一个技术撰稿人。
        你的任务是将分析结果整理成清晰的报告。
        使用简洁专业的语言。""",
        llm_config=llm_config,
    )

    # 创建群聊
    groupchat = GroupChat(
        agents=[researcher, analyst, writer],
        messages=[],
        max_round=6,  # 最大轮次
    )

    # 创建群聊管理器
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 创建用户代理
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    print("✅ Group Chat 配置完成")
    print(f"   成员: {[a.name for a in groupchat.agents]}")
    print(f"   最大轮次: {groupchat.max_round}")

    # 发起群聊
    print("\n发起群聊任务...")
    user_proxy.initiate_chat(
        manager,
        message="请讨论并总结 LangGraph 框架的主要特点和应用场景。",
    )


# ==========================================
# 示例 5: 人机交互模式
# ==========================================


def example_human_in_loop():
    """演示人机交互模式"""
    print("\n" + "=" * 50)
    print("示例 5: 人机交互模式")
    print("=" * 50)

    if not HAS_AUTOGEN:
        print("请先安装 AutoGen: pip install pyautogen")
        return

    llm_config = get_llm_config()

    print("""
    AutoGen 支持三种人机交互模式：

    1. NEVER - 完全自动，不请求人类输入
       human_input_mode="NEVER"

    2. ALWAYS - 每次都请求人类输入
       human_input_mode="ALWAYS"

    3. TERMINATE - 只在特定条件时请求人类输入
       human_input_mode="TERMINATE"
       需要配置 is_termination_msg 函数

    示例：使用 TERMINATE 模式
    当检测到任务完成关键词时暂停，等待人类确认。
    """)

    # 创建助手
    assistant = AssistantAgent(
        name="assistant",
        system_message="""你是一个有帮助的助手。
        完成任务后，请在回复末尾加上 [TASK_COMPLETE]。""",
        llm_config=llm_config,
    )

    # 配置终止条件
    def is_termination_msg(msg: Dict) -> bool:
        """检测是否应该终止"""
        content = msg.get("content", "")
        return "[TASK_COMPLETE]" in content or "TERMINATE" in content

    # 创建用户代理（TERMINATE 模式）
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="TERMINATE",
        is_termination_msg=is_termination_msg,
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )

    print("✅ 人机交互模式配置完成")
    print("   模式: TERMINATE")
    print("   终止条件: 检测到 [TASK_COMPLETE]")


# ==========================================
# 示例 6: 自定义 Agent
# ==========================================


def example_custom_agent():
    """演示自定义 Agent"""
    print("\n" + "=" * 50)
    print("示例 6: 自定义 Agent")
    print("=" * 50)

    print("""
    AutoGen 支持创建自定义 Agent，通过继承 ConversableAgent 实现。

    自定义 Agent 可以：
    1. 定义特定的消息处理逻辑
    2. 添加自定义工具和功能
    3. 控制回复生成策略

    示例自定义 Agent 结构：

    class MyCustomAgent(ConversableAgent):
        def __init__(self, name, **kwargs):
            super().__init__(name, **kwargs)
            # 自定义初始化

        def generate_reply(self, messages, sender, **kwargs):
            # 自定义回复逻辑
            # 可以调用外部 API、处理特定格式等
            return super().generate_reply(messages, sender, **kwargs)

    使用场景：
    - 特定领域的 Agent（如法律、医疗）
    - 集成外部工具的 Agent
    - 自定义对话策略的 Agent
    """)

    if HAS_AUTOGEN:
        # 展示自定义 Agent 的简单示例
        from autogen import ConversableAgent

        class CounterAgent(ConversableAgent):
            """一个简单的计数 Agent 示例"""

            def __init__(self, name, **kwargs):
                super().__init__(name, **kwargs)
                self.counter = 0

            def generate_reply(self, messages, sender, **kwargs):
                self.counter += 1
                reply = f"这是第 {self.counter} 次交互。"
                return reply

        # 创建自定义 Agent
        counter = CounterAgent(
            name="counter",
            llm_config=False,  # 不使用 LLM
        )

        print("\n✅ 自定义 CounterAgent 创建成功")

        # 模拟交互
        counter.send("你好", sender=None)


# ==========================================
# 示例 7: 工具集成
# ==========================================


def example_tool_integration():
    """演示工具集成"""
    print("\n" + "=" * 50)
    print("示例 7: 工具集成")
    print("=" * 50)

    if not HAS_AUTOGEN:
        print("请先安装 AutoGen: pip install pyautogen")
        return

    print("""
    AutoGen 支持为 Agent 注册自定义工具（函数）。

    工具注册方式：
    1. 使用 @agent.register_for_llm 装饰器注册为 LLM 工具
    2. 使用 @agent.register_for_execution 装饰器注册执行函数

    示例：为 Agent 添加计算器工具
    """)

    llm_config = get_llm_config()

    # 创建助手
    assistant = AssistantAgent(
        name="tool_assistant",
        system_message="你是一个可以执行计算的工具助手。",
        llm_config=llm_config,
    )

    # 定义工具函数
    def calculate(expression: str) -> str:
        """
        执行数学计算

        Args:
            expression: 数学表达式，如 "2 + 3 * 4"

        Returns:
            计算结果
        """
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    # 注册工具（概念演示）
    print("✅ 工具函数 'calculate' 定义完成")
    print("   功能: 执行数学计算表达式")

    # 在实际使用中，需要使用装饰器注册工具：
    # @assistant.register_for_llm(description="执行数学计算")
    # def calculate(expression: str) -> str:
    #     ...

    # @user_proxy.register_for_execution()
    # def calculate(expression: str) -> str:
    #     ...


# ==========================================
# 示例 8: 对话模式对比
# ==========================================


def example_conversation_modes():
    """演示不同的对话模式"""
    print("\n" + "=" * 50)
    print("示例 8: 对话模式对比")
    print("=" * 50)

    print("""
    AutoGen 支持多种对话模式：

    ┌─────────────────────────────────────────────────────┐
    │                   对话模式对比                       │
    ├─────────────────────────────────────────────────────┤
    │                                                     │
    │  1. 双 Agent 对话 (Two-Agent Chat)                  │
    │     ┌────────┐        ┌────────┐                   │
    │     │  User  │ ←────→ │Assistant│                   │
    │     └────────┘        └────────┘                   │
    │     适用：简单问答、代码生成                         │
    │                                                     │
    │  2. 群聊模式 (Group Chat)                           │
    │     ┌────────┐  ┌────────┐  ┌────────┐            │
    │     │ Agent A│←→│ Agent B│←→│ Agent C│            │
    │     └────────┘  └────────┘  └────────┘            │
    │          ↑            ↑            ↑               │
    │          └────────────┼────────────┘               │
    │                       │                            │
    │                  GroupChatManager                  │
    │     适用：多角度讨论、复杂任务分解                   │
    │                                                     │
    │  3. 层级模式 (Hierarchical)                         │
    │                 ┌────────┐                         │
    │                 │Manager │                         │
    │                 └────────┘                         │
    │                   ↙    ↘                           │
    │             ┌────────┐ ┌────────┐                 │
    │             │Worker1 │ │Worker2 │                 │
    │             └────────┘ └────────┘                 │
    │     适用：任务分配、工作流管理                       │
    │                                                     │
    │  4. 人机协作 (Human-in-the-Loop)                    │
    │     ┌────────┐        ┌────────┐                   │
    │     │ Human  │ ←────→ │  AI    │                   │
    │     └────────┘        └────────┘                   │
    │     适用：需要人类决策的场景                         │
    │                                                     │
    └─────────────────────────────────────────────────────┘
    """)


# ==========================================
# 示例 9: 最佳实践
# ==========================================


def example_best_practices():
    """演示 AutoGen 最佳实践"""
    print("\n" + "=" * 50)
    print("示例 9: AutoGen 最佳实践")
    print("=" * 50)

    print("""
    AutoGen 使用最佳实践：

    1. Agent 系统提示设计
       ✅ 清晰定义 Agent 角色和职责
       ✅ 提供明确的任务说明
       ✅ 指定输出格式要求
       ❌ 避免模糊不清的角色定义

    2. 对话轮次控制
       ✅ 设置合理的 max_round 限制
       ✅ 配置终止条件防止无限循环
       ✅ 监控对话进度
       ❌ 避免不设置上限的对话

    3. 代码执行安全
       ✅ 使用 Docker 容器隔离
       ✅ 设置执行超时
       ✅ 限制工作目录访问
       ❌ 避免在无沙箱环境执行代码

    4. 成本控制
       ✅ 使用较小模型处理简单任务
       ✅ 缓存常见回复
       ✅ 监控 Token 使用量
       ❌ 避免不必要的重复调用

    5. 错误处理
       ✅ 配置重试机制
       ✅ 处理 API 超时
       ✅ 记录错误日志
       ❌ 避免忽略异常情况

    6. 性能优化
       ✅ 并行处理独立任务
       ✅ 预加载常用模型
       ✅ 使用流式响应
       ❌ 避免串行处理可并行任务
    """)


# ==========================================
# 示例 10: 实际应用场景
# ==========================================


def example_use_cases():
    """演示 AutoGen 实际应用场景"""
    print("\n" + "=" * 50)
    print("示例 10: 实际应用场景")
    print("=" * 50)

    print("""
    AutoGen 实际应用场景：

    1. 代码开发和调试
       ┌─────────────────────────────────────────────────┐
       │   User → CodeAgent → 执行代码 → 修复错误        │
       │       ↑                                │        │
       │       └────────── 迭代优化 ←───────────┘        │
       └─────────────────────────────────────────────────┘
       场景：自动化编程、Bug 修复、测试生成

    2. 数据分析流程
       ┌─────────────────────────────────────────────────┐
       │   DataCollector → DataAnalyst → ReportWriter   │
       │        (收集)        (分析)        (报告)       │
       └─────────────────────────────────────────────────┘
       场景：自动化报告生成、数据挖掘

    3. 研究辅助
       ┌─────────────────────────────────────────────────┐
       │   Researcher → Reviewer → Writer               │
       │     (研究)      (审核)     (撰写)               │
       └─────────────────────────────────────────────────┘
       场景：文献综述、论文辅助、知识整理

    4. 多角色辩论
       ┌─────────────────────────────────────────────────┐
       │   Proponent ←→ Skeptic ←→ Moderator            │
       │     (支持)       (质疑)      (主持)             │
       └─────────────────────────────────────────────────┘
       场景：决策分析、方案评审、风险识别

    5. 复杂任务分解
       ┌─────────────────────────────────────────────────┐
       │              Manager (任务分解)                  │
       │              ↙         ↘                        │
       │        Specialist1    Specialist2               │
       │           ↓                ↓                    │
       │        Aggregator (结果整合)                     │
       └─────────────────────────────────────────────────┘
       场景：复杂项目、多步骤任务
    """)


# ==========================================
# 模拟运行（无 AutoGen 时的演示）
# ==========================================


def run_mock_demo():
    """无 AutoGen 时的模拟演示"""
    print("\n" + "=" * 60)
    print("AutoGen 模拟演示模式")
    print("=" * 60)

    print("""
    ⚠️ AutoGen 未安装，使用模拟演示模式

    要安装 AutoGen，请运行：
    pip install pyautogen

    安装后可以运行完整的示例代码。

    AutoGen 核心功能：
    1. Two-Agent 对话 - 两个 Agent 之间的自动对话
    2. Group Chat - 多 Agent 群聊协作
    3. 代码执行 - 自动生成和执行代码
    4. 人机交互 - 支持人类参与决策

    典型工作流程：
    ┌───────────────────────────────────────────────────┐
    │                                                   │
    │  1. 配置 LLM (DeepSeek/OpenAI/本地模型)          │
    │     ↓                                             │
    │  2. 创建 Agent (定义角色和系统提示)              │
    │     ↓                                             │
    │  3. 发起对话 (设置初始任务/问题)                  │
    │     ↓                                             │
    │  4. Agent 自主协作 (自动多轮对话)                │
    │     ↓                                             │
    │  5. 获取结果 (任务完成或达到轮次限制)            │
    │                                                   │
    └───────────────────────────────────────────────────┘

    示例代码结构：
    ```python
    import autogen

    # 配置 LLM
    llm_config = {
        "config_list": [{
            "model": "deepseek-chat",
            "api_key": "your_key",
            "base_url": "https://api.deepseek.com"
        }]
    }

    # 创建 Agent
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config
    )

    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "coding"}
    )

    # 发起对话
    user_proxy.initiate_chat(
        assistant,
        message="帮我写一个 Python 脚本"
    )
    ```
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 19: AutoGen 多 Agent 示例")
    print("=" * 60)

    # 运行概念介绍
    example_autogen_concept()

    if not HAS_AUTOGEN:
        run_mock_demo()
        print("\n" + "=" * 60)
        print("提示: 安装 pyautogen 后可运行完整示例")
        print("=" * 60)
        return

    # 运行各示例
    try:
        example_two_agent_chat()
    except Exception as e:
        print(f"示例 2 跳过: {e}")

    try:
        example_code_execution()
    except Exception as e:
        print(f"示例 3 跳过: {e}")

    try:
        example_group_chat()
    except Exception as e:
        print(f"示例 4 跳过: {e}")

    example_human_in_loop()
    example_custom_agent()
    example_tool_integration()
    example_conversation_modes()
    example_best_practices()
    example_use_cases()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)
    print("\n关键要点:")
    print("  1. AutoGen 支持多 Agent 自主对话")
    print("  2. 可以自动生成和执行代码")
    print("  3. 支持人机协作模式")
    print("  4. 适合研究和实验复杂 Agent 系统")


if __name__ == "__main__":
    main()