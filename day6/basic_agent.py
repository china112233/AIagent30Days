"""
基础 Agent 实现
实现 Agent 核心循环、消息管理、简单记忆功能
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

from tools import ToolRegistry, create_default_registry

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== Agent 配置 ====================

@dataclass
class AgentConfig:
    """Agent 配置"""
    model: str = "deepseek-chat"
    max_steps: int = 10
    temperature: float = 0.7
    verbose: bool = True


# ==================== Agent 记忆 ====================

class AgentMemory:
    """
    Agent 记忆系统
    管理对话历史和执行记录
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.messages: List[Dict] = []
        self.execution_history: List[Dict] = []

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({"role": role, "content": content})
        # 保持历史在限制内
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def add_execution(self, thought: str, action: str, observation: str):
        """添加执行记录"""
        self.execution_history.append({
            "thought": thought,
            "action": action,
            "observation": observation
        })

    def get_messages(self) -> List[Dict]:
        """获取消息历史"""
        return self.messages.copy()

    def get_last_executions(self, n: int = 3) -> List[Dict]:
        """获取最近 n 条执行记录"""
        return self.execution_history[-n:]

    def clear(self):
        """清空记忆"""
        self.messages = []
        self.execution_history = []


# ==================== 基础 Agent ====================

class BasicAgent:
    """
    基础 Agent 实现
    支持 ReAct 模式的思考和工具调用
    """

    SYSTEM_PROMPT = """你是一个智能助手，能够使用工具来完成任务。

## 可用工具
{tool_descriptions}

## 工作模式
使用 ReAct 模式工作：
1. Thought: 分析当前情况，思考下一步
2. Action: 选择要执行的工具
3. Action Input: 工具参数（JSON 格式）

当你认为可以回答用户问题时：
Thought: 我已经可以回答用户的问题
Final Answer: 你的回答

## 注意事项
- 每次只执行一个动作
- 仔细分析工具参数
- 如果工具执行失败，尝试修正

开始！"""

    def __init__(
        self,
        tools: ToolRegistry = None,
        config: AgentConfig = None
    ):
        """
        初始化 Agent

        Args:
            tools: 工具注册表
            config: Agent 配置
        """
        self.tools = tools or create_default_registry()
        self.config = config or AgentConfig()
        self.memory = AgentMemory()

    def run(self, task: str) -> str:
        """
        执行用户任务

        Args:
            task: 用户任务描述

        Returns:
            任务结果
        """
        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"[任务] {task}")
            print(f"{'='*50}")

        # 重置记忆
        self.memory.clear()
        self.step_count = 0

        # 添加系统提示
        system_prompt = self.SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_descriptions()
        )
        self.memory.add_message("system", system_prompt)

        # 添加用户任务
        self.memory.add_message("user", task)

        # 执行循环
        return self._execute_loop()

    def _execute_loop(self) -> str:
        """执行主循环"""
        while self.step_count < self.config.max_steps:
            if self.config.verbose:
                print(f"\n--- 步骤 {self.step_count + 1} ---")

            # 思考并决策
            response = self._think()

            # 解析响应
            parsed = self._parse_response(response)

            if self.config.verbose:
                print(f"[思考] {parsed.get('thought', 'N/A')}")

            # 检查是否完成
            if parsed.get("type") == "final_answer":
                return parsed.get("answer", "任务完成")

            # 执行动作
            action = parsed.get("action")
            action_input = parsed.get("action_input", {})

            if action:
                observation = self._execute_action(action, action_input)

                if self.config.verbose:
                    print(f"[动作] {action}({action_input})")
                    print(f"[观察] {observation}")

                # 记录执行
                self.memory.add_execution(
                    parsed.get("thought", ""),
                    f"{action}({action_input})",
                    observation
                )

                # 添加观察结果到消息
                self.memory.add_message("user", f"观察结果: {observation}")

            self.step_count += 1

        return "达到最大步数限制，任务未完成"

    def _think(self) -> str:
        """
        思考过程

        Returns:
            LLM 响应
        """
        response = client.chat.completions.create(
            model=self.config.model,
            messages=self.memory.get_messages(),
            temperature=self.config.temperature
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Dict:
        """
        解析 LLM 响应

        Args:
            response: LLM 响应文本

        Returns:
            解析后的字典
        """
        result = {}

        # 提取思考内容
        if "Thought:" in response:
            thought_match = response.split("Thought:")[-1]
            if "Action:" in thought_match:
                result["thought"] = thought_match.split("Action:")[0].strip()
            elif "Final Answer:" in thought_match:
                result["thought"] = thought_match.split("Final Answer:")[0].strip()
                result["type"] = "final_answer"
                answer = thought_match.split("Final Answer:")[-1].strip()
                result["answer"] = answer
                return result
            else:
                result["thought"] = thought_match.strip()

        # 提取动作
        if "Action:" in response:
            action_match = response.split("Action:")[-1]
            if "Action Input:" in action_match:
                result["action"] = action_match.split("Action Input:")[0].strip()
            else:
                result["action"] = action_match.strip()

        # 提取动作输入
        if "Action Input:" in response:
            input_str = response.split("Action Input:")[-1].strip()
            # 尝试解析 JSON
            try:
                # 处理可能的格式问题
                if input_str.startswith("{"):
                    result["action_input"] = json.loads(input_str)
                else:
                    # 简单参数
                    result["action_input"] = {"query": input_str} if result.get("action") == "search" else {"expression": input_str}
            except json.JSONDecodeError:
                # 尝试修复常见问题
                result["action_input"] = self._fix_json_input(input_str)

        return result

    def _fix_json_input(self, input_str: str) -> Dict:
        """修复 JSON 输入"""
        # 简单的参数映射
        return {"expression": input_str, "query": input_str, "text": input_str}

    def _execute_action(self, action: str, action_input: Dict) -> str:
        """
        执行动作

        Args:
            action: 动作名称
            action_input: 动作参数

        Returns:
            执行结果
        """
        return self.tools.execute(action, **action_input)


# ==================== 演示函数 ====================

def demo_basic_agent():
    """演示基础 Agent"""
    print("\n" + "=" * 50)
    print("基础 Agent 演示")
    print("=" * 50)

    # 创建 Agent
    agent = BasicAgent(config=AgentConfig(verbose=True))

    # 测试任务
    tasks = [
        "请计算 23 * 45 + 67 的结果",
        "现在几点了？",
        "搜索一下 Python 的相关信息"
    ]

    for task in tasks:
        result = agent.run(task)
        print(f"\n[结果] {result}")


def demo_interactive():
    """交互式 Agent"""
    print("\n" + "=" * 50)
    print("交互式 Agent")
    print("=" * 50)
    print("输入任务，Agent 将自主完成")
    print("输入 'quit' 退出")
    print("=" * 50)

    agent = BasicAgent(config=AgentConfig(verbose=True))

    while True:
        user_input = input("\n请输入任务: ").strip()

        if user_input.lower() == 'quit':
            print("再见！")
            break

        if not user_input:
            continue

        result = agent.run(user_input)
        print(f"\n[最终结果] {result}")


if __name__ == "__main__":
    # 演示基础用法
    demo_basic_agent()

    # 交互模式
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()