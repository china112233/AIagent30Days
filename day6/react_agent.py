"""
ReAct Agent 实现
完整的 ReAct 模式，支持结构化思维链、多步推理、自我修正
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv

from tools import ToolRegistry, create_default_registry

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 枚举和配置 ====================

class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class ReActConfig:
    """ReAct Agent 配置"""
    model: str = "deepseek-chat"
    max_steps: int = 15
    temperature: float = 0.7
    verbose: bool = True
    enable_reflection: bool = True


# ==================== 执行记录 ====================

@dataclass
class ExecutionStep:
    """单步执行记录"""
    step_number: int
    thought: str
    action: str
    action_input: Dict
    observation: str
    success: bool = True

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"""
步骤 {self.step_number} [{status}]:
  Thought: {self.thought}
  Action: {self.action}
  Action Input: {json.dumps(self.action_input, ensure_ascii=False)}
  Observation: {self.observation[:100]}..."""


# ==================== ReAct Agent ====================

class ReActAgent:
    """
    ReAct 模式 Agent

    特点:
    - 结构化思维链
    - 多步骤推理
    - 自我反思和修正
    - 详细执行日志
    """

    REACT_PROMPT = """你是一个智能助手，使用 ReAct (Reasoning + Acting) 模式工作。

## 可用工具
{tool_descriptions}

## 工作流程
每次思考请遵循以下格式：

Thought: 分析当前情况，思考下一步该做什么
Action: 选择一个工具名称
Action Input: 工具参数（必须是有效的 JSON 格式）

执行后会收到观察结果，然后继续思考下一步。

当你可以回答用户问题时，使用：
Thought: 我已经有了足够的信息来回答用户问题
Final Answer: 你的完整回答

## 规则
1. 每次只调用一个工具
2. Action Input 必须是有效的 JSON
3. 如果工具调用失败，分析原因并尝试修正
4. 复杂任务分解为多个步骤

## 示例
用户: 计算 15 的平方根加上 10

Thought: 用户需要计算 sqrt(15) + 10，我需要用计算器工具
Action: calculator
Action Input: {{"expression": "sqrt(15) + 10"}}

观察结果: 13.872983346207417

Thought: 我已经得到了计算结果
Final Answer: 15 的平方根加上 10 等于约 13.87

---

现在开始处理用户的任务！"""

    def __init__(
        self,
        tools: ToolRegistry = None,
        config: ReActConfig = None
    ):
        self.tools = tools or create_default_registry()
        self.config = config or ReActConfig()
        self.state = AgentState.IDLE
        self.execution_history: List[ExecutionStep] = []
        self.messages: List[Dict] = []

    def run(self, task: str) -> Tuple[str, List[ExecutionStep]]:
        """
        执行任务

        Args:
            task: 用户任务

        Returns:
            (最终结果, 执行历史)
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[ReAct Agent] 开始任务: {task}")
            print(f"{'='*60}")

        # 初始化
        self.state = AgentState.THINKING
        self.execution_history = []
        self.messages = []

        # 设置系统提示
        system_prompt = self.REACT_PROMPT.format(
            tool_descriptions=self._format_tool_descriptions()
        )
        self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": task})

        # 执行循环
        step_count = 0
        while step_count < self.config.max_steps:
            step_count += 1
            self.state = AgentState.THINKING

            if self.config.verbose:
                print(f"\n--- 步骤 {step_count} ---")

            # 思考
            response = self._call_llm()

            # 解析响应
            parsed = self._parse_response(response)

            # 检查是否完成
            if parsed.get("is_final", False):
                self.state = AgentState.FINISHED
                final_answer = parsed.get("final_answer", "任务完成")
                if self.config.verbose:
                    print(f"\n[完成] {final_answer}")
                return final_answer, self.execution_history

            # 执行动作
            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})

            if not action:
                if self.config.verbose:
                    print("[警告] 未识别到有效动作，重新思考")
                self.messages.append({
                    "role": "user",
                    "content": "请按照格式重新组织你的思考，确保包含 Action 和 Action Input。"
                })
                continue

            # 执行工具
            self.state = AgentState.ACTING
            observation = self._execute_tool(action, action_input)

            # 记录执行步骤
            step = ExecutionStep(
                step_number=step_count,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                success="错误" not in observation
            )
            self.execution_history.append(step)

            if self.config.verbose:
                print(f"Thought: {thought}")
                print(f"Action: {action}")
                print(f"Action Input: {json.dumps(action_input, ensure_ascii=False)}")
                print(f"Observation: {observation[:200]}{'...' if len(observation) > 200 else ''}")

            # 添加观察结果
            self.state = AgentState.OBSERVING
            self.messages.append({
                "role": "assistant",
                "content": response
            })
            self.messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        # 达到步数限制
        self.state = AgentState.ERROR
        return "达到最大步数限制，任务未能完成", self.execution_history

    def _call_llm(self) -> str:
        """调用 LLM"""
        response = client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Dict:
        """
        解析 ReAct 响应

        Args:
            response: LLM 响应

        Returns:
            解析后的字典
        """
        result = {
            "thought": "",
            "action": "",
            "action_input": {},
            "is_final": False,
            "final_answer": ""
        }

        # 提取 Thought
        thought_patterns = [
            r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)",
            r"思考:\\s*(.+?)(?=动作:|最终答案:|$)"
        ]
        for pattern in thought_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result["thought"] = match.group(1).strip()
                break

        # 检查是否是最终答案
        final_patterns = [
            r"Final Answer:\s*(.+)$",
            r"最终答案:\s*(.+)$"
        ]
        for pattern in final_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result["is_final"] = True
                result["final_answer"] = match.group(1).strip()
                return result

        # 提取 Action
        action_patterns = [
            r"Action:\s*(\w+)",
            r"动作:\s*(\w+)"
        ]
        for pattern in action_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["action"] = match.group(1).strip()
                break

        # 提取 Action Input
        input_patterns = [
            r"Action Input:\s*(\{.+?\})",
            r"动作输入:\s*(\{.+?\})"
        ]
        for pattern in input_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    result["action_input"] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    # 尝试修复 JSON
                    result["action_input"] = self._repair_json(match.group(1))
                break

        return result

    def _repair_json(self, json_str: str) -> Dict:
        """修复损坏的 JSON"""
        # 移除可能的注释
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        # 尝试添加缺失的引号
        if '"' not in json_str and "'" not in json_str:
            json_str = json_str.replace(':', '":"').replace('{', '{"').replace('}', '"}')
        try:
            return json.loads(json_str)
        except:
            return {}

    def _execute_tool(self, action: str, action_input: Dict) -> str:
        """执行工具"""
        return self.tools.execute(action, **action_input)

    def _format_tool_descriptions(self) -> str:
        """格式化工具描述"""
        descriptions = []
        for name, tool in self.tools._tools.items():
            desc = f"- {name}: {tool.description}"
            params = tool.parameters.get("properties", {})
            if params:
                param_str = ", ".join(params.keys())
                desc += f"\n  参数: {param_str}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def reflect(self) -> str:
        """
        反思执行过程

        Returns:
            反思结果
        """
        if not self.execution_history:
            return "没有执行历史可供反思"

        history_str = "\n".join([str(step) for step in self.execution_history])

        reflection_prompt = f"""请回顾以下执行历史，分析：

执行历史:
{history_str}

请回答：
1. 哪些步骤执行得很好？
2. 遇到了什么问题？
3. 如何改进？
"""
        self.messages.append({"role": "user", "content": reflection_prompt})
        return self._call_llm()

    def get_execution_summary(self) -> Dict:
        """获取执行摘要"""
        return {
            "total_steps": len(self.execution_history),
            "successful_steps": sum(1 for s in self.execution_history if s.success),
            "failed_steps": sum(1 for s in self.execution_history if not s.success),
            "tools_used": list(set(s.action for s in self.execution_history)),
            "final_state": self.state.value
        }


# ==================== 演示函数 ====================

def demo_react_agent():
    """演示 ReAct Agent"""
    print("\n" + "=" * 60)
    print("ReAct Agent 演示")
    print("=" * 60)

    agent = ReActAgent(config=ReActConfig(verbose=True))

    # 测试任务
    tasks = [
        "帮我计算 sqrt(144) + sqrt(81) 的结果",
        "先获取当前时间，然后告诉我今天是星期几",
        "搜索 Python 的信息，然后用字符串工具将结果转换为大写"
    ]

    for task in tasks:
        result, history = agent.run(task)
        print(f"\n[最终结果] {result}")

        # 显示执行摘要
        summary = agent.get_execution_summary()
        print(f"\n[执行摘要]")
        print(f"  总步数: {summary['total_steps']}")
        print(f"  成功: {summary['successful_steps']}, 失败: {summary['failed_steps']}")
        print(f"  使用工具: {summary['tools_used']}")


def demo_reflection():
    """演示反思功能"""
    print("\n" + "=" * 60)
    print("ReAct Agent 反思功能演示")
    print("=" * 60)

    agent = ReActAgent(config=ReActConfig(verbose=True, enable_reflection=True))

    # 执行任务
    task = "计算 (100 + 200) * 3 / 5 的结果"
    result, history = agent.run(task)

    print(f"\n[结果] {result}")

    # 反思
    print("\n[反思过程]")
    reflection = agent.reflect()
    print(reflection)


def demo_interactive():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("ReAct Agent 交互模式")
    print("=" * 60)
    print("输入任务，Agent 将使用 ReAct 模式自主完成")
    print("命令: /history - 查看执行历史, /summary - 查看摘要, /quit - 退出")
    print("=" * 60)

    agent = ReActAgent(config=ReActConfig(verbose=True))

    while True:
        user_input = input("\n请输入任务: ").strip()

        if user_input.lower() == '/quit':
            print("再见！")
            break

        if user_input.lower() == '/history':
            for step in agent.execution_history:
                print(step)
            continue

        if user_input.lower() == '/summary':
            print(agent.get_execution_summary())
            continue

        if not user_input:
            continue

        result, history = agent.run(user_input)
        print(f"\n[结果] {result}")


if __name__ == "__main__":
    # 基础演示
    demo_react_agent()

    # 反思演示
    demo_reflection()

    # 交互模式
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()