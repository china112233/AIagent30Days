"""
任务规划 Agent 实现
支持目标分解、计划生成、步骤执行、进度跟踪
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
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


# ==================== 数据结构 ====================

class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """计划步骤"""
    step_id: int
    description: str
    tool: Optional[str] = None
    parameters: Dict = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "tool": self.tool,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result
        }


@dataclass
class Plan:
    """执行计划"""
    goal: str
    steps: List[PlanStep]
    current_step: int = 0

    def get_next_step(self) -> Optional[PlanStep]:
        """获取下一个可执行的步骤"""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # 检查依赖
                if self._check_dependencies(step):
                    return step
        return None

    def _check_dependencies(self, step: PlanStep) -> bool:
        """检查步骤依赖是否满足"""
        for dep_id in step.dependencies:
            dep_step = self.get_step_by_id(dep_id)
            if dep_step is None or dep_step.status != StepStatus.COMPLETED:
                return False
        return True

    def get_step_by_id(self, step_id: int) -> Optional[PlanStep]:
        """根据 ID 获取步骤"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_progress(self) -> Tuple[int, int]:
        """获取进度"""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed, len(self.steps)

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "progress": f"{self.get_progress()[0]}/{self.get_progress()[1]}"
        }


# ==================== 规划 Agent ====================

class PlanningAgent:
    """
    任务规划 Agent

    特点:
    - 将复杂目标分解为步骤
    - 生成可执行计划
    - 自动执行步骤
    - 进度跟踪和调整
    """

    PLANNING_PROMPT = """你是一个任务规划专家。请将用户目标分解为具体的执行步骤。

## 可用工具
{tool_descriptions}

## 任务分解要求
1. 每个步骤应该是一个清晰、可执行的任务
2. 标明每个步骤需要使用的工具（如果有）
3. 标明步骤之间的依赖关系
4. 步骤数量适中（通常 3-7 步）

## 输出格式
请输出 JSON 格式的计划：

```json
{{
    "steps": [
        {{
            "step_id": 1,
            "description": "步骤描述",
            "tool": "工具名称或null",
            "parameters": {{}},
            "dependencies": []
        }}
    ]
}}
```

## 示例
目标: 查询北京天气并给出穿衣建议

```json
{{
    "steps": [
        {{
            "step_id": 1,
            "description": "查询北京当前天气",
            "tool": "search",
            "parameters": {{"query": "北京天气"}},
            "dependencies": []
        }},
        {{
            "step_id": 2,
            "description": "根据天气生成穿衣建议",
            "tool": null,
            "parameters": {{}},
            "dependencies": [1]
        }}
    ]
}}
```

---

请为以下目标生成执行计划："""

    EXECUTION_PROMPT = """你是一个任务执行助手。请根据给定的步骤描述和上下文执行任务。

## 当前步骤
{step_description}

## 前序步骤结果
{previous_results}

## 请执行当前步骤
如果需要使用工具，请说明工具名称和参数。
如果不需要工具，请直接给出执行结果。"""

    def __init__(
        self,
        tools: ToolRegistry = None,
        verbose: bool = True
    ):
        self.tools = tools or create_default_registry()
        self.verbose = verbose
        self.current_plan: Optional[Plan] = None

    def run(self, goal: str) -> str:
        """
        执行用户目标

        Args:
            goal: 用户目标

        Returns:
            最终结果
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[规划 Agent] 目标: {goal}")
            print(f"{'='*60}")

        # 1. 生成计划
        if self.verbose:
            print("\n[阶段1] 生成执行计划...")
        self.current_plan = self._create_plan(goal)

        if not self.current_plan:
            return "无法生成执行计划"

        if self.verbose:
            self._display_plan()

        # 2. 执行计划
        if self.verbose:
            print("\n[阶段2] 执行计划...")
        return self._execute_plan()

    def _create_plan(self, goal: str) -> Optional[Plan]:
        """生成执行计划"""
        tool_descriptions = self.tools.get_descriptions()

        prompt = self.PLANNING_PROMPT.format(
            tool_descriptions=tool_descriptions
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": goal}
            ],
            temperature=0.3
        )

        # 解析计划
        content = response.choices[0].message.content
        plan_data = self._extract_json(content)

        if not plan_data:
            return None

        # 构建 Plan 对象
        steps = []
        for step_data in plan_data.get("steps", []):
            step = PlanStep(
                step_id=step_data.get("step_id", len(steps) + 1),
                description=step_data.get("description", ""),
                tool=step_data.get("tool"),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", [])
            )
            steps.append(step)

        return Plan(goal=goal, steps=steps)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 尝试提取 JSON 块
        import re
        json_patterns = [
            r'```json\s*(.+?)\s*```',
            r'```\s*(.+?)\s*```',
            r'\{[\s\S]*\}'
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue

        return None

    def _execute_plan(self) -> str:
        """执行计划"""
        while True:
            # 获取下一个步骤
            step = self.current_plan.get_next_step()
            if step is None:
                break

            step.status = StepStatus.RUNNING

            if self.verbose:
                completed, total = self.current_plan.get_progress()
                print(f"\n--- 步骤 {step.step_id}/{total} ---")
                print(f"任务: {step.description}")

            # 执行步骤
            result = self._execute_step(step)

            # 更新状态
            step.result = result
            step.status = StepStatus.COMPLETED if "错误" not in result else StepStatus.FAILED

            if self.verbose:
                print(f"结果: {result[:200]}{'...' if len(result) > 200 else ''}")

            # 检查是否需要重新规划
            if step.status == StepStatus.FAILED:
                if self.verbose:
                    print("[警告] 步骤执行失败，尝试重新规划...")
                # 可以在这里添加重新规划逻辑

        # 生成最终结果
        return self._synthesize_results()

    def _execute_step(self, step: PlanStep) -> str:
        """执行单个步骤"""
        # 如果有指定工具，直接执行
        if step.tool:
            return self.tools.execute(step.tool, **step.parameters)

        # 否则使用 LLM 执行
        previous_results = self._get_previous_results(step)

        prompt = self.EXECUTION_PROMPT.format(
            step_description=step.description,
            previous_results=previous_results
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content

    def _get_previous_results(self, current_step: PlanStep) -> str:
        """获取前序步骤的结果"""
        results = []
        for step in self.current_plan.steps:
            if step.step_id in current_step.dependencies and step.result:
                results.append(f"步骤{step.step_id}结果: {step.result}")

        if not results:
            return "无前序步骤"

        return "\n".join(results)

    def _synthesize_results(self) -> str:
        """综合所有结果生成最终答案"""
        results = []
        for step in self.current_plan.steps:
            if step.result:
                results.append(f"步骤{step.step_id}: {step.result}")

        prompt = f"""请根据以下执行结果，给用户一个完整的回复：

目标: {self.current_plan.goal}

执行结果:
{chr(10).join(results)}

请生成一个清晰、有帮助的最终回答："""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content

    def _display_plan(self):
        """显示计划"""
        print(f"\n执行计划 (共 {len(self.current_plan.steps)} 步):")
        for step in self.current_plan.steps:
            deps = f" (依赖: {step.dependencies})" if step.dependencies else ""
            tool = f" [{step.tool}]" if step.tool else ""
            print(f"  {step.step_id}. {step.description}{tool}{deps}")

    def get_plan_status(self) -> Dict:
        """获取计划状态"""
        if not self.current_plan:
            return {"status": "no_plan"}

        completed, total = self.current_plan.get_progress()
        return {
            "goal": self.current_plan.goal,
            "total_steps": total,
            "completed_steps": completed,
            "progress": f"{completed}/{total}",
            "steps": [s.to_dict() for s in self.current_plan.steps]
        }


# ==================== 演示函数 ====================

def demo_planning_agent():
    """演示规划 Agent"""
    print("\n" + "=" * 60)
    print("规划 Agent 演示")
    print("=" * 60)

    agent = PlanningAgent(verbose=True)

    # 测试目标
    goals = [
        "帮我获取当前时间，然后告诉我这个时间对应的分钟数",
        "计算 100 的平方根，然后将结果转换为大写字符串"
    ]

    for goal in goals:
        result = agent.run(goal)
        print(f"\n[最终结果] {result}")

        # 显示状态
        status = agent.get_plan_status()
        print(f"\n[计划状态] 进度: {status['progress']}")


def demo_complex_planning():
    """演示复杂规划"""
    print("\n" + "=" * 60)
    print("复杂任务规划演示")
    print("=" * 60)

    agent = PlanningAgent(verbose=True)

    goal = """
    请帮我完成以下任务：
    1. 获取当前日期时间
    2. 计算 256 的平方根
    3. 将计算结果和日期时间组合成一个报告
    """

    result = agent.run(goal)
    print(f"\n[最终结果] {result}")


def demo_interactive():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("规划 Agent 交互模式")
    print("=" * 60)
    print("输入目标，Agent 将自动分解并执行")
    print("命令: /status - 查看计划状态, /plan - 查看计划详情, /quit - 退出")
    print("=" * 60)

    agent = PlanningAgent(verbose=True)

    while True:
        user_input = input("\n请输入目标: ").strip()

        if user_input.lower() == '/quit':
            print("再见！")
            break

        if user_input.lower() == '/status':
            print(agent.get_plan_status())
            continue

        if user_input.lower() == '/plan':
            if agent.current_plan:
                agent._display_plan()
            else:
                print("暂无计划")
            continue

        if not user_input:
            continue

        result = agent.run(user_input)
        print(f"\n[最终结果] {result}")


if __name__ == "__main__":
    # 基础演示
    demo_planning_agent()

    # 复杂任务演示
    demo_complex_planning()

    # 交互模式
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        demo_interactive()