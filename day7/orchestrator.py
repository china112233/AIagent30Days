"""
编排者模块
实现任务分析、分解、分配和结果整合
"""

import os
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 枚举和数据结构 ====================

class SubtaskStatus(Enum):
    """子任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"      # 串行执行
    PARALLEL = "parallel"          # 并行执行
    HYBRID = "hybrid"              # 混合执行


@dataclass
class Subtask:
    """
    子任务定义

    Attributes:
        id: 子任务 ID
        name: 子任务名称
        description: 详细描述
        agent_type: 所需 Agent 类型
        dependencies: 依赖的子任务 ID 列表
        priority: 优先级
        status: 状态
        result: 执行结果
        assigned_agent: 分配的 Agent ID
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    agent_type: str = ""
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Any = None
    assigned_agent: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent
        }


@dataclass
class ExecutionPlan:
    """
    执行计划

    Attributes:
        goal: 总目标
        subtasks: 子任务列表
        execution_mode: 执行模式
        estimated_time: 预估时间
    """
    goal: str
    subtasks: List[Subtask]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    estimated_time: float = 0.0

    def get_ready_subtasks(self) -> List[Subtask]:
        """获取可以执行的子任务"""
        ready = []
        for subtask in self.subtasks:
            if subtask.status != SubtaskStatus.PENDING:
                continue

            # 检查依赖
            deps_satisfied = True
            for dep_id in subtask.dependencies:
                dep = self.get_subtask_by_id(dep_id)
                if not dep or dep.status != SubtaskStatus.COMPLETED:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(subtask)

        return ready

    def get_subtask_by_id(self, subtask_id: str) -> Optional[Subtask]:
        """根据 ID 获取子任务"""
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask
        return None

    def get_progress(self) -> Tuple[int, int]:
        """获取进度"""
        completed = sum(1 for s in self.subtasks
                       if s.status == SubtaskStatus.COMPLETED)
        return completed, len(self.subtasks)

    def is_complete(self) -> bool:
        """是否完成"""
        return all(s.status in [SubtaskStatus.COMPLETED, SubtaskStatus.SKIPPED]
                  for s in self.subtasks)

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal,
            "execution_mode": self.execution_mode.value,
            "progress": f"{self.get_progress()[0]}/{self.get_progress()[1]}",
            "subtasks": [s.to_dict() for s in self.subtasks]
        }


# ==================== 任务分解器 ====================

class TaskDecomposer:
    """
    任务分解器

    将复杂任务分解为子任务
    """

    DECOMPOSITION_PROMPT = """你是一个任务分解专家。请将以下任务分解为具体的子任务。

任务: {task}

## 可用的 Agent 类型
{agent_types}

## 分解要求
1. 每个子任务应该是一个独立的、可执行的任务
2. 明确每个子任务需要的 Agent 类型
3. 标明子任务之间的依赖关系
4. 子任务数量适中（通常 3-7 个）

## 输出格式
请输出 JSON 格式：

```json
{{
    "execution_mode": "sequential 或 parallel 或 hybrid",
    "subtasks": [
        {{
            "name": "子任务名称",
            "description": "详细描述",
            "agent_type": "researcher 或 coder 或 writer 或 analyst 或 architect",
            "dependencies": [],
            "priority": 5
        }}
    ]
}}
```

请分析任务并输出分解结果："""

    def __init__(self, available_agents: List[str] = None):
        """
        初始化任务分解器

        Args:
            available_agents: 可用的 Agent 类型列表
        """
        self.available_agents = available_agents or [
            "researcher", "coder", "writer", "analyst", "architect"
        ]

    def decompose(self, task: str) -> ExecutionPlan:
        """
        分解任务

        Args:
            task: 任务描述

        Returns:
            执行计划
        """
        # 构建提示
        agent_types_desc = self._format_agent_types()
        prompt = self.DECOMPOSITION_PROMPT.format(
            task=task,
            agent_types=agent_types_desc
        )

        # 调用 LLM
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # 解析结果
        plan_data = self._parse_response(content)

        if plan_data:
            return self._build_plan(task, plan_data)

        # 回退：创建默认计划
        return self._create_default_plan(task)

    def _format_agent_types(self) -> str:
        """格式化 Agent 类型描述"""
        descriptions = {
            "researcher": "研究员 - 信息检索、数据分析、知识整合",
            "coder": "程序员 - 代码生成、调试、代码审查",
            "writer": "作家 - 内容创作、文档编写",
            "analyst": "分析师 - 质量评估、反馈生成",
            "architect": "架构师 - 系统设计、架构规划"
        }

        lines = []
        for agent_type in self.available_agents:
            desc = descriptions.get(agent_type, agent_type)
            lines.append(f"- {desc}")

        return "\n".join(lines)

    def _parse_response(self, content: str) -> Optional[Dict]:
        """解析 LLM 响应"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except:
            pass

        # 尝试提取 JSON 块
        import re
        patterns = [
            r'```json\s*(.+?)\s*```',
            r'```\s*(.+?)\s*```',
            r'\{[\s\S]*\}'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue

        return None

    def _build_plan(self, task: str, plan_data: Dict) -> ExecutionPlan:
        """构建执行计划"""
        # 解析执行模式
        mode_str = plan_data.get("execution_mode", "sequential")
        execution_mode = ExecutionMode(mode_str)

        # 构建子任务
        subtasks = []
        subtask_id_map = {}  # 用于处理依赖关系

        for i, st_data in enumerate(plan_data.get("subtasks", [])):
            subtask = Subtask(
                id=str(i + 1),
                name=st_data.get("name", f"子任务{i+1}"),
                description=st_data.get("description", ""),
                agent_type=st_data.get("agent_type", "researcher"),
                dependencies=[],  # 后续处理
                priority=st_data.get("priority", 5)
            )
            subtasks.append(subtask)
            subtask_id_map[subtask.name] = subtask.id

        # 处理依赖关系
        for i, st_data in enumerate(plan_data.get("subtasks", [])):
            deps = st_data.get("dependencies", [])
            # 将依赖名称转换为 ID
            dep_ids = []
            for dep in deps:
                if isinstance(dep, int):
                    dep_ids.append(str(dep))
                elif dep in subtask_id_map:
                    dep_ids.append(subtask_id_map[dep])
            subtasks[i].dependencies = dep_ids

        return ExecutionPlan(
            goal=task,
            subtasks=subtasks,
            execution_mode=execution_mode
        )

    def _create_default_plan(self, task: str) -> ExecutionPlan:
        """创建默认计划"""
        subtasks = [
            Subtask(
                id="1",
                name="分析任务",
                description=f"分析任务需求: {task}",
                agent_type="researcher",
                dependencies=[],
                priority=8
            ),
            Subtask(
                id="2",
                name="执行任务",
                description=task,
                agent_type="coder",
                dependencies=["1"],
                priority=5
            ),
            Subtask(
                id="3",
                name="审查结果",
                description="审查执行结果并提供反馈",
                agent_type="analyst",
                dependencies=["2"],
                priority=3
            )
        ]

        return ExecutionPlan(
            goal=task,
            subtasks=subtasks,
            execution_mode=ExecutionMode.SEQUENTIAL
        )


# ==================== 任务分配器 ====================

class TaskAssigner:
    """
    任务分配器

    将子任务分配给合适的 Agent
    """

    def __init__(self, agent_pool):
        """
        初始化任务分配器

        Args:
            agent_pool: Agent 池
        """
        self.agent_pool = agent_pool
        self.assignment_history: List[Dict] = []

    def assign(self, subtask: Subtask) -> Optional[str]:
        """
        分配子任务

        Args:
            subtask: 子任务

        Returns:
            分配的 Agent ID 或 None
        """
        # 根据类型查找 Agent
        from worker_agents import AgentCapability, TaskContext, TaskPriority

        # 构建能力映射
        capability_map = {
            "researcher": AgentCapability.INFORMATION_RETRIEVAL,
            "coder": AgentCapability.CODE_GENERATION,
            "writer": AgentCapability.CONTENT_WRITING,
            "analyst": AgentCapability.QUALITY_ASSESSMENT,
            "architect": AgentCapability.ARCHITECTURE_DESIGN
        }

        # 查找 Agent
        required_cap = capability_map.get(subtask.agent_type)
        if required_cap:
            candidates = self.agent_pool.find_by_capability(required_cap)

            # 选择最佳 Agent
            for agent in candidates:
                if agent.status.value == "idle":
                    agent_id = agent.profile.id

                    # 记录分配
                    self.assignment_history.append({
                        "subtask_id": subtask.id,
                        "agent_id": agent_id,
                        "timestamp": time.time()
                    })

                    return agent_id

        return None

    def get_assignment_stats(self) -> Dict:
        """获取分配统计"""
        agent_counts = {}
        for record in self.assignment_history:
            agent_id = record["agent_id"]
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

        return {
            "total_assignments": len(self.assignment_history),
            "agent_distribution": agent_counts
        }


# ==================== 结果整合器 ====================

class ResultSynthesizer:
    """
    结果整合器

    整合多个子任务的结果
    """

    SYNTHESIS_PROMPT = """你是一个结果整合专家。请整合以下子任务的执行结果。

## 总目标
{goal}

## 子任务结果
{results}

## 整合要求
1. 保持逻辑连贯性
2. 突出关键信息
3. 提供完整的最终答案

请输出整合后的结果："""

    def synthesize(self, plan: ExecutionPlan) -> str:
        """
        整合结果

        Args:
            plan: 执行计划

        Returns:
            整合后的结果
        """
        # 收集所有结果
        results = []
        for subtask in plan.subtasks:
            if subtask.result:
                results.append(f"### {subtask.name}\n{subtask.result}")

        if not results:
            return "没有可用的执行结果"

        # 调用 LLM 整合
        prompt = self.SYNTHESIS_PROMPT.format(
            goal=plan.goal,
            results="\n\n".join(results)
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content

    def simple_merge(self, plan: ExecutionPlan) -> str:
        """
        简单合并结果

        Args:
            plan: 执行计划

        Returns:
            合并后的结果
        """
        merged = []
        merged.append(f"# 任务执行报告\n")
        merged.append(f"## 目标\n{plan.goal}\n")
        merged.append(f"## 执行进度\n{plan.get_progress()[0]}/{plan.get_progress()[1]} 完成\n")
        merged.append("## 子任务结果\n")

        for subtask in plan.subtasks:
            status = "✓" if subtask.status == SubtaskStatus.COMPLETED else "✗"
            merged.append(f"\n### {status} {subtask.name}\n")
            if subtask.result:
                merged.append(str(subtask.result)[:500])
                if len(str(subtask.result)) > 500:
                    merged.append("...")

        return "\n".join(merged)


# ==================== 编排者 ====================

class Orchestrator:
    """
    编排者

    协调任务分解、分配和执行
    """

    def __init__(self, agent_pool, verbose: bool = True):
        """
        初始化编排者

        Args:
            agent_pool: Agent 池
            verbose: 是否输出详细日志
        """
        self.agent_pool = agent_pool
        self.verbose = verbose

        self.decomposer = TaskDecomposer()
        self.assigner = TaskAssigner(agent_pool)
        self.synthesizer = ResultSynthesizer()

        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_history: List[Dict] = []

    def analyze_task(self, task: str) -> ExecutionPlan:
        """
        分析任务

        Args:
            task: 任务描述

        Returns:
            执行计划
        """
        if self.verbose:
            print(f"\n[Orchestrator] 分析任务: {task}")

        # 分解任务
        plan = self.decomposer.decompose(task)
        self.current_plan = plan

        if self.verbose:
            print(f"\n执行计划 ({plan.execution_mode.value}):")
            for subtask in plan.subtasks:
                deps = f" (依赖: {subtask.dependencies})" if subtask.dependencies else ""
                print(f"  [{subtask.id}] {subtask.name} -> {subtask.agent_type}{deps}")

        return plan

    def assign_subtask(self, subtask: Subtask) -> bool:
        """
        分配子任务

        Args:
            subtask: 子任务

        Returns:
            是否成功分配
        """
        agent_id = self.assigner.assign(subtask)

        if agent_id:
            subtask.assigned_agent = agent_id
            subtask.status = SubtaskStatus.ASSIGNED

            if self.verbose:
                print(f"  分配: {subtask.name} -> {agent_id}")

            return True

        if self.verbose:
            print(f"  警告: 无法为 {subtask.name} 找到合适的 Agent")

        return False

    def execute_plan(self, plan: ExecutionPlan = None) -> Dict:
        """
        执行计划

        Args:
            plan: 执行计划 (可选，默认使用当前计划)

        Returns:
            执行结果
        """
        if plan:
            self.current_plan = plan

        if not self.current_plan:
            return {"status": "error", "message": "没有执行计划"}

        plan = self.current_plan

        if self.verbose:
            print(f"\n[Orchestrator] 开始执行计划...")
            print(f"目标: {plan.goal}")

        start_time = time.time()

        # 执行循环
        while not plan.is_complete():
            # 获取可执行的子任务
            ready_tasks = plan.get_ready_subtasks()

            if not ready_tasks:
                # 检查是否有失败的任务
                failed = [s for s in plan.subtasks
                         if s.status == SubtaskStatus.FAILED]
                if failed:
                    break

                # 检查是否有卡住的任务
                pending = [s for s in plan.subtasks
                          if s.status == SubtaskStatus.PENDING]
                if pending:
                    if self.verbose:
                        print("  警告: 存在无法执行的待处理任务")
                    break

                break

            # 根据执行模式处理
            if plan.execution_mode == ExecutionMode.PARALLEL:
                # 并行执行所有就绪任务
                for subtask in ready_tasks:
                    self._execute_subtask(subtask)
            else:
                # 串行执行
                subtask = ready_tasks[0]
                self._execute_subtask(subtask)

        # 整合结果
        final_result = self.synthesizer.synthesize(plan)

        execution_time = time.time() - start_time

        # 记录历史
        self.execution_history.append({
            "goal": plan.goal,
            "status": "completed" if plan.is_complete() else "partial",
            "execution_time": execution_time,
            "timestamp": time.time()
        })

        if self.verbose:
            completed, total = plan.get_progress()
            print(f"\n[Orchestrator] 执行完成: {completed}/{total}")
            print(f"耗时: {execution_time:.2f}s")

        return {
            "status": "completed" if plan.is_complete() else "partial",
            "result": final_result,
            "plan": plan.to_dict(),
            "execution_time": execution_time
        }

    def _execute_subtask(self, subtask: Subtask):
        """执行单个子任务"""
        from worker_agents import TaskContext

        # 分配 Agent
        if not self.assign_subtask(subtask):
            subtask.status = SubtaskStatus.FAILED
            return

        # 获取依赖结果
        dependencies = {}
        for dep_id in subtask.dependencies:
            dep_subtask = self.current_plan.get_subtask_by_id(dep_id)
            if dep_subtask and dep_subtask.result:
                dependencies[dep_subtask.name] = dep_subtask.result

        # 构建 TaskContext
        task_context = TaskContext(
            task_id=subtask.id,
            description=subtask.description,
            dependencies=dependencies
        )

        # 获取 Agent 并执行
        agent = self.agent_pool.get(subtask.assigned_agent)
        if agent:
            subtask.status = SubtaskStatus.RUNNING

            if self.verbose:
                print(f"\n  执行: {subtask.name} (by {subtask.assigned_agent})")

            result = agent.execute(task_context)

            subtask.result = result.output
            subtask.status = SubtaskStatus.COMPLETED if result.status == "success" else SubtaskStatus.FAILED

            if self.verbose:
                status = "✓" if subtask.status == SubtaskStatus.COMPLETED else "✗"
                print(f"  {status} 完成: {subtask.name}")
        else:
            subtask.status = SubtaskStatus.FAILED

    def run(self, task: str) -> Dict:
        """
        运行任务（分析+执行）

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        # 分析任务
        plan = self.analyze_task(task)

        # 执行计划
        return self.execute_plan(plan)

    def get_execution_status(self) -> Dict:
        """获取执行状态"""
        if not self.current_plan:
            return {"status": "no_plan"}

        completed, total = self.current_plan.get_progress()

        return {
            "goal": self.current_plan.goal,
            "progress": f"{completed}/{total}",
            "execution_mode": self.current_plan.execution_mode.value,
            "subtasks": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "assigned_to": s.assigned_agent
                }
                for s in self.current_plan.subtasks
            ]
        }


# ==================== 演示函数 ====================

def demo_task_decomposition():
    """演示任务分解"""
    print("\n" + "=" * 60)
    print("任务分解演示")
    print("=" * 60)

    decomposer = TaskDecomposer()

    task = """
    开发一个简单的待办事项应用：
    1. 研究现有类似应用的功能特点
    2. 设计应用架构
    3. 编写核心代码
    4. 撰写用户文档
    """

    plan = decomposer.decompose(task)

    print(f"\n目标: {plan.goal[:50]}...")
    print(f"执行模式: {plan.execution_mode.value}")
    print(f"\n子任务:")
    for subtask in plan.subtasks:
        deps = f" (依赖: {subtask.dependencies})" if subtask.dependencies else ""
        print(f"  [{subtask.id}] {subtask.name} -> {subtask.agent_type}{deps}")


def demo_orchestrator():
    """演示编排者"""
    print("\n" + "=" * 60)
    print("编排者演示")
    print("=" * 60)

    from worker_agents import AgentPool, AgentFactory

    # 创建 Agent 池
    pool = AgentPool()

    # 注册 Agent
    for agent_type in ["researcher", "coder", "writer", "analyst"]:
        agent = AgentFactory.create(agent_type)
        pool.register(agent)

    # 创建编排者
    orchestrator = Orchestrator(pool, verbose=True)

    # 执行任务
    task = "研究 Python 装饰器的用法，编写一个计时装饰器，并撰写使用说明"

    result = orchestrator.run(task)

    print(f"\n[最终结果]")
    print(result["result"][:500] + "...")

    print(f"\n[执行状态]")
    status = orchestrator.get_execution_status()
    print(f"进度: {status['progress']}")


if __name__ == "__main__":
    demo_task_decomposition()
    demo_orchestrator()