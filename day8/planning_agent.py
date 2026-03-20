"""
规划型 Agent 模块
实现任务分解、计划生成、执行监控和动态重规划
"""

import uuid
import time
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
import re

# 导入记忆系统
from agent_memory import MemoryManager, MemoryItem, MemoryType


# ==================== 任务状态枚举 ====================

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"           # 待执行
    READY = "ready"               # 准备就绪
    RUNNING = "running"           # 执行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消
    BLOCKED = "blocked"           # 被阻塞


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ==================== 任务数据结构 ====================

@dataclass
class Task:
    """
    任务数据结构
    
    Attributes:
        id: 任务唯一ID
        name: 任务名称
        description: 任务描述
        status: 任务状态
        priority: 优先级
        dependencies: 依赖任务ID列表
        subtasks: 子任务ID列表
        parent_id: 父任务ID
        estimated_time: 预估时间（秒）
        actual_time: 实际时间
        result: 执行结果
        error: 错误信息
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
        metadata: 元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    estimated_time: Optional[float] = None
    actual_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """
        检查任务是否准备就绪
        
        Args:
            completed_tasks: 已完成的任务ID集合
            
        Returns:
            是否准备就绪
        """
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def start(self) -> None:
        """开始执行任务"""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
    
    def complete(self, result: Any = None) -> None:
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()
        self.actual_time = self.completed_at - (self.started_at or self.created_at)
    
    def fail(self, error: str) -> None:
        """任务失败"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()
    
    def cancel(self) -> None:
        """取消任务"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "subtasks": self.subtasks,
            "parent_id": self.parent_id,
            "estimated_time": self.estimated_time,
            "actual_time": self.actual_time,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 2)),
            dependencies=data.get("dependencies", []),
            subtasks=data.get("subtasks", []),
            parent_id=data.get("parent_id"),
            estimated_time=data.get("estimated_time"),
            actual_time=data.get("actual_time"),
            result=data.get("result"),
            error=data.get("error"),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {})
        )


# ==================== 计划数据结构 ====================

@dataclass
class Plan:
    """
    计划数据结构
    
    Attributes:
        id: 计划唯一ID
        goal: 目标描述
        tasks: 任务字典 {task_id: Task}
        execution_order: 执行顺序
        current_task: 当前执行的任务ID
        status: 计划状态
        created_at: 创建时间
        updated_at: 更新时间
        metadata: 元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.id] = task
        self.updated_at = time.time()
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            if task_id in self.execution_order:
                self.execution_order.remove(task_id)
            self.updated_at = time.time()
            return True
        return False
    
    def get_ready_tasks(self) -> List[Task]:
        """
        获取准备就绪的任务
        
        Returns:
            准备就绪的任务列表
        """
        completed_ids = {
            tid for tid, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }
        
        ready = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and task.is_ready(completed_ids):
                ready.append(task)
        
        # 按优先级排序
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready
    
    def get_progress(self) -> Dict[str, Any]:
        """
        获取进度信息
        
        Returns:
            进度统计
        """
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            "total": total,
            "completed": completed,
            "running": running,
            "failed": failed,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }
    
    def is_complete(self) -> bool:
        """计划是否完成"""
        return all(
            t.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]
            for t in self.tasks.values()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "execution_order": self.execution_order,
            "current_task": self.current_task,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }


# ==================== 规划策略 ====================

class PlanningStrategy(Enum):
    """规划策略枚举"""
    HIERARCHICAL = "hierarchical"     # 层次分解
    SEQUENTIAL = "sequential"         # 顺序执行
    PARALLEL = "parallel"             # 并行执行
    ADAPTIVE = "adaptive"             # 自适应


# ==================== 规划型 Agent ====================

class PlanningAgent:
    """
    规划型 Agent
    
    实现：
    - 目标理解和任务分解
    - 计划生成和优化
    - 执行监控和动态重规划
    - 与记忆系统集成
    """
    
    def __init__(self, 
                 memory: Optional[MemoryManager] = None,
                 strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL,
                 max_depth: int = 3):
        """
        初始化规划 Agent
        
        Args:
            memory: 记忆管理器
            strategy: 规划策略
            max_depth: 最大分解深度
        """
        self.memory = memory or MemoryManager()
        self.strategy = strategy
        self.max_depth = max_depth
        
        # 存储的计划
        self.plans: Dict[str, Plan] = {}
        
        # 任务执行器注册表
        self._executors: Dict[str, Callable] = {}
        
        # 规划模板
        self._templates = self._init_templates()
    
    def _init_templates(self) -> Dict[str, Dict]:
        """初始化规划模板"""
        return {
            "report": {
                "subtasks": [
                    {"name": "收集信息", "description": "收集相关资料和数据"},
                    {"name": "分析数据", "description": "整理和分析收集的数据"},
                    {"name": "撰写报告", "description": "编写报告内容"},
                    {"name": "审核修改", "description": "检查并优化报告"}
                ],
                "dependencies": [
                    [],  # 收集信息无依赖
                    ["收集信息"],  # 分析数据依赖收集信息
                    ["分析数据"],  # 撰写报告依赖分析数据
                    ["撰写报告"]   # 审核修改依赖撰写报告
                ]
            },
            "research": {
                "subtasks": [
                    {"name": "确定研究方向", "description": "明确研究问题和范围"},
                    {"name": "文献调研", "description": "查阅相关文献资料"},
                    {"name": "方法设计", "description": "设计研究方法"},
                    {"name": "数据分析", "description": "分析研究数据"},
                    {"name": "撰写论文", "description": "编写研究成果"}
                ],
                "dependencies": [
                    [],
                    ["确定研究方向"],
                    ["文献调研", "确定研究方向"],
                    ["方法设计"],
                    ["数据分析"]
                ]
            },
            "coding": {
                "subtasks": [
                    {"name": "需求分析", "description": "理解功能需求"},
                    {"name": "设计架构", "description": "设计系统架构"},
                    {"name": "编码实现", "description": "编写代码"},
                    {"name": "测试验证", "description": "测试功能正确性"},
                    {"name": "优化重构", "description": "优化代码质量"}
                ],
                "dependencies": [
                    [],
                    ["需求分析"],
                    ["设计架构"],
                    ["编码实现"],
                    ["测试验证"]
                ]
            },
            "general": {
                "subtasks": [
                    {"name": "理解目标", "description": "明确任务目标"},
                    {"name": "制定计划", "description": "规划执行步骤"},
                    {"name": "执行任务", "description": "按计划执行"},
                    {"name": "验证结果", "description": "检查执行结果"}
                ],
                "dependencies": [
                    [],
                    ["理解目标"],
                    ["制定计划"],
                    ["执行任务"]
                ]
            }
        }
    
    def register_executor(self, task_type: str, executor: Callable) -> None:
        """
        注册任务执行器
        
        Args:
            task_type: 任务类型
            executor: 执行函数
        """
        self._executors[task_type] = executor
    
    def plan(self, goal: str, context: Optional[Dict] = None) -> Plan:
        """
        根据目标生成计划
        
        Args:
            goal: 目标描述
            context: 上下文信息
            
        Returns:
            生成的计划
        """
        # 从记忆中检索相关经验
        relevant_memories = self.memory.recall(goal, limit=5)
        
        # 确定任务类型
        task_type = self._classify_goal(goal)
        
        # 分解任务
        plan = self._decompose_goal(goal, task_type, context)
        
        # 存储计划
        self.plans[plan.id] = plan
        
        # 记忆存储
        self.memory.remember(
            content=f"创建了计划: {goal}",
            memory_type="episodic",
            importance=0.6,
            metadata={"plan_id": plan.id, "task_type": task_type}
        )
        
        return plan
    
    def _classify_goal(self, goal: str) -> str:
        """
        分类目标类型
        
        Args:
            goal: 目标描述
            
        Returns:
            任务类型
        """
        goal_lower = goal.lower()
        
        # 关键词匹配
        if any(kw in goal_lower for kw in ["报告", "总结", "汇报", "文档"]):
            return "report"
        elif any(kw in goal_lower for kw in ["研究", "调研", "分析", "调查"]):
            return "research"
        elif any(kw in goal_lower for kw in ["编程", "代码", "开发", "实现", "编写程序"]):
            return "coding"
        else:
            return "general"
    
    def _decompose_goal(self, goal: str, task_type: str, context: Optional[Dict]) -> Plan:
        """
        分解目标为任务
        
        Args:
            goal: 目标描述
            task_type: 任务类型
            context: 上下文
            
        Returns:
            生成的计划
        """
        plan = Plan(goal=goal)
        
        # 获取模板
        template = self._templates.get(task_type, self._templates["general"])
        
        # 创建任务
        task_id_map = {}
        for i, subtask_def in enumerate(template["subtasks"]):
            task = Task(
                name=subtask_def["name"],
                description=subtask_def["description"],
                priority=TaskPriority.MEDIUM,
                metadata={"index": i}
            )
            plan.add_task(task)
            task_id_map[subtask_def["name"]] = task.id
        
        # 设置依赖关系
        for i, dep_names in enumerate(template["dependencies"]):
            task_name = template["subtasks"][i]["name"]
            task_id = task_id_map[task_name]
            
            for dep_name in dep_names:
                if dep_name in task_id_map:
                    plan.tasks[task_id].dependencies.append(task_id_map[dep_name])
        
        # 计算执行顺序（拓扑排序）
        plan.execution_order = self._topological_sort(plan)
        
        # 设置状态
        plan.status = "ready"
        
        return plan
    
    def _topological_sort(self, plan: Plan) -> List[str]:
        """
        拓扑排序确定执行顺序
        
        Args:
            plan: 计划
            
        Returns:
            任务ID列表（执行顺序）
        """
        # 计算入度
        in_degree = {tid: 0 for tid in plan.tasks}
        for task in plan.tasks.values():
            for dep_id in task.dependencies:
                if dep_id in in_degree:
                    in_degree[task.id] += 1
        
        # 广度优先搜索
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # 按优先级排序
            queue.sort(key=lambda tid: plan.tasks[tid].priority.value, reverse=True)
            
            current = queue.pop(0)
            result.append(current)
            
            # 更新依赖此任务的节点的入度
            for tid, task in plan.tasks.items():
                if current in task.dependencies:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0 and tid not in result:
                        queue.append(tid)
        
        return result
    
    def replan(self, plan: Plan, reason: str = "") -> Plan:
        """
        动态重规划
        
        Args:
            plan: 原计划
            reason: 重规划原因
            
        Returns:
            新计划
        """
        # 分析当前状态
        progress = plan.get_progress()
        
        # 创建新计划
        new_plan = Plan(
            goal=plan.goal,
            metadata={
                "original_plan_id": plan.id,
                "replan_reason": reason,
                "previous_progress": progress
            }
        )
        
        # 复制未完成的任务
        for task_id, task in plan.tasks.items():
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
                new_task = Task(
                    name=task.name,
                    description=task.description,
                    priority=task.priority,
                    dependencies=task.dependencies.copy(),
                    metadata=task.metadata.copy()
                )
                new_plan.add_task(new_task)
        
        # 重新计算执行顺序
        new_plan.execution_order = self._topological_sort(new_plan)
        new_plan.status = "ready"
        
        # 存储新计划
        self.plans[new_plan.id] = new_plan
        
        # 记忆存储
        self.memory.remember(
            content=f"重规划: {reason}",
            memory_type="episodic",
            importance=0.7,
            metadata={"plan_id": new_plan.id}
        )
        
        return new_plan
    
    def get_next_task(self, plan: Plan) -> Optional[Task]:
        """
        获取下一个待执行的任务
        
        Args:
            plan: 计划
            
        Returns:
            下一个任务或None
        """
        ready_tasks = plan.get_ready_tasks()
        return ready_tasks[0] if ready_tasks else None
    
    def estimate_completion_time(self, plan: Plan) -> float:
        """
        估算完成时间
        
        Args:
            plan: 计划
            
        Returns:
            估算时间（秒）
        """
        total = 0.0
        for task in plan.tasks.values():
            if task.status != TaskStatus.COMPLETED:
                if task.estimated_time:
                    total += task.estimated_time
                else:
                    # 默认估算
                    total += 60  # 默认1分钟
        
        return total
    
    def get_plan_summary(self, plan: Plan) -> str:
        """
        获取计划摘要
        
        Args:
            plan: 计划
            
        Returns:
            摘要字符串
        """
        progress = plan.get_progress()
        lines = [
            f"目标: {plan.goal}",
            f"状态: {plan.status}",
            f"进度: {progress['completed']}/{progress['total']} ({progress['progress_percent']:.1f}%)",
            "",
            "任务列表:"
        ]
        
        for i, task_id in enumerate(plan.execution_order, 1):
            task = plan.tasks.get(task_id)
            if task:
                status_icon = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.CANCELLED: "🚫"
                }.get(task.status, "❓")
                
                lines.append(f"  {i}. {status_icon} {task.name}: {task.description}")
        
        return "\n".join(lines)


# ==================== 示例和测试 ====================

def demo_planning_agent():
    """演示规划 Agent"""
    print("=" * 60)
    print("规划型 Agent 演示")
    print("=" * 60)
    
    # 创建规划 Agent
    planner = PlanningAgent(strategy=PlanningStrategy.HIERARCHICAL)
    
    # 1. 生成计划
    print("\n1. 生成计划")
    print("-" * 40)
    
    goal = "帮我准备一份季度销售报告"
    plan = planner.plan(goal)
    
    print(f"目标: {goal}")
    print(f"计划ID: {plan.id}")
    print(f"任务数量: {len(plan.tasks)}")
    
    # 2. 查看计划详情
    print("\n2. 计划详情")
    print("-" * 40)
    print(planner.get_plan_summary(plan))
    
    # 3. 模拟执行
    print("\n3. 模拟任务执行")
    print("-" * 40)
    
    # 获取下一个任务
    task = planner.get_next_task(plan)
    if task:
        print(f"执行任务: {task.name}")
        task.start()
        # 模拟执行
        time.sleep(0.1)
        task.complete(result="已完成信息收集")
        print(f"任务状态: {task.status.value}")
    
    # 查看进度
    print("\n进度:")
    progress = plan.get_progress()
    print(f"  已完成: {progress['completed']}/{progress['total']}")
    print(f"  进度: {progress['progress_percent']:.1f}%")
    
    # 4. 动态重规划
    print("\n4. 动态重规划")
    print("-" * 40)
    
    new_plan = planner.replan(plan, reason="需求变更，需要添加市场分析")
    print(f"新计划ID: {new_plan.id}")
    print(f"新计划任务数: {len(new_plan.tasks)}")
    
    # 5. 不同类型的计划
    print("\n5. 不同类型的计划")
    print("-" * 40)
    
    goals = [
        "开发一个用户登录功能",
        "调研人工智能在教育领域的应用",
        "组织一次团队活动"
    ]
    
    for g in goals:
        p = planner.plan(g)
        task_type = planner._classify_goal(g)
        print(f"\n目标: {g}")
        print(f"类型: {task_type}")
        print(f"任务: {[t.name for t in p.tasks.values()]}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_planning_agent()