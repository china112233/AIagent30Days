"""
工作者 Agent 实现
包含基类定义、专家型 Agent、能力声明和任务执行
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable
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


# ==================== 枚举和常量 ====================

class AgentCapability(Enum):
    """Agent 能力枚举"""
    # 信息处理
    INFORMATION_RETRIEVAL = "information_retrieval"
    DATA_ANALYSIS = "data_analysis"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"

    # 编程开发
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"

    # 内容创作
    CONTENT_WRITING = "content_writing"
    DOCUMENTATION = "documentation"
    TRANSLATION = "translation"

    # 规划设计
    TASK_PLANNING = "task_planning"
    ARCHITECTURE_DESIGN = "architecture_design"

    # 评估审核
    QUALITY_ASSESSMENT = "quality_assessment"
    FEEDBACK_GENERATION = "feedback_generation"


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# ==================== 数据结构 ====================

@dataclass
class AgentProfile:
    """
    Agent 档案

    Attributes:
        id: Agent 唯一标识
        name: Agent 名称
        role: 角色描述
        capabilities: 能力列表
        specialties: 专业领域
        max_concurrent_tasks: 最大并发任务数
        performance_score: 性能评分 (0-1)
    """
    id: str
    name: str
    role: str
    capabilities: List[AgentCapability]
    specialties: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    performance_score: float = 0.8


@dataclass
class TaskContext:
    """
    任务上下文

    Attributes:
        task_id: 任务 ID
        description: 任务描述
        requirements: 需求列表
        dependencies: 依赖任务结果
        constraints: 约束条件
        priority: 优先级
        deadline: 截止时间
    """
    task_id: str
    description: str
    requirements: List[str] = field(default_factory=list)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[float] = None


@dataclass
class TaskResult:
    """
    任务结果

    Attributes:
        task_id: 任务 ID
        agent_id: 执行 Agent ID
        status: 执行状态
        output: 输出结果
        artifacts: 生成的制品
        metrics: 执行指标
        feedback: 反馈信息
        execution_time: 执行时间
    """
    task_id: str
    agent_id: str
    status: str  # success, failed, partial
    output: Any = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    execution_time: float = 0.0


# ==================== 基础 Agent 类 ====================

class BaseWorkerAgent:
    """
    基础工作者 Agent 类

    所有专业 Agent 的基类，定义通用接口和行为
    """

    def __init__(self, profile: AgentProfile):
        """
        初始化 Agent

        Args:
            profile: Agent 档案
        """
        self.profile = profile
        self.status = AgentStatus.IDLE
        self.current_tasks: List[str] = []
        self.task_history: List[TaskResult] = []
        self.communication_manager = None

    def set_communication_manager(self, manager):
        """设置通信管理器"""
        self.communication_manager = manager

    def can_handle(self, task: TaskContext) -> float:
        """
        判断是否能处理任务

        Args:
            task: 任务上下文

        Returns:
            处理能力评分 (0-1)
        """
        score = 0.0

        # 检查能力匹配
        required_caps = self._extract_required_capabilities(task)
        for cap in required_caps:
            if cap in self.profile.capabilities:
                score += 0.2

        # 检查专业领域匹配
        for specialty in self.profile.specialties:
            if specialty.lower() in task.description.lower():
                score += 0.2

        return min(score, 1.0)

    def _extract_required_capabilities(self, task: TaskContext) -> List[AgentCapability]:
        """从任务中提取所需能力"""
        caps = []
        desc_lower = task.description.lower()

        capability_keywords = {
            AgentCapability.INFORMATION_RETRIEVAL: ["搜索", "查找", "检索", "search", "find"],
            AgentCapability.DATA_ANALYSIS: ["分析", "数据", "analysis", "data"],
            AgentCapability.CODE_GENERATION: ["编写", "代码", "开发", "code", "develop"],
            AgentCapability.CONTENT_WRITING: ["写作", "撰写", "文档", "write", "document"],
            AgentCapability.DEBUGGING: ["调试", "修复", "bug", "debug", "fix"],
        }

        for cap, keywords in capability_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                caps.append(cap)

        return caps

    def execute(self, task: TaskContext) -> TaskResult:
        """
        执行任务

        Args:
            task: 任务上下文

        Returns:
            任务结果
        """
        start_time = time.time()

        # 检查状态
        if self.status == AgentStatus.BUSY:
            if len(self.current_tasks) >= self.profile.max_concurrent_tasks:
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.profile.id,
                    status="failed",
                    feedback="Agent 正忙，无法接受新任务"
                )

        # 更新状态
        self.status = AgentStatus.BUSY
        self.current_tasks.append(task.task_id)

        try:
            # 执行具体任务
            result = self._do_execute(task)

            # 更新状态
            result.execution_time = time.time() - start_time
            self.task_history.append(result)

            return result

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.profile.id,
                status="failed",
                feedback=f"执行错误: {str(e)}",
                execution_time=time.time() - start_time
            )

        finally:
            if task.task_id in self.current_tasks:
                self.current_tasks.remove(task.task_id)
            if not self.current_tasks:
                self.status = AgentStatus.IDLE

    def _do_execute(self, task: TaskContext) -> TaskResult:
        """
        具体执行逻辑（子类实现）

        Args:
            task: 任务上下文

        Returns:
            任务结果
        """
        raise NotImplementedError("子类必须实现此方法")

    def get_status(self) -> Dict[str, Any]:
        """获取 Agent 状态"""
        return {
            "id": self.profile.id,
            "name": self.profile.name,
            "status": self.status.value,
            "current_tasks": len(self.current_tasks),
            "performance_score": self.profile.performance_score
        }

    def send_message(self, receiver: str, content: Dict[str, Any], msg_type):
        """发送消息"""
        if self.communication_manager:
            from communication import Message
            msg = Message(
                sender=self.profile.id,
                receiver=receiver,
                type=msg_type,
                content=content
            )
            self.communication_manager.send_message(msg)

    def receive_message(self, timeout: float = 1.0):
        """接收消息"""
        if self.communication_manager:
            return self.communication_manager.receive_message(self.profile.id, timeout)
        return None


# ==================== 专家型 Agent ====================

class ResearcherAgent(BaseWorkerAgent):
    """
    研究专家 Agent

    负责信息检索、数据分析、知识整合
    """

    RESEARCH_PROMPT = """你是一个专业的研究员。请根据任务要求进行研究分析。

任务: {task_description}

要求:
1. 系统性地收集相关信息
2. 分析关键要点
3. 整合成清晰的报告

请提供研究结果:"""

    def __init__(self, agent_id: str = "researcher"):
        profile = AgentProfile(
            id=agent_id,
            name="研究员",
            role="负责信息检索、分析和知识整合",
            capabilities=[
                AgentCapability.INFORMATION_RETRIEVAL,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            ],
            specialties=["数据分析", "文献研究", "市场调研", "技术调研"]
        )
        super().__init__(profile)

    def _do_execute(self, task: TaskContext) -> TaskResult:
        """执行研究任务"""
        # 构建研究提示
        prompt = self.RESEARCH_PROMPT.format(
            task_description=task.description
        )

        # 添加依赖信息
        if task.dependencies:
            prompt += f"\n\n前置任务结果:\n{json.dumps(task.dependencies, ensure_ascii=False, indent=2)}"

        # 调用 LLM
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        result_content = response.choices[0].message.content

        return TaskResult(
            task_id=task.task_id,
            agent_id=self.profile.id,
            status="success",
            output=result_content,
            artifacts={"research_report": result_content},
            metrics={"response_length": len(result_content)}
        )


class CoderAgent(BaseWorkerAgent):
    """
    编程专家 Agent

    负责代码生成、调试、代码审查
    """

    CODING_PROMPT = """你是一个专业的程序员。请根据任务要求编写代码。

任务: {task_description}

要求:
1. 代码清晰、规范
2. 添加必要的注释
3. 考虑错误处理

请提供代码实现:"""

    def __init__(self, agent_id: str = "coder"):
        profile = AgentProfile(
            id=agent_id,
            name="程序员",
            role="负责代码开发、调试和审查",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.DEBUGGING,
                AgentCapability.CODE_REVIEW
            ],
            specialties=["Python", "JavaScript", "后端开发", "前端开发", "API开发"]
        )
        super().__init__(profile)

    def _do_execute(self, task: TaskContext) -> TaskResult:
        """执行编程任务"""
        prompt = self.CODING_PROMPT.format(
            task_description=task.description
        )

        # 添加约束
        if task.constraints:
            prompt += f"\n\n约束条件:\n{json.dumps(task.constraints, ensure_ascii=False, indent=2)}"

        # 添加依赖
        if task.dependencies:
            prompt += f"\n\n参考上下文:\n{json.dumps(task.dependencies, ensure_ascii=False, indent=2)}"

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        result_content = response.choices[0].message.content

        # 提取代码块
        code_blocks = self._extract_code_blocks(result_content)

        return TaskResult(
            task_id=task.task_id,
            agent_id=self.profile.id,
            status="success",
            output=result_content,
            artifacts={"code": code_blocks, "full_response": result_content},
            metrics={"code_blocks": len(code_blocks)}
        )

    def _extract_code_blocks(self, text: str) -> List[str]:
        """提取代码块"""
        import re
        pattern = r'```(?:\w+)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches]


class WriterAgent(BaseWorkerAgent):
    """
    写作专家 Agent

    负责内容创作、文档编写
    """

    WRITING_PROMPT = """你是一个专业的内容创作者。请根据任务要求创作内容。

任务: {task_description}

要求:
1. 内容清晰、有逻辑
2. 语言流畅、易读
3. 结构合理

请创作内容:"""

    def __init__(self, agent_id: str = "writer"):
        profile = AgentProfile(
            id=agent_id,
            name="作家",
            role="负责内容创作和文档编写",
            capabilities=[
                AgentCapability.CONTENT_WRITING,
                AgentCapability.DOCUMENTATION,
                AgentCapability.TRANSLATION
            ],
            specialties=["技术文档", "用户手册", "博客文章", "报告撰写"]
        )
        super().__init__(profile)

    def _do_execute(self, task: TaskContext) -> TaskResult:
        """执行写作任务"""
        prompt = self.WRITING_PROMPT.format(
            task_description=task.description
        )

        # 添加参考信息
        if task.dependencies:
            prompt += f"\n\n参考资料:\n{json.dumps(task.dependencies, ensure_ascii=False, indent=2)}"

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )

        result_content = response.choices[0].message.content

        return TaskResult(
            task_id=task.task_id,
            agent_id=self.profile.id,
            status="success",
            output=result_content,
            artifacts={"document": result_content},
            metrics={"word_count": len(result_content)}
        )


class AnalystAgent(BaseWorkerAgent):
    """
    分析专家 Agent

    负责质量评估、反馈生成
    """

    ANALYSIS_PROMPT = """你是一个专业的分析师。请分析以下内容:

内容: {content}

分析维度:
1. 完整性
2. 准确性
3. 清晰度
4. 改进建议

请提供分析报告:"""

    def __init__(self, agent_id: str = "analyst"):
        profile = AgentProfile(
            id=agent_id,
            name="分析师",
            role="负责质量评估和反馈生成",
            capabilities=[
                AgentCapability.QUALITY_ASSESSMENT,
                AgentCapability.FEEDBACK_GENERATION,
                AgentCapability.DATA_ANALYSIS
            ],
            specialties=["质量评估", "代码审查", "内容审核", "数据分析"]
        )
        super().__init__(profile)

    def _do_execute(self, task: TaskContext) -> TaskResult:
        """执行分析任务"""
        content = task.description
        if task.dependencies:
            content = json.dumps(task.dependencies, ensure_ascii=False, indent=2)

        prompt = self.ANALYSIS_PROMPT.format(content=content)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        result_content = response.choices[0].message.content

        return TaskResult(
            task_id=task.task_id,
            agent_id=self.profile.id,
            status="success",
            output=result_content,
            artifacts={"analysis_report": result_content},
            metrics={"report_length": len(result_content)}
        )


class ArchitectAgent(BaseWorkerAgent):
    """
    架构师 Agent

    负责系统设计、架构规划
    """

    ARCHITECTURE_PROMPT = """你是一个系统架构师。请根据需求设计系统架构。

需求: {requirements}

设计要求:
1. 清晰的架构图描述
2. 关键组件说明
3. 技术选型理由
4. 扩展性考虑

请提供架构设计:"""

    def __init__(self, agent_id: str = "architect"):
        profile = AgentProfile(
            id=agent_id,
            name="架构师",
            role="负责系统架构设计和技术规划",
            capabilities=[
                AgentCapability.ARCHITECTURE_DESIGN,
                AgentCapability.TASK_PLANNING
            ],
            specialties=["系统架构", "微服务", "分布式系统", "API设计"]
        )
        super().__init__(profile)

    def _do_execute(self, task: TaskContext) -> TaskResult:
        """执行架构设计任务"""
        prompt = self.ARCHITECTURE_PROMPT.format(
            requirements=task.description
        )

        if task.constraints:
            prompt += f"\n\n约束条件:\n{json.dumps(task.constraints, ensure_ascii=False, indent=2)}"

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )

        result_content = response.choices[0].message.content

        return TaskResult(
            task_id=task.task_id,
            agent_id=self.profile.id,
            status="success",
            output=result_content,
            artifacts={"architecture_design": result_content},
            metrics={"design_length": len(result_content)}
        )


# ==================== Agent 工厂 ====================

class AgentFactory:
    """
    Agent 工厂

    用于创建和管理 Agent 实例
    """

    _agent_classes = {
        "researcher": ResearcherAgent,
        "coder": CoderAgent,
        "writer": WriterAgent,
        "analyst": AnalystAgent,
        "architect": ArchitectAgent
    }

    @classmethod
    def create(cls, agent_type: str, agent_id: str = None) -> BaseWorkerAgent:
        """
        创建 Agent 实例

        Args:
            agent_type: Agent 类型
            agent_id: 自定义 ID (可选)

        Returns:
            Agent 实例
        """
        if agent_type not in cls._agent_classes:
            raise ValueError(f"未知的 Agent 类型: {agent_type}")

        agent_class = cls._agent_classes[agent_type]
        return agent_class(agent_id or agent_type)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取可用的 Agent 类型"""
        return list(cls._agent_classes.keys())

    @classmethod
    def register(cls, agent_type: str, agent_class: type):
        """
        注册新的 Agent 类型

        Args:
            agent_type: Agent 类型名称
            agent_class: Agent 类
        """
        cls._agent_classes[agent_type] = agent_class


# ==================== Agent 池 ====================

class AgentPool:
    """
    Agent 池

    管理多个 Agent 实例，支持负载均衡
    """

    def __init__(self):
        """初始化 Agent 池"""
        self.agents: Dict[str, BaseWorkerAgent] = {}
        self.capability_index: Dict[AgentCapability, List[str]] = {}

    def register(self, agent: BaseWorkerAgent):
        """
        注册 Agent

        Args:
            agent: Agent 实例
        """
        self.agents[agent.profile.id] = agent

        # 建立能力索引
        for cap in agent.profile.capabilities:
            if cap not in self.capability_index:
                self.capability_index[cap] = []
            self.capability_index[cap].append(agent.profile.id)

    def unregister(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            # 移除能力索引
            for cap in agent.profile.capabilities:
                if cap in self.capability_index and agent_id in self.capability_index[cap]:
                    self.capability_index[cap].remove(agent_id)
            del self.agents[agent_id]

    def get(self, agent_id: str) -> Optional[BaseWorkerAgent]:
        """获取 Agent"""
        return self.agents.get(agent_id)

    def find_by_capability(self, capability: AgentCapability) -> List[BaseWorkerAgent]:
        """根据能力查找 Agent"""
        agent_ids = self.capability_index.get(capability, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def find_best_for_task(self, task: TaskContext) -> Optional[BaseWorkerAgent]:
        """
        为任务找到最佳 Agent

        Args:
            task: 任务上下文

        Returns:
            最佳 Agent 或 None
        """
        best_agent = None
        best_score = 0

        for agent in self.agents.values():
            # 检查状态
            if agent.status == AgentStatus.BUSY:
                if len(agent.current_tasks) >= agent.profile.max_concurrent_tasks:
                    continue

            # 计算匹配分数
            score = agent.can_handle(task)

            # 考虑性能分数
            score *= agent.profile.performance_score

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    def get_all_status(self) -> List[Dict[str, Any]]:
        """获取所有 Agent 状态"""
        return [agent.get_status() for agent in self.agents.values()]


# ==================== 演示函数 ====================

def demo_worker_agents():
    """演示工作者 Agent"""
    print("\n" + "=" * 60)
    print("工作者 Agent 演示")
    print("=" * 60)

    # 创建 Agent
    researcher = ResearcherAgent()
    coder = CoderAgent()
    writer = WriterAgent()

    print("\n[1] Agent 信息:")
    for agent in [researcher, coder, writer]:
        print(f"  {agent.profile.name}: {agent.profile.role}")
        print(f"    能力: {[c.value for c in agent.profile.capabilities]}")

    # 执行任务
    print("\n[2] 执行任务:")

    # 研究任务
    task1 = TaskContext(
        task_id="T001",
        description="研究 Python 异步编程的最佳实践"
    )

    result1 = researcher.execute(task1)
    print(f"\n  研究员执行结果 (状态: {result1.status}):")
    print(f"    {result1.output[:200]}...")

    # 编程任务
    task2 = TaskContext(
        task_id="T002",
        description="编写一个简单的异步 HTTP 请求函数",
        dependencies={"研究结果": result1.output[:100]}
    )

    result2 = coder.execute(task2)
    print(f"\n  程序员执行结果 (状态: {result2.status}):")
    print(f"    代码块数量: {result2.metrics.get('code_blocks', 0)}")


def demo_agent_pool():
    """演示 Agent 池"""
    print("\n" + "=" * 60)
    print("Agent 池演示")
    print("=" * 60)

    pool = AgentPool()

    # 注册 Agent
    print("\n[1] 注册 Agent:")
    for agent_type in AgentFactory.get_available_types():
        agent = AgentFactory.create(agent_type)
        pool.register(agent)
        print(f"  已注册: {agent.profile.name}")

    # 查找 Agent
    print("\n[2] 按能力查找:")
    agents = pool.find_by_capability(AgentCapability.CODE_GENERATION)
    print(f"  有代码生成能力的 Agent: {[a.profile.name for a in agents]}")

    # 任务匹配
    print("\n[3] 任务匹配:")
    task = TaskContext(
        task_id="T003",
        description="编写一个数据分析脚本"
    )
    best_agent = pool.find_best_for_task(task)
    if best_agent:
        print(f"  最佳 Agent: {best_agent.profile.name}")

    # 状态
    print("\n[4] Agent 状态:")
    for status in pool.get_all_status():
        print(f"  {status['name']}: {status['status']}")


if __name__ == "__main__":
    demo_worker_agents()
    demo_agent_pool()