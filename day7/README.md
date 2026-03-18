# Day 7: Multi-Agent Systems（多智能体系统）

## 概述

第七天学习多智能体系统（Multi-Agent Systems, MAS），让多个 Agent 协作完成复杂任务。多智能体系统通过协调、分工和通信，实现单个 Agent 难以完成的复杂目标。

## 学习目标

- 理解多智能体系统的核心架构和设计模式
- 掌握 Agent 间的通信机制和消息传递
- 学会任务分解和智能分配
- 实现协作工作流和共识机制
- 构建可扩展的多 Agent 协作系统

## 核心概念

### 1. 什么是多智能体系统？

多智能体系统是由多个智能 Agent 组成的系统，它们通过协作、竞争或协商来完成任务：

```
用户请求 → Orchestrator（编排者）→ 分解任务 → 分配给多个 Agent → 协作执行 → 汇总结果
```

**与单 Agent 的对比**：

| 特性 | 单 Agent | 多 Agent 系统 |
|------|----------|---------------|
| 任务复杂度 | 简单任务 | 复杂、多领域任务 |
| 专业性 | 通用能力 | 专业化分工 |
| 容错性 | 单点故障 | 冗余和备份 |
| 可扩展性 | 有限 | 水平扩展 |
| 解决方案 | 单一视角 | 多角度分析 |

### 2. 多 Agent 架构模式

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent System 架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         Orchestrator                               │ │
│  │                    (任务编排与协调中心)                              │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  任务分析 │ 任务分解 │ Agent分配 │ 进度监控 │ 结果汇总              │ │
│  └───────────────────────────────┬───────────────────────────────────┘ │
│                                  │                                      │
│              ┌───────────────────┼───────────────────┐                  │
│              │                   │                   │                  │
│              ▼                   ▼                   ▼                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Research Agent │  │  Coder Agent    │  │  Writer Agent   │         │
│  │  (研究专家)      │  │  (编程专家)      │  │  (写作专家)     │         │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤         │
│  │ - 信息搜索      │  │ - 代码生成      │  │ - 内容创作      │         │
│  │ - 数据分析      │  │ - 调试修复      │  │ - 文档编写      │         │
│  │ - 知识整合      │  │ - 代码审查      │  │ - 报告撰写      │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                  │
│           └────────────────────┼────────────────────┘                  │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Communication Layer                           │ │
│  │                        (通信层)                                     │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  Message Queue │ Shared Memory │ Event Bus │ Consensus Protocol   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. 核心角色定义

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent 角色分类                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Orchestrator (编排者)                                       │
│  ├── 任务理解和分解                                          │
│  ├── Agent 选择和分配                                        │
│  ├── 执行流程控制                                            │
│  └── 结果整合和输出                                          │
│                                                             │
│  Worker Agents (工作者)                                      │
│  ├── Specialist (专家型): 特定领域的专业能力                  │
│  ├── Generalist (通才型): 广泛的任务处理能力                  │
│  └── Hybrid (混合型): 专业能力 + 协作能力                     │
│                                                             │
│  Support Agents (支持者)                                     │
│  ├── Critic (评审): 质量控制和反馈                           │
│  ├── Validator (验证): 结果验证和确认                        │
│  └── Monitor (监控): 状态监控和异常处理                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. 通信模式

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent 通信模式                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 点对点通信 (Point-to-Point)                              │
│     Agent A ──────────────────── Agent B                    │
│     - 直接消息传递                                           │
│     - 请求-响应模式                                          │
│                                                             │
│  2. 发布-订阅 (Publish-Subscribe)                            │
│     Publisher ──┬──→ Subscriber A                           │
│                 ├──→ Subscriber B                           │
│                 └──→ Subscriber C                           │
│     - 事件驱动                                               │
│     - 解耦通信                                               │
│                                                             │
│  3. 黑板模式 (Blackboard)                                    │
│     ┌─────────────────────────────────┐                     │
│     │        Shared Blackboard         │                     │
│     │  ┌─────┬─────┬─────┬─────┐      │                     │
│     │  │Task1│Task2│Task3│ ... │      │                     │
│     │  └─────┴─────┴─────┴─────┘      │                     │
│     └─────────────────────────────────┘                     │
│        ↑         ↑         ↑                                │
│     Agent A   Agent B   Agent C                             │
│     - 共享工作空间                                           │
│     - 异步协作                                               │
│                                                             │
│  4. 层级通信 (Hierarchical)                                  │
│     Orchestrator                                            │
│         ├── Manager A ──┬── Worker 1                        │
│         │               └── Worker 2                        │
│         └── Manager B ──┬── Worker 3                        │
│                         └── Worker 4                        │
│     - 分层管理                                               │
│     - 任务委派                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. 任务分解策略

```python
# 任务分解示例
original_task = "开发一个天气查询应用，包含前端界面、后端API和数据存储"

# 分解结果
subtasks = [
    {
        "id": 1,
        "name": "需求分析",
        "agent": "analyst_agent",
        "dependencies": []
    },
    {
        "id": 2,
        "name": "后端API开发",
        "agent": "backend_agent",
        "dependencies": [1]
    },
    {
        "id": 3,
        "name": "前端界面开发",
        "agent": "frontend_agent",
        "dependencies": [1]
    },
    {
        "id": 4,
        "name": "数据库设计",
        "agent": "database_agent",
        "dependencies": [1]
    },
    {
        "id": 5,
        "name": "集成测试",
        "agent": "tester_agent",
        "dependencies": [2, 3, 4]
    }
]
```

### 6. 共识机制

```
┌─────────────────────────────────────────────────────────────┐
│                     共识机制类型                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 投票机制 (Voting)                                        │
│     ┌───┐   ┌───┐   ┌───┐                                   │
│     │ A │──▶│ B │──▶│ C │  → 多数决定                       │
│     └───┘   └───┘   └───┘                                   │
│                                                             │
│  2. 协商机制 (Negotiation)                                   │
│     Agent A ←→ Agent B ←→ Agent C                           │
│     - 提议-反提议-接受/拒绝                                   │
│                                                             │
│  3. 拍卖机制 (Auction)                                       │
│     Task → Agent A (bid: 0.8)                               │
│          → Agent B (bid: 0.9) → Winner!                     │
│          → Agent C (bid: 0.7)                               │
│                                                             │
│  4. 联邦学习 (Federated)                                     │
│     Local Model A + Local Model B + Local Model C           │
│              ↓                                              │
│         Aggregated Model                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装依赖

```bash
pip install openai python-dotenv
```

### 基础示例

```python
from multi_agent import MultiAgentSystem
from orchestrator import Orchestrator
from worker_agents import ResearcherAgent, CoderAgent, WriterAgent

# 1. 创建多智能体系统
system = MultiAgentSystem()

# 2. 注册专业 Agent
system.register_agent("researcher", ResearcherAgent())
system.register_agent("coder", CoderAgent())
system.register_agent("writer", WriterAgent())

# 3. 执行复杂任务
task = """
研究 Python 异步编程的最佳实践，
然后编写一个异步爬虫示例代码，
最后撰写一份技术文档。
"""

result = system.run(task)
print(result)
```

## 练习文件

### `communication.py`
Agent 通信模块：
- 消息定义和封装
- 消息队列实现
- 发布-订阅机制
- 共享内存管理

### `worker_agents.py`
工作者 Agent 实现：
- 专家型 Agent（研究员、程序员、作家等）
- Agent 能力声明
- 任务执行接口
- 结果反馈机制

### `orchestrator.py`
编排者实现：
- 任务分析和分解
- Agent 选择和分配
- 执行流程控制
- 结果整合

### `multi_agent.py`
核心框架：
- 系统初始化和管理
- Agent 注册和发现
- 协作工作流引擎
- 共识机制实现

## 系统架构详解

### 1. 消息系统

```python
from communication import Message, MessageBus, MessageType

# 创建消息总线
bus = MessageBus()

# 定义消息
message = Message(
    sender="orchestrator",
    receiver="researcher",
    type=MessageType.TASK_ASSIGN,
    content={
        "task_id": "T001",
        "description": "研究 Python 并发编程模式",
        "priority": "high"
    }
)

# 发送消息
bus.send(message)

# Agent 接收消息
received = bus.receive("researcher")
```

### 2. Agent 定义

```python
from worker_agents import BaseWorkerAgent, AgentCapability

class ResearcherAgent(BaseWorkerAgent):
    """研究专家 Agent"""

    def __init__(self):
        super().__init__(
            name="researcher",
            capabilities=[
                AgentCapability.INFORMATION_RETRIEVAL,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.KNOWLEDGE_SYNTHESIS
            ]
        )

    def execute(self, task: dict) -> dict:
        """执行研究任务"""
        # 实现研究逻辑
        research_result = self._do_research(task["description"])
        return {
            "status": "completed",
            "result": research_result
        }
```

### 3. 编排器

```python
from orchestrator import Orchestrator, TaskDecomposer

orchestrator = Orchestrator()

# 分析任务
analysis = orchestrator.analyze_task("""
创建一个 Web 应用：
1. 设计数据库模型
2. 实现后端 API
3. 开发前端界面
""")

# 显示分解结果
for subtask in analysis.subtasks:
    print(f"Subtask: {subtask.name}")
    print(f"  Agent: {subtask.assigned_agent}")
    print(f"  Dependencies: {subtask.dependencies}")
```

## 协作模式

### 1. 串行执行

```
Task → Agent A → Agent B → Agent C → Result

适用场景：
- 有明确依赖关系的任务
- 需要逐步处理的流程
```

### 2. 并行执行

```
         ┌→ Agent A ─┐
Task ────┼→ Agent B ─┼──→ Merge → Result
         └→ Agent C ─┘

适用场景：
- 独立可并行的任务
- 需要加快处理速度
```

### 3. 迭代协作

```
Task → Agent A → Agent B → Review →
         ↑                    │
         └────────────────────┘

适用场景：
- 需要持续改进的任务
- 评审和修改流程
```

### 4. 竞争协作

```
         ┌→ Agent A ─┐
Task ────┼→ Agent B ─┼──→ Best Selection → Result
         └→ Agent C ─┘

适用场景：
- 需要最佳方案的任务
- 创意生成和选择
```

## 共识实现

### 1. 投票共识

```python
from multi_agent import VotingConsensus

# 创建投票共识机制
voting = VotingConsensus()

# 收集各 Agent 的方案
proposals = {
    "agent_a": {"solution": "方案A", "score": 0.85},
    "agent_b": {"solution": "方案B", "score": 0.90},
    "agent_c": {"solution": "方案C", "score": 0.80}
}

# 执行投票
winner = voting.vote(proposals)
print(f"获胜方案: {winner}")
```

### 2. 协商共识

```python
from multi_agent import NegotiationProtocol

# 创建协商协议
negotiation = NegotiationProtocol(max_rounds=5)

# 执行协商
result = negotiation.negotiate(
    agents=["agent_a", "agent_b", "agent_c"],
    initial_proposal=initial_solution
)
```

## 最佳实践

### 1. Agent 设计原则

```python
# 好的 Agent 设计
class GoodAgent:
    def __init__(self):
        # 明确的能力声明
        self.capabilities = ["research", "analysis"]
        # 清晰的职责边界
        self.responsibilities = "信息研究和分析"

    def can_handle(self, task) -> bool:
        # 明确的任务匹配逻辑
        return any(cap in task.type for cap in self.capabilities)

# 避免的 Agent 设计
class BadAgent:
    def __init__(self):
        # 过于宽泛的能力
        self.capabilities = ["everything"]  # 不推荐
```

### 2. 任务分解技巧

```python
# 好的分解
def decompose_task(task):
    """
    分解原则：
    1. 每个子任务有明确的目标
    2. 子任务之间有清晰的边界
    3. 依赖关系明确
    4. 可以独立验证结果
    """
    subtasks = []
    # 分析任务类型
    if "研究" in task:
        subtasks.append(create_research_subtask(task))
    if "编程" in task:
        subtasks.append(create_coding_subtask(task))
    return subtasks
```

### 3. 错误处理

```python
class MultiAgentSystem:
    def execute_with_retry(self, agent, task, max_retries=3):
        """带重试的执行"""
        for i in range(max_retries):
            try:
                result = agent.execute(task)
                if result["status"] == "success":
                    return result
                # 尝试其他 Agent
                alternative = self.find_alternative_agent(agent)
                if alternative:
                    return alternative.execute(task)
            except Exception as e:
                self.logger.error(f"Agent {agent} failed: {e}")
        return {"status": "failed", "error": "Max retries exceeded"}
```

### 4. 常见问题解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Agent 间冲突 | 任务边界不清 | 明确职责划分，添加协调机制 |
| 通信延迟 | 消息积压 | 使用异步通信，添加超时机制 |
| 结果不一致 | 缺乏共识 | 实现共识机制，添加验证步骤 |
| 系统僵死 | 死锁或循环依赖 | 添加超时和死锁检测 |

## 进阶主题

### 1. 动态 Agent 组合

```python
class DynamicAgentPool:
    """动态 Agent 池"""

    def __init__(self):
        self.agents = {}
        self.load_balancer = LoadBalancer()

    def get_agent(self, capability: str) -> Agent:
        """根据能力获取 Agent"""
        candidates = self.find_by_capability(capability)
        return self.load_balancer.select(candidates)

    def scale(self, capability: str, count: int):
        """动态扩展 Agent 数量"""
        for _ in range(count):
            new_agent = self.create_agent(capability)
            self.register(new_agent)
```

### 2. 学习型协作

```python
class LearningCollaboration:
    """学习型协作系统"""

    def __init__(self):
        self.experience_store = {}

    def learn_from_execution(self, task, agents, result):
        """从执行中学习"""
        # 记录成功的协作模式
        if result["success"]:
            self.experience_store[task.type] = {
                "agents": agents,
                "pattern": self.extract_pattern(task)
            }

    def suggest_agents(self, task):
        """基于经验推荐 Agent 组合"""
        if task.type in self.experience_store:
            return self.experience_store[task.type]["agents"]
        return self.default_selection(task)
```

### 3. 人机协作

```python
class HumanInTheLoop:
    """人机协作模式"""

    def __init__(self):
        self.agents = {}
        self.human_feedback = None

    def request_human_input(self, context):
        """请求人类输入"""
        return self.human_interface.ask(context)

    def incorporate_feedback(self, agent_result, human_feedback):
        """整合人类反馈"""
        # 根据反馈调整 Agent 行为
        for agent in self.agents.values():
            agent.update_with_feedback(human_feedback)
```

## 学习成果

完成本天学习后，你将能够：
- 理解多智能体系统的核心架构和设计模式
- 实现 Agent 间的通信和协作机制
- 设计和实现任务分解与分配策略
- 构建可扩展的多 Agent 协作系统
- 应用共识机制解决 Agent 间的决策问题

## 下一步

第八天将学习 Agent 工具集成，让 Agent 能够调用外部 API 和服务。

## 参考资料

- [Multi-Agent Systems: A Survey](https://arxiv.org/abs/1911.09667)
- [LangGraph: Multi-Agent Orchestration](https://langchain-ai.github.io/langgraph/)
- [AutoGen: Multi-Agent Conversation Framework](https://microsoft.github.io/autogen/)
- [CrewAI: Platform for Multi-AI Agents](https://www.crewai.com/)