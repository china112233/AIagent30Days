"""
多智能体系统核心框架
实现系统初始化、Agent 注册发现、协作工作流和共识机制
"""

import os
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv

from communication import MessageBus, Message, MessageType
from worker_agents import (
    AgentPool, AgentFactory, BaseWorkerAgent,
    AgentCapability, TaskContext, TaskResult
)
from orchestrator import Orchestrator, ExecutionPlan, Subtask, SubtaskStatus

load_dotenv()

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==================== 枚举和数据结构 ====================

class WorkflowStatus(Enum):
    """工作流状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsensusType(Enum):
    """共识类型"""
    VOTING = "voting"           # 投票
    NEGOTIATION = "negotiation"  # 协商
    AUCTION = "auction"         # 拍卖
    WEIGHTED = "weighted"       # 加权


@dataclass
class WorkflowConfig:
    """
    工作流配置

    Attributes:
        max_iterations: 最大迭代次数
        timeout: 超时时间(秒)
        parallel_limit: 并行任务上限
        retry_count: 重试次数
        verbose: 是否输出详细日志
    """
    max_iterations: int = 10
    timeout: float = 300.0
    parallel_limit: int = 5
    retry_count: int = 3
    verbose: bool = True


@dataclass
class Proposal:
    """
    提案

    Attributes:
        id: 提案 ID
        proposer: 提出者 Agent ID
        content: 提案内容
        score: 自评分
        votes: 投票记录
        timestamp: 时间戳
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    proposer: str = ""
    content: Any = None
    score: float = 0.0
    votes: Dict[str, bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ==================== 共识机制 ====================

class VotingConsensus:
    """
    投票共识机制

    通过多数投票决定最终方案
    """

    def __init__(self, threshold: float = 0.5):
        """
        初始化投票共识

        Args:
            threshold: 通过阈值 (0-1)
        """
        self.threshold = threshold
        self.voting_history: List[Dict] = []

    def vote(self, proposals: Dict[str, Proposal], agents: List[str]) -> Proposal:
        """
        执行投票

        Args:
            proposals: 提案字典 {agent_id: proposal}
            agents: 有投票权的 Agent 列表

        Returns:
            获胜提案
        """
        if not proposals:
            return None

        # 收集投票
        votes_count = {pid: 0 for pid in proposals}

        for agent_id in agents:
            # Agent 根据提案质量投票
            best_proposal = self._evaluate_proposals(proposals, agent_id)
            if best_proposal and best_proposal.id in votes_count:
                votes_count[best_proposal.id] += 1

        # 找出获胜者
        winner_id = max(votes_count, key=votes_count.get)
        winner = proposals[winner_id]

        # 检查是否达到阈值
        total_votes = len(agents)
        win_votes = votes_count[winner_id]
        win_ratio = win_votes / total_votes if total_votes > 0 else 0

        # 记录历史
        self.voting_history.append({
            "timestamp": time.time(),
            "proposals": len(proposals),
            "voters": len(agents),
            "winner": winner_id,
            "ratio": win_ratio,
            "passed": win_ratio >= self.threshold
        })

        return winner if win_ratio >= self.threshold else None

    def _evaluate_proposals(self, proposals: Dict[str, Proposal], agent_id: str) -> Optional[Proposal]:
        """
        Agent 评估提案

        Args:
            proposals: 提案列表
            agent_id: 评估 Agent ID

        Returns:
            最佳提案
        """
        # 简单实现：选择评分最高的
        return max(proposals.values(), key=lambda p: p.score)

    def get_voting_stats(self) -> Dict:
        """获取投票统计"""
        if not self.voting_history:
            return {"total_votes": 0}

        passed = sum(1 for v in self.voting_history if v["passed"])
        return {
            "total_votes": len(self.voting_history),
            "passed": passed,
            "pass_rate": passed / len(self.voting_history)
        }


class NegotiationProtocol:
    """
    协商共识机制

    通过多轮协商达成一致
    """

    NEGOTIATION_PROMPT = """你是一个协商专家。请评估以下提案并提出你的意见。

当前提案: {proposal}

其他 Agent 的意见:
{other_opinions}

请回答:
1. 你是否同意这个提案？(同意/反对)
2. 如果反对，请提出修改建议。

请用以下格式回答:
决策: [同意/反对]
理由: [你的理由]
建议: [如有]"""

    def __init__(self, max_rounds: int = 3):
        """
        初始化协商协议

        Args:
            max_rounds: 最大协商轮数
        """
        self.max_rounds = max_rounds
        self.negotiation_history: List[Dict] = []

    def negotiate(self, initial_proposal: str, agents: List[str], 
                  evaluate_func: Callable = None) -> Dict:
        """
        执行协商

        Args:
            initial_proposal: 初始提案
            agents: 参与协商的 Agent 列表
            evaluate_func: 评估函数 (可选)

        Returns:
            协商结果
        """
        current_proposal = initial_proposal
        round_num = 0

        for round_num in range(1, self.max_rounds + 1):
            # 收集各 Agent 意见
            opinions = {}

            for agent_id in agents:
                opinion = self._get_agent_opinion(
                    agent_id, current_proposal, opinions
                )
                opinions[agent_id] = opinion

            # 统计同意数
            agree_count = sum(1 for o in opinions.values() if o.get("agree", False))

            # 检查是否达成共识
            if agree_count >= len(agents) * 0.6:  # 60% 同意即可
                result = {
                    "status": "agreed",
                    "proposal": current_proposal,
                    "rounds": round_num,
                    "agree_count": agree_count,
                    "total_agents": len(agents)
                }
                self.negotiation_history.append(result)
                return result

            # 整合建议，生成新提案
            suggestions = [o.get("suggestion", "") for o in opinions.values() if o.get("suggestion")]
            if suggestions:
                current_proposal = self._integrate_suggestions(current_proposal, suggestions)

        # 未达成共识
        result = {
            "status": "disagreed",
            "proposal": current_proposal,
            "rounds": round_num,
            "agree_count": agree_count,
            "total_agents": len(agents)
        }
        self.negotiation_history.append(result)
        return result

    def _get_agent_opinion(self, agent_id: str, proposal: str, 
                           other_opinions: Dict) -> Dict:
        """获取 Agent 意见"""
        # 格式化其他意见
        others_str = "\n".join([
            f"- {aid}: {op.get('reason', '无意见')}"
            for aid, op in other_opinions.items()
        ]) or "暂无"

        # 使用 LLM 生成意见
        prompt = self.NEGOTIATION_PROMPT.format(
            proposal=proposal,
            other_opinions=others_str
        )

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            content = response.choices[0].message.content

            # 解析结果
            agree = "同意" in content

            # 提取理由和建议
            reason = ""
            suggestion = ""
            lines = content.split("\n")
            for line in lines:
                if "理由" in line:
                    reason = line.split(":", 1)[-1].strip()
                elif "建议" in line:
                    suggestion = line.split(":", 1)[-1].strip()

            return {
                "agree": agree,
                "reason": reason,
                "suggestion": suggestion if not agree else ""
            }

        except Exception as e:
            return {"agree": False, "reason": str(e), "suggestion": ""}

    def _integrate_suggestions(self, proposal: str, suggestions: List[str]) -> str:
        """整合建议生成新提案"""
        prompt = f"""请根据以下建议修改提案:

原提案: {proposal}

建议:
{chr(10).join(f'- {s}' for s in suggestions)}

请输出修改后的提案:"""

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content
        except:
            return proposal


class AuctionMechanism:
    """
    拍卖机制

    通过竞标方式分配任务
    """

    def __init__(self):
        """初始化拍卖机制"""
        self.auction_history: List[Dict] = []

    def auction_task(self, task: TaskContext, agents: List[BaseWorkerAgent]) -> Dict:
        """
        拍卖任务

        Args:
            task: 任务上下文
            agents: 竞标 Agent 列表

        Returns:
            拍卖结果
        """
        bids = {}

        # 收集竞标
        for agent in agents:
            bid = self._get_bid(agent, task)
            if bid is not None:
                bids[agent.profile.id] = bid

        if not bids:
            return {"status": "no_bids", "winner": None}

        # 选择获胜者 (最高竞标者)
        winner_id = max(bids, key=lambda x: bids[x]["value"])
        winner_bid = bids[winner_id]

        result = {
            "status": "success",
            "winner": winner_id,
            "bid_value": winner_bid["value"],
            "all_bids": bids,
            "task_id": task.task_id
        }

        self.auction_history.append(result)
        return result

    def _get_bid(self, agent: BaseWorkerAgent, task: TaskContext) -> Optional[Dict]:
        """
        获取 Agent 竞标

        Args:
            agent: Agent 实例
            task: 任务上下文

        Returns:
            竞标信息
        """
        # 检查能力
        capability_score = agent.can_handle(task)

        if capability_score < 0.3:
            return None  # 能力不足，不参与竞标

        # 计算竞标值
        # 考虑因素: 能力匹配度、当前负载、性能分数
        load_factor = 1 - (len(agent.current_tasks) / agent.profile.max_concurrent_tasks)
        bid_value = capability_score * load_factor * agent.profile.performance_score

        return {
            "agent_id": agent.profile.id,
            "value": bid_value,
            "capability_score": capability_score,
            "load_factor": load_factor
        }

    def get_auction_stats(self) -> Dict:
        """获取拍卖统计"""
        if not self.auction_history:
            return {"total_auctions": 0}

        return {
            "total_auctions": len(self.auction_history),
            "successful": sum(1 for a in self.auction_history if a["status"] == "success"),
            "no_bid_count": sum(1 for a in self.auction_history if a["status"] == "no_bids")
        }


# ==================== 协作工作流引擎 ====================

class CollaborationWorkflow:
    """
    协作工作流引擎

    管理多 Agent 协作的工作流程
    """

    WORKFLOW_PROMPT = """你是一个工作流协调专家。请分析以下任务并确定协作模式。

任务: {task}

可用 Agent:
{agents}

请确定:
1. 协作模式 (串行/并行/混合)
2. 每个 Agent 的分工
3. 协作流程

请输出 JSON 格式的工作流配置。"""

    def __init__(self, agent_pool: AgentPool, message_bus: MessageBus, 
                 config: WorkflowConfig = None):
        """
        初始化工作流引擎

        Args:
            agent_pool: Agent 池
            message_bus: 消息总线
            config: 工作流配置
        """
        self.agent_pool = agent_pool
        self.message_bus = message_bus
        self.config = config or WorkflowConfig()

        self.status = WorkflowStatus.IDLE
        self.current_task: Optional[str] = None
        self.execution_log: List[Dict] = []

    def create_workflow(self, task: str) -> Dict:
        """
        创建工作流

        Args:
            task: 任务描述

        Returns:
            工作流配置
        """
        # 获取可用 Agent
        available_agents = self.agent_pool.get_all_status()
        agents_desc = "\n".join([
            f"- {a['name']}: {a['status']}"
            for a in available_agents
        ])

        # 调用 LLM 生成工作流
        prompt = self.WORKFLOW_PROMPT.format(
            task=task,
            agents=agents_desc
        )

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = response.choices[0].message.content

            # 解析工作流配置
            workflow_config = self._parse_workflow(content)
            return workflow_config

        except Exception as e:
            # 返回默认工作流
            return self._default_workflow(task)

    def execute_workflow(self, task: str, workflow_config: Dict = None) -> Dict:
        """
        执行工作流

        Args:
            task: 任务描述
            workflow_config: 工作流配置 (可选)

        Returns:
            执行结果
        """
        self.status = WorkflowStatus.RUNNING
        self.current_task = task
        start_time = time.time()

        if self.config.verbose:
            print(f"\n[Workflow] 开始执行: {task[:50]}...")

        # 创建或使用工作流配置
        if not workflow_config:
            workflow_config = self.create_workflow(task)

        # 创建 Orchestrator 执行
        orchestrator = Orchestrator(self.agent_pool, verbose=self.config.verbose)
        result = orchestrator.run(task)

        # 记录执行日志
        self.execution_log.append({
            "task": task,
            "status": result["status"],
            "execution_time": result.get("execution_time", 0),
            "timestamp": time.time()
        })

        self.status = WorkflowStatus.COMPLETED if result["status"] == "completed" else WorkflowStatus.FAILED
        self.current_task = None

        return result

    def pause(self):
        """暂停工作流"""
        self.status = WorkflowStatus.PAUSED

    def resume(self):
        """恢复工作流"""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING

    def get_status(self) -> Dict:
        """获取工作流状态"""
        return {
            "status": self.status.value,
            "current_task": self.current_task,
            "execution_count": len(self.execution_log)
        }

    def _parse_workflow(self, content: str) -> Dict:
        """解析工作流配置"""
        try:
            # 尝试提取 JSON
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
        except:
            pass

        return self._default_workflow("")

    def _default_workflow(self, task: str) -> Dict:
        """默认工作流"""
        return {
            "mode": "sequential",
            "steps": [
                {"agent": "researcher", "task": "分析任务需求"},
                {"agent": "coder", "task": "执行主要任务"},
                {"agent": "analyst", "task": "审查结果"}
            ]
        }


# ==================== 多智能体系统 ====================

class MultiAgentSystem:
    """
    多智能体系统

    整合 Agent 池、消息系统、编排器和工作流
    """

    def __init__(self, config: WorkflowConfig = None):
        """
        初始化多智能体系统

        Args:
            config: 工作流配置
        """
        self.config = config or WorkflowConfig()

        # 核心组件
        self.agent_pool = AgentPool()
        self.message_bus = MessageBus()
        self.orchestrator = None
        self.workflow_engine = None

        # 共识机制
        self.voting = VotingConsensus()
        self.negotiation = NegotiationProtocol()
        self.auction = AuctionMechanism()

        # 系统状态
        self.initialized = False
        self.system_log: List[Dict] = []

    def initialize(self, agent_types: List[str] = None):
        """
        初始化系统

        Args:
            agent_types: 要创建的 Agent 类型列表
        """
        if self.config.verbose:
            print("\n[MultiAgentSystem] 初始化系统...")

        # 创建默认 Agent
        agent_types = agent_types or ["researcher", "coder", "writer", "analyst", "architect"]

        for agent_type in agent_types:
            agent = AgentFactory.create(agent_type)
            agent.set_communication_manager(self.message_bus)
            self.agent_pool.register(agent)

            if self.config.verbose:
                print(f"  已注册: {agent.profile.name}")

        # 创建编排器和工作流引擎
        self.orchestrator = Orchestrator(self.agent_pool, verbose=self.config.verbose)
        self.workflow_engine = CollaborationWorkflow(
            self.agent_pool, self.message_bus, self.config
        )

        self.initialized = True

        if self.config.verbose:
            print(f"  系统初始化完成，共 {len(self.agent_pool.agents)} 个 Agent")

    def register_agent(self, agent_id: str, agent: BaseWorkerAgent):
        """
        注册 Agent

        Args:
            agent_id: Agent ID
            agent: Agent 实例
        """
        agent.set_communication_manager(self.message_bus)
        self.agent_pool.register(agent)

        if self.config.verbose:
            print(f"[MultiAgentSystem] 注册 Agent: {agent.profile.name}")

    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        self.agent_pool.unregister(agent_id)

        if self.config.verbose:
            print(f"[MultiAgentSystem] 注销 Agent: {agent_id}")

    def find_agent(self, capability: AgentCapability) -> List[BaseWorkerAgent]:
        """
        根据能力查找 Agent

        Args:
            capability: 能力类型

        Returns:
            匹配的 Agent 列表
        """
        return self.agent_pool.find_by_capability(capability)

    def run(self, task: str, mode: str = "auto") -> Dict:
        """
        运行任务

        Args:
            task: 任务描述
            mode: 执行模式 (auto/orchestrate/workflow/consensus)

        Returns:
            执行结果
        """
        if not self.initialized:
            self.initialize()

        start_time = time.time()

        if self.config.verbose:
            print(f"\n[MultiAgentSystem] 开始执行任务")
            print(f"  任务: {task[:50]}...")
            print(f"  模式: {mode}")

        # 根据模式选择执行方式
        if mode == "orchestrate":
            result = self.orchestrator.run(task)
        elif mode == "workflow":
            result = self.workflow_engine.execute_workflow(task)
        elif mode == "consensus":
            result = self._run_with_consensus(task)
        else:  # auto
            result = self.orchestrator.run(task)

        # 记录日志
        self.system_log.append({
            "task": task,
            "mode": mode,
            "status": result.get("status"),
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        })

        return result

    def _run_with_consensus(self, task: str) -> Dict:
        """
        使用共识机制执行任务

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        # 让多个 Agent 提出方案
        proposals = {}
        agents = list(self.agent_pool.agents.values())[:3]  # 取前3个

        for agent in agents:
            task_ctx = TaskContext(
                task_id=f"propose_{agent.profile.id}",
                description=f"提出解决方案: {task}"
            )
            result = agent.execute(task_ctx)

            proposals[agent.profile.id] = Proposal(
                proposer=agent.profile.id,
                content=result.output,
                score=self._evaluate_proposal(result.output)
            )

        # 投票决定最佳方案
        agent_ids = [a.profile.id for a in agents]
        winner = self.voting.vote(proposals, agent_ids)

        if winner:
            return {
                "status": "completed",
                "result": winner.content,
                "winner": winner.proposer,
                "consensus": True
            }

        # 如果投票未通过，使用协商
        if proposals:
            first_proposal = list(proposals.values())[0].content
            negotiation_result = self.negotiation.negotiate(first_proposal, agent_ids)

            return {
                "status": "completed" if negotiation_result["status"] == "agreed" else "partial",
                "result": negotiation_result["proposal"],
                "consensus": negotiation_result["status"] == "agreed"
            }

        return {"status": "failed", "message": "无法达成共识"}

    def _evaluate_proposal(self, content: str) -> float:
        """评估提案质量"""
        # 简单评估：基于长度和结构
        if not content:
            return 0.0

        score = 0.5

        # 长度适中加分
        if 100 < len(content) < 2000:
            score += 0.2

        # 有结构加分
        if "```" in content or "#" in content:
            score += 0.2

        return min(score, 1.0)

    def broadcast(self, message: str, exclude: List[str] = None):
        """
        广播消息给所有 Agent

        Args:
            message: 消息内容
            exclude: 排除的 Agent ID 列表
        """
        exclude = exclude or []

        for agent_id in self.agent_pool.agents:
            if agent_id not in exclude:
                msg = Message(
                    sender="system",
                    receiver=agent_id,
                    type=MessageType.BROADCAST,
                    content={"message": message}
                )
                self.message_bus.send(msg)

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "initialized": self.initialized,
            "agent_count": len(self.agent_pool.agents),
            "agents": self.agent_pool.get_all_status(),
            "message_queue_size": self.message_bus.get_queue_size(),
            "execution_count": len(self.system_log)
        }

    def get_statistics(self) -> Dict:
        """获取系统统计"""
        return {
            "total_executions": len(self.system_log),
            "successful_executions": sum(1 for log in self.system_log if log.get("status") == "completed"),
            "voting_stats": self.voting.get_voting_stats(),
            "auction_stats": self.auction.get_auction_stats(),
            "agent_pool": {
                "total_agents": len(self.agent_pool.agents),
                "idle_agents": sum(1 for a in self.agent_pool.agents.values() 
                                   if a.status.value == "idle")
            }
        }


# ==================== 演示函数 ====================

def demo_multi_agent_system():
    """演示多智能体系统"""
    print("\n" + "=" * 60)
    print("多智能体系统演示")
    print("=" * 60)

    # 创建系统
    system = MultiAgentSystem(config=WorkflowConfig(verbose=True))

    # 初始化
    system.initialize()

    # 执行任务
    task = """
    研究 Python 装饰器的工作原理，
    编写一个缓存装饰器示例，
    并撰写使用文档。
    """

    result = system.run(task)

    print("\n[执行结果]")
    print(f"状态: {result['status']}")
    if 'result' in result:
        print(f"结果预览: {result['result'][:300]}...")

    # 系统状态
    print("\n[系统状态]")
    status = system.get_system_status()
    print(f"Agent 数量: {status['agent_count']}")
    print(f"消息队列: {status['message_queue_size']}")


def demo_consensus():
    """演示共识机制"""
    print("\n" + "=" * 60)
    print("共识机制演示")
    print("=" * 60)

    # 创建投票共识
    voting = VotingConsensus(threshold=0.5)

    # 创建提案
    proposals = {
        "agent_a": Proposal(proposer="agent_a", content="方案A", score=0.8),
        "agent_b": Proposal(proposer="agent_b", content="方案B", score=0.9),
        "agent_c": Proposal(proposer="agent_c", content="方案C", score=0.7)
    }

    # 执行投票
    agents = ["voter_1", "voter_2", "voter_3", "voter_4", "voter_5"]
    winner = voting.vote(proposals, agents)

    if winner:
        print(f"\n获胜提案: {winner.content} (来自 {winner.proposer})")

    print(f"\n投票统计: {voting.get_voting_stats()}")


def demo_auction():
    """演示拍卖机制"""
    print("\n" + "=" * 60)
    print("拍卖机制演示")
    print("=" * 60)

    # 创建 Agent 池和拍卖机制
    pool = AgentPool()
    auction = AuctionMechanism()

    # 注册 Agent
    for agent_type in ["researcher", "coder", "writer"]:
        agent = AgentFactory.create(agent_type)
        pool.register(agent)

    # 创建任务
    task = TaskContext(
        task_id="auction_test",
        description="编写一个数据处理脚本"
    )

    # 执行拍卖
    agents = list(pool.agents.values())
    result = auction.auction_task(task, agents)

    print(f"\n拍卖结果:")
    print(f"  状态: {result['status']}")
    if result['winner']:
        print(f"  获胜者: {result['winner']}")
        print(f"  竞标值: {result['bid_value']:.3f}")

    print(f"\n拍卖统计: {auction.get_auction_stats()}")


if __name__ == "__main__":
    demo_multi_agent_system()
    demo_consensus()
    demo_auction()