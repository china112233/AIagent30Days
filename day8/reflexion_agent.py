"""
反思型 Agent 模块
实现反思机制、经验总结和学习改进
"""

import uuid
import time
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# 导入记忆系统和规划系统
from agent_memory import MemoryManager, MemoryItem, MemoryType
from planning_agent import Task, TaskStatus, Plan, PlanningAgent


# ==================== 反思类型枚举 ====================

class ReflectionType(Enum):
    """反思类型枚举"""
    SUCCESS_ANALYSIS = "success_analysis"     # 成功分析
    FAILURE_ANALYSIS = "failure_analysis"     # 失败分析
    IMPROVEMENT = "improvement"               # 改进建议
    KNOWLEDGE_EXTRACTION = "knowledge"        # 知识提取
    PATTERN_RECOGNITION = "pattern"           # 模式识别


# ==================== 反思结果数据结构 ====================

@dataclass
class Reflection:
    """
    反思结果数据结构
    
    Attributes:
        id: 反思唯一ID
        task_id: 关联的任务ID
        reflection_type: 反思类型
        success: 是否成功
        analysis: 分析内容
        lessons_learned: 学到的教训
        improvements: 改进建议
        knowledge_gained: 获得的知识
        confidence: 置信度 (0.0-1.0)
        created_at: 创建时间
        metadata: 元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    reflection_type: ReflectionType = ReflectionType.SUCCESS_ANALYSIS
    success: bool = True
    analysis: str = ""
    lessons_learned: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    knowledge_gained: List[str] = field(default_factory=list)
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "reflection_type": self.reflection_type.value,
            "success": self.success,
            "analysis": self.analysis,
            "lessons_learned": self.lessons_learned,
            "improvements": self.improvements,
            "knowledge_gained": self.knowledge_gained,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reflection":
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            task_id=data.get("task_id", ""),
            reflection_type=ReflectionType(data.get("reflection_type", "success_analysis")),
            success=data.get("success", True),
            analysis=data.get("analysis", ""),
            lessons_learned=data.get("lessons_learned", []),
            improvements=data.get("improvements", []),
            knowledge_gained=data.get("knowledge_gained", []),
            confidence=data.get("confidence", 0.8),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {})
        )
    
    def get_summary(self) -> str:
        """获取摘要"""
        status = "成功" if self.success else "失败"
        lines = [
            f"反思 [{self.id}] - 任务 {self.task_id}",
            f"结果: {status}",
            f"分析: {self.analysis[:100]}..." if len(self.analysis) > 100 else f"分析: {self.analysis}",
            f"教训: {', '.join(self.lessons_learned[:3])}",
            f"改进: {', '.join(self.improvements[:3])}"
        ]
        return "\n".join(lines)


# ==================== 经验数据结构 ====================

@dataclass
class Experience:
    """
    经验数据结构
    
    Attributes:
        id: 经验唯一ID
        situation: 情境描述
        action: 采取的行动
        result: 结果
        outcome: 结果评价 (正面/负面/中性)
        applicability: 适用条件
        confidence: 置信度
        usage_count: 使用次数
        created_at: 创建时间
        last_used_at: 最后使用时间
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    situation: str = ""
    action: str = ""
    result: str = ""
    outcome: str = "neutral"  # positive, negative, neutral
    applicability: List[str] = field(default_factory=list)
    confidence: float = 0.5
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    
    def use(self) -> None:
        """使用经验"""
        self.usage_count += 1
        self.last_used_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "situation": self.situation,
            "action": self.action,
            "result": self.result,
            "outcome": self.outcome,
            "applicability": self.applicability,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            situation=data.get("situation", ""),
            action=data.get("action", ""),
            result=data.get("result", ""),
            outcome=data.get("outcome", "neutral"),
            applicability=data.get("applicability", []),
            confidence=data.get("confidence", 0.5),
            usage_count=data.get("usage_count", 0),
            created_at=data.get("created_at", time.time()),
            last_used_at=data.get("last_used_at")
        )


# ==================== 反思型 Agent ====================

class ReflexionAgent:
    """
    反思型 Agent
    
    实现：
    - 执行后反思
    - 经验总结
    - 知识提取
    - 策略改进
    
    核心理念：通过反思从经验中学习，不断改进
    """
    
    def __init__(self, 
                 memory: Optional[MemoryManager] = None,
                 planner: Optional[PlanningAgent] = None,
                 reflection_depth: int = 2):
        """
        初始化反思 Agent
        
        Args:
            memory: 记忆管理器
            planner: 规划 Agent
            reflection_depth: 反思深度
        """
        self.memory = memory or MemoryManager()
        self.planner = planner or PlanningAgent(memory=self.memory)
        self.reflection_depth = reflection_depth
        
        # 反思历史
        self.reflections: Dict[str, Reflection] = {}
        
        # 经验库
        self.experiences: Dict[str, Experience] = {}
        
        # 执行器注册表
        self._executors: Dict[str, Callable] = {}
        
        # 反思策略
        self._reflection_strategies = self._init_strategies()
    
    def _init_strategies(self) -> Dict[str, Callable]:
        """初始化反思策略"""
        return {
            "success": self._reflect_on_success,
            "failure": self._reflect_on_failure,
            "partial": self._reflect_on_partial
        }
    
    def register_executor(self, task_type: str, executor: Callable) -> None:
        """
        注册任务执行器
        
        Args:
            task_type: 任务类型
            executor: 执行函数
        """
        self._executors[task_type] = executor
    
    def execute_and_reflect(self, task: Task) -> Tuple[Any, Reflection]:
        """
        执行任务并反思
        
        Args:
            task: 任务
            
        Returns:
            (执行结果, 反思结果)
        """
        # 执行任务
        result = self._execute_task(task)
        
        # 评估结果
        success = self._evaluate_result(task, result)
        
        # 反思
        reflection = self.reflect(task, result, success)
        
        return result, reflection
    
    def _execute_task(self, task: Task) -> Any:
        """
        执行任务
        
        Args:
            task: 任务
            
        Returns:
            执行结果
        """
        task.start()
        
        # 检索相关经验
        relevant_experiences = self._get_relevant_experiences(task)
        
        # 查找执行器
        task_type = task.metadata.get("type", "default")
        executor = self._executors.get(task_type) or self._executors.get("default")
        
        if executor:
            try:
                result = executor(task, relevant_experiences)
                task.complete(result)
                return result
            except Exception as e:
                task.fail(str(e))
                return {"error": str(e)}
        else:
            # 默认执行
            result = f"任务 '{task.name}' 执行完成"
            task.complete(result)
            return result
    
    def _evaluate_result(self, task: Task, result: Any) -> str:
        """
        评估结果
        
        Args:
            task: 任务
            result: 执行结果
            
        Returns:
            评估结果 ("success", "failure", "partial")
        """
        if task.status == TaskStatus.FAILED:
            return "failure"
        elif task.status == TaskStatus.COMPLETED:
            # 检查结果质量
            if isinstance(result, dict) and "error" in result:
                return "failure"
            elif isinstance(result, dict) and "partial" in result.get("status", ""):
                return "partial"
            else:
                return "success"
        else:
            return "partial"
    
    def reflect(self, task: Task, result: Any, outcome: str) -> Reflection:
        """
        执行反思
        
        Args:
            task: 任务
            result: 执行结果
            outcome: 结果评价
            
        Returns:
            反思结果
        """
        # 选择反思策略
        strategy = self._reflection_strategies.get(outcome, self._reflect_on_partial)
        
        # 执行反思
        reflection = strategy(task, result)
        
        # 存储反思
        self.reflections[reflection.id] = reflection
        
        # 存储到记忆系统
        self.memory.remember(
            content={
                "type": "reflection",
                "task_name": task.name,
                "outcome": outcome,
                "lessons": reflection.lessons_learned,
                "improvements": reflection.improvements
            },
            memory_type="episodic",
            importance=0.8 if outcome == "failure" else 0.6,
            metadata={"reflection_id": reflection.id, "task_id": task.id}
        )
        
        # 提取经验
        experience = self._extract_experience(task, result, reflection)
        if experience:
            self.experiences[experience.id] = experience
        
        # 更新知识
        self._update_knowledge(reflection)
        
        return reflection
    
    def _reflect_on_success(self, task: Task, result: Any) -> Reflection:
        """
        成功反思
        
        Args:
            task: 任务
            result: 执行结果
            
        Returns:
            反思结果
        """
        # 分析成功因素
        analysis = self._analyze_success_factors(task, result)
        
        # 提取可复用的模式
        patterns = self._extract_patterns(task, result)
        
        return Reflection(
            task_id=task.id,
            reflection_type=ReflectionType.SUCCESS_ANALYSIS,
            success=True,
            analysis=analysis,
            lessons_learned=patterns["lessons"],
            improvements=["保持当前策略"],
            knowledge_gained=patterns["knowledge"],
            confidence=0.9
        )
    
    def _reflect_on_failure(self, task: Task, result: Any) -> Reflection:
        """
        失败反思
        
        Args:
            task: 任务
            result: 执行结果
            
        Returns:
            反思结果
        """
        # 分析失败原因
        analysis = self._analyze_failure_causes(task, result)
        
        # 生成改进建议
        improvements = self._generate_improvements(task, result)
        
        # 提取教训
        lessons = self._extract_lessons(task, result)
        
        return Reflection(
            task_id=task.id,
            reflection_type=ReflectionType.FAILURE_ANALYSIS,
            success=False,
            analysis=analysis,
            lessons_learned=lessons,
            improvements=improvements,
            knowledge_gained=["需要避免此类错误"],
            confidence=0.85
        )
    
    def _reflect_on_partial(self, task: Task, result: Any) -> Reflection:
        """
        部分成功反思
        
        Args:
            task: 任务
            result: 执行结果
            
        Returns:
            反思结果
        """
        analysis = f"任务 '{task.name}' 部分成功，需要进一步改进"
        
        return Reflection(
            task_id=task.id,
            reflection_type=ReflectionType.IMPROVEMENT,
            success=False,
            analysis=analysis,
            lessons_learned=["部分结果需要验证"],
            improvements=["优化执行策略", "增加验证步骤"],
            knowledge_gained=["需要更详细的检查"],
            confidence=0.7
        )
    
    def _analyze_success_factors(self, task: Task, result: Any) -> str:
        """分析成功因素"""
        factors = []
        
        # 检查任务属性
        if task.priority.value >= 3:
            factors.append("高优先级任务得到了优先处理")
        
        if not task.dependencies:
            factors.append("无依赖关系，执行顺畅")
        
        # 检查相关经验
        relevant_exp = self._get_relevant_experiences(task)
        if relevant_exp:
            factors.append(f"参考了 {len(relevant_exp)} 条相关经验")
        
        if factors:
            return f"成功因素: {'; '.join(factors)}"
        else:
            return "任务执行顺利，策略有效"
    
    def _analyze_failure_causes(self, task: Task, result: Any) -> str:
        """分析失败原因"""
        causes = []
        
        # 检查错误信息
        if task.error:
            causes.append(f"错误: {task.error}")
        
        if isinstance(result, dict) and "error" in result:
            causes.append(f"执行错误: {result['error']}")
        
        # 检查依赖问题
        if task.dependencies:
            causes.append("可能存在依赖问题")
        
        if causes:
            return f"失败原因: {'; '.join(causes)}"
        else:
            return "失败原因未知，需要进一步分析"
    
    def _extract_patterns(self, task: Task, result: Any) -> Dict[str, List[str]]:
        """提取模式"""
        patterns = {
            "lessons": [],
            "knowledge": []
        }
        
        # 基于任务类型提取
        task_type = task.metadata.get("type", "general")
        patterns["lessons"].append(f"{task_type}类型任务的有效策略")
        
        # 基于结果提取
        if result:
            patterns["knowledge"].append(f"成功结果模式: {type(result).__name__}")
        
        return patterns
    
    def _generate_improvements(self, task: Task, result: Any) -> List[str]:
        """生成改进建议"""
        improvements = []
        
        # 基于错误类型
        if task.error:
            improvements.append(f"解决错误: {task.error}")
        
        # 基于任务属性
        if task.priority.value < 3:
            improvements.append("考虑提高任务优先级")
        
        # 通用改进
        improvements.extend([
            "增加执行前的验证步骤",
            "添加更详细的错误处理",
            "优化执行策略"
        ])
        
        return improvements[:5]  # 限制数量
    
    def _extract_lessons(self, task: Task, result: Any) -> List[str]:
        """提取教训"""
        lessons = []
        
        # 基于任务特征
        lessons.append(f"任务 '{task.name}' 执行失败的经验")
        
        # 基于错误
        if task.error:
            lessons.append(f"避免错误: {task.error}")
        
        return lessons
    
    def _get_relevant_experiences(self, task: Task) -> List[Experience]:
        """获取相关经验"""
        relevant = []
        
        # 按任务类型搜索
        task_type = task.metadata.get("type", "")
        
        for exp in self.experiences.values():
            # 检查适用性
            if task_type in exp.applicability or not exp.applicability:
                relevant.append(exp)
        
        # 按置信度和使用次数排序
        relevant.sort(key=lambda e: e.confidence * 0.6 + min(e.usage_count * 0.1, 0.4), reverse=True)
        
        return relevant[:5]  # 返回前5条
    
    def _extract_experience(self, task: Task, result: Any, reflection: Reflection) -> Optional[Experience]:
        """提取经验"""
        if not reflection.lessons_learned and not reflection.improvements:
            return None
        
        experience = Experience(
            situation=f"任务: {task.name}",
            action=f"执行任务类型: {task.metadata.get('type', 'general')}",
            result="成功" if reflection.success else "失败",
            outcome="positive" if reflection.success else "negative",
            applicability=[task.metadata.get("type", "general")],
            confidence=reflection.confidence
        )
        
        return experience
    
    def _update_knowledge(self, reflection: Reflection) -> None:
        """更新知识库"""
        # 存储学到的知识
        for knowledge in reflection.knowledge_gained:
            self.memory.remember(
                content=knowledge,
                memory_type="semantic",
                importance=0.7,
                metadata={"source": "reflection", "reflection_id": reflection.id}
            )
    
    def get_reflection_summary(self) -> str:
        """获取反思摘要"""
        total = len(self.reflections)
        successes = sum(1 for r in self.reflections.values() if r.success)
        failures = total - successes
        
        # 常见教训
        all_lessons = []
        for r in self.reflections.values():
            all_lessons.extend(r.lessons_learned)
        
        lesson_counts = {}
        for lesson in all_lessons:
            lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
        
        top_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        lines = [
            "反思摘要",
            "=" * 40,
            f"总反思次数: {total}",
            f"成功: {successes}, 失败: {failures}",
            f"成功率: {successes/total*100:.1f}%" if total > 0 else "成功率: N/A",
            "",
            "主要教训:"
        ]
        
        for lesson, count in top_lessons:
            lines.append(f"  - {lesson} (出现{count}次)")
        
        return "\n".join(lines)
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议汇总"""
        all_improvements = []
        for r in self.reflections.values():
            all_improvements.extend(r.improvements)
        
        # 去重并计数
        improvement_counts = {}
        for imp in all_improvements:
            improvement_counts[imp] = improvement_counts.get(imp, 0) + 1
        
        # 按出现次数排序
        sorted_improvements = sorted(
            improvement_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [imp for imp, _ in sorted_improvements[:10]]
    
    def learn_from_history(self, task_history: List[Tuple[Task, Any, str]]) -> List[Reflection]:
        """
        从历史中学习
        
        Args:
            task_history: 任务历史 [(任务, 结果, 评价)]
            
        Returns:
            反思结果列表
        """
        reflections = []
        
        for task, result, outcome in task_history:
            reflection = self.reflect(task, result, outcome)
            reflections.append(reflection)
        
        return reflections
    
    def export_reflections(self) -> Dict[str, Any]:
        """导出反思数据"""
        return {
            "reflections": [r.to_dict() for r in self.reflections.values()],
            "experiences": [e.to_dict() for e in self.experiences.values()]
        }
    
    def import_reflections(self, data: Dict[str, Any]) -> int:
        """导入反思数据"""
        count = 0
        
        for r_data in data.get("reflections", []):
            reflection = Reflection.from_dict(r_data)
            self.reflections[reflection.id] = reflection
            count += 1
        
        for e_data in data.get("experiences", []):
            experience = Experience.from_dict(e_data)
            self.experiences[experience.id] = experience
            count += 1
        
        return count


# ==================== 示例执行器 ====================

def sample_executor_with_reflection(task: Task, experiences: List[Experience]) -> Any:
    """示例执行器（带经验参考）"""
    print(f"  执行任务: {task.name}")
    
    # 参考经验
    if experiences:
        print(f"  参考 {len(experiences)} 条经验")
        for exp in experiences[:2]:
            exp.use()  # 标记使用
    
    # 模拟执行
    time.sleep(0.3)
    
    return f"任务 '{task.name}' 执行成功"


# ==================== 示例和测试 ====================

def demo_reflexion_agent():
    """演示反思 Agent"""
    print("=" * 60)
    print("反思型 Agent 演示")
    print("=" * 60)
    
    # 创建反思 Agent
    agent = ReflexionAgent(reflection_depth=2)
    
    # 注册执行器
    agent.register_executor("default", sample_executor_with_reflection)
    
    # 1. 执行成功任务并反思
    print("\n1. 执行成功任务并反思")
    print("-" * 40)
    
    task1 = Task(
        name="数据分析任务",
        description="分析销售数据",
        metadata={"type": "analysis"}
    )
    
    result1, reflection1 = agent.execute_and_reflect(task1)
    print(f"任务结果: {result1}")
    print(f"\n反思结果:")
    print(reflection1.get_summary())
    
    # 2. 执行失败任务并反思
    print("\n2. 执行失败任务并反思")
    print("-" * 40)
    
    task2 = Task(
        name="失败的任务",
        description="模拟失败",
        metadata={"type": "test"}
    )
    
    # 模拟失败
    def failing_executor(task: Task, exp: List[Experience]) -> Any:
        print(f"  执行任务: {task.name}")
        raise Exception("模拟执行错误")
    
    agent.register_executor("test", failing_executor)
    
    result2, reflection2 = agent.execute_and_reflect(task2)
    print(f"\n反思结果:")
    print(reflection2.get_summary())
    
    # 3. 利用经验执行新任务
    print("\n3. 利用经验执行新任务")
    print("-" * 40)
    
    task3 = Task(
        name="新的分析任务",
        description="分析用户数据",
        metadata={"type": "analysis"}
    )
    
    agent.register_executor("analysis", sample_executor_with_reflection)
    
    result3, reflection3 = agent.execute_and_reflect(task3)
    print(f"\n经验库大小: {len(agent.experiences)}")
    
    # 4. 获取反思摘要
    print("\n4. 反思摘要")
    print("-" * 40)
    print(agent.get_reflection_summary())
    
    # 5. 获取改进建议
    print("\n5. 改进建议")
    print("-" * 40)
    improvements = agent.get_improvement_suggestions()
    for i, imp in enumerate(improvements[:5], 1):
        print(f"  {i}. {imp}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_reflexion_agent()