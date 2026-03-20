"""
任务管理器模块
实现任务队列、调度、依赖管理和执行控制
"""

import uuid
import time
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue, Queue, Empty
import heapq

# 导入任务定义
from planning_agent import Task, TaskStatus, TaskPriority


# ==================== 调度策略 ====================

class ScheduleStrategy(Enum):
    """调度策略"""
    FIFO = "fifo"                   # 先进先出
    PRIORITY = "priority"           # 优先级调度
    SHORTEST_FIRST = "shortest"     # 最短任务优先
    DEADLINE_FIRST = "deadline"     # 截止时间优先


# ==================== 任务执行结果 ====================

@dataclass
class ExecutionResult:
    """
    任务执行结果
    
    Attributes:
        task_id: 任务ID
        success: 是否成功
        result: 执行结果
        error: 错误信息
        execution_time: 执行时间
        retries: 重试次数
    """
    task_id: str = ""
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0


# ==================== 任务队列 ====================

class TaskQueue:
    """
    任务队列
    
    支持优先级排序和依赖检查
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化任务队列
        
        Args:
            max_size: 最大容量
        """
        self.max_size = max_size
        self._queue: List[Tuple[int, int, Task]] = []  # (优先级, 序号, 任务)
        self._counter = 0  # 序号计数器
        self._lock = threading.Lock()
        self._task_index: Dict[str, Task] = {}  # 任务ID索引
    
    def put(self, task: Task) -> bool:
        """
        添加任务
        
        Args:
            task: 任务
            
        Returns:
            是否成功添加
        """
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            
            # 优先级取负数（因为heapq是最小堆）
            priority = -task.priority.value
            heapq.heappush(self._queue, (priority, self._counter, task))
            self._counter += 1
            self._task_index[task.id] = task
            return True
    
    def get(self) -> Optional[Task]:
        """
        获取下一个任务
        
        Returns:
            任务或None
        """
        with self._lock:
            if not self._queue:
                return None
            
            _, _, task = heapq.heappop(self._queue)
            self._task_index.pop(task.id, None)
            return task
    
    def peek(self) -> Optional[Task]:
        """
        查看下一个任务（不移除）
        
        Returns:
            任务或None
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0][2]
    
    def get_by_id(self, task_id: str) -> Optional[Task]:
        """
        按ID获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务或None
        """
        return self._task_index.get(task_id)
    
    def remove(self, task_id: str) -> Optional[Task]:
        """
        移除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            被移除的任务或None
        """
        with self._lock:
            task = self._task_index.pop(task_id, None)
            if task:
                self._queue = [(p, c, t) for p, c, t in self._queue if t.id != task_id]
                heapq.heapify(self._queue)
            return task
    
    def size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """是否为空"""
        with self._lock:
            return len(self._queue) == 0
    
    def clear(self) -> None:
        """清空队列"""
        with self._lock:
            self._queue.clear()
            self._task_index.clear()
    
    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        with self._lock:
            return [t for _, _, t in self._queue]


# ==================== 任务调度器 ====================

class TaskScheduler:
    """
    任务调度器
    
    负责：
    - 依赖检查
    - 任务调度
    - 执行控制
    """
    
    def __init__(self, strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY):
        """
        初始化调度器
        
        Args:
            strategy: 调度策略
        """
        self.strategy = strategy
        
        # 任务存储
        self._all_tasks: Dict[str, Task] = {}
        
        # 就绪队列
        self._ready_queue = TaskQueue()
        
        # 状态追踪
        self._completed: Set[str] = set()
        self._running: Set[str] = set()
        self._failed: Set[str] = set()
        self._blocked: Dict[str, Set[str]] = {}  # task_id -> blocking_task_ids
        
        # 锁
        self._lock = threading.Lock()
    
    def submit(self, task: Task) -> bool:
        """
        提交任务
        
        Args:
            task: 任务
            
        Returns:
            是否成功提交
        """
        with self._lock:
            self._all_tasks[task.id] = task
            
            # 检查依赖
            unsatisfied = self._check_unsatisfied_dependencies(task)
            
            if unsatisfied:
                # 任务被阻塞
                task.status = TaskStatus.BLOCKED
                self._blocked[task.id] = unsatisfied
            else:
                # 任务就绪
                task.status = TaskStatus.READY
                self._ready_queue.put(task)
            
            return True
    
    def _check_unsatisfied_dependencies(self, task: Task) -> Set[str]:
        """
        检查未满足的依赖
        
        Args:
            task: 任务
            
        Returns:
            未完成的依赖任务ID集合
        """
        unsatisfied = set()
        for dep_id in task.dependencies:
            if dep_id not in self._completed:
                unsatisfied.add(dep_id)
        return unsatisfied
    
    def get_next_task(self) -> Optional[Task]:
        """
        获取下一个待执行的任务
        
        Returns:
            任务或None
        """
        with self._lock:
            task = self._ready_queue.get()
            if task:
                self._running.add(task.id)
            return task
    
    def task_completed(self, task_id: str, result: Any = None) -> None:
        """
        标记任务完成
        
        Args:
            task_id: 任务ID
            result: 执行结果
        """
        with self._lock:
            if task_id in self._all_tasks:
                task = self._all_tasks[task_id]
                task.complete(result)
                
                self._completed.add(task_id)
                self._running.discard(task_id)
                self._failed.discard(task_id)
                
                # 解除阻塞的任务
                self._unblock_tasks(task_id)
    
    def task_failed(self, task_id: str, error: str) -> None:
        """
        标记任务失败
        
        Args:
            task_id: 任务ID
            error: 错误信息
        """
        with self._lock:
            if task_id in self._all_tasks:
                task = self._all_tasks[task_id]
                task.fail(error)
                
                self._running.discard(task_id)
                self._failed.add(task_id)
    
    def _unblock_tasks(self, completed_task_id: str) -> None:
        """
        解除被指定任务阻塞的其他任务
        
        Args:
            completed_task_id: 完成的任务ID
        """
        to_unblock = []
        
        for task_id, blocking_ids in list(self._blocked.items()):
            blocking_ids.discard(completed_task_id)
            
            if not blocking_ids:
                # 所有依赖都已满足
                to_unblock.append(task_id)
        
        for task_id in to_unblock:
            del self._blocked[task_id]
            if task_id in self._all_tasks:
                task = self._all_tasks[task_id]
                task.status = TaskStatus.READY
                self._ready_queue.put(task)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            状态信息
        """
        with self._lock:
            return {
                "total_tasks": len(self._all_tasks),
                "ready": self._ready_queue.size(),
                "running": len(self._running),
                "completed": len(self._completed),
                "failed": len(self._failed),
                "blocked": len(self._blocked)
            }
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务或None
        """
        return self._all_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        with self._lock:
            if task_id in self._all_tasks:
                task = self._all_tasks[task_id]
                task.cancel()
                
                # 从各种集合中移除
                self._running.discard(task_id)
                self._completed.discard(task_id)
                self._failed.discard(task_id)
                self._blocked.pop(task_id, None)
                self._ready_queue.remove(task_id)
                
                return True
            return False


# ==================== 任务管理器 ====================

class TaskManager:
    """
    任务管理器
    
    统一管理任务的提交、调度和执行
    """
    
    def __init__(self, 
                 strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY,
                 max_workers: int = 4,
                 max_retries: int = 3):
        """
        初始化任务管理器
        
        Args:
            strategy: 调度策略
            max_workers: 最大并发数
            max_retries: 最大重试次数
        """
        self.scheduler = TaskScheduler(strategy)
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # 执行器注册表
        self._executors: Dict[str, Callable] = {}
        
        # 执行历史
        self._history: List[ExecutionResult] = []
        
        # 工作线程
        self._workers: List[threading.Thread] = []
        self._running = False
        
        # 结果回调
        self._on_task_complete: Optional[Callable] = None
    
    def register_executor(self, task_type: str, executor: Callable) -> None:
        """
        注册任务执行器
        
        Args:
            task_type: 任务类型
            executor: 执行函数
        """
        self._executors[task_type] = executor
    
    def set_completion_callback(self, callback: Callable) -> None:
        """
        设置任务完成回调
        
        Args:
            callback: 回调函数
        """
        self._on_task_complete = callback
    
    def submit_task(self, task: Task) -> bool:
        """
        提交任务
        
        Args:
            task: 任务
            
        Returns:
            是否成功提交
        """
        return self.scheduler.submit(task)
    
    def submit_tasks(self, tasks: List[Task]) -> int:
        """
        批量提交任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            成功提交的数量
        """
        count = 0
        for task in tasks:
            if self.submit_task(task):
                count += 1
        return count
    
    def execute_task(self, task: Task) -> ExecutionResult:
        """
        执行单个任务
        
        Args:
            task: 任务
            
        Returns:
            执行结果
        """
        result = ExecutionResult(task_id=task.id)
        start_time = time.time()
        
        try:
            task.start()
            
            # 查找执行器
            task_type = task.metadata.get("type", "default")
            executor = self._executors.get(task_type) or self._executors.get("default")
            
            if executor:
                output = executor(task)
                result.success = True
                result.result = output
                task.complete(output)
            else:
                # 没有执行器，使用默认行为
                result.success = True
                result.result = f"任务 '{task.name}' 执行完成（无执行器）"
                task.complete(result.result)
        
        except Exception as e:
            result.success = False
            result.error = str(e)
            task.fail(str(e))
        
        finally:
            result.execution_time = time.time() - start_time
            self._history.append(result)
            
            # 更新调度器状态
            if result.success:
                self.scheduler.task_completed(task.id, result.result)
            else:
                self.scheduler.task_failed(task.id, result.error or "未知错误")
            
            # 回调
            if self._on_task_complete:
                self._on_task_complete(task, result)
        
        return result
    
    def execute_with_retry(self, task: Task) -> ExecutionResult:
        """
        带重试的执行
        
        Args:
            task: 任务
            
        Returns:
            执行结果
        """
        result = ExecutionResult(task_id=task.id)
        
        for attempt in range(self.max_retries):
            result.retries = attempt
            result = self.execute_task(task)
            
            if result.success:
                break
            
            # 重试前等待
            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))  # 指数退避
        
        return result
    
    def run_all(self) -> List[ExecutionResult]:
        """
        执行所有任务
        
        Returns:
            执行结果列表
        """
        results = []
        
        while True:
            task = self.scheduler.get_next_task()
            if task is None:
                # 检查是否还有阻塞的任务
                status = self.scheduler.get_status()
                if status["blocked"] == 0 and status["running"] == 0:
                    break
                time.sleep(0.1)
                continue
            
            result = self.execute_with_retry(task)
            results.append(result)
        
        return results
    
    def run_async(self) -> None:
        """
        异步执行任务（后台线程）
        """
        def worker():
            while self._running:
                task = self.scheduler.get_next_task()
                if task:
                    self.execute_with_retry(task)
                else:
                    time.sleep(0.1)
        
        self._running = True
        for _ in range(self.max_workers):
            worker_thread = threading.Thread(target=worker, daemon=True)
            worker_thread.start()
            self._workers.append(worker_thread)
    
    def stop(self) -> None:
        """停止执行"""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return self.scheduler.get_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计数据
        """
        total = len(self._history)
        success = sum(1 for r in self._history if r.success)
        failed = total - success
        
        total_time = sum(r.execution_time for r in self._history)
        avg_time = total_time / total if total > 0 else 0
        
        total_retries = sum(r.retries for r in self._history)
        
        return {
            "total_executed": total,
            "successful": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "total_retries": total_retries
        }
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionResult]:
        """
        获取执行历史
        
        Args:
            limit: 返回数量
            
        Returns:
            执行结果列表
        """
        return self._history[-limit:]
    
    def create_task(self, 
                    name: str,
                    description: str = "",
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    dependencies: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None) -> Task:
        """
        创建任务的便捷方法
        
        Args:
            name: 任务名称
            description: 任务描述
            priority: 优先级
            dependencies: 依赖
            metadata: 元数据
            
        Returns:
            创建的任务
        """
        task = Task(
            name=name,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        return task


# ==================== 示例执行器 ====================

def sample_executor(task: Task) -> str:
    """示例执行器"""
    print(f"  执行任务: {task.name}")
    time.sleep(0.5)  # 模拟执行
    return f"任务 '{task.name}' 执行成功"


# ==================== 示例和测试 ====================

def demo_task_manager():
    """演示任务管理器"""
    print("=" * 60)
    print("任务管理器演示")
    print("=" * 60)
    
    # 创建任务管理器
    manager = TaskManager(
        strategy=ScheduleStrategy.PRIORITY,
        max_workers=2,
        max_retries=2
    )
    
    # 注册执行器
    manager.register_executor("default", sample_executor)
    
    # 设置回调
    def on_complete(task: Task, result: ExecutionResult):
        status = "✅" if result.success else "❌"
        print(f"  {status} 任务完成: {task.name}")
    
    manager.set_completion_callback(on_complete)
    
    # 1. 创建并提交任务
    print("\n1. 创建并提交任务")
    print("-" * 40)
    
    tasks = [
        manager.create_task(
            name="任务A",
            description="第一个任务",
            priority=TaskPriority.HIGH
        ),
        manager.create_task(
            name="任务B",
            description="第二个任务",
            priority=TaskPriority.MEDIUM,
            dependencies=[]  # 无依赖
        ),
        manager.create_task(
            name="任务C",
            description="依赖任务A的任务",
            priority=TaskPriority.LOW,
            dependencies=[]  # 稍后设置
        ),
        manager.create_task(
            name="任务D",
            description="低优先级任务",
            priority=TaskPriority.LOW
        ),
        manager.create_task(
            name="任务E",
            description="紧急任务",
            priority=TaskPriority.CRITICAL
        )
    ]
    
    # 设置任务C依赖任务A
    tasks[2].dependencies = [tasks[0].id]
    
    # 提交任务
    count = manager.submit_tasks(tasks)
    print(f"提交了 {count} 个任务")
    
    # 2. 查看调度状态
    print("\n2. 调度状态")
    print("-" * 40)
    status = manager.get_status()
    print(f"总任务数: {status['total_tasks']}")
    print(f"就绪: {status['ready']}")
    print(f"运行中: {status['running']}")
    print(f"已完成: {status['completed']}")
    print(f"阻塞: {status['blocked']}")
    
    # 3. 执行所有任务
    print("\n3. 执行任务")
    print("-" * 40)
    
    results = manager.run_all()
    
    # 4. 查看执行结果
    print("\n4. 执行结果统计")
    print("-" * 40)
    stats = manager.get_statistics()
    print(f"执行总数: {stats['total_executed']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"总执行时间: {stats['total_execution_time']:.2f}秒")
    print(f"平均执行时间: {stats['average_execution_time']:.2f}秒")
    
    # 5. 执行历史
    print("\n5. 执行历史（最近5条）")
    print("-" * 40)
    for result in manager.get_execution_history(limit=5):
        status = "✅" if result.success else "❌"
        print(f"  {status} 任务 {result.task_id}: {result.execution_time:.2f}秒, 重试{result.retries}次")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_task_manager()