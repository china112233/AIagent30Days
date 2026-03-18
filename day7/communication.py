"""
Agent 通信模块
实现消息定义、消息总线、发布-订阅机制、共享内存
"""

import uuid
import time
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty
from collections import defaultdict
import json


# ==================== 消息类型 ====================

class MessageType(Enum):
    """消息类型枚举"""
    # 任务相关
    TASK_ASSIGN = "task_assign"           # 任务分配
    TASK_STATUS = "task_status"           # 任务状态更新
    TASK_RESULT = "task_result"           # 任务结果
    TASK_ERROR = "task_error"             # 任务错误

    # 协作相关
    COLLABORATION_REQUEST = "collab_request"   # 协作请求
    COLLABORATION_RESPONSE = "collab_response" # 协作响应
    INFORMATION_SHARE = "info_share"           # 信息共享

    # 控制相关
    HEARTBEAT = "heartbeat"               # 心跳
    SHUTDOWN = "shutdown"                 # 关闭指令
    STATUS_QUERY = "status_query"         # 状态查询
    STATUS_RESPONSE = "status_response"   # 状态响应

    # 共识相关
    PROPOSAL = "proposal"                 # 提议
    VOTE = "vote"                         # 投票
    CONSENSUS_RESULT = "consensus_result" # 共识结果


# ==================== 消息定义 ====================

@dataclass
class Message:
    """
    消息数据结构

    Attributes:
        id: 消息唯一ID
        sender: 发送者ID
        receiver: 接收者ID (广播时为 "broadcast")
        type: 消息类型
        content: 消息内容
        timestamp: 时间戳
        priority: 优先级 (1-10, 数字越大优先级越高)
        requires_response: 是否需要响应
        correlation_id: 关联消息ID (用于请求-响应模式)
        metadata: 元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    receiver: str = ""
    type: MessageType = MessageType.INFORMATION_SHARE
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 5
    requires_response: bool = False
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            type=MessageType(data.get("type", "info_share")),
            content=data.get("content", {}),
            timestamp=data.get("timestamp", time.time()),
            priority=data.get("priority", 5),
            requires_response=data.get("requires_response", False),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {})
        )

    def create_response(self, content: Dict[str, Any]) -> "Message":
        """
        创建响应消息

        Args:
            content: 响应内容

        Returns:
            响应消息
        """
        return Message(
            sender=self.receiver,
            receiver=self.sender,
            type=MessageType.INFORMATION_SHARE,
            content=content,
            correlation_id=self.id,
            requires_response=False
        )


# ==================== 消息队列 ====================

class MessageQueue:
    """
    消息队列
    支持优先级排序和阻塞获取
    """

    def __init__(self, max_size: int = 1000):
        """
        初始化消息队列

        Args:
            max_size: 队列最大容量
        """
        self.queue: List[Message] = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

    def put(self, message: Message) -> bool:
        """
        放入消息

        Args:
            message: 消息对象

        Returns:
            是否成功
        """
        with self.lock:
            if len(self.queue) >= self.max_size:
                return False

            # 按优先级插入
            inserted = False
            for i, m in enumerate(self.queue):
                if message.priority > m.priority:
                    self.queue.insert(i, message)
                    inserted = True
                    break

            if not inserted:
                self.queue.append(message)

            self.not_empty.notify()
            return True

    def get(self, timeout: float = None) -> Optional[Message]:
        """
        获取消息

        Args:
            timeout: 超时时间（秒）

        Returns:
            消息对象或 None
        """
        with self.not_empty:
            if not self.queue:
                self.not_empty.wait(timeout)

            if self.queue:
                return self.queue.pop(0)
            return None

    def peek(self) -> Optional[Message]:
        """查看队首消息（不移除）"""
        with self.lock:
            return self.queue[0] if self.queue else None

    def size(self) -> int:
        """获取队列大小"""
        with self.lock:
            return len(self.queue)

    def clear(self):
        """清空队列"""
        with self.lock:
            self.queue.clear()


# ==================== 消息总线 ====================

class MessageBus:
    """
    消息总线
    实现 Agent 间的消息传递
    """

    def __init__(self):
        """初始化消息总线"""
        # 每个 Agent 的消息队列
        self.agent_queues: Dict[str, MessageQueue] = {}
        # 广播订阅者
        self.broadcast_subscribers: List[str] = []
        # 消息处理器
        self.handlers: Dict[str, Callable] = {}
        # 消息历史
        self.message_history: List[Message] = []
        self.max_history = 1000
        # 锁
        self.lock = threading.Lock()

    def register_agent(self, agent_id: str) -> bool:
        """
        注册 Agent

        Args:
            agent_id: Agent ID

        Returns:
            是否成功
        """
        with self.lock:
            if agent_id in self.agent_queues:
                return False
            self.agent_queues[agent_id] = MessageQueue()
            self.broadcast_subscribers.append(agent_id)
            return True

    def unregister_agent(self, agent_id: str):
        """
        注销 Agent

        Args:
            agent_id: Agent ID
        """
        with self.lock:
            if agent_id in self.agent_queues:
                del self.agent_queues[agent_id]
            if agent_id in self.broadcast_subscribers:
                self.broadcast_subscribers.remove(agent_id)

    def send(self, message: Message) -> bool:
        """
        发送消息

        Args:
            message: 消息对象

        Returns:
            是否成功
        """
        with self.lock:
            # 记录历史
            self.message_history.append(message)
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history:]

            # 广播消息
            if message.receiver == "broadcast":
                for agent_id in self.broadcast_subscribers:
                    if agent_id != message.sender:
                        queue = self.agent_queues.get(agent_id)
                        if queue:
                            queue.put(message)
                return True

            # 点对点消息
            if message.receiver in self.agent_queues:
                return self.agent_queues[message.receiver].put(message)

            return False

    def receive(self, agent_id: str, timeout: float = None) -> Optional[Message]:
        """
        接收消息

        Args:
            agent_id: Agent ID
            timeout: 超时时间

        Returns:
            消息对象或 None
        """
        queue = self.agent_queues.get(agent_id)
        if queue:
            return queue.get(timeout)
        return None

    def broadcast(self, sender: str, content: Dict[str, Any],
                  msg_type: MessageType = MessageType.INFORMATION_SHARE) -> bool:
        """
        广播消息

        Args:
            sender: 发送者
            content: 消息内容
            msg_type: 消息类型

        Returns:
            是否成功
        """
        message = Message(
            sender=sender,
            receiver="broadcast",
            type=msg_type,
            content=content
        )
        return self.send(message)

    def get_queue_size(self, agent_id: str) -> int:
        """获取 Agent 消息队列大小"""
        queue = self.agent_queues.get(agent_id)
        return queue.size() if queue else 0

    def get_history(self, agent_id: str = None, limit: int = 50) -> List[Message]:
        """
        获取消息历史

        Args:
            agent_id: Agent ID (可选，过滤特定 Agent 的消息)
            limit: 返回数量限制

        Returns:
            消息列表
        """
        with self.lock:
            if agent_id:
                history = [m for m in self.message_history
                          if m.sender == agent_id or m.receiver == agent_id]
            else:
                history = self.message_history.copy()

            return history[-limit:]


# ==================== 发布-订阅系统 ====================

class PubSubSystem:
    """
    发布-订阅系统
    支持事件驱动的 Agent 通信
    """

    def __init__(self):
        """初始化发布-订阅系统"""
        # 主题订阅者映射
        self.topics: Dict[str, List[str]] = defaultdict(list)
        # Agent 回调函数
        self.callbacks: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        # 消息总线
        self.message_bus: Optional[MessageBus] = None

    def set_message_bus(self, bus: MessageBus):
        """设置消息总线"""
        self.message_bus = bus

    def subscribe(self, agent_id: str, topic: str, callback: Callable = None):
        """
        订阅主题

        Args:
            agent_id: Agent ID
            topic: 主题名称
            callback: 回调函数 (可选)
        """
        if agent_id not in self.topics[topic]:
            self.topics[topic].append(agent_id)

        if callback:
            self.callbacks[agent_id][topic] = callback

    def unsubscribe(self, agent_id: str, topic: str = None):
        """
        取消订阅

        Args:
            agent_id: Agent ID
            topic: 主题名称 (None 表示取消所有订阅)
        """
        if topic:
            if agent_id in self.topics[topic]:
                self.topics[topic].remove(agent_id)
            if topic in self.callbacks[agent_id]:
                del self.callbacks[agent_id][topic]
        else:
            for t in list(self.topics.keys()):
                if agent_id in self.topics[t]:
                    self.topics[t].remove(agent_id)
            self.callbacks[agent_id].clear()

    def publish(self, topic: str, content: Dict[str, Any], sender: str = "system"):
        """
        发布消息

        Args:
            topic: 主题名称
            content: 消息内容
            sender: 发送者
        """
        subscribers = self.topics.get(topic, [])

        for agent_id in subscribers:
            if agent_id == sender:
                continue

            # 调用回调函数
            if agent_id in self.callbacks and topic in self.callbacks[agent_id]:
                try:
                    self.callbacks[agent_id][topic](topic, content)
                except Exception as e:
                    print(f"[PubSub] Callback error for {agent_id}: {e}")

            # 发送消息
            if self.message_bus:
                message = Message(
                    sender=sender,
                    receiver=agent_id,
                    type=MessageType.INFORMATION_SHARE,
                    content={"topic": topic, "data": content}
                )
                self.message_bus.send(message)

    def get_topics(self) -> List[str]:
        """获取所有主题"""
        return list(self.topics.keys())

    def get_subscribers(self, topic: str) -> List[str]:
        """获取主题订阅者"""
        return self.topics.get(topic, []).copy()


# ==================== 共享内存 ====================

class SharedMemory:
    """
    共享内存
    实现 Agent 间的数据共享
    """

    def __init__(self):
        """初始化共享内存"""
        self.data: Dict[str, Any] = {}
        self.version: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()
        # 数据变更监听器
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)

    def set(self, key: str, value: Any, agent_id: str = None) -> bool:
        """
        设置值

        Args:
            key: 键
            value: 值
            agent_id: 设置者 ID (可选)

        Returns:
            是否成功
        """
        with self.lock:
            self.data[key] = value
            self.version[key] += 1

            # 触发监听器
            for listener in self.listeners[key]:
                try:
                    listener(key, value, agent_id)
                except Exception as e:
                    print(f"[SharedMemory] Listener error: {e}")

            return True

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取值

        Args:
            key: 键
            default: 默认值

        Returns:
            值
        """
        with self.lock:
            return self.data.get(key, default)

    def delete(self, key: str) -> bool:
        """
        删除值

        Args:
            key: 键

        Returns:
            是否成功
        """
        with self.lock:
            if key in self.data:
                del self.data[key]
                del self.version[key]
                return True
            return False

    def update(self, updates: Dict[str, Any], agent_id: str = None):
        """
        批量更新

        Args:
            updates: 更新字典
            agent_id: 设置者 ID
        """
        with self.lock:
            for key, value in updates.items():
                self.set(key, value, agent_id)

    def listen(self, key: str, callback: Callable):
        """
        监听数据变更

        Args:
            key: 键
            callback: 回调函数 (key, value, agent_id)
        """
        self.listeners[key].append(callback)

    def unlisten(self, key: str, callback: Callable = None):
        """
        取消监听

        Args:
            key: 键
            callback: 回调函数 (None 表示移除所有)
        """
        if callback:
            if callback in self.listeners[key]:
                self.listeners[key].remove(callback)
        else:
            self.listeners[key].clear()

    def get_version(self, key: str) -> int:
        """获取数据版本"""
        return self.version.get(key, 0)

    def snapshot(self) -> Dict[str, Any]:
        """获取内存快照"""
        with self.lock:
            return {
                "data": self.data.copy(),
                "version": dict(self.version)
            }

    def clear(self):
        """清空共享内存"""
        with self.lock:
            self.data.clear()
            self.version.clear()


# ==================== 黑板系统 ====================

class Blackboard:
    """
    黑板系统
    实现基于共享工作空间的协作
    """

    def __init__(self):
        """初始化黑板"""
        self.entries: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        # 黑板观察者
        self.observers: List[Callable] = []

    def write(self, key: str, value: Any, agent_id: str,
              entry_type: str = "data") -> bool:
        """
        写入黑板

        Args:
            key: 键
            value: 值
            agent_id: 写入者 ID
            entry_type: 条目类型

        Returns:
            是否成功
        """
        with self.lock:
            entry = {
                "key": key,
                "value": value,
                "author": agent_id,
                "type": entry_type,
                "timestamp": time.time(),
                "version": self.entries.get(key, {}).get("version", 0) + 1
            }
            self.entries[key] = entry

            # 通知观察者
            for observer in self.observers:
                try:
                    observer("write", entry)
                except Exception as e:
                    print(f"[Blackboard] Observer error: {e}")

            return True

    def read(self, key: str) -> Optional[Dict]:
        """
        读取黑板条目

        Args:
            key: 键

        Returns:
            条目字典或 None
        """
        with self.lock:
            return self.entries.get(key)

    def read_all(self) -> Dict[str, Dict]:
        """读取所有条目"""
        with self.lock:
            return self.entries.copy()

    def delete(self, key: str, agent_id: str = None) -> bool:
        """
        删除条目

        Args:
            key: 键
            agent_id: 删除者 ID

        Returns:
            是否成功
        """
        with self.lock:
            if key in self.entries:
                entry = self.entries[key]
                del self.entries[key]

                for observer in self.observers:
                    try:
                        observer("delete", {"key": key, "entry": entry, "by": agent_id})
                    except Exception as e:
                        print(f"[Blackboard] Observer error: {e}")

                return True
            return False

    def search(self, entry_type: str = None, author: str = None) -> List[Dict]:
        """
        搜索条目

        Args:
            entry_type: 条目类型
            author: 作者

        Returns:
            匹配的条目列表
        """
        with self.lock:
            results = []
            for entry in self.entries.values():
                if entry_type and entry.get("type") != entry_type:
                    continue
                if author and entry.get("author") != author:
                    continue
                results.append(entry)
            return results

    def subscribe(self, observer: Callable):
        """订阅黑板变更"""
        self.observers.append(observer)

    def unsubscribe(self, observer: Callable):
        """取消订阅"""
        if observer in self.observers:
            self.observers.remove(observer)


# ==================== 通信管理器 ====================

class CommunicationManager:
    """
    通信管理器
    统一管理所有通信组件
    """

    def __init__(self):
        """初始化通信管理器"""
        self.message_bus = MessageBus()
        self.pubsub = PubSubSystem()
        self.shared_memory = SharedMemory()
        self.blackboard = Blackboard()

        # 关联组件
        self.pubsub.set_message_bus(self.message_bus)

    def register_agent(self, agent_id: str):
        """注册 Agent 到通信系统"""
        self.message_bus.register_agent(agent_id)

    def unregister_agent(self, agent_id: str):
        """从通信系统注销 Agent"""
        self.message_bus.unregister_agent(agent_id)
        self.pubsub.unsubscribe(agent_id)

    def send_message(self, message: Message) -> bool:
        """发送消息"""
        return self.message_bus.send(message)

    def receive_message(self, agent_id: str, timeout: float = None) -> Optional[Message]:
        """接收消息"""
        return self.message_bus.receive(agent_id, timeout)

    def publish_event(self, topic: str, content: Dict, sender: str = "system"):
        """发布事件"""
        self.pubsub.publish(topic, content, sender)

    def subscribe_topic(self, agent_id: str, topic: str, callback: Callable = None):
        """订阅主题"""
        self.pubsub.subscribe(agent_id, topic, callback)

    def set_shared_data(self, key: str, value: Any, agent_id: str = None):
        """设置共享数据"""
        self.shared_memory.set(key, value, agent_id)

    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """获取共享数据"""
        return self.shared_memory.get(key, default)

    def write_blackboard(self, key: str, value: Any, agent_id: str):
        """写入黑板"""
        self.blackboard.write(key, value, agent_id)

    def read_blackboard(self, key: str) -> Optional[Dict]:
        """读取黑板"""
        return self.blackboard.read(key)


# ==================== 演示函数 ====================

def demo_message_bus():
    """演示消息总线"""
    print("\n" + "=" * 60)
    print("消息总线演示")
    print("=" * 60)

    bus = MessageBus()

    # 注册 Agent
    bus.register_agent("agent_a")
    bus.register_agent("agent_b")
    bus.register_agent("agent_c")

    print("\n[1] 点对点消息")
    msg = Message(
        sender="agent_a",
        receiver="agent_b",
        type=MessageType.TASK_ASSIGN,
        content={"task": "分析数据", "priority": "high"}
    )
    bus.send(msg)

    received = bus.receive("agent_b", timeout=1)
    if received:
        print(f"  agent_b 收到消息: {received.content}")

    print("\n[2] 广播消息")
    bus.broadcast("agent_a", {"event": "任务更新", "status": "进行中"})
    print(f"  agent_b 队列大小: {bus.get_queue_size('agent_b')}")
    print(f"  agent_c 队列大小: {bus.get_queue_size('agent_c')}")


def demo_pubsub():
    """演示发布-订阅"""
    print("\n" + "=" * 60)
    print("发布-订阅系统演示")
    print("=" * 60)

    pubsub = PubSubSystem()

    # 定义回调函数
    def on_event(topic, data):
        print(f"  [回调] 收到主题 '{topic}': {data}")

    # 订阅
    pubsub.subscribe("agent_a", "task_updates", on_event)
    pubsub.subscribe("agent_b", "task_updates")
    pubsub.subscribe("agent_c", "system_alerts")

    print("\n[1] 订阅情况:")
    print(f"  task_updates 订阅者: {pubsub.get_subscribers('task_updates')}")
    print(f"  system_alerts 订阅者: {pubsub.get_subscribers('system_alerts')}")

    print("\n[2] 发布消息:")
    pubsub.publish("task_updates", {"task_id": "T001", "status": "completed"}, "system")


def demo_shared_memory():
    """演示共享内存"""
    print("\n" + "=" * 60)
    print("共享内存演示")
    print("=" * 60)

    memory = SharedMemory()

    # 监听变更
    def on_change(key, value, agent_id):
        print(f"  [变更] {key} = {value} (by {agent_id})")

    memory.listen("project_status", on_change)

    print("\n[1] 写入数据:")
    memory.set("project_status", "进行中", "agent_a")
    memory.set("progress", 0.6, "agent_b")

    print("\n[2] 读取数据:")
    print(f"  project_status: {memory.get('project_status')}")
    print(f"  progress: {memory.get('progress')}")

    print("\n[3] 版本信息:")
    print(f"  project_status 版本: {memory.get_version('project_status')}")


def demo_blackboard():
    """演示黑板系统"""
    print("\n" + "=" * 60)
    print("黑板系统演示")
    print("=" * 60)

    bb = Blackboard()

    # 观察者
    def observer(action, entry):
        print(f"  [观察] {action}: {entry}")

    bb.subscribe(observer)

    print("\n[1] 写入黑板:")
    bb.write("task_analysis", {"complexity": "high", "domain": "NLP"}, "researcher")
    bb.write("code_design", {"architecture": "microservices"}, "architect")

    print("\n[2] 读取黑板:")
    entry = bb.read("task_analysis")
    print(f"  task_analysis: {entry}")

    print("\n[3] 搜索黑板:")
    all_entries = bb.read_all()
    print(f"  所有条目: {list(all_entries.keys())}")


def demo_communication_manager():
    """演示通信管理器"""
    print("\n" + "=" * 60)
    print("通信管理器演示")
    print("=" * 60)

    cm = CommunicationManager()

    # 注册 Agent
    cm.register_agent("orchestrator")
    cm.register_agent("researcher")
    cm.register_agent("coder")

    print("\n[1] 发送任务消息:")
    msg = Message(
        sender="orchestrator",
        receiver="researcher",
        type=MessageType.TASK_ASSIGN,
        content={"task": "研究异步编程模式"}
    )
    cm.send_message(msg)

    received = cm.receive_message("researcher", timeout=1)
    print(f"  研究员收到: {received.content if received else '无消息'}")

    print("\n[2] 使用共享内存:")
    cm.set_shared_data("global_config", {"model": "gpt-4"}, "orchestrator")
    config = cm.get_shared_data("global_config")
    print(f"  全局配置: {config}")

    print("\n[3] 使用黑板:")
    cm.write_blackboard("analysis_result", {"findings": ["...", "..."]}, "researcher")
    result = cm.read_blackboard("analysis_result")
    print(f"  分析结果: {result}")


if __name__ == "__main__":
    # 演示各组件
    demo_message_bus()
    demo_pubsub()
    demo_shared_memory()
    demo_blackboard()
    demo_communication_manager()