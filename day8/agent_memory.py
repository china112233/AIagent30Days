"""
Agent 记忆系统模块
实现短期记忆、长期记忆、记忆检索和管理
"""

import uuid
import time
import json
import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import heapq


# ==================== 记忆类型枚举 ====================

class MemoryType(Enum):
    """记忆类型枚举"""
    # 短期记忆
    WORKING = "working"           # 工作记忆
    SENSORY = "sensory"           # 感知缓冲
    
    # 长期记忆
    EPISODIC = "episodic"         # 情景记忆
    SEMANTIC = "semantic"         # 语义记忆
    PROCEDURAL = "procedural"     # 程序记忆


class MemoryImportance(Enum):
    """记忆重要性级别"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ==================== 记忆项数据结构 ====================

@dataclass
class MemoryItem:
    """
    记忆项数据结构
    
    Attributes:
        id: 记忆唯一ID
        content: 记忆内容
        memory_type: 记忆类型
        importance: 重要性评分 (0.0-1.0)
        created_at: 创建时间
        last_accessed: 最后访问时间
        access_count: 访问次数
        decay_rate: 衰减率
        embedding: 向量嵌入 (用于相似性检索)
        associations: 关联记忆ID列表
        metadata: 元数据
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: Any = None
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    decay_rate: float = 0.1
    embedding: Optional[List[float]] = None
    associations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """访问记忆，更新访问信息"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_strength(self) -> float:
        """
        计算记忆强度
        
        综合考虑重要性、访问频率和时间衰减
        
        Returns:
            记忆强度值 (0.0-1.0)
        """
        # 时间衰减
        time_elapsed = time.time() - self.created_at
        time_decay = math.exp(-self.decay_rate * time_elapsed / 3600)  # 小时为单位
        
        # 访问增强
        access_boost = min(1.0, self.access_count * 0.1)
        
        # 综合强度
        strength = self.importance * 0.5 + time_decay * 0.3 + access_boost * 0.2
        return min(1.0, strength)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "decay_rate": self.decay_rate,
            "embedding": self.embedding,
            "associations": self.associations,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            content=data.get("content"),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
            decay_rate=data.get("decay_rate", 0.1),
            embedding=data.get("embedding"),
            associations=data.get("associations", []),
            metadata=data.get("metadata", {})
        )


# ==================== 短期记忆 ====================

class ShortTermMemory:
    """
    短期记忆系统
    
    实现：
    - 工作记忆：当前正在处理的信息
    - 感知缓冲：最近的感知输入
    
    特点：
    - 容量有限 (Miller's Law: 7±2)
    - 快速访问
    - 自动衰减
    """
    
    def __init__(self, capacity: int = 7, buffer_size: int = 20):
        """
        初始化短期记忆
        
        Args:
            capacity: 工作记忆容量 (默认7)
            buffer_size: 感知缓冲大小 (默认20)
        """
        self.capacity = capacity
        self.buffer_size = buffer_size
        
        # 工作记忆 (容量有限，按重要性排序)
        self.working_memory: Dict[str, MemoryItem] = {}
        
        # 感知缓冲 (滑动窗口)
        self.sensory_buffer: deque = deque(maxlen=buffer_size)
        
        # 记忆ID到位置的映射
        self._attention_weights: Dict[str, float] = {}
    
    def add_to_working(self, item: MemoryItem) -> bool:
        """
        添加到工作记忆
        
        Args:
            item: 记忆项
            
        Returns:
            是否成功添加
        """
        # 如果已满，移除最不重要的
        if len(self.working_memory) >= self.capacity:
            self._evict_least_important()
        
        item.memory_type = MemoryType.WORKING
        self.working_memory[item.id] = item
        self._attention_weights[item.id] = item.importance
        return True
    
    def add_to_sensory(self, item: MemoryItem) -> None:
        """
        添加到感知缓冲
        
        Args:
            item: 记忆项
        """
        item.memory_type = MemoryType.SENSORY
        self.sensory_buffer.append(item)
    
    def _evict_least_important(self) -> Optional[MemoryItem]:
        """
        驱逐最不重要的记忆
        
        Returns:
            被驱逐的记忆项
        """
        if not self.working_memory:
            return None
        
        # 找到最不重要的
        min_id = min(self._attention_weights, key=self._attention_weights.get)
        evicted = self.working_memory.pop(min_id, None)
        self._attention_weights.pop(min_id, None)
        return evicted
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """
        获取记忆项
        
        Args:
            item_id: 记忆ID
            
        Returns:
            记忆项或None
        """
        # 先查工作记忆
        if item_id in self.working_memory:
            item = self.working_memory[item_id]
            item.access()
            return item
        
        # 再查感知缓冲
        for item in self.sensory_buffer:
            if item.id == item_id:
                item.access()
                return item
        
        return None
    
    def recall_recent(self, n: int = 5) -> List[MemoryItem]:
        """
        回忆最近的记忆
        
        Args:
            n: 数量
            
        Returns:
            最近的记忆列表
        """
        # 从感知缓冲获取最近的
        recent = list(self.sensory_buffer)[-n:]
        
        # 补充工作记忆
        if len(recent) < n:
            working_items = sorted(
                self.working_memory.values(),
                key=lambda x: x.last_accessed,
                reverse=True
            )[:n - len(recent)]
            recent.extend(working_items)
        
        return recent
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        搜索记忆
        
        Args:
            query: 查询内容
            top_k: 返回数量
            
        Returns:
            (记忆项, 相似度) 列表
        """
        results = []
        query_lower = query.lower()
        
        # 搜索工作记忆
        for item in self.working_memory.values():
            content_str = str(item.content).lower()
            if query_lower in content_str:
                score = self._calculate_relevance(item, query)
                results.append((item, score))
        
        # 搜索感知缓冲
        for item in self.sensory_buffer:
            content_str = str(item.content).lower()
            if query_lower in content_str:
                score = self._calculate_relevance(item, query)
                results.append((item, score))
        
        # 按相关度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _calculate_relevance(self, item: MemoryItem, query: str) -> float:
        """
        计算相关度
        
        Args:
            item: 记忆项
            query: 查询
            
        Returns:
            相关度分数
        """
        # 简单的关键词匹配 + 时间衰减
        content_str = str(item.content).lower()
        query_lower = query.lower()
        
        # 关键词匹配度
        match_score = 1.0 if query_lower in content_str else 0.0
        
        # 时间衰减
        time_elapsed = time.time() - item.created_at
        time_decay = math.exp(-item.decay_rate * time_elapsed / 60)  # 分钟为单位
        
        return match_score * item.importance * time_decay
    
    def consolidate(self, threshold: float = 0.7) -> List[MemoryItem]:
        """
        记忆巩固 - 将重要的短期记忆转为长期记忆候选
        
        Args:
            threshold: 巩固阈值
            
        Returns:
            需要巩固的记忆列表
        """
        to_consolidate = []
        
        for item in self.working_memory.values():
            if item.get_strength() >= threshold:
                to_consolidate.append(item)
        
        return to_consolidate
    
    def clear(self) -> None:
        """清空短期记忆"""
        self.working_memory.clear()
        self.sensory_buffer.clear()
        self._attention_weights.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "working_memory_size": len(self.working_memory),
            "capacity": self.capacity,
            "sensory_buffer_size": len(self.sensory_buffer),
            "buffer_capacity": self.buffer_size,
            "utilization": len(self.working_memory) / self.capacity
        }


# ==================== 长期记忆 ====================

class LongTermMemory:
    """
    长期记忆系统
    
    实现：
    - 情景记忆：个人经历和事件
    - 语义记忆：知识和概念
    - 程序记忆：技能和方法
    
    特点：
    - 容量几乎无限
    - 持久存储
    - 多种检索方式
    """
    
    def __init__(self, max_size: int = 10000):
        """
        初始化长期记忆
        
        Args:
            max_size: 最大存储数量
        """
        self.max_size = max_size
        
        # 按类型组织的记忆存储
        self.episodic_memory: Dict[str, MemoryItem] = {}  # 情景记忆
        self.semantic_memory: Dict[str, MemoryItem] = {}   # 语义记忆
        self.procedural_memory: Dict[str, MemoryItem] = {} # 程序记忆
        
        # 索引
        self._keyword_index: Dict[str, List[str]] = {}     # 关键词索引
        self._time_index: Dict[str, List[str]] = {}        # 时间索引
        self._association_graph: Dict[str, List[str]] = {} # 关联图
    
    def store(self, item: MemoryItem) -> bool:
        """
        存储记忆
        
        Args:
            item: 记忆项
            
        Returns:
            是否成功存储
        """
        if self._get_total_count() >= self.max_size:
            self._forget_least_important()
        
        # 根据类型存储
        if item.memory_type == MemoryType.EPISODIC:
            self.episodic_memory[item.id] = item
        elif item.memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[item.id] = item
        elif item.memory_type == MemoryType.PROCEDURAL:
            self.procedural_memory[item.id] = item
        else:
            # 默认存储为情景记忆
            item.memory_type = MemoryType.EPISODIC
            self.episodic_memory[item.id] = item
        
        # 更新索引
        self._update_indices(item)
        
        return True
    
    def _get_total_count(self) -> int:
        """获取总记忆数量"""
        return (
            len(self.episodic_memory) + 
            len(self.semantic_memory) + 
            len(self.procedural_memory)
        )
    
    def _forget_least_important(self) -> Optional[MemoryItem]:
        """遗忘最不重要的记忆"""
        all_items = [
            *self.episodic_memory.values(),
            *self.semantic_memory.values(),
            *self.procedural_memory.values()
        ]
        
        if not all_items:
            return None
        
        # 找到最不重要的
        min_item = min(all_items, key=lambda x: x.get_strength())
        
        # 从对应存储中删除
        if min_item.id in self.episodic_memory:
            del self.episodic_memory[min_item.id]
        elif min_item.id in self.semantic_memory:
            del self.semantic_memory[min_item.id]
        elif min_item.id in self.procedural_memory:
            del self.procedural_memory[min_item.id]
        
        return min_item
    
    def _update_indices(self, item: MemoryItem) -> None:
        """更新索引"""
        # 关键词索引
        content_str = str(item.content).lower()
        words = content_str.split()
        for word in words:
            if len(word) > 2:  # 忽略短词
                if word not in self._keyword_index:
                    self._keyword_index[word] = []
                self._keyword_index[word].append(item.id)
        
        # 时间索引
        date_key = datetime.fromtimestamp(item.created_at).strftime("%Y-%m-%d")
        if date_key not in self._time_index:
            self._time_index[date_key] = []
        self._time_index[date_key].append(item.id)
        
        # 关联图
        for assoc_id in item.associations:
            if item.id not in self._association_graph:
                self._association_graph[item.id] = []
            self._association_graph[item.id].append(assoc_id)
            
            if assoc_id not in self._association_graph:
                self._association_graph[assoc_id] = []
            self._association_graph[assoc_id].append(item.id)
    
    def recall_by_type(self, memory_type: MemoryType, limit: int = 10) -> List[MemoryItem]:
        """
        按类型检索记忆
        
        Args:
            memory_type: 记忆类型
            limit: 返回数量
            
        Returns:
            记忆列表
        """
        if memory_type == MemoryType.EPISODIC:
            storage = self.episodic_memory
        elif memory_type == MemoryType.SEMANTIC:
            storage = self.semantic_memory
        elif memory_type == MemoryType.PROCEDURAL:
            storage = self.procedural_memory
        else:
            return []
        
        items = sorted(
            storage.values(),
            key=lambda x: x.get_strength(),
            reverse=True
        )[:limit]
        
        for item in items:
            item.access()
        
        return items
    
    def recall_by_keyword(self, keyword: str, limit: int = 10) -> List[MemoryItem]:
        """
        按关键词检索
        
        Args:
            keyword: 关键词
            limit: 返回数量
            
        Returns:
            记忆列表
        """
        keyword_lower = keyword.lower()
        
        # 从索引中查找
        matched_ids = set()
        for word, ids in self._keyword_index.items():
            if keyword_lower in word:
                matched_ids.update(ids)
        
        # 获取记忆项
        items = []
        all_storage = {
            **self.episodic_memory,
            **self.semantic_memory,
            **self.procedural_memory
        }
        
        for item_id in matched_ids:
            if item_id in all_storage:
                items.append(all_storage[item_id])
        
        # 按强度排序
        items.sort(key=lambda x: x.get_strength(), reverse=True)
        
        for item in items[:limit]:
            item.access()
        
        return items[:limit]
    
    def recall_by_time(self, start_time: float, end_time: float) -> List[MemoryItem]:
        """
        按时间范围检索
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            记忆列表
        """
        items = []
        all_storage = [
            *self.episodic_memory.values(),
            *self.semantic_memory.values(),
            *self.procedural_memory.values()
        ]
        
        for item in all_storage:
            if start_time <= item.created_at <= end_time:
                items.append(item)
        
        items.sort(key=lambda x: x.created_at, reverse=True)
        
        for item in items:
            item.access()
        
        return items
    
    def recall_by_association(self, item_id: str, depth: int = 1) -> List[MemoryItem]:
        """
        按关联检索
        
        Args:
            item_id: 起始记忆ID
            depth: 搜索深度
            
        Returns:
            关联的记忆列表
        """
        visited = set()
        result = []
        queue = [(item_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            
            # 查找记忆项
            all_storage = {
                **self.episodic_memory,
                **self.semantic_memory,
                **self.procedural_memory
            }
            
            if current_id in all_storage:
                result.append(all_storage[current_id])
                all_storage[current_id].access()
            
            # 添加关联记忆
            if current_id in self._association_graph:
                for assoc_id in self._association_graph[current_id]:
                    if assoc_id not in visited:
                        queue.append((assoc_id, current_depth + 1))
        
        return result
    
    def consolidate_from_short_term(self, items: List[MemoryItem]) -> int:
        """
        从短期记忆巩固到长期记忆
        
        Args:
            items: 需要巩固的记忆列表
            
        Returns:
            成功巩固的数量
        """
        count = 0
        for item in items:
            # 提升重要性
            item.importance = min(1.0, item.importance + 0.1)
            # 降低衰减率
            item.decay_rate = max(0.01, item.decay_rate - 0.02)
            
            if self.store(item):
                count += 1
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_count": self._get_total_count(),
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "procedural_count": len(self.procedural_memory),
            "keyword_index_size": len(self._keyword_index),
            "time_index_size": len(self._time_index),
            "association_count": len(self._association_graph)
        }


# ==================== 记忆管理器 ====================

class MemoryManager:
    """
    记忆管理器
    
    统一管理短期记忆和长期记忆
    提供统一的记忆存储和检索接口
    """
    
    def __init__(self, stm_capacity: int = 7, ltm_capacity: int = 10000):
        """
        初始化记忆管理器
        
        Args:
            stm_capacity: 短期记忆容量
            ltm_capacity: 长期记忆容量
        """
        self.short_term = ShortTermMemory(capacity=stm_capacity)
        self.long_term = LongTermMemory(max_size=ltm_capacity)
        
        # 巩固配置
        self._consolidation_threshold = 0.7
        self._consolidation_interval = 100  # 操作次数
        self._operation_count = 0
    
    def remember(self, 
                 content: Any, 
                 memory_type: str = "episodic",
                 importance: float = 0.5,
                 metadata: Optional[Dict] = None) -> MemoryItem:
        """
        记忆存储的统一接口
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 ("working", "sensory", "episodic", "semantic", "procedural")
            importance: 重要性 (0.0-1.0)
            metadata: 元数据
            
        Returns:
            创建的记忆项
        """
        # 创建记忆项
        item = MemoryItem(
            content=content,
            memory_type=MemoryType(memory_type),
            importance=importance,
            metadata=metadata or {}
        )
        
        # 根据类型存储
        if memory_type in ["working", "sensory"]:
            if memory_type == "working":
                self.short_term.add_to_working(item)
            else:
                self.short_term.add_to_sensory(item)
        else:
            self.long_term.store(item)
            # 同时添加到感知缓冲
            self.short_term.add_to_sensory(item)
        
        # 检查是否需要巩固
        self._operation_count += 1
        if self._operation_count % self._consolidation_interval == 0:
            self._auto_consolidate()
        
        return item
    
    def recall(self, 
               query: str, 
               search_type: str = "all",
               limit: int = 10) -> List[MemoryItem]:
        """
        记忆检索的统一接口
        
        Args:
            query: 查询内容
            search_type: 搜索类型 ("all", "short_term", "long_term", "episodic", "semantic", "procedural")
            limit: 返回数量
            
        Returns:
            记忆列表
        """
        results = []
        
        if search_type in ["all", "short_term"]:
            stm_results = self.short_term.search(query, top_k=limit)
            results.extend([item for item, _ in stm_results])
        
        if search_type in ["all", "long_term"]:
            ltm_results = self.long_term.recall_by_keyword(query, limit=limit)
            results.extend(ltm_results)
        
        if search_type == "episodic":
            results.extend(self.long_term.recall_by_type(MemoryType.EPISODIC, limit))
        elif search_type == "semantic":
            results.extend(self.long_term.recall_by_type(MemoryType.SEMANTIC, limit))
        elif search_type == "procedural":
            results.extend(self.long_term.recall_by_type(MemoryType.PROCEDURAL, limit))
        
        # 去重并排序
        unique_results = {}
        for item in results:
            if item.id not in unique_results:
                unique_results[item.id] = item
        
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.get_strength(),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def recall_recent(self, n: int = 5) -> List[MemoryItem]:
        """
        回忆最近的记忆
        
        Args:
            n: 数量
            
        Returns:
            最近的记忆列表
        """
        return self.short_term.recall_recent(n)
    
    def recall_by_time(self, start_time: float, end_time: float) -> List[MemoryItem]:
        """
        按时间范围检索
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            记忆列表
        """
        return self.long_term.recall_by_time(start_time, end_time)
    
    def forget(self, item_id: str) -> bool:
        """
        主动遗忘
        
        Args:
            item_id: 记忆ID
            
        Returns:
            是否成功遗忘
        """
        # 从短期记忆删除
        if item_id in self.short_term.working_memory:
            del self.short_term.working_memory[item_id]
            return True
        
        # 从长期记忆删除
        for storage in [
            self.long_term.episodic_memory,
            self.long_term.semantic_memory,
            self.long_term.procedural_memory
        ]:
            if item_id in storage:
                del storage[item_id]
                return True
        
        return False
    
    def associate(self, item_id1: str, item_id2: str) -> bool:
        """
        建立记忆关联
        
        Args:
            item_id1: 记忆ID 1
            item_id2: 记忆ID 2
            
        Returns:
            是否成功建立关联
        """
        # 查找两个记忆项
        all_items = {
            **self.short_term.working_memory,
            **self.long_term.episodic_memory,
            **self.long_term.semantic_memory,
            **self.long_term.procedural_memory
        }
        
        if item_id1 not in all_items or item_id2 not in all_items:
            return False
        
        # 添加关联
        item1 = all_items[item_id1]
        item2 = all_items[item_id2]
        
        if item_id2 not in item1.associations:
            item1.associations.append(item_id2)
        if item_id1 not in item2.associations:
            item2.associations.append(item_id1)
        
        return True
    
    def _auto_consolidate(self) -> int:
        """
        自动记忆巩固
        
        Returns:
            巩固的记忆数量
        """
        to_consolidate = self.short_term.consolidate(self._consolidation_threshold)
        return self.long_term.consolidate_from_short_term(to_consolidate)
    
    def get_context(self, query: str, max_tokens: int = 1000) -> str:
        """
        获取上下文（用于LLM输入）
        
        Args:
            query: 当前查询
            max_tokens: 最大token数
            
        Returns:
            上下文字符串
        """
        # 检索相关记忆
        relevant_memories = self.recall(query, limit=10)
        
        # 构建上下文
        context_parts = []
        current_length = 0
        
        for item in relevant_memories:
            content_str = str(item.content)
            estimated_tokens = len(content_str.split())
            
            if current_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(f"- {content_str}")
            current_length += estimated_tokens
        
        return "\n".join(context_parts)
    
    def get_state(self) -> Dict[str, Any]:
        """获取完整状态"""
        return {
            "short_term": self.short_term.get_state(),
            "long_term": self.long_term.get_statistics(),
            "operation_count": self._operation_count
        }
    
    def export_memories(self) -> Dict[str, Any]:
        """导出所有记忆"""
        return {
            "short_term": {
                "working": [item.to_dict() for item in self.short_term.working_memory.values()],
                "sensory": [item.to_dict() for item in self.short_term.sensory_buffer]
            },
            "long_term": {
                "episodic": [item.to_dict() for item in self.long_term.episodic_memory.values()],
                "semantic": [item.to_dict() for item in self.long_term.semantic_memory.values()],
                "procedural": [item.to_dict() for item in self.long_term.procedural_memory.values()]
            }
        }
    
    def import_memories(self, data: Dict[str, Any]) -> int:
        """
        导入记忆
        
        Args:
            data: 记忆数据
            
        Returns:
            导入的数量
        """
        count = 0
        
        # 导入短期记忆
        for item_data in data.get("short_term", {}).get("working", []):
            item = MemoryItem.from_dict(item_data)
            self.short_term.add_to_working(item)
            count += 1
        
        for item_data in data.get("short_term", {}).get("sensory", []):
            item = MemoryItem.from_dict(item_data)
            self.short_term.add_to_sensory(item)
            count += 1
        
        # 导入长期记忆
        for item_data in data.get("long_term", {}).get("episodic", []):
            item = MemoryItem.from_dict(item_data)
            self.long_term.store(item)
            count += 1
        
        for item_data in data.get("long_term", {}).get("semantic", []):
            item = MemoryItem.from_dict(item_data)
            self.long_term.store(item)
            count += 1
        
        for item_data in data.get("long_term", {}).get("procedural", []):
            item = MemoryItem.from_dict(item_data)
            self.long_term.store(item)
            count += 1
        
        return count


# ==================== 示例和测试 ====================

def demo_memory_system():
    """演示记忆系统"""
    print("=" * 60)
    print("Agent 记忆系统演示")
    print("=" * 60)
    
    # 创建记忆管理器
    memory = MemoryManager(stm_capacity=5, ltm_capacity=100)
    
    # 1. 存储记忆
    print("\n1. 存储记忆")
    print("-" * 40)
    
    # 存储不同类型的记忆
    memory.remember(
        content="用户询问了关于Python编程的问题",
        memory_type="episodic",
        importance=0.7,
        metadata={"topic": "programming", "language": "Python"}
    )
    
    memory.remember(
        content="Python是一种高级编程语言，由Guido van Rossum创建",
        memory_type="semantic",
        importance=0.8
    )
    
    memory.remember(
        content="使用def关键字定义函数",
        memory_type="procedural",
        importance=0.6
    )
    
    memory.remember(
        content="用户喜欢简洁的回答",
        memory_type="semantic",
        importance=0.9
    )
    
    # 2. 检索记忆
    print("\n2. 检索记忆")
    print("-" * 40)
    
    # 按关键词检索
    results = memory.recall("Python")
    print(f"搜索 'Python' 的结果:")
    for item in results:
        print(f"  - [{item.memory_type.value}] {item.content}")
    
    # 回忆最近的记忆
    print("\n最近的记忆:")
    recent = memory.recall_recent(3)
    for item in recent:
        print(f"  - {item.content}")
    
    # 3. 建立关联
    print("\n3. 建立记忆关联")
    print("-" * 40)
    
    # 获取两个记忆项的ID
    all_memories = memory.recall("Python")
    if len(all_memories) >= 2:
        id1 = all_memories[0].id
        id2 = all_memories[1].id
        memory.associate(id1, id2)
        print(f"已建立记忆关联: {id1} <-> {id2}")
    
    # 4. 获取状态
    print("\n4. 记忆系统状态")
    print("-" * 40)
    state = memory.get_state()
    print(f"短期记忆利用率: {state['short_term']['utilization']:.1%}")
    print(f"长期记忆总数: {state['long_term']['total_count']}")
    
    # 5. 导出/导入
    print("\n5. 记忆导出")
    print("-" * 40)
    exported = memory.export_memories()
    print(f"导出的记忆数量: {sum(len(v) for v in exported['short_term'].values()) + sum(len(v) for v in exported['long_term'].values())}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_memory_system()