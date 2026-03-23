"""
工具使用 Agent 模块

实现能够使用工具的 Agent。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import json
import re

# 导入本地模块
try:
    from tool_definition import Tool, ToolResult
    from tool_registry import ToolRegistry
except ImportError:
    # 直接运行时的导入
    import sys
    sys.path.insert(0, '.')
    from tool_definition import Tool, ToolResult
    from tool_registry import ToolRegistry


@dataclass
class Message:
    """消息"""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None  # 工具名称（role=tool 时）
    tool_calls: Optional[List[Dict]] = None  # 工具调用（role=assistant 时）


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[ToolResult] = None


@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finished: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseToolAgent(ABC):
    """
    工具使用 Agent 基类
    """
    
    def __init__(
        self,
        tools: Union[ToolRegistry, List[Tool]],
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None
    ):
        """
        初始化 Agent
        
        Args:
            tools: 工具注册表或工具列表
            model: 模型名称
            system_prompt: 系统提示
        """
        # 处理工具
        if isinstance(tools, ToolRegistry):
            self.registry = tools
        else:
            self.registry = ToolRegistry()
            for tool in tools:
                self.registry.register(tool)
        
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.messages: List[Message] = []
        self.tool_call_history: List[ToolCall] = []
    
    def _default_system_prompt(self) -> str:
        """默认系统提示"""
        tools_desc = self._format_tools_description()
        return f"""你是一个有用的 AI 助手，可以使用工具来帮助用户。

可用工具：
{tools_desc}

当需要使用工具时，请按照指定格式输出工具调用。
"""
    
    def _format_tools_description(self) -> str:
        """格式化工具描述"""
        descriptions = []
        for tool in self.registry.list_tools():
            params_desc = ", ".join(
                f"{p.name}: {p.type}" + ("" if p.required else " (可选)")
                for p in tool.parameters
            )
            descriptions.append(f"- {tool.name}({params_desc}): {tool.description}")
        return "\n".join(descriptions)
    
    def add_message(self, role: str, content: str, **kwargs):
        """添加消息"""
        self.messages.append(Message(role=role, content=content, **kwargs))
    
    def clear_history(self):
        """清空历史"""
        self.messages = []
        self.tool_call_history = []
    
    @abstractmethod
    def run(self, user_input: str) -> str:
        """
        运行 Agent
        
        Args:
            user_input: 用户输入
            
        Returns:
            Agent 响应
        """
        pass
    
    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        return self.registry.execute(name, **kwargs)


class SimpleToolAgent(BaseToolAgent):
    """
    简单的工具使用 Agent
    
    使用模式匹配来决定是否调用工具。
    """
    
    def __init__(
        self,
        tools: Union[ToolRegistry, List[Tool]],
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None
    ):
        super().__init__(tools, model, system_prompt)
        self.tool_patterns = self._build_tool_patterns()
    
    def _build_tool_patterns(self) -> Dict[str, re.Pattern]:
        """构建工具匹配模式"""
        patterns = {}
        for tool in self.registry.list_tools():
            # 基于工具名称和描述构建关键词
            keywords = [tool.name] + tool.tags
            pattern = re.compile(
                r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b',
                re.IGNORECASE
            )
            patterns[tool.name] = pattern
        return patterns
    
    def _detect_tool_need(self, user_input: str) -> Optional[str]:
        """检测是否需要使用工具"""
        for tool_name, pattern in self.tool_patterns.items():
            if pattern.search(user_input):
                return tool_name
        return None
    
    def _extract_parameters(self, user_input: str, tool: Tool) -> Dict[str, Any]:
        """从用户输入中提取参数"""
        params = {}
        
        for param in tool.parameters:
            # 尝试匹配参数值
            patterns = {
                "city": r'(\w+(?:市|省)?)',  # 城市
                "expression": r'[\d+\-*/.()a-z]+',  # 数学表达式
                "query": r'["\'](.+?)["\']|搜索\s*(.+?)(?:\s|$)',  # 搜索词
            }
            
            if param.name in patterns:
                match = re.search(patterns[param.name], user_input, re.IGNORECASE)
                if match:
                    params[param.name] = match.group(1) or match.group(2)
        
        return params
    
    def run(self, user_input: str) -> str:
        """
        运行 Agent（简单模式）
        """
        # 检测是否需要工具
        tool_name = self._detect_tool_need(user_input)
        
        if tool_name:
            tool = self.registry.get_tool(tool_name)
            
            # 提取参数
            params = self._extract_parameters(user_input, tool)
            
            # 执行工具
            result = self.execute_tool(tool_name, **params)
            
            if result.success:
                return f"我使用了 {tool_name} 工具，结果是：{result.data}"
            else:
                return f"工具执行失败：{result.error}"
        else:
            return "我没有找到合适的工具来处理您的请求。"


class LLMToolAgent(BaseToolAgent):
    """
    基于 LLM 的工具使用 Agent
    
    使用 LLM 来决定工具调用（模拟）。
    """
    
    def __init__(
        self,
        tools: Union[ToolRegistry, List[Tool]],
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        max_tool_calls: int = 5
    ):
        super().__init__(tools, model, system_prompt)
        self.max_tool_calls = max_tool_calls
    
    def _simulate_llm_response(self, user_input: str) -> AgentResponse:
        """
        模拟 LLM 响应（实际应用中会调用真正的 LLM API）
        
        这里使用简单的规则来模拟 LLM 的决策过程。
        """
        # 简单的关键词匹配来决定工具调用
        tool_calls = []
        
        # 检查天气相关
        if "天气" in user_input or "温度" in user_input:
            import re
            city_match = re.search(r'(\w+)(?:的)?天气', user_input)
            city = city_match.group(1) if city_match else "北京"
            tool_calls.append(ToolCall(
                id="call_1",
                name="get_weather",
                arguments={"city": city}
            ))
        
        # 检查计算相关
        if "计算" in user_input or "+" in user_input or "*" in user_input:
            import re
            expr_match = re.search(r'计算\s*(.+?)(?:\s|$)|([\d+\-*/.()]+)', user_input)
            expression = expr_match.group(1) or expr_match.group(2) if expr_match else "1+1"
            tool_calls.append(ToolCall(
                id="call_2",
                name="calculate",
                arguments={"expression": expression.strip()}
            ))
        
        # 检查搜索相关
        if "搜索" in user_input or "查找" in user_input:
            import re
            query_match = re.search(r'(?:搜索|查找)\s*["\']?(.+?)["\']?(?:\s|$)', user_input)
            query = query_match.group(1) if query_match else "test"
            tool_calls.append(ToolCall(
                id="call_3",
                name="search_web",
                arguments={"query": query.strip()}
            ))
        
        # 生成响应
        if tool_calls:
            return AgentResponse(
                content="让我使用工具来帮您处理。",
                tool_calls=tool_calls,
                finished=False
            )
        else:
            return AgentResponse(
                content=f"我理解您的请求：{user_input}。请问有什么我可以帮助您的吗？",
                finished=True
            )
    
    def run(self, user_input: str) -> str:
        """
        运行 Agent
        """
        self.add_message("user", user_input)
        
        tool_call_count = 0
        
        while tool_call_count < self.max_tool_calls:
            # 获取 LLM 响应
            response = self._simulate_llm_response(user_input)
            
            # 如果没有工具调用，直接返回
            if not response.tool_calls:
                self.add_message("assistant", response.content)
                return response.content
            
            # 执行工具调用
            tool_results = []
            for tool_call in response.tool_calls:
                result = self.execute_tool(
                    tool_call.name, 
                    **tool_call.arguments
                )
                tool_call.result = result
                self.tool_call_history.append(tool_call)
                tool_results.append(f"{tool_call.name}: {result.to_dict()}")
                tool_call_count += 1
            
            # 模拟 LLM 根据工具结果生成最终响应
            if response.tool_calls:
                results_str = "\n".join(tool_results)
                final_response = f"根据工具调用结果：\n{results_str}\n\n"
                
                # 根据工具结果生成友好的响应
                for tool_call in response.tool_calls:
                    if tool_call.result and tool_call.result.success:
                        data = tool_call.result.data
                        if tool_call.name == "get_weather":
                            final_response += f"{data.get('city', '未知')}今天的天气是{data.get('condition', '未知')}，温度{data.get('temperature', '未知')}{data.get('unit', '°C')}。"
                        elif tool_call.name == "calculate":
                            final_response += f"计算结果：{data.get('expression', '')} = {data.get('result', '未知')}"
                        elif tool_call.name == "search_web":
                            results = data.get('results', [])
                            final_response += f"找到 {len(results)} 个相关结果。"
                
                return final_response
            
            # 更新用户输入（用于下一轮）
            user_input = f"工具结果：{results_str}"
        
        return "达到最大工具调用次数限制。"


# ============ 演示 ============

def demo():
    """演示工具使用 Agent"""
    print("=" * 60)
    print("工具使用 Agent 演示")
    print("=" * 60)
    
    from tool_definition import get_weather, calculate, search_web
    
    # 创建工具列表
    tools = [get_weather, calculate, search_web]
    
    # 创建简单 Agent
    print("\n1. 简单工具 Agent:")
    simple_agent = SimpleToolAgent(tools)
    
    # 测试
    test_inputs = [
        "北京今天的天气怎么样？",
        "帮我计算 2 + 3 * 4",
        "搜索 Python 教程"
    ]
    
    for user_input in test_inputs:
        print(f"\n用户: {user_input}")
        response = simple_agent.run(user_input)
        print(f"Agent: {response}")
    
    # 创建 LLM Agent
    print("\n" + "=" * 60)
    print("2. LLM 工具 Agent:")
    llm_agent = LLMToolAgent(tools)
    
    for user_input in test_inputs:
        print(f"\n用户: {user_input}")
        response = llm_agent.run(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    demo()