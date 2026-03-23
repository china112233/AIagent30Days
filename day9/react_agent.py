"""
ReAct Agent 模块

实现 ReAct (Reasoning + Acting) 模式的 Agent。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import re
import json

# 导入本地模块
try:
    from tool_definition import Tool, ToolResult
    from tool_registry import ToolRegistry
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from tool_definition import Tool, ToolResult
    from tool_registry import ToolRegistry


class ReActStepType(Enum):
    """ReAct 步骤类型"""
    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"
    ANSWER = "Answer"


@dataclass
class ReActStep:
    """ReAct 步骤"""
    type: ReActStepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    result: Optional[ToolResult] = None


@dataclass
class ReActTrace:
    """ReAct 执行轨迹"""
    steps: List[ReActStep] = field(default_factory=list)
    
    def add_thought(self, thought: str):
        """添加思考步骤"""
        self.steps.append(ReActStep(
            type=ReActStepType.THOUGHT,
            content=thought
        ))
    
    def add_action(self, tool_name: str, tool_args: Dict[str, Any]):
        """添加行动步骤"""
        self.steps.append(ReActStep(
            type=ReActStepType.ACTION,
            content=f"{tool_name}({tool_args})",
            tool_name=tool_name,
            tool_args=tool_args
        ))
    
    def add_observation(self, result: ToolResult):
        """添加观察步骤"""
        self.steps.append(ReActStep(
            type=ReActStepType.OBSERVATION,
            content=str(result.data) if result.success else f"Error: {result.error}",
            result=result
        ))
    
    def add_answer(self, answer: str):
        """添加回答步骤"""
        self.steps.append(ReActStep(
            type=ReActStepType.ANSWER,
            content=answer
        ))
    
    def format_trace(self) -> str:
        """格式化轨迹"""
        output = []
        for i, step in enumerate(self.steps, 1):
            prefix = f"Step {i} - {step.type.value}:"
            output.append(f"{prefix}\n{step.content}\n")
        return "\n".join(output)


class ReActAgent:
    """
    ReAct Agent
    
    实现 Thought -> Action -> Observation 循环。
    """
    
    def __init__(
        self,
        tools: Union[ToolRegistry, List[Tool]],
        model: str = "gpt-4",
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        初始化 ReAct Agent
        
        Args:
            tools: 工具注册表或工具列表
            model: 模型名称
            max_iterations: 最大迭代次数
            verbose: 是否打印详细过程
        """
        # 处理工具
        if isinstance(tools, ToolRegistry):
            self.registry = tools
        else:
            self.registry = ToolRegistry()
            for tool in tools:
                self.registry.register(tool)
        
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 构建工具描述
        self._tools_description = self._build_tools_description()
        self._system_prompt = self._build_system_prompt()
    
    def _build_tools_description(self) -> str:
        """构建工具描述"""
        descriptions = []
        for tool in self.registry.list_tools():
            params_desc = []
            for p in tool.parameters:
                if p.required:
                    params_desc.append(f"{p.name}: {p.type}")
                else:
                    params_desc.append(f"{p.name}: {p.type} (可选, 默认: {p.default})")
            
            descriptions.append(f"""{tool.name}:
  描述: {tool.description}
  参数: {', '.join(params_desc)}""")
        
        return "\n\n".join(descriptions)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return f"""你是一个遵循 ReAct 模式的 AI 助手。

ReAct 模式要求你交替进行思考和行动：
1. Thought: 分析当前情况，决定下一步行动
2. Action: 调用工具获取信息
3. Observation: 观察工具返回结果
4. 重复以上步骤直到可以给出最终答案

可用工具：

{self._tools_description}

回答格式：
Thought: [你的思考过程]
Action: [工具名称]
Action Input: {{"param1": "value1", "param2": "value2"}}

或者当你可以给出最终答案时：
Thought: [你的思考过程]
Final Answer: [你的最终答案]
"""
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应
        
        Returns:
            {
                "thought": str,
                "action": Optional[str],
                "action_input": Optional[Dict],
                "final_answer": Optional[str]
            }
        """
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }
        
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final)|$)', response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # 提取 Action 和 Action Input
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            result["action"] = action_match.group(1).strip()
            
            # 提取 Action Input
            input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL)
            if input_match:
                try:
                    result["action_input"] = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    result["action_input"] = {}
        
        # 提取 Final Answer
        answer_match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if answer_match:
            result["final_answer"] = answer_match.group(1).strip()
        
        return result
    
    def _simulate_llm(self, user_input: str, trace: ReActTrace) -> str:
        """
        模拟 LLM 响应
        
        实际应用中会调用真正的 LLM API。
        这里使用简单的规则来模拟。
        """
        # 根据用户输入和当前轨迹模拟思考
        last_observation = None
        for step in reversed(trace.steps):
            if step.type == ReActStepType.OBSERVATION:
                last_observation = step
                break
        
        # 第一轮：决定使用什么工具
        if not last_observation:
            # 检查关键词决定工具
            if "天气" in user_input or "温度" in user_input:
                city_match = re.search(r'(\w+)(?:的)?天气', user_input)
                city = city_match.group(1) if city_match else "北京"
                
                # 检查是否需要温度转换
                if "华氏" in user_input:
                    return f"""Thought: 用户想知道{city}的天气，并且需要华氏温度。我需要先获取摄氏温度，然后转换。
Action: get_weather
Action Input: {{"city": "{city}", "unit": "celsius"}}"""
                else:
                    return f"""Thought: 用户想知道{city}的天气。我应该使用 get_weather 工具来获取信息。
Action: get_weather
Action Input: {{"city": "{city}"}}"""
            
            elif "计算" in user_input or "+" in user_input or "*" in user_input:
                expr_match = re.search(r'计算\s*(.+?)(?:\s|$)|([\d+\-*/.()a-z]+)', user_input)
                expression = expr_match.group(1) or expr_match.group(2) if expr_match else "1+1"
                
                return f"""Thought: 用户想要计算数学表达式。我应该使用 calculate 工具。
Action: calculate
Action Input: {{"expression": "{expression.strip()}"}}}"""
            
            elif "搜索" in user_input:
                query_match = re.search(r'(?:搜索|查找)\s*["\']?(.+?)["\']?(?:\s|$)', user_input)
                query = query_match.group(1) if query_match else user_input
                
                return f"""Thought: 用户想要搜索信息。我应该使用 search_web 工具。
Action: search_web
Action Input: {{"query": "{query.strip()}"}}"""
            
            else:
                return f"""Thought: 我理解用户的请求，但可能不需要使用工具。
Final Answer: 我理解您的问题：{user_input}。请问有什么具体需要我帮助的吗？"""
        
        # 后续轮：根据观察结果继续
        else:
            observation_data = last_observation.result.data if last_observation.result else {}
            
            # 检查是否需要进一步操作
            if "华氏" in user_input and "temperature" in str(observation_data):
                temp_c = observation_data.get("temperature", 0)
                return f"""Thought: 我已经获得了摄氏温度 {temp_c}°C，现在需要转换为华氏温度。
Action: calculate
Action Input: {{"expression": "({temp_c} * 9/5) + 32"}}"""
            
            # 检查是否可以给出最终答案
            if last_observation.result and last_observation.result.success:
                data = observation_data
                
                # 天气结果
                if "city" in data and "temperature" in data:
                    return f"""Thought: 我已经获得了所有需要的信息，可以给用户一个完整的回答了。
Final Answer: {data.get('city', '未知')}今天的天气是{data.get('condition', '未知')}，温度{data.get('temperature', '未知')}{data.get('unit', '°C')}。"""
                
                # 计算结果
                elif "result" in data:
                    expr = data.get("expression", "")
                    result = data.get("result", "")
                    return f"""Thought: 计算完成，我可以给出答案了。
Final Answer: {expr} = {result}"""
                
                # 搜索结果
                elif "results" in data:
                    results = data.get("results", [])
                    summary = f"找到 {len(results)} 个相关结果。"
                    for i, r in enumerate(results[:3], 1):
                        summary += f"\n{i}. {r.get('title', '')}"
                    return f"""Thought: 搜索完成，我可以总结结果给用户了。
Final Answer: {summary}"""
            
            # 出错情况
            else:
                error = last_observation.result.error if last_observation.result else "未知错误"
                return f"""Thought: 工具执行出错了：{error}。我需要告诉用户。
Final Answer: 抱歉，处理您的请求时遇到问题：{error}"""
    
    def run(self, user_input: str) -> str:
        """
        运行 ReAct 循环
        
        Args:
            user_input: 用户输入
            
        Returns:
            最终答案
        """
        trace = ReActTrace()
        iteration = 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"任务: {user_input}")
            print(f"{'='*60}\n")
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"--- 迭代 {iteration} ---")
            
            # 获取 LLM 响应（模拟）
            response = self._simulate_llm(user_input, trace)
            
            # 解析响应
            parsed = self._parse_response(response)
            
            # 记录思考
            if parsed["thought"]:
                trace.add_thought(parsed["thought"])
                if self.verbose:
                    print(f"Thought: {parsed['thought']}")
            
            # 检查是否是最终答案
            if parsed["final_answer"]:
                trace.add_answer(parsed["final_answer"])
                if self.verbose:
                    print(f"\nFinal Answer: {parsed['final_answer']}")
                return parsed["final_answer"]
            
            # 执行工具调用
            if parsed["action"] and parsed["action_input"]:
                tool_name = parsed["action"]
                tool_args = parsed["action_input"]
                
                if self.verbose:
                    print(f"Action: {tool_name}")
                    print(f"Action Input: {tool_args}")
                
                # 记录行动
                trace.add_action(tool_name, tool_args)
                
                # 执行工具
                result = self.registry.execute(tool_name, **tool_args)
                
                # 记录观察
                trace.add_observation(result)
                
                if self.verbose:
                    print(f"Observation: {result.data if result.success else result.error}")
        
        # 达到最大迭代次数
        return "抱歉，我无法在有限的步骤内完成这个任务。"
    
    def get_trace(self, user_input: str) -> ReActTrace:
        """
        运行并获取完整轨迹
        
        Args:
            user_input: 用户输入
            
        Returns:
            执行轨迹
        """
        trace = ReActTrace()
        # ... 与 run 类似，但返回 trace
        self._verbose_backup = self.verbose
        self.verbose = False
        
        # 简化版本，实际应该重构
        original_trace = ReActTrace()
        self.run(user_input)
        
        self.verbose = self._verbose_backup
        return original_trace


# ============ 演示 ============

def demo():
    """演示 ReAct Agent"""
    print("=" * 60)
    print("ReAct Agent 演示")
    print("=" * 60)
    
    from tool_definition import get_weather, calculate, search_web
    
    # 创建工具列表
    tools = [get_weather, calculate, search_web]
    
    # 创建 ReAct Agent
    agent = ReActAgent(tools, verbose=True)
    
    # 测试任务
    tasks = [
        "北京今天的天气怎么样？",
        "计算 (10 + 5) * 2",
        "搜索 Python 教程",
        "上海今天天气如何？温度是多少华氏度？"
    ]
    
    for task in tasks:
        result = agent.run(task)
        print(f"\n结果: {result}\n")
        print("-" * 60)


if __name__ == "__main__":
    demo()