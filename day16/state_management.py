"""
Day 16: LangGraph 状态管理示例

本文件演示 LangGraph 的状态管理机制：
- 状态定义与类型注解
- 状态更新与合并策略
- 消息历史管理
- 检查点持久化
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 加载环境变量
load_dotenv()


def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7,
    )


# ==========================================
# 示例 1: 基础状态定义
# ==========================================


def example_basic_state():
    """演示最基本的状态定义和使用"""
    print("\n" + "=" * 50)
    print("示例 1: 基础状态定义")
    print("=" * 50)

    # 使用 TypedDict 定义状态类型
    class BasicState(TypedDict):
        input: str
        counter: int
        result: str

    # 节点函数
    def increment_node(state: BasicState) -> dict:
        """增加计数器"""
        print(f"  当前计数: {state['counter']}")
        # 返回部分状态更新（会与原状态合并）
        return {"counter": state["counter"] + 1}

    def process_node(state: BasicState) -> dict:
        """处理节点"""
        return {"result": f"处理完成，计数={state['counter']}"}

    # 构建图
    builder = StateGraph(BasicState)

    builder.add_node("increment", increment_node)
    builder.add_node("process", process_node)

    builder.add_edge("increment", "process")
    builder.set_entry_point("increment")
    builder.set_finish_point("process")

    graph = builder.compile()

    # 执行
    # 注意：初始状态需要包含所有字段的初始值
    result = graph.invoke({
        "input": "测试",
        "counter": 0,
        "result": ""
    })

    print(f"\n最终状态: {result}")
    print(f"计数器从 0 变为 {result['counter']}")


# ==========================================
# 示例 2: 状态更新策略（覆盖 vs 合并）
# ==========================================


def example_state_update_strategy():
    """演示不同的状态更新策略"""
    print("\n" + "=" * 50)
    print("示例 2: 状态更新策略")
    print("=" * 50)

    # 方式 1: 默认合并策略（字典合并）
    print("\n--- 方式 1: 默认合并 ---")

    class MergeState(TypedDict):
        field_a: str
        field_b: str

    def update_a(state: MergeState) -> dict:
        """只更新 field_a"""
        return {"field_a": "新值A"}

    def update_b(state: MergeState) -> dict:
        """只更新 field_b"""
        return {"field_b": "新值B"}

    builder = StateGraph(MergeState)
    builder.add_node("node_a", update_a)
    builder.add_node("node_b", update_b)
    builder.add_edge("node_a", "node_b")
    builder.set_entry_point("node_a")
    builder.set_finish_point("node_b")

    graph = builder.compile()

    result = graph.invoke({"field_a": "原值A", "field_b": "原值B"})
    print(f"初始: field_a='原值A', field_b='原值B'")
    print(f"最终: field_a='{result['field_a']}', field_b='{result['field_b']}'")
    print("结论: 每个节点只更新它返回的字段，其他字段保持不变")


# ==========================================
# 示例 3: 消息历史状态（使用 add_messages）
# ==========================================


def example_messages_state():
    """演示消息列表的状态管理"""
    print("\n" + "=" * 50)
    print("示例 3: 消息历史状态")
    print("=" * 50)

    # 使用 Annotated 和 add_messages reducer
    # add_messages 会智能合并消息（添加新消息，不覆盖旧消息）
    class MessagesState(TypedDict):
        messages: Annotated[list, add_messages]

    llm = create_llm()

    def assistant_node(state: MessagesState) -> dict:
        """助手节点"""
        print(f"  当前消息数: {len(state['messages'])}")

        # 调用 LLM
        response = llm.invoke(state["messages"])

        # 返回新消息（会自动添加到 messages 列表）
        return {"messages": [response]}

    # 构建图
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant_node)
    builder.set_entry_point("assistant")
    builder.set_finish_point("assistant")

    graph = builder.compile()

    # 执行对话
    print("\n第一次对话:")
    result1 = graph.invoke({
        "messages": [HumanMessage(content="你好，请介绍一下你自己")]
    })
    print(f"消息历史: {len(result1['messages'])} 条")
    for msg in result1["messages"]:
        print(f"  {msg.type}: {msg.content[:50]}...")

    print("\n第二次对话（追加消息）:")
    # 使用第一次的结果作为初始状态
    result2 = graph.invoke({
        "messages": result1["messages"] + [HumanMessage(content="你能做什么？")]
    })
    print(f"消息历史: {len(result2['messages'])} 条")
    for msg in result2["messages"]:
        print(f"  {msg.type}: {msg.content[:50]}...")


# ==========================================
# 示例 4: 完整对话 Agent 状态
# ==========================================


def example_agent_state():
    """演示完整 Agent 的状态管理"""
    print("\n" + "=" * 50)
    print("示例 4: 完整对话 Agent 状态")
    print("=" * 50)

    # 定义完整的 Agent 状态
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        current_step: str
        iterations: int
        max_iterations: int

    llm = create_llm()

    def think_node(state: AgentState) -> dict:
        """思考节点"""
        print(f"  [思考] 第 {state['iterations'] + 1} 次迭代")

        # 添加思考消息
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if last_message and isinstance(last_message, HumanMessage):
            # 生成思考过程
            think_prompt = ChatPromptTemplate.from_template(
                "请分析这个问题并给出思考过程：{question}"
            )
            chain = think_prompt | llm | StrOutputParser()
            thinking = chain.invoke({"question": last_message.content})

            print(f"    思考结果: {thinking[:100]}...")

            return {
                "messages": [AIMessage(content=f"[思考] {thinking}")],
                "current_step": "think",
                "iterations": state["iterations"] + 1
            }

        return {
            "current_step": "think",
            "iterations": state["iterations"] + 1
        }

    def respond_node(state: AgentState) -> dict:
        """响应节点"""
        print("  [响应] 生成最终回答")

        # 简化：直接回答最后一个用户消息
        messages = state["messages"]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]

        if human_messages:
            last_question = human_messages[-1].content
            respond_prompt = ChatPromptTemplate.from_template(
                "请回答以下问题：{question}"
            )
            chain = respond_prompt | llm | StrOutputParser()
            answer = chain.invoke({"question": last_question})

            print(f"    回答: {answer[:100]}...")

            return {
                "messages": [AIMessage(content=answer)],
                "current_step": "respond"
            }

        return {"current_step": "respond"}

    # 路由函数
    def should_continue(state: AgentState) -> str:
        """判断是否继续迭代"""
        if state["iterations"] >= state["max_iterations"]:
            print("  [路由] 达到最大迭代次数，结束")
            return "respond"
        if state["current_step"] == "think":
            print("  [路由] 思考完成，继续响应")
            return "respond"
        return "think"

    # 构建图
    builder = StateGraph(AgentState)

    builder.add_node("think", think_node)
    builder.add_node("respond", respond_node)

    builder.add_conditional_edges(
        "think",
        should_continue,
        {"respond": "respond", "think": "think"}
    )

    builder.set_entry_point("think")
    builder.set_finish_point("respond")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [HumanMessage(content="什么是 RAG 技术？")],
        "current_step": "",
        "iterations": 0,
        "max_iterations": 2
    })

    print(f"\n最终状态:")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  消息数: {len(result['messages'])}")


# ==========================================
# 示例 5: 检查点持久化
# ==========================================


def example_checkpoint():
    """演示使用检查点实现状态持久化"""
    print("\n" + "=" * 50)
    print("示例 5: 检查点持久化")
    print("=" * 50)

    # 定义状态
    class CheckpointState(TypedDict):
        step: int
        data: str
        history: list[str]

    # 定义节点
    def step_node(state: CheckpointState) -> dict:
        """执行一个步骤"""
        print(f"  执行步骤 {state['step'] + 1}")
        new_data = f"步骤{state['step'] + 1}的数据"
        return {
            "step": state["step"] + 1,
            "data": new_data,
            "history": state["history"] + [new_data]
        }

    # 路由函数
    def should_continue(state: CheckpointState) -> str:
        """判断是否继续"""
        if state["step"] >= 3:
            return END
        return "step"

    # 构建图
    builder = StateGraph(CheckpointState)
    builder.add_node("step", step_node)
    builder.add_conditional_edges("step", should_continue, {END: END, "step": "step"})
    builder.set_entry_point("step")

    # 使用检查点
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # 第一次执行（会保存检查点）
    print("\n第一次执行:")
    thread_id = "thread-001"
    result1 = graph.invoke(
        {"step": 0, "data": "", "history": []},
        config={"configurable": {"thread_id": thread_id}}
    )
    print(f"  结果: step={result1['step']}, history={result1['history']}")

    # 查看当前检查点状态
    print("\n查看检查点状态:")
    checkpoint_state = graph.get_state({"configurable": {"thread_id": thread_id}})
    print(f"  当前状态: {checkpoint_state.values}")

    # 从检查点恢复继续执行（如果图中未完成）
    # 这里演示获取历史状态
    print("\n检查点历史:")
    checkpoint_history = list(graph.get_state_history({"configurable": {"thread_id": thread_id}}))
    print(f"  检查点数量: {len(checkpoint_history)}")

    print("\n说明: 检查点可用于:")
    print("  - 中断后恢复执行")
    print("  - 回滚到之前的状态")
    print("  - 多线程/多用户隔离状态")


# ==========================================
# 示例 6: 状态更新覆盖策略
# ==========================================


def example_override_state():
    """演示如何完全覆盖状态字段"""
    print("\n" + "=" * 50)
    print("示例 6: 状态覆盖策略")
    print("=" * 50)

    # 默认情况下，列表会被覆盖而不是合并
    class OverrideState(TypedDict):
        items: list[str]  # 列表类型默认是覆盖
        count: int

    def add_items_node(state: OverrideState) -> dict:
        """添加项目（会覆盖）"""
        new_items = ["item1", "item2"]
        print(f"  原有项目: {state['items']}")
        print(f"  新项目: {new_items}")
        return {
            "items": new_items,  # 会完全覆盖，不是追加
            "count": len(new_items)
        }

    builder = StateGraph(OverrideState)
    builder.add_node("add_items", add_items_node)
    builder.set_entry_point("add_items")
    builder.set_finish_point("add_items")

    graph = builder.compile()

    result = graph.invoke({"items": ["old_item"], "count": 1})
    print(f"\n结果:")
    print(f"  原状态: items=['old_item'], count=1")
    print(f"  新状态: items={result['items']}, count={result['count']}")
    print("  说明: 普通列表会被完全覆盖")


# ==========================================
# 示例 7: 复合状态设计
# ==========================================


def example_complex_state():
    """演示复杂状态结构的设计"""
    print("\n" + "=" * 50)
    print("示例 7: 复合状态设计")
    print("=" * 50)

    # 定义嵌套状态结构
    class TaskInfo(TypedDict):
        """任务信息"""
        id: str
        type: str
        priority: int

    class UserInfo(TypedDict):
        """用户信息"""
        id: str
        name: str

    class ComplexState(TypedDict):
        """复合状态"""
        task: TaskInfo
        user: UserInfo
        messages: Annotated[list, add_messages]
        status: str
        progress: float

    def init_node(state: ComplexState) -> dict:
        """初始化节点"""
        print("  [初始化] 设置任务和用户信息")
        return {
            "task": {"id": "task-001", "type": "analysis", "priority": 1},
            "user": {"id": "user-001", "name": "测试用户"},
            "status": "initialized",
            "progress": 0.0
        }

    def process_node(state: ComplexState) -> dict:
        """处理节点"""
        print(f"  [处理] 任务ID: {state['task']['id']}")
        print(f"  [处理] 用户: {state['user']['name']}")
        return {
            "status": "processing",
            "progress": 0.5
        }

    def complete_node(state: ComplexState) -> dict:
        """完成节点"""
        print("  [完成] 任务处理完成")
        return {
            "status": "completed",
            "progress": 1.0,
            "messages": [AIMessage(content="任务完成！")]
        }

    # 构建图
    builder = StateGraph(ComplexState)

    builder.add_node("init", init_node)
    builder.add_node("process", process_node)
    builder.add_node("complete", complete_node)

    builder.add_edge("init", "process")
    builder.add_edge("process", "complete")

    builder.set_entry_point("init")
    builder.set_finish_point("complete")

    graph = builder.compile()

    # 执行（嵌套结构需要初始化）
    result = graph.invoke({
        "task": {"id": "", "type": "", "priority": 0},
        "user": {"id": "", "name": ""},
        "messages": [],
        "status": "",
        "progress": 0.0
    })

    print(f"\n最终状态:")
    print(f"  任务: {result['task']}")
    print(f"  用户: {result['user']}")
    print(f"  状态: {result['status']}")
    print(f"  进度: {result['progress']}")
    print(f"  消息数: {len(result['messages'])}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 16: LangGraph 状态管理示例")
    print("=" * 60)

    # 运行各示例
    example_basic_state()
    example_state_update_strategy()
    example_messages_state()
    example_agent_state()
    example_checkpoint()
    example_override_state()
    example_complex_state()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()