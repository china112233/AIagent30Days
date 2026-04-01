"""
Day 16: LangGraph 多 Agent 工作流示例

本文件演示多 Agent 协作的工作流：
- Agent 间通信
- 层级管理架构
- 专家团队路由
- 复杂任务分解
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
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
# 示例 1: 顺序协作模式
# ==========================================


def example_sequential_collaboration():
    """演示顺序协作的多 Agent 工作流"""
    print("\n" + "=" * 50)
    print("示例 1: 顺序协作模式")
    print("=" * 50)

    # 定义状态
    class SequentialState(TypedDict):
        messages: Annotated[list, add_messages]
        research_result: str
        analysis_result: str
        final_report: str

    llm = create_llm()

    def researcher_agent(state: SequentialState) -> dict:
        """研究员 Agent：收集信息"""
        print("  [研究员] 正在收集信息...")

        # 获取用户问题
        user_msg = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        if user_msg:
            prompt = ChatPromptTemplate.from_template(
                "作为研究员，请收集关于 '{topic}' 的关键信息（列出3-5个要点）："
            )
            chain = prompt | llm | StrOutputParser()
            research = chain.invoke({"topic": user_msg})

            print(f"    研究结果: {research[:100]}...")
            return {"research_result": research}

        return {"research_result": ""}

    def analyst_agent(state: SequentialState) -> dict:
        """分析师 Agent：分析信息"""
        print("  [分析师] 正在分析信息...")

        research = state["research_result"]
        if research:
            prompt = ChatPromptTemplate.from_template(
                """作为分析师，请分析以下研究结果：

研究结果：
{research}

请给出分析结论（2-3个要点）："""
            )
            chain = prompt | llm | StrOutputParser()
            analysis = chain.invoke({"research": research})

            print(f"    分析结果: {analysis[:100]}...")
            return {"analysis_result": analysis}

        return {"analysis_result": ""}

    def writer_agent(state: SequentialState) -> dict:
        """撰稿人 Agent：生成报告"""
        print("  [撰稿人] 正在撰写报告...")

        research = state["research_result"]
        analysis = state["analysis_result"]

        if research and analysis:
            prompt = ChatPromptTemplate.from_template(
                """作为撰稿人，请基于以下信息撰写一份简短报告：

研究结果：
{research}

分析结论：
{analysis}

请写出一份完整的报告（100字左右）："""
            )
            chain = prompt | llm | StrOutputParser()
            report = chain.invoke({"research": research, "analysis": analysis})

            print(f"    报告: {report[:150]}...")
            return {
                "final_report": report,
                "messages": [AIMessage(content=report)]
            }

        return {"final_report": ""}

    # 构建图
    builder = StateGraph(SequentialState)

    builder.add_node("researcher", researcher_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)

    # 顺序连接
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "writer")

    builder.set_entry_point("researcher")
    builder.set_finish_point("writer")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [HumanMessage(content="LangGraph 框架")],
        "research_result": "",
        "analysis_result": "",
        "final_report": ""
    })

    print(f"\n最终报告:")
    print(f"  {result['final_report'][:200]}...")


# ==========================================
# 示例 2: 层级管理架构（Supervisor 模式）
# ==========================================


def example_supervisor_mode():
    """演示层级管理的 Supervisor 模式"""
    print("\n" + "=" * 50)
    print("示例 2: Supervisor 层级管理")
    print("=" * 50)

    # 定义状态
    class SupervisorState(TypedDict):
        messages: Annotated[list, add_messages]
        task: str
        subtasks: list[str]
        assigned_agent: str
        worker_results: dict[str, str]
        supervisor_decision: str
        final_result: str

    llm = create_llm()

    def supervisor_agent(state: SupervisorState) -> dict:
        """主控 Agent：分配任务"""
        print("  [Supervisor] 分析任务...")

        user_msg = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        if user_msg:
            # 分析并分解任务
            prompt = ChatPromptTemplate.from_template(
                """作为 Supervisor，请分析以下任务：

任务：{task}

请将任务分解为 2-3 个子任务，并指明每个子任务应该分配给哪个专家：
- 研究员（researcher）：负责信息收集
- 分析师（analyst）：负责数据分析
- 撰稿人（writer）：负责内容撰写

输出格式：
子任务1: [内容] -> 分配给: [专家名]
子任务2: [内容] -> 分配给: [专家名]
"""
            )
            chain = prompt | llm | StrOutputParser()
            decision = chain.invoke({"task": user_msg})

            print(f"    决策: {decision[:150]}...")

            # 模拟子任务列表
            subtasks = [
                "收集 LangGraph 相关信息",
                "分析 LangGraph 特点",
                "撰写总结报告"
            ]

            return {
                "task": user_msg,
                "subtasks": subtasks,
                "supervisor_decision": decision,
                "assigned_agent": "researcher"  # 先分配给研究员
            }

        return {"assigned_agent": END}

    def researcher_worker(state: SupervisorState) -> dict:
        """研究员 Worker"""
        print("  [Worker-研究员] 执行子任务...")

        task = state["task"]
        prompt = ChatPromptTemplate.from_template(
            "作为研究员，请收集关于 '{task}' 的关键信息："
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"task": task})

        print(f"    完成: {result[:80]}...")

        # 更新 worker 结果
        worker_results = state.get("worker_results", {})
        worker_results["researcher"] = result

        return {
            "worker_results": worker_results,
            "assigned_agent": "analyst"
        }

    def analyst_worker(state: SupervisorState) -> dict:
        """分析师 Worker"""
        print("  [Worker-分析师] 执行子任务...")

        research_result = state["worker_results"].get("researcher", "")

        prompt = ChatPromptTemplate.from_template(
            "作为分析师，请分析以下信息：\n{research}\n\n给出分析结论："
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"research": research_result})

        print(f"    完成: {result[:80]}...")

        worker_results = state["worker_results"]
        worker_results["analyst"] = result

        return {
            "worker_results": worker_results,
            "assigned_agent": "writer"
        }

    def writer_worker(state: SupervisorState) -> dict:
        """撰稿人 Worker"""
        print("  [Worker-撰稿人] 执行子任务...")

        all_results = state["worker_results"]

        prompt = ChatPromptTemplate.from_template(
            """作为撰稿人，请整合以下工作成果：

研究结果：{research}
分析结论：{analysis}

请撰写最终报告："""
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "research": all_results.get("researcher", ""),
            "analysis": all_results.get("analyst", "")
        })

        print(f"    完成: {result[:80]}...")

        worker_results = state["worker_results"]
        worker_results["writer"] = result

        return {
            "worker_results": worker_results,
            "assigned_agent": END,
            "final_result": result,
            "messages": [AIMessage(content=result)]
        }

    # 路由函数
    def route_to_worker(state: SupervisorState) -> str:
        """路由到对应的 Worker"""
        return state["assigned_agent"]

    # 构建图
    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("researcher", researcher_worker)
    builder.add_node("analyst", analyst_worker)
    builder.add_node("writer", writer_worker)

    # Supervisor 分配任务
    builder.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )

    # Worker 完成后汇报给 Supervisor（这里简化为直接流转）
    builder.add_conditional_edges(
        "researcher",
        route_to_worker,
        {"analyst": "analyst", END: END}
    )
    builder.add_conditional_edges(
        "analyst",
        route_to_worker,
        {"writer": "writer", END: END}
    )
    builder.add_conditional_edges(
        "writer",
        route_to_worker,
        {END: END}
    )

    builder.set_entry_point("supervisor")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [HumanMessage(content="请分析 LangGraph 的优势")],
        "task": "",
        "subtasks": [],
        "assigned_agent": "",
        "worker_results": {},
        "supervisor_decision": "",
        "final_result": ""
    })

    print(f"\n最终结果:")
    print(f"  任务: {result['task']}")
    print(f"  报告: {result['final_result'][:150]}...")


# ==========================================
# 示例 3: 专家团队路由
# ==========================================


def example_expert_team():
    """演示专家团队路由模式"""
    print("\n" + "=" * 50)
    print("示例 3: 专家团队路由")
    print("=" * 50)

    # 定义专家类型
    ExpertType = Literal["code_expert", "data_expert", "writing_expert", "general_expert"]

    # 定义状态
    class ExpertState(TypedDict):
        messages: Annotated[list, add_messages]
        question: str
        expert_type: ExpertType
        expert_response: str
        final_answer: str

    llm = create_llm()

    def router_agent(state: ExpertState) -> dict:
        """路由 Agent：决定哪个专家处理"""
        print("  [路由器] 分析问题类型...")

        user_msg = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        if user_msg:
            # 简化路由逻辑
            question_lower = user_msg.lower()

            if any(word in question_lower for word in ["代码", "code", "编程", "python", "函数"]):
                expert_type = "code_expert"
            elif any(word in question_lower for word in ["数据", "data", "分析", "统计"]):
                expert_type = "data_expert"
            elif any(word in question_lower for word in ["写作", "writing", "文章", "文案"]):
                expert_type = "writing_expert"
            else:
                expert_type = "general_expert"

            print(f"    路由到: {expert_type}")
            return {
                "question": user_msg,
                "expert_type": expert_type
            }

        return {"expert_type": "general_expert"}

    def code_expert(state: ExpertState) -> dict:
        """代码专家"""
        print("  [代码专家] 回答问题...")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "作为代码专家，请回答以下问题（给出代码示例）：\n{question}"
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})

        print(f"    回答: {response[:100]}...")
        return {"expert_response": response}

    def data_expert(state: ExpertState) -> dict:
        """数据专家"""
        print("  [数据专家] 回答问题...")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "作为数据分析专家，请回答以下问题（给出分析思路）：\n{question}"
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})

        print(f"    回答: {response[:100]}...")
        return {"expert_response": response}

    def writing_expert(state: ExpertState) -> dict:
        """写作专家"""
        print("  [写作专家] 回答问题...")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "作为写作专家，请回答以下问题（给出写作建议）：\n{question}"
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})

        print(f"    回答: {response[:100]}...")
        return {"expert_response": response}

    def general_expert(state: ExpertState) -> dict:
        """通用专家"""
        print("  [通用专家] 回答问题...")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "作为通用知识专家，请回答以下问题：\n{question}"
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})

        print(f"    回答: {response[:100]}...")
        return {"expert_response": response}

    def aggregator(state: ExpertState) -> dict:
        """汇总节点"""
        print("  [汇总] 生成最终回答")

        response = state["expert_response"]
        return {
            "final_answer": response,
            "messages": [AIMessage(content=response)]
        }

    # 构建图
    builder = StateGraph(ExpertState)

    builder.add_node("router", router_agent)
    builder.add_node("code_expert", code_expert)
    builder.add_node("data_expert", data_expert)
    builder.add_node("writing_expert", writing_expert)
    builder.add_node("general_expert", general_expert)
    builder.add_node("aggregator", aggregator)

    # 路由条件边
    builder.add_conditional_edges(
        "router",
        lambda state: state["expert_type"],
        {
            "code_expert": "code_expert",
            "data_expert": "data_expert",
            "writing_expert": "writing_expert",
            "general_expert": "general_expert"
        }
    )

    # 所有专家连接到汇总
    builder.add_edge("code_expert", "aggregator")
    builder.add_edge("data_expert", "aggregator")
    builder.add_edge("writing_expert", "aggregator")
    builder.add_edge("general_expert", "aggregator")

    builder.set_entry_point("router")
    builder.set_finish_point("aggregator")

    graph = builder.compile()

    # 测试不同类型问题
    test_questions = [
        "如何在 Python 中实现一个简单的 Agent？",
        "如何分析用户行为数据？",
        "如何写一篇技术博客？",
        "什么是 LangGraph？"
    ]

    print("\n测试不同类型问题:")
    for question in test_questions:
        print(f"\n问题: {question}")
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "question": "",
            "expert_type": "general_expert",
            "expert_response": "",
            "final_answer": ""
        })
        print(f"专家类型: {result['expert_type']}")
        print(f"回答: {result['final_answer'][:100]}...")


# ==========================================
# 示例 4: 循环迭代生成
# ==========================================


def example_iterative_generation():
    """演示循环迭代的生成模式"""
    print("\n" + "=" * 50)
    print("示例 4: 循环迭代生成")
    print("=" * 50)

    # 定义状态
    class IterativeState(TypedDict):
        messages: Annotated[list, add_messages]
        draft: str
        critique: str
        revision_count: int
        max_revisions: int
        is_good: bool
        final_content: str

    llm = create_llm()

    def generator_agent(state: IterativeState) -> dict:
        """生成器 Agent"""
        print("  [生成器] 创建内容...")

        user_msg = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        if user_msg:
            # 如果是第一次生成或有修改建议
            context = ""
            if state["critique"]:
                context = f"\n修改建议：{state['critique']}"

            prompt = ChatPromptTemplate.from_template(
                """请写一段关于 '{topic}' 的介绍。

{context}

要求：简洁、专业、有说服力（50-100字）："""
            )
            chain = prompt | llm | StrOutputParser()
            draft = chain.invoke({"topic": user_msg, "context": context})

            print(f"    草稿: {draft[:80]}...")
            return {"draft": draft}

        return {"draft": state["draft"]}

    def critic_agent(state: IterativeState) -> dict:
        """批评者 Agent：评估内容"""
        print("  [批评者] 评估内容...")

        draft = state["draft"]
        prompt = ChatPromptTemplate.from_template(
            """请评估以下内容的优劣：

内容：{draft}

评估标准：
1. 语言流畅性
2. 信息准确性
3. 专业程度

请给出评分（1-10）和改进建议：
评分：X
建议：..."""
        )
        chain = prompt | llm | StrOutputParser()
        critique = chain.invoke({"draft": draft})

        print(f"    批评: {critique[:100]}...")

        # 模拟评分（实际应用中需要解析输出）
        # 简化：第三次迭代后认为足够好
        is_good = state["revision_count"] >= 2

        return {
            "critique": critique,
            "is_good": is_good,
            "revision_count": state["revision_count"] + 1
        }

    def optimizer_agent(state: IterativeState) -> dict:
        """优化器 Agent：修改内容"""
        print("  [优化器] 改进内容...")

        draft = state["draft"]
        critique = state["critique"]

        prompt = ChatPromptTemplate.from_template(
            """请根据批评意见优化以下内容：

原内容：{draft}
批评意见：{critique}

请给出优化后的版本："""
        )
        chain = prompt | llm | StrOutputParser()
        revised = chain.invoke({"draft": draft, "critique": critique})

        print(f"    优化后: {revised[:80]}...")

        return {
            "draft": revised,
            "critique": ""  # 清空批评，准备重新评估
        }

    def finalize_agent(state: IterativeState) -> dict:
        """最终确定"""
        print("  [完成] 内容最终确定")

        return {
            "final_content": state["draft"],
            "messages": [AIMessage(content=state["draft"])]
        }

    # 路由函数
    def route_after_critique(state: IterativeState) -> str:
        """批评后的路由"""
        if state["is_good"]:
            print("    路由: 内容足够好，完成")
            return "finalize"
        if state["revision_count"] >= state["max_revisions"]:
            print("    路由: 达到最大修改次数，完成")
            return "finalize"
        print("    路由: 需要优化")
        return "optimizer"

    # 构建图
    builder = StateGraph(IterativeState)

    builder.add_node("generator", generator_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("optimizer", optimizer_agent)
    builder.add_node("finalize", finalize_agent)

    # 线性连接 + 循环
    builder.add_edge("generator", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critique,
        {"finalize": "finalize", "optimizer": "optimizer"}
    )
    builder.add_edge("optimizer", "generator")  # 优化后回到生成器

    builder.set_entry_point("generator")
    builder.set_finish_point("finalize")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [HumanMessage(content="人工智能的未来")],
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "max_revisions": 3,
        "is_good": False,
        "final_content": ""
    })

    print(f"\n最终结果:")
    print(f"  修改次数: {result['revision_count']}")
    print(f"  最终内容: {result['final_content']}")


# ==========================================
# 示例 5: 任务分解与并行执行
# ==========================================


def example_task_decomposition():
    """演示任务分解后并行执行"""
    print("\n" + "=" * 50)
    print("示例 5: 任务分解与并行执行")
    print("=" * 50)

    # 定义状态
    class ParallelState(TypedDict):
        messages: Annotated[list, add_messages]
        original_task: str
        subtask_results: dict[str, str]
        final_summary: str

    llm = create_llm()

    def decomposer_agent(state: ParallelState) -> dict:
        """任务分解 Agent"""
        print("  [分解器] 分解任务...")

        user_msg = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        return {"original_task": user_msg}

    def subtask_a_agent(state: ParallelState) -> dict:
        """子任务 A Agent"""
        print("  [子任务A] 执行...")

        task = state["original_task"]
        prompt = ChatPromptTemplate.from_template(
            "请回答 '{task}' 的第一个方面：技术特点"
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"task": task})

        print(f"    完成: {result[:60]}...")

        subtask_results = state.get("subtask_results", {})
        subtask_results["subtask_a"] = result

        return {"subtask_results": subtask_results}

    def subtask_b_agent(state: ParallelState) -> dict:
        """子任务 B Agent"""
        print("  [子任务B] 执行...")

        task = state["original_task"]
        prompt = ChatPromptTemplate.from_template(
            "请回答 '{task}' 的第二个方面：应用场景"
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"task": task})

        print(f"    完成: {result[:60]}...")

        subtask_results = state.get("subtask_results", {})
        subtask_results["subtask_b"] = result

        return {"subtask_results": subtask_results}

    def subtask_c_agent(state: ParallelState) -> dict:
        """子任务 C Agent"""
        print("  [子任务C] 执行...")

        task = state["original_task"]
        prompt = ChatPromptTemplate.from_template(
            "请回答 '{task}' 的第三个方面：发展趋势"
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"task": task})

        print(f"    完成: {result[:60]}...")

        subtask_results = state.get("subtask_results", {})
        subtask_results["subtask_c"] = result

        return {"subtask_results": subtask_results}

    def aggregator_agent(state: ParallelState) -> dict:
        """汇总 Agent"""
        print("  [汇总器] 整合结果...")

        results = state["subtask_results"]
        all_results = "\n".join([
            f"方面{i}: {r}"
            for i, r in enumerate(results.values(), 1)
        ])

        prompt = ChatPromptTemplate.from_template(
            """请整合以下各方面的回答，生成完整回答：

{results}

请给出完整的总结："""
        )
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"results": all_results})

        print(f"    总结: {summary[:100]}...")

        return {
            "final_summary": summary,
            "messages": [AIMessage(content=summary)]
        }

    # 构建图
    builder = StateGraph(ParallelState)

    builder.add_node("decomposer", decomposer_agent)
    builder.add_node("subtask_a", subtask_a_agent)
    builder.add_node("subtask_b", subtask_b_agent)
    builder.add_node("subtask_c", subtask_c_agent)
    builder.add_node("aggregator", aggregator_agent)

    # 分解后并行分发到三个子任务
    builder.add_edge("decomposer", "subtask_a")
    builder.add_edge("decomposer", "subtask_b")
    builder.add_edge("decomposer", "subtask_c")

    # 所有子任务完成后汇总
    builder.add_edge("subtask_a", "aggregator")
    builder.add_edge("subtask_b", "aggregator")
    builder.add_edge("subtask_c", "aggregator")

    builder.set_entry_point("decomposer")
    builder.set_finish_point("aggregator")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [HumanMessage(content="LangGraph 框架")],
        "original_task": "",
        "subtask_results": {},
        "final_summary": ""
    })

    print(f"\n最终结果:")
    print(f"  任务: {result['original_task']}")
    print(f"  汇总: {result['final_summary'][:150]}...")


# ==========================================
# 示例 6: 多 Agent 对话模式
# ==========================================


def example_agent_conversation():
    """演示多 Agent 之间的对话模式"""
    print("\n" + "=" * 50)
    print("示例 6: 多 Agent 对话")
    print("=" * 50)

    # 定义状态
    class ConversationState(TypedDict):
        messages: Annotated[list, add_messages]
        current_speaker: str
        topic: str
        turn_count: int
        max_turns: int

    llm = create_llm()

    def agent_alpha(state: ConversationState) -> dict:
        """Agent Alpha：主张者"""
        print(f"  [Alpha - 第 {state['turn_count'] + 1} 轮] 发言...")

        topic = state["topic"]
        history = state["messages"]

        # 构建 Alpha 的发言
        context = "\n".join([f"{m.type}: {m.content[:50]}..." for m in history[-3:]] if len(history) > 3 else [])

        prompt = ChatPromptTemplate.from_template(
            """你 Alpha，主张 '{topic}' 是一个重要技术。

历史对话：
{history}

请发表你的观点（50字左右）："""
        )
        chain = prompt | llm | StrOutputParser()
        message = chain.invoke({"topic": topic, "history": context})

        print(f"    发言: {message[:80]}...")

        return {
            "messages": [AIMessage(content=f"[Alpha] {message}")],
            "current_speaker": "beta",
            "turn_count": state["turn_count"] + 1
        }

    def agent_beta(state: ConversationState) -> dict:
        """Agent Beta：质疑者"""
        print(f"  [Beta - 第 {state['turn_count']} 轮] 发言...")

        topic = state["topic"]
        history = state["messages"]

        context = "\n".join([f"{m.type}: {m.content[:50]}..." for m in history[-3:]] if len(history) > 3 else [])

        prompt = ChatPromptTemplate.from_template(
            """你是 Beta，对 '{topic}' 持谨慎态度。

历史对话：
{history}

请提出你的质疑或反驳（50字左右）："""
        )
        chain = prompt | llm | StrOutputParser()
        message = chain.invoke({"topic": topic, "history": context})

        print(f"    发言: {message[:80]}...")

        return {
            "messages": [AIMessage(content=f"[Beta] {message}")],
            "current_speaker": "alpha",
            "turn_count": state["turn_count"]
        }

    def moderator(state: ConversationState) -> dict:
        """主持人：总结对话"""
        print("  [主持人] 总结对话...")

        history = state["messages"]
        all_messages = "\n".join([m.content for m in history])

        prompt = ChatPromptTemplate.from_template(
            """请总结以下对话的主要内容：

{conversation}

请给出简洁的总结（100字左右）："""
        )
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"conversation": all_messages})

        print(f"    总结: {summary[:100]}...")

        return {
            "messages": [AIMessage(content=f"[总结] {summary}")]
        }

    # 路由函数
    def route_conversation(state: ConversationState) -> str:
        """路由对话"""
        if state["turn_count"] >= state["max_turns"]:
            print("    对话结束，生成总结")
            return "moderator"

        # Alpha 和 Beta 轮流发言
        return state["current_speaker"]

    # 构建图
    builder = StateGraph(ConversationState)

    builder.add_node("alpha", agent_alpha)
    builder.add_node("beta", agent_beta)
    builder.add_node("moderator", moderator)

    # Alpha 和 Beta 的对话循环
    builder.add_conditional_edges(
        "alpha",
        route_conversation,
        {"beta": "beta", "moderator": "moderator"}
    )
    builder.add_conditional_edges(
        "beta",
        route_conversation,
        {"alpha": "alpha", "moderator": "moderator"}
    )

    builder.set_entry_point("alpha")
    builder.set_finish_point("moderator")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [],
        "current_speaker": "alpha",
        "topic": "AI Agent 技术",
        "turn_count": 0,
        "max_turns": 3  # 每人发言 3 次
    })

    print(f"\n对话总结:")
    print(f"  轮次: {result['turn_count']}")
    print(f"  消息数: {len(result['messages'])}")
    for msg in result["messages"]:
        print(f"  {msg.content[:80]}...")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 16: LangGraph 多 Agent 工作流示例")
    print("=" * 60)

    # 运行各示例
    example_sequential_collaboration()
    example_supervisor_mode()
    example_expert_team()
    example_iterative_generation()
    example_task_decomposition()
    example_agent_conversation()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()