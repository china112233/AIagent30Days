"""
Day 16: LangGraph 循环与分支示例

本文件演示 LangGraph 的控制流设计：
- 条件边与动态路由
- 循环与迭代执行
- 提前终止条件
- 错误处理与重试
"""

import os
from typing import TypedDict, Annotated
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
# 示例 1: 简单循环
# ==========================================


def example_simple_loop():
    """演示最基本的循环结构"""
    print("\n" + "=" * 50)
    print("示例 1: 简单循环")
    print("=" * 50)

    # 定义状态
    class LoopState(TypedDict):
        counter: int
        max_iterations: int
        history: list[str]

    # 定义节点
    def increment_node(state: LoopState) -> dict:
        """增加计数"""
        current = state["counter"]
        print(f"  [循环] 第 {current + 1} 次")

        return {
            "counter": current + 1,
            "history": state["history"] + [f"迭代 {current + 1}"]
        }

    # 路由函数
    def should_continue(state: LoopState) -> str:
        """判断是否继续循环"""
        if state["counter"] >= state["max_iterations"]:
            print("  [路由] 达到上限，结束循环")
            return END
        print("  [路由] 继续循环")
        return "increment"

    # 构建图
    builder = StateGraph(LoopState)

    builder.add_node("increment", increment_node)

    # 添加循环条件边
    builder.add_conditional_edges(
        "increment",
        should_continue,
        {
            END: END,
            "increment": "increment"  # 循环回自己
        }
    )

    builder.set_entry_point("increment")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "counter": 0,
        "max_iterations": 5,
        "history": []
    })

    print(f"\n结果:")
    print(f"  最终计数: {result['counter']}")
    print(f"  历史记录: {result['history']}")


# ==========================================
# 示例 2: 条件分支循环
# ==========================================


def example_conditional_loop():
    """演示带条件判断的循环"""
    print("\n" + "=" * 50)
    print("示例 2: 条件分支循环")
    print("=" * 50)

    # 定义状态
    class ConditionalState(TypedDict):
        value: int
        target: int
        steps: list[str]

    # 定义节点
    def increase_node(state: ConditionalState) -> dict:
        """增加值的节点"""
        new_value = state["value"] + 1
        print(f"  [增加] {state['value']} -> {new_value}")
        return {
            "value": new_value,
            "steps": state["steps"] + [f"增加: {new_value}"]
        }

    def decrease_node(state: ConditionalState) -> dict:
        """减少值的节点"""
        new_value = state["value"] - 1
        print(f"  [减少] {state['value']} -> {new_value}")
        return {
            "value": new_value,
            "steps": state["steps"] + [f"减少: {new_value}"]
        }

    # 决策路由
    def decide_action(state: ConditionalState) -> str:
        """决定下一步操作"""
        if state["value"] == state["target"]:
            print("  [决策] 达到目标值，结束")
            return END
        elif state["value"] < state["target"]:
            print("  [决策] 当前值小于目标，增加")
            return "increase"
        else:
            print("  [决策] 当前值大于目标，减少")
            return "decrease"

    # 构建图
    builder = StateGraph(ConditionalState)

    builder.add_node("increase", increase_node)
    builder.add_node("decrease", decrease_node)

    # 从两个节点都添加条件边
    builder.add_conditional_edges("increase", decide_action, {END: END, "increase": "increase", "decrease": "decrease"})
    builder.add_conditional_edges("decrease", decide_action, {END: END, "increase": "increase", "decrease": "decrease"})

    builder.set_entry_point("increase")

    graph = builder.compile()

    # 测试：从 0 调整到 3
    print("\n测试 1: 从 0 调整到 3")
    result1 = graph.invoke({"value": 0, "target": 3, "steps": []})
    print(f"  结果: value={result1['value']}, steps={result1['steps']}")

    # 测试：从 5 调整到 2
    print("\n测试 2: 从 5 调整到 2")
    result2 = graph.invoke({"value": 5, "target": 2, "steps": []})
    print(f"  结果: value={result2['value']}, steps={result2['steps']}")


# ==========================================
# 示例 3: ReAct 循环（推理-行动循环）
# ==========================================


def example_react_loop():
    """演示 ReAct 模式的推理-行动循环"""
    print("\n" + "=" * 50)
    print("示例 3: ReAct 循环")
    print("=" * 50)

    # 定义状态
    class ReActState(TypedDict):
        messages: Annotated[list, add_messages]
        thought: str
        action: str
        action_result: str
        iterations: int
        max_iterations: int
        final_answer: str

    llm = create_llm()

    def think_node(state: ReActState) -> dict:
        """推理节点"""
        print("  [思考] 分析当前情况...")

        # 获取最后一条用户消息
        messages = state["messages"]
        user_msg = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        if user_msg:
            # 生成思考
            think_prompt = ChatPromptTemplate.from_template(
                """你是一个智能助手。分析以下问题：

问题：{question}

当前迭代次数：{iterations}

请给出你的思考过程（一句话）："""
            )
            chain = think_prompt | llm | StrOutputParser()
            thought = chain.invoke({
                "question": user_msg,
                "iterations": state["iterations"]
            })

            print(f"    思考: {thought[:100]}...")

            return {
                "thought": thought,
                "iterations": state["iterations"] + 1
            }

        return {"iterations": state["iterations"] + 1}

    def decide_node(state: ReActState) -> dict:
        """决策节点：决定是否需要执行工具"""
        print("  [决策] 判断是否需要行动...")

        # 简化逻辑：超过最大迭代或已有答案就结束
        if state["iterations"] >= state["max_iterations"]:
            print("    决策: 达到最大迭代，生成最终答案")
            return {"action": "finish"}

        # 模拟决策
        if state["iterations"] == 1:
            print("    决策: 需要查询信息")
            return {"action": "search"}
        elif state["iterations"] == 2:
            print("    决策: 信息已获取，可以回答")
            return {"action": "finish"}
        else:
            return {"action": "finish"}

    def act_node(state: ReActState) -> dict:
        """行动节点"""
        print("  [行动] 执行模拟工具...")
        action = state["action"]

        if action == "search":
            # 模拟搜索结果
            result = "搜索结果：LangGraph 是一个用于构建有状态多角色应用的框架..."
            print(f"    搜索完成")
            return {"action_result": result}
        else:
            return {"action_result": ""}

    def answer_node(state: ReActState) -> dict:
        """生成最终答案"""
        print("  [回答] 生成最终答案...")

        messages = state["messages"]
        user_msg = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_msg = msg.content

        if user_msg:
            context = state.get("action_result", "")
            answer_prompt = ChatPromptTemplate.from_template(
                """基于以下信息回答问题：

问题：{question}
参考信息：{context}

请给出简洁的回答："""
            )
            chain = answer_prompt | llm | StrOutputParser()
            answer = chain.invoke({"question": user_msg, "context": context})

            print(f"    回答: {answer[:100]}...")
            return {"final_answer": answer}

        return {"final_answer": ""}

    # 路由函数
    def route_after_think(state: ReActState) -> str:
        """思考后的路由"""
        if state["iterations"] >= state["max_iterations"]:
            return "answer"
        return "decide"

    def route_after_decide(state: ReActState) -> str:
        """决策后的路由"""
        action = state["action"]
        if action == "finish":
            return "answer"
        return "act"

    def route_after_act(state: ReActState) -> str:
        """行动后的路由"""
        # 行动后继续思考
        return "think"

    # 构建图
    builder = StateGraph(ReActState)

    builder.add_node("think", think_node)
    builder.add_node("decide", decide_node)
    builder.add_node("act", act_node)
    builder.add_node("answer", answer_node)

    # 添加条件边
    builder.add_conditional_edges("think", route_after_think, {"answer": "answer", "decide": "decide"})
    builder.add_conditional_edges("decide", route_after_decide, {"answer": "answer", "act": "act"})
    builder.add_edge("act", "think")  # 行动后回到思考

    builder.set_entry_point("think")
    builder.set_finish_point("answer")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "messages": [HumanMessage(content="什么是 LangGraph？")],
        "thought": "",
        "action": "",
        "action_result": "",
        "iterations": 0,
        "max_iterations": 3,
        "final_answer": ""
    })

    print(f"\n最终结果:")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  最终答案: {result['final_answer'][:150]}...")


# ==========================================
# 示例 4: 带中断的人机交互循环
# ==========================================


def example_human_in_loop():
    """演示人机交互循环（简化版）"""
    print("\n" + "=" * 50)
    print("示例 4: 人机交互循环")
    print("=" * 50)

    # 定义状态
    class HumanLoopState(TypedDict):
        content: str
        review_status: str  # "pending", "approved", "rejected"
        revision_count: int
        max_revisions: int
        final_content: str

    llm = create_llm()

    def generate_node(state: HumanLoopState) -> dict:
        """生成内容"""
        print("  [生成] 创建内容...")

        prompt = ChatPromptTemplate.from_template(
            "请写一段关于 {topic} 的简短介绍（50字左右）"
        )
        chain = prompt | llm | StrOutputParser()
        content = chain.invoke({"topic": "人工智能"})

        print(f"    生成内容: {content[:80]}...")

        return {
            "content": content,
            "review_status": "pending"
        }

    def human_review_node(state: HumanLoopState) -> dict:
        """模拟人工审核节点"""
        print("  [审核] 等待人工审核...")
        print("    (实际应用中，这里会中断等待用户输入)")

        # 模拟审核结果
        if state["revision_count"] < 2:
            # 模拟第一次被拒绝
            print("    模拟审核结果: 需要修改")
            return {"review_status": "rejected"}
        else:
            # 模拟最终被批准
            print("    模拟审核结果: 已批准")
            return {"review_status": "approved"}

    def revise_node(state: HumanLoopState) -> dict:
        """修改内容"""
        print("  [修改] 根据反馈修改内容...")

        revision_prompt = ChatPromptTemplate.from_template(
            "请改进以下内容，使其更加专业：\n{content}"
        )
        chain = revision_prompt | llm | StrOutputParser()
        revised = chain.invoke({"content": state["content"]})

        print(f"    修改后内容: {revised[:80]}...")

        return {
            "content": revised,
            "revision_count": state["revision_count"] + 1,
            "review_status": "pending"
        }

    def finalize_node(state: HumanLoopState) -> dict:
        """最终确定"""
        print("  [完成] 内容已批准")
        return {"final_content": state["content"]}

    # 路由函数
    def route_after_review(state: HumanLoopState) -> str:
        """审核后的路由"""
        if state["review_status"] == "approved":
            return "finalize"
        if state["revision_count"] >= state["max_revisions"]:
            print("    达到最大修改次数，强制完成")
            return "finalize"
        return "revise"

    # 构建图
    builder = StateGraph(HumanLoopState)

    builder.add_node("generate", generate_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("revise", revise_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge("generate", "human_review")
    builder.add_conditional_edges("human_review", route_after_review, {"finalize": "finalize", "revise": "revise"})
    builder.add_edge("revise", "human_review")

    builder.set_entry_point("generate")
    builder.set_finish_point("finalize")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "content": "",
        "review_status": "",
        "revision_count": 0,
        "max_revisions": 3,
        "final_content": ""
    })

    print(f"\n最终结果:")
    print(f"  修改次数: {result['revision_count']}")
    print(f"  最终内容: {result['final_content'][:100]}...")


# ==========================================
# 示例 5: 错误处理与重试
# ==========================================


def example_retry_loop():
    """演示错误处理和自动重试"""
    print("\n" + "=" * 50)
    print("示例 5: 错误处理与重试")
    print("=" * 50)

    # 定义状态
    class RetryState(TypedDict):
        attempt: int
        max_attempts: int
        success: bool
        error_message: str
        result: str

    def risky_operation_node(state: RetryState) -> dict:
        """模拟可能失败的操作"""
        print(f"  [尝试] 第 {state['attempt'] + 1} 次执行...")

        # 模拟成功率随次数提高
        success_rate = 0.3 + (state["attempt"] * 0.2)  # 30%, 50%, 70%...

        # 简化：第三次尝试必定成功
        if state["attempt"] >= 2:
            print("    执行成功！")
            return {
                "attempt": state["attempt"] + 1,
                "success": True,
                "result": "操作成功完成",
                "error_message": ""
            }
        else:
            # 模拟失败
            print("    执行失败")
            return {
                "attempt": state["attempt"] + 1,
                "success": False,
                "result": "",
                "error_message": "临时错误"
            }

    def handle_error_node(state: RetryState) -> dict:
        """处理错误"""
        print("  [错误处理] 分析错误...")
        print(f"    错误信息: {state['error_message']}")
        print("    准备重试...")
        return {}

    def success_node(state: RetryState) -> dict:
        """成功完成"""
        print("  [成功] 操作完成")
        return {}

    # 路由函数
    def route_after_operation(state: RetryState) -> str:
        """操作后的路由"""
        if state["success"]:
            return "success"

        if state["attempt"] >= state["max_attempts"]:
            print("    达到最大尝试次数")
            return "success"  # 即使失败也要结束

        return "handle_error"

    def route_after_error_handling(state: RetryState) -> str:
        """错误处理后的路由"""
        return "risky_operation"

    # 构建图
    builder = StateGraph(RetryState)

    builder.add_node("risky_operation", risky_operation_node)
    builder.add_node("handle_error", handle_error_node)
    builder.add_node("success", success_node)

    builder.add_conditional_edges("risky_operation", route_after_operation, {"success": "success", "handle_error": "handle_error"})
    builder.add_edge("handle_error", "risky_operation")

    builder.set_entry_point("risky_operation")
    builder.set_finish_point("success")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "attempt": 0,
        "max_attempts": 5,
        "success": False,
        "error_message": "",
        "result": ""
    })

    print(f"\n最终结果:")
    print(f"  尝试次数: {result['attempt']}")
    print(f"  成功状态: {result['success']}")
    print(f"  结果: {result['result']}")


# ==========================================
# 示例 6: 复杂多路径分支
# ==========================================


def example_multi_path_branch():
    """演示复杂的多路径分支"""
    print("\n" + "=" * 50)
    print("示例 6: 复杂多路径分支")
    print("=" * 50)

    # 定义状态
    class MultiPathState(TypedDict):
        input_type: str  # "query", "command", "feedback"
        content: str
        processed_content: str
        output: str

    def classify_node(state: MultiPathState) -> dict:
        """分类输入"""
        print("  [分类] 分析输入类型...")

        content = state["content"]
        content_lower = content.lower()

        # 简化分类逻辑
        if content.startswith("/") or "执行" in content_lower:
            input_type = "command"
        elif "?" in content or "什么" in content or "如何" in content_lower:
            input_type = "query"
        else:
            input_type = "feedback"

        print(f"    类型: {input_type}")
        return {"input_type": input_type}

    def query_handler_node(state: MultiPathState) -> dict:
        """处理查询"""
        print("  [查询处理] 回答问题...")
        return {"output": f"[查询回答] {state['content']}"}

    def command_handler_node(state: MultiPathState) -> dict:
        """处理命令"""
        print("  [命令处理] 执行命令...")
        return {"output": f"[命令执行] {state['content']}"}

    def feedback_handler_node(state: MultiPathState) -> dict:
        """处理反馈"""
        print("  [反馈处理] 记录反馈...")
        return {"output": f"[反馈记录] {state['content']}"}

    def finalize_node(state: MultiPathState) -> dict:
        """最终处理"""
        print("  [完成] 输出结果")
        return {"processed_content": state["output"]}

    # 路由函数
    def route_by_type(state: MultiPathState) -> str:
        """根据类型路由"""
        return f"{state['input_type']}_handler"

    # 构建图
    builder = StateGraph(MultiPathState)

    builder.add_node("classify", classify_node)
    builder.add_node("query_handler", query_handler_node)
    builder.add_node("command_handler", command_handler_node)
    builder.add_node("feedback_handler", feedback_handler_node)
    builder.add_node("finalize", finalize_node)

    # 分类后的条件边
    builder.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "query_handler": "query_handler",
            "command_handler": "command_handler",
            "feedback_handler": "feedback_handler"
        }
    )

    # 所有处理器都连接到最终节点
    builder.add_edge("query_handler", "finalize")
    builder.add_edge("command_handler", "finalize")
    builder.add_edge("feedback_handler", "finalize")

    builder.set_entry_point("classify")
    builder.set_finish_point("finalize")

    graph = builder.compile()

    # 测试不同类型输入
    test_inputs = [
        ("什么是 LangGraph？", "query"),
        ("执行数据备份", "command"),
        ("这个功能很有用", "feedback"),
    ]

    print("\n测试不同类型输入:")
    for content, expected_type in test_inputs:
        print(f"\n输入: {content}")
        result = graph.invoke({
            "input_type": "",
            "content": content,
            "processed_content": "",
            "output": ""
        })
        print(f"分类结果: {result['input_type']} (预期: {expected_type})")
        print(f"输出: {result['output']}")


# ==========================================
# 示例 7: 提前终止模式
# ==========================================


def example_early_exit():
    """演示提前终止模式"""
    print("\n" + "=" * 50)
    print("示例 7: 提前终止模式")
    print("=" * 50)

    # 定义状态
    class EarlyExitState(TypedDict):
        value: int
        threshold: int
        steps: list[str]
        early_exit_triggered: bool

    def step1_node(state: EarlyExitState) -> dict:
        """步骤 1"""
        print("  [步骤1] 执行...")
        new_value = state["value"] + 10

        # 检查是否超过阈值
        if new_value > state["threshold"]:
            print("    触发提前终止条件")
            return {
                "value": new_value,
                "steps": state["steps"] + ["步骤1-提前终止"],
                "early_exit_triggered": True
            }

        return {
            "value": new_value,
            "steps": state["steps"] + ["步骤1"]
        }

    def step2_node(state: EarlyExitState) -> dict:
        """步骤 2"""
        print("  [步骤2] 执行...")
        return {
            "value": state["value"] + 5,
            "steps": state["steps"] + ["步骤2"]
        }

    def step3_node(state: EarlyExitState) -> dict:
        """步骤 3"""
        print("  [步骤3] 执行...")
        return {
            "value": state["value"] + 3,
            "steps": state["steps"] + ["步骤3"]
        }

    def final_node(state: EarlyExitState) -> dict:
        """最终节点"""
        print("  [完成] 流程结束")
        return {}

    # 路由函数
    def route_after_step1(state: EarlyExitState) -> str:
        """步骤1后的路由"""
        if state["early_exit_triggered"]:
            return "final"
        return "step2"

    # 构建图
    builder = StateGraph(EarlyExitState)

    builder.add_node("step1", step1_node)
    builder.add_node("step2", step2_node)
    builder.add_node("step3", step3_node)
    builder.add_node("final", final_node)

    builder.add_conditional_edges("step1", route_after_step1, {"final": "final", "step2": "step2"})
    builder.add_edge("step2", "step3")
    builder.add_edge("step3", "final")

    builder.set_entry_point("step1")
    builder.set_finish_point("final")

    graph = builder.compile()

    # 测试 1: 正常流程
    print("\n测试 1: 正常流程（不触发提前终止）")
    result1 = graph.invoke({
        "value": 0,
        "threshold": 100,
        "steps": [],
        "early_exit_triggered": False
    })
    print(f"  最终值: {result1['value']}")
    print(f"  步骤: {result1['steps']}")

    # 测试 2: 提前终止
    print("\n测试 2: 提前终止（触发条件）")
    result2 = graph.invoke({
        "value": 0,
        "threshold": 5,
        "steps": [],
        "early_exit_triggered": False
    })
    print(f"  最终值: {result2['value']}")
    print(f"  步骤: {result2['steps']}")
    print(f"  提前终止: {result2['early_exit_triggered']}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 16: LangGraph 循环与分支示例")
    print("=" * 60)

    # 运行各示例
    example_simple_loop()
    example_conditional_loop()
    example_react_loop()
    example_human_in_loop()
    example_retry_loop()
    example_multi_path_branch()
    example_early_exit()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()