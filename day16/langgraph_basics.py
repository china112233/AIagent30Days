"""
Day 16: LangGraph 基础示例

本文件演示 LangGraph 的核心概念：
- StateGraph 创建与配置
- 节点定义和添加
- 边的连接方式
- 图的编译和执行
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
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
# 示例 1: 最简单的 StateGraph
# ==========================================


def example_simple_graph():
    """演示最基本的状态图创建"""
    print("\n" + "=" * 50)
    print("示例 1: 最简单的 StateGraph")
    print("=" * 50)

    # 定义状态类型
    class SimpleState(TypedDict):
        input: str
        output: str

    # 定义节点函数
    def process_node(state: SimpleState) -> dict:
        """处理节点"""
        print(f"  [处理节点] 输入: {state['input']}")
        # 对输入进行处理（这里简单添加前缀）
        return {"output": f"处理结果: {state['input']}"}

    # 创建状态图
    builder = StateGraph(SimpleState)

    # 添加节点
    builder.add_node("process", process_node)

    # 设置入口和出口
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    # 编译图
    graph = builder.compile()

    # 执行图
    result = graph.invoke({"input": "Hello LangGraph!", "output": ""})
    print(f"\n最终结果: {result}")


# ==========================================
# 示例 2: 多节点线性流程
# ==========================================


def example_linear_flow():
    """演示多个节点的线性执行"""
    print("\n" + "=" * 50)
    print("示例 2: 多节点线性流程")
    print("=" * 50)

    # 定义状态
    class LinearState(TypedDict):
        input: str
        step1_result: str
        step2_result: str
        final_output: str

    # 定义节点
    def step1_node(state: LinearState) -> dict:
        """步骤 1: 预处理"""
        print(f"  [步骤1] 预处理输入...")
        return {"step1_result": f"[预处理] {state['input']}"}

    def step2_node(state: LinearState) -> dict:
        """步骤 2: 分析"""
        print(f"  [步骤2] 分析内容...")
        content = state["step1_result"]
        return {"step2_result": f"[分析] {content} -> 已理解"}

    def step3_node(state: LinearState) -> dict:
        """步骤 3: 输出"""
        print(f"  [步骤3] 生成输出...")
        return {"final_output": f"[最终] {state['step2_result']}"}

    # 构建图
    builder = StateGraph(LinearState)

    # 添加节点
    builder.add_node("step1", step1_node)
    builder.add_node("step2", step2_node)
    builder.add_node("step3", step3_node)

    # 添加边（线性连接）
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")

    # 设置入口和出口
    builder.set_entry_point("step1")
    builder.set_finish_point("step3")

    # 编译并执行
    graph = builder.compile()

    result = graph.invoke({
        "input": "测试数据",
        "step1_result": "",
        "step2_result": "",
        "final_output": ""
    })

    print(f"\n执行流程:")
    print(f"  输入: {result['input']}")
    print(f"  步骤1: {result['step1_result']}")
    print(f"  步骤2: {result['step2_result']}")
    print(f"  步骤3: {result['final_output']}")


# ==========================================
# 示例 3: 集成 LLM 的节点
# ==========================================


def example_llm_node():
    """演示在节点中调用 LLM"""
    print("\n" + "=" * 50)
    print("示例 3: 集成 LLM 的节点")
    print("=" * 50)

    llm = create_llm()

    # 定义状态
    class LLMState(TypedDict):
        question: str
        analysis: str
        answer: str

    # 定义节点
    def analyze_node(state: LLMState) -> dict:
        """分析问题的节点"""
        print("  [分析节点] 正在分析问题...")

        prompt = ChatPromptTemplate.from_template(
            "请分析以下问题的类型和难度（用一句话）：\n问题：{question}"
        )
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({"question": state["question"]})

        print(f"    分析结果: {analysis}")
        return {"analysis": analysis}

    def answer_node(state: LLMState) -> dict:
        """回答问题的节点"""
        print("  [回答节点] 正在生成回答...")

        prompt = ChatPromptTemplate.from_template(
            """基于以下分析结果回答问题：

分析：{analysis}
问题：{question}

请提供详细回答："""
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question": state["question"],
            "analysis": state["analysis"]
        })

        return {"answer": answer}

    # 构建图
    builder = StateGraph(LLMState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("answer", answer_node)

    builder.add_edge("analyze", "answer")
    builder.set_entry_point("analyze")
    builder.set_finish_point("answer")

    # 编译并执行
    graph = builder.compile()

    result = graph.invoke({
        "question": "什么是 LangGraph？它有什么优势？",
        "analysis": "",
        "answer": ""
    })

    print(f"\n问题: {result['question']}")
    print(f"分析: {result['analysis']}")
    print(f"回答: {result['answer'][:200]}...")


# ==========================================
# 示例 4: 条件路由
# ==========================================


def example_conditional_routing():
    """演示基于状态的条件路由"""
    print("\n" + "=" * 50)
    print("示例 4: 条件路由")
    print("=" * 50)

    # 定义状态
    class RouterState(TypedDict):
        input: str
        route: str
        output: str

    # 定义节点
    def classify_node(state: RouterState) -> dict:
        """分类节点：判断输入类型"""
        print(f"  [分类节点] 分析输入类型...")

        # 简单分类逻辑
        input_lower = state["input"].lower()
        if "翻译" in input_lower or "translate" in input_lower:
            route = "translate"
        elif "总结" in input_lower or "summarize" in input_lower:
            route = "summarize"
        else:
            route = "general"

        print(f"    路由到: {route}")
        return {"route": route}

    def translate_node(state: RouterState) -> dict:
        """翻译节点"""
        print("  [翻译节点] 执行翻译...")
        return {"output": f"[翻译结果] {state['input']}"}

    def summarize_node(state: RouterState) -> dict:
        """总结节点"""
        print("  [总结节点] 执行总结...")
        return {"output": f"[总结结果] {state['input']}"}

    def general_node(state: RouterState) -> dict:
        """通用处理节点"""
        print("  [通用节点] 通用处理...")
        return {"output": f"[通用处理结果] {state['input']}"}

    # 路由函数
    def route_function(state: RouterState) -> str:
        """根据状态决定下一步"""
        return state["route"]

    # 构建图
    builder = StateGraph(RouterState)

    # 添加节点
    builder.add_node("classify", classify_node)
    builder.add_node("translate", translate_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("general", general_node)

    # 添加条件边
    builder.add_conditional_edges(
        "classify",
        route_function,
        {
            "translate": "translate",
            "summarize": "summarize",
            "general": "general"
        }
    )

    # 设置出口
    builder.set_entry_point("classify")
    builder.set_finish_point("translate")
    builder.set_finish_point("summarize")
    builder.set_finish_point("general")

    # 编译图
    graph = builder.compile()

    # 测试不同输入
    test_inputs = [
        "请翻译这段话：Hello World",
        "请总结一下今天的新闻",
        "请介绍一下 LangGraph"
    ]

    for input_text in test_inputs:
        print(f"\n测试输入: {input_text}")
        result = graph.invoke({"input": input_text, "route": "", "output": ""})
        print(f"路由: {result['route']}")
        print(f"输出: {result['output']}")


# ==========================================
# 示例 5: 图的可视化
# ==========================================


def example_graph_visualization():
    """演示如何可视化图结构"""
    print("\n" + "=" * 50)
    print("示例 5: 图的可视化")
    print("=" * 50)

    # 定义状态
    class VisualState(TypedDict):
        input: str
        step1: str
        step2: str
        output: str

    # 定义节点
    def node_a(state: VisualState) -> dict:
        return {"step1": "A processed"}

    def node_b(state: VisualState) -> dict:
        return {"step2": "B processed"}

    def node_c(state: VisualState) -> dict:
        return {"output": "C processed"}

    # 构建图
    builder = StateGraph(VisualState)

    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)

    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", "node_c")

    builder.set_entry_point("node_a")
    builder.set_finish_point("node_c")

    graph = builder.compile()

    # 获取图的 ASCII 表示
    print("图结构（ASCII）:")
    try:
        # 尝试获取图的结构信息
        print(f"  节点: {list(graph.nodes.keys())}")
        print(f"  边: {graph.edges}")
    except Exception as e:
        print(f"  无法获取详细结构: {e}")

    # 执行图
    result = graph.invoke({
        "input": "test",
        "step1": "",
        "step2": "",
        "output": ""
    })
    print(f"\n执行结果: {result}")


# ==========================================
# 示例 6: 流式执行
# ==========================================


def example_stream_execution():
    """演示图的流式执行"""
    print("\n" + "=" * 50)
    print("示例 6: 流式执行")
    print("=" * 50)

    llm = create_llm()

    # 定义状态
    class StreamState(TypedDict):
        prompt: str
        response: str

    # 定义节点
    def generate_node(state: StreamState) -> dict:
        """生成节点（流式）"""
        print("  [生成节点] 开始流式生成...")

        # 流式调用 LLM
        prompt = ChatPromptTemplate.from_template("{prompt}")
        chain = prompt | llm | StrOutputParser()

        full_response = ""
        for chunk in chain.stream({"prompt": state["prompt"]}):
            print(chunk, end="", flush=True)
            full_response += chunk

        print()  # 换行
        return {"response": full_response}

    # 构建图
    builder = StateGraph(StreamState)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("generate")
    builder.set_finish_point("generate")

    graph = builder.compile()

    # 执行
    result = graph.invoke({
        "prompt": "请简短介绍 LangGraph 的三个主要特点",
        "response": ""
    })

    print(f"\n完成，响应长度: {len(result['response'])} 字符")


# ==========================================
# 示例 7: 批量处理
# ==========================================


def example_batch_processing():
    """演示批量处理多个输入"""
    print("\n" + "=" * 50)
    print("示例 7: 批量处理")
    print("=" * 50)

    # 定义状态
    class BatchState(TypedDict):
        item: str
        processed: str

    # 定义节点
    def process_node(state: BatchState) -> dict:
        """处理单个项目"""
        return {"processed": f"[已处理] {state['item']}"}

    # 构建图
    builder = StateGraph(BatchState)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    graph = builder.compile()

    # 批量输入
    items = ["项目A", "项目B", "项目C", "项目D"]

    print("批量处理:")
    for item in items:
        result = graph.invoke({"item": item, "processed": ""})
        print(f"  {item} -> {result['processed']}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 16: LangGraph 基础示例")
    print("=" * 60)

    # 运行各示例
    example_simple_graph()
    example_linear_flow()
    example_llm_node()
    example_conditional_routing()
    example_graph_visualization()
    example_stream_execution()
    example_batch_processing()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()