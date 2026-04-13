"""
框架对比示例 - 展示各框架的核心用法差异

本文件演示六大主流 LLM 框架的典型使用方式：
- LangChain: LCEL 链示例
- LangGraph: 状态图示例
- LlamaIndex: 索引查询示例
- Semantic Kernel: 技能调用示例
- AutoGen: 多 Agent 对话示例

帮助理解各框架的代码风格和设计理念差异。
"""

import os
from dotenv import load_dotenv

load_dotenv()


def example_langchain_lcel():
    """
    LangChain LCEL 链示例
    
    特点：
    - 使用管道语法 | 组合组件
    - 支持流式输出
    - 声明式链定义
    """
    print("\n=== LangChain LCEL 示例 ===")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 创建模型
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0.7
        )
        
        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的技术顾问。"),
            ("user", "{input}")
        ])
        
        # 创建输出解析器
        parser = StrOutputParser()
        
        # LCEL 组合链（管道语法）
        chain = prompt | llm | parser
        
        # 执行
        result = chain.invoke({"input": "LangChain 的核心优势是什么？"})
        print(f"结果: {result[:200]}...")
        
        # LangChain 特点总结
        print("\nLangChain 特点：")
        print("  - 代码风格：管道语法组合组件")
        print("  - 适合场景：通用 LLM 应用开发")
        print("  - 灵活性：高（组件可自由组合）")
        print("  - 学习难度：中等")
        
    except ImportError as e:
        print(f"需要安装 langchain: {e}")
        print("pip install langchain langchain-openai langchain-core")


def example_langgraph_state():
    """
    LangGraph 状态图示例
    
    特点：
    - 显式状态定义
    - 支持循环和条件分支
    - 图状编排结构
    """
    print("\n=== LangGraph 状态图示例 ===")
    
    try:
        from typing import TypedDict
        from langgraph.graph import StateGraph, END
        
        # 定义状态结构
        class AnalysisState(TypedDict):
            input: str
            analysis: str | None
            result: str | None
            iterations: int
        
        # 定义节点函数
        def analyze_node(state: AnalysisState) -> dict:
            """分析节点：处理输入"""
            return {
                "analysis": f"分析结果: {state['input'][:50]}...",
                "iterations": state.get("iterations", 0) + 1
            }
        
        def process_node(state: AnalysisState) -> dict:
            """处理节点：生成结果"""
            return {
                "result": f"处理完成: {state['analysis']}"
            }
        
        def should_continue(state: AnalysisState) -> str:
            """条件路由：决定是否继续"""
            if state["iterations"] < 2:
                return "analyze"
            return END
        
        # 创建状态图
        graph = StateGraph(AnalysisState)
        
        # 添加节点
        graph.add_node("analyze", analyze_node)
        graph.add_node("process", process_node)
        
        # 设置入口
        graph.set_entry_point("analyze")
        
        # 添加边（条件 + 普通）
        graph.add_conditional_edges(
            "analyze",
            should_continue,
            {"analyze": "analyze", END: "process"}
        )
        graph.add_edge("process", END)
        
        # 编译图
        app = graph.compile()
        
        # 执行
        initial_state = {"input": "测试输入数据", "iterations": 0}
        result = app.invoke(initial_state)
        print(f"最终状态: {result}")
        
        # LangGraph 特点总结
        print("\nLangGraph 特点：")
        print("  - 代码风格：显式状态 + 图结构定义")
        print("  - 适合场景：复杂 Agent 工作流")
        print("  - 灵活性：高（循环/分支/状态管理）")
        print("  - 学习难度：中等")
        
    except ImportError as e:
        print(f"需要安装 langgraph: {e}")
        print("pip install langgraph")


def example_llamaindex_query():
    """
    LlamaIndex 索引查询示例
    
    特点：
    - 数据为核心
    - 专业索引类型
    - 检索优先设计
    """
    print("\n=== LlamaIndex 索引查询示例 ===")
    
    try:
        from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core.settings import Settings
        from llama_index.llms.openai_like import OpenAILikeLLM
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        # 配置模型
        llm = OpenAILikeLLM(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL"),
        )
        Settings.llm = llm
        
        # 创建模拟文档
        documents = [
            Document(text="LangChain 是一个通用的 LLM 应用开发框架。"),
            Document(text="LlamaIndex 专注于数据索引和检索。"),
            Document(text="LangGraph 支持复杂的状态图编排。"),
        ]
        
        # 创建向量索引
        index = VectorStoreIndex.from_documents(documents)
        
        # 创建查询引擎
        query_engine = index.as_query_engine()
        
        # 执行查询
        response = query_engine.query("哪个框架专注于数据检索？")
        print(f"查询结果: {response}")
        
        # LlamaIndex 特点总结
        print("\nLlamaIndex 特点：")
        print("  - 代码风格：数据 → 索引 → 查询")
        print("  - 适合场景：RAG、知识库应用")
        print("  - 灵活性：中等（检索专业化）")
        print("  - 学习难度：中等")
        
    except ImportError as e:
        print(f"需要安装 llama-index: {e}")
        print("pip install llama-index llama-index-core")


def example_semantic_kernel_skills():
    """
    Semantic Kernel 技能调用示例
    
    特点：
    - 技能驱动设计
    - 自动规划执行
    - 语义函数概念
    """
    print("\n=== Semantic Kernel 技能调用示例 ===")
    
    print("Semantic Kernel 概念示例（需要实际安装）：")
    
    # 概念代码（展示设计理念）
    concept_code = '''
# Semantic Kernel 技能定义示例

# 语义函数（基于提示）
skill = kernel.create_semantic_function(
    "Translate",
    "将 {{input}} 翻译成中文",
    max_tokens=100
)

# 原生函数（Python 代码）
@kernel_function
def get_current_time() -> str:
    """获取当前时间"""
    import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")

# Planner 自动编排
planner = SequentialPlanner(kernel)
plan = planner.create_plan("分析数据并生成报告")

# 执行计划
result = await plan.invoke_async()
'''
    print(concept_code)
    
    # Semantic Kernel 特点总结
    print("\nSemantic Kernel 特点：")
    print("  - 代码风格：技能定义 + Planner 编排")
    print("  - 适合场景：企业应用集成（Azure）")
    print("  - 灵活性：高（技能复用）")
    print("  - 学习难度：中等")
    
    print("\n注意: Semantic Kernel 需安装 semantic-kernel 包")


def example_autogen_agents():
    """
    AutoGen 多 Agent 对话示例
    
    特点：
    - 多 Agent 对话机制
    - Agent 自主协作
    - 代码执行沙箱
    """
    print("\n=== AutoGen 多 Agent 对话示例 ===")
    
    print("AutoGen 概念示例（需要实际安装）：")
    
    # 概念代码（展示设计理念）
    concept_code = '''
# AutoGen 多 Agent 配置示例

from autogen import AssistantAgent, UserProxyAgent

# 配置 LLM
llm_config = {
    "model": "deepseek-chat",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL"),
}

# 创建助手 Agent
assistant = AssistantAgent(
    name="assistant",
    system_message="你是专业的 Python 编程助手",
    llm_config=llm_config
)

# 创建用户代理
user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",  # 自动模式
    code_execution_config={"work_dir": "sandbox"}
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="写一个计算斐波那契数列的函数"
)
'''
    print(concept_code)
    
    # AutoGen 特点总结
    print("\nAutoGen 特点：")
    print("  - 代码风格：Agent 定义 → 对话流程")
    print("  - 适合场景：多 Agent 协作、研究实验")
    print("  - 灵活性：高（Agent 自主协作）")
    print("  - 学习难度：较高")
    
    print("\n注意: AutoGen 需安装 pyautogen 包")


def example_dify_workflow():
    """
    Dify 工作流概念示例
    
    特点：
    - 可视化构建
    - 低代码平台
    - 开箱即用
    """
    print("\n=== Dify 工作流概念示例 ===")
    
    print("Dify 是低代码可视化平台，主要通过 Web 界面使用：")
    
    # Dify API 调用示例
    api_code = '''
# Dify API 调用示例

import requests

DIFY_API_URL = "https://api.dify.ai"
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

# 调用 Dify Chat App
response = requests.post(
    f"{DIFY_API_URL}/v1/chat-messages",
    headers={"Authorization": f"Bearer {DIFY_API_KEY}"},
    json={
        "query": "什么是 RAG？",
        "user": "user-123",
        "response_mode": "blocking"
    }
)

print(response.json())
'''
    print(api_code)
    
    # Dify 特点总结
    print("\nDify 特点：")
    print("  - 使用方式：Web 可视化界面")
    print("  - 适合场景：快速原型、非技术人员")
    print("  - 灵活性：中等（可视化限制）")
    print("  - 学习难度：低")


def compare_framework_styles():
    """
    框架代码风格对比
    
    展示各框架处理同一任务的不同代码风格
    """
    print("\n=== 框架代码风格对比 ===")
    
    print("场景：构建一个简单的问答链")
    print()
    
    # LangChain 风格
    print("1. LangChain 风格（管道组合）：")
    print("   chain = prompt | llm | parser")
    print("   result = chain.invoke({'input': 'question'})")
    print()
    
    # LangGraph 风格
    print("2. LangGraph 风格（状态图）：")
    print("   graph = StateGraph(State)")
    print("   graph.add_node('llm', llm_node)")
    print("   app = graph.compile()")
    print("   result = app.invoke({'input': 'question'})")
    print()
    
    # LlamaIndex 风格
    print("3. LlamaIndex 风格（索引查询）：")
    print("   index = VectorStoreIndex.from_documents(docs)")
    print("   engine = index.as_query_engine()")
    print("   response = engine.query('question')")
    print()
    
    # Semantic Kernel 风格
    print("4. Semantic Kernel 风格（技能规划）：")
    print("   planner = SequentialPlanner(kernel)")
    print("   plan = planner.create_plan('answer question')")
    print("   result = await plan.invoke_async()")
    print()
    
    # AutoGen 风格
    print("5. AutoGen 风格（Agent 对话）：")
    print("   assistant = AssistantAgent(...)")
    print("   user.initiate_chat(assistant, message='question')")
    print()
    
    # Dify 风格
    print("6. Dify 风格（可视化节点）：")
    print("   [Start Node] → [LLM Node] → [End Node]")
    print("   （通过 Web 界面拖拽连接）")
    print()


def main():
    """运行所有示例"""
    print("=" * 60)
    print("框架对比示例 - Day 20")
    print("=" * 60)
    
    # 运行各框架示例
    example_langchain_lcel()
    example_langgraph_state()
    example_llamaindex_query()
    example_semantic_kernel_skills()
    example_autogen_agents()
    example_dify_workflow()
    
    # 代码风格对比
    compare_framework_styles()
    
    print("\n" + "=" * 60)
    print("框架对比总结")
    print("=" * 60)
    
    print("""
各框架核心差异：

| 框架          | 核心概念      | 代码风格       | 最佳场景          |
|---------------|---------------|----------------|-------------------|
| LangChain     | Chain/组件    | 管道组合       | 通用 LLM 应用     |
| LangGraph     | StateGraph    | 显式状态       | 复杂 Agent 工作流 |
| LlamaIndex    | Index/Query   | 数据流         | RAG/知识库        |
| Semantic Kernel | Skill/Planner | 技能规划     | 企业集成          |
| AutoGen       | Agent/Chat    | 对话协作       | 多 Agent 系统     |
| Dify          | Workflow/Node | 可视化节点    | 快速原型          |

选型建议：
- RAG 应用：LlamaIndex（首选）
- 复杂 Agent：LangGraph
- 多 Agent 协作：AutoGen
- 企业集成：Semantic Kernel
- 快速原型：Dify
- 通用开发：LangChain
""")


if __name__ == "__main__":
    main()