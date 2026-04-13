"""
选型指南示例 - 展示不同场景下的框架选择

本文件演示六大主流框架在不同应用场景下的选择策略：
- RAG 应用场景
- Agent 应用场景
- 多 Agent 系统场景
- 企业应用场景
- 快速原型场景
- 框架组合示例

帮助理解技术选型的决策过程。
"""

import os
from dotenv import load_dotenv

load_dotenv()


def selection_rag_application():
    """
    RAG 应用场景选型
    
    场景：构建知识库问答系统
    """
    print("\n=== RAG 应用场景选型 ===")
    
    print("场景描述：构建一个企业知识库问答系统")
    print("需求：")
    print("  - 处理大量文档数据")
    print("  - 高质量语义检索")
    print("  - 支持多种文档格式")
    print("  - 低延迟响应")
    print()
    
    # 选型分析
    print("框架分析：")
    print()
    
    print("1. LlamaIndex（推荐）：")
    print("   ✅ 专业数据处理")
    print("   ✅ 多种索引类型")
    print("   ✅ 高级检索策略")
    print("   ✅ 文档处理丰富")
    print()
    
    print("2. LangChain（备选）：")
    print("   ✅ 基础 RAG 支持")
    print("   ⚠️ 数据处理不如 LlamaIndex 专业")
    print("   ✅ 工具集成丰富")
    print()
    
    print("3. Dify（快速验证）：")
    print("   ✅ 内置知识库功能")
    print("   ✅ 可视化配置")
    print("   ⚠️ 定制能力有限")
    print()
    
    # 选型决策
    print("选型决策：")
    print("  首选：LlamaIndex")
    print("  理由：数据处理专业化、检索质量高")
    print()
    
    # 示例代码
    print("LlamaIndex RAG 示例：")
    print("""
# LlamaIndex RAG 最佳实践
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 1. 加载文档
documents = SimpleDirectoryReader("./knowledge").load_data()

# 2. 创建索引
index = VectorStoreIndex.from_documents(documents)

# 3. 配置检索器（带重排序）
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[CohereRerank(top_n=3)]
)

# 4. 查询
response = query_engine.query("公司政策是什么？")
""")


def selection_agent_application():
    """
    Agent 应用场景选型
    
    场景：构建任务执行 Agent
    """
    print("\n=== Agent 应用场景选型 ===")
    
    print("场景描述：构建一个能执行多工具任务的 Agent")
    print("需求：")
    print("  - 多工具调用能力")
    print("  - 自主决策执行")
    print("  - 对话式交互")
    print("  - 结果验证反馈")
    print()
    
    # 选型分析
    print("框架分析：")
    print()
    
    print("1. LangChain + LangGraph（推荐）：")
    print("   ✅ 工具定义丰富")
    print("   ✅ Agent 类型多样")
    print("   ✅ 状态管理完善")
    print("   ✅ 可视化流程")
    print()
    
    print("2. Semantic Kernel（企业场景）：")
    print("   ✅ 技能规划机制")
    print("   ✅ Azure 集成")
    print("   ⚠️ 学习曲线中等")
    print()
    
    print("3. AutoGen（研究场景）：")
    print("   ✅ Agent 对话灵活")
    print("   ⚠️ 配置复杂")
    print("   ⚠️ 生产适用性待验证")
    print()
    
    # 选型决策
    print("选型决策：")
    print("  首选：LangChain + LangGraph")
    print("  理由：工具丰富、状态完善、生态成熟")
    print()
    
    # 示例代码
    print("LangGraph Agent 示例：")
    print("""
# LangGraph Agent 最佳实践
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# 1. 定义工具
tools = [search_tool, calculator_tool, email_tool]

# 2. 创建 Agent 状态图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# 3. 配置路由
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)

# 4. 执行
app = graph.compile()
result = app.invoke({"input": "搜索并分析数据"})
""")


def selection_multi_agent():
    """
    多 Agent 系统场景选型
    
    场景：构建多 Agent 协作系统
    """
    print("\n=== 多 Agent 系统场景选型 ===")
    
    print("场景描述：构建多个 Agent 协作的复杂系统")
    print("需求：")
    print("  - Agent 自主对话")
    print("  - 角色分工明确")
    print("  - 任务协作执行")
    print("  - 结果综合输出")
    print()
    
    # 选型分析
    print("框架分析：")
    print()
    
    print("1. AutoGen（推荐）：")
    print("   ✅ 多 Agent 原生支持")
    print("   ✅ 对话机制完善")
    print("   ✅ GroupChat 管理")
    print("   ✅ 研究场景验证")
    print()
    
    print("2. LangGraph（生产场景）：")
    print("   ✅ 图状编排")
    print("   ✅ 状态持久化")
    print("   ✅ 生产适用")
    print("   ⚠️ Agent 对话不如 AutoGen 直观")
    print()
    
    print("3. Semantic Kernel：")
    print("   ⚠️ 多 Agent 支持较弱")
    print("   ✅ 技能规划可组合")
    print()
    
    # 选型决策
    print("选型决策：")
    print("  研究：AutoGen（多 Agent 对话专业）")
    print("  生产：LangGraph（状态管理完善）")
    print("  组合：AutoGen（实验）→ LangGraph（生产）")
    print()
    
    # 示例代码
    print("AutoGen 多 Agent 示例：")
    print("""
# AutoGen 多 Agent 最佳实践
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# 1. 定义多个 Agent
researcher = AssistantAgent("researcher", system_message="你是研究员")
analyst = AssistantAgent("analyst", system_message="你是分析师")
writer = AssistantAgent("writer", system_message="你是撰稿人")

# 2. 创建群聊
groupchat = GroupChat(
    agents=[researcher, analyst, writer],
    max_round=10
)

# 3. 创建管理器
manager = GroupChatManager(groupchat=groupchat)

# 4. 启动协作
user.initiate_chat(manager, message="研究并撰写报告")
""")


def selection_enterprise_application():
    """
    企业应用场景选型
    
    场景：构建企业级 LLM 应用
    """
    print("\n=== 企业应用场景选型 ===")
    
    print("场景描述：构建企业内部 LLM 应用")
    print("需求：")
    print("  - 企业系统集成")
    print("  - 权限和安全控制")
    print("  - 可观测性")
    print("  - 长期维护支持")
    print()
    
    # 选型分析
    print("框架分析：")
    print()
    
    print("1. Semantic Kernel（Azure 企业推荐）：")
    print("   ✅ Azure 生态无缝集成")
    print("   ✅ 企业级安全")
    print("   ✅ Microsoft 支持")
    print("   ✅ 多语言支持（C#/Python）")
    print()
    
    print("2. LangChain + LangGraph：")
    print("   ✅ 灵活定制")
    print("   ✅ 社区活跃")
    print("   ⚠️ 需自行集成企业服务")
    print()
    
    print("3. Dify（快速部署）：")
    print("   ✅ 开箱即用")
    print("   ✅ 低运维成本")
    print("   ⚠️ 定制能力有限")
    print()
    
    # 选型决策
    print("选型决策：")
    print("  Azure 环境：Semantic Kernel")
    print("  其他云环境：LangChain + LangGraph")
    print("  快速部署：Dify（企业版）")
    print()
    
    # 示例代码
    print("Semantic Kernel 企业集成示例：")
    print("""
# Semantic Kernel + Azure 最佳实践
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# 1. 连接 Azure OpenAI
kernel.add_chat_service(
    "azure_chat",
    AzureChatCompletion(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        deployment_name="gpt-4"
    )
)

# 2. 注册企业技能
kernel.register_skill(EnterpriseSkills())

# 3. 使用 Planner 自动编排
planner = SequentialPlanner(kernel)
plan = planner.create_plan("处理企业数据")

# 4. 执行
result = await plan.invoke_async()
""")


def selection_prototype():
    """
    快速原型场景选型
    
    场景：快速验证想法
    """
    print("\n=== 快速原型场景选型 ===")
    
    print("场景描述：快速验证一个 LLM 应用想法")
    print("需求：")
    print("  - 快速搭建")
    print("  - 低门槛")
    print("  - 可视化调试")
    print("  - 功能验证")
    print()
    
    # 选型分析
    print("框架分析：")
    print()
    
    print("1. Dify（推荐）：")
    print("   ✅ 可视化构建")
    print("   ✅ 零代码启动")
    print("   ✅ 内置模板")
    print("   ✅ 快速迭代")
    print()
    
    print("2. LangChain（技术验证）：")
    print("   ✅ 灵活实验")
    print("   ✅ 组件可替换")
    print("   ⚠️ 需编程能力")
    print()
    
    print("3. 组合模式：")
    print("   Dify（原型）→ LangChain（定制）")
    print()
    
    # 选型决策
    print("选型决策：")
    print("  首选：Dify")
    print("  理由：可视化、零门槛、快速验证")
    print("  后续：验证成功后转 LangChain 定制")
    print()
    
    # 示例代码
    print("Dify + LangChain 组合示例：")
    print("""
# 原型阶段：使用 Dify 可视化构建
# [在 Dify Web 界面拖拽创建工作流]

# 定制阶段：使用 LangChain 代码实现
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 将 Dify 原型转化为代码
llm = ChatOpenAI(model="deepseek-chat")
prompt = ChatPromptTemplate.from_template(
    "根据原型设计实现功能: {requirement}"
)
chain = prompt | llm

# API 部署
from fastapi import FastAPI
app = FastAPI()

@app.post("/api/chat")
async def chat(message: str):
    return chain.invoke({"requirement": message})
""")


def selection_decision_flowchart():
    """
    选型决策流程图
    
    展示完整的选型决策过程
    """
    print("\n=== 选型决策流程图 ===")
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                    选型决策流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Step 1: 确定应用类型                                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │   RAG      │  │   Agent    │  │   多Agent   │          │
│   └───────┬─────┘  └───────┬─────┘  └───────┬─────┘          │
│           │                │                │                │
│           ▼                ▼                ▼                │
│   LlamaIndex      LangChain        AutoGen                  │
│                    LangGraph                                │
│                                                                 │
│   Step 2: 确定部署环境                                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │   Azure    │  │   其他云   │  │   本地     │          │
│   └───────┬─────┘  └───────┬─────┘  └───────┬─────┘          │
│           │                │                │                │
│           ▼                ▼                ▼                │
│   Semantic Kernel  LangChain    LangChain                 │
│                    LlamaIndex   LlamaIndex                 │
│                                                                 │
│   Step 3: 确定团队能力                                          │
│   ┌─────────────┐  ┌─────────────┐                          │
│   │   有编程   │  │   无编程   │                          │
│   └───────┬─────┘  └───────┬─────┘                          │
│           │                │                                │
│           ▼                ▼                                │
│   编程框架         Dify                               │
│   (LangChain/                                       │
│    LlamaIndex/etc.)                                      │
│                                                                 │
│   Step 4: 确定时间限制                                          │
│   ┌─────────────┐  ┌─────────────┐                          │
│   │   快速     │  │   长期     │                          │
│   └───────┬─────┘  └───────┬─────┘                          │
│           │                │                                │
│           ▼                ▼                                │
│   Dify            定制框架                             │
│   LangChain       (深度定制)                            │
│                                                                 │
│   Step 5: 综合决策                                              │
│   根据以上维度，选择最匹配的框架                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")


def framework_combination_example():
    """
    框架组合示例
    
    展示如何组合使用多个框架
    """
    print("\n=== 框架组合示例 ===")
    
    print("组合 1: LangChain + LlamaIndex")
    print("场景：Agent + 高质量 RAG")
    print("""
# LangChain Agent 调用 LlamaIndex 检索器
from langchain.agents import AgentExecutor
from llama_index.core import VectorStoreIndex

# LlamaIndex 创建高质量索引
index = VectorStoreIndex.from_documents(docs)
retriever = index.as_retriever()

# LangChain Agent 使用检索器
def search_knowledge(query: str) -> str:
    nodes = retriever.retrieve(query)
    return "\n".join([n.text for n in nodes])

# 定义工具
from langchain.tools import Tool
tools = [
    Tool(name="KnowledgeSearch", func=search_knowledge)
]

# 创建 Agent
agent = create_tool_calling_agent(llm, tools)
executor = AgentExecutor(agent=agent, tools=tools)
""")
    
    print("\n组合 2: Dify + Python 定制")
    print("场景：可视化原型 + 定制扩展")
    print("""
# Dify 可视化创建基础流程
# [在 Web 界面设计]

# Python 定制扩展复杂逻辑
import requests

def call_dify_workflow(input_data):
    response = requests.post(
        "https://api.dify.ai/v1/workflows/run",
        headers={"Authorization": f"Bearer {DIFY_API_KEY}"},
        json={"input": input_data}
    )
    return process_result(response.json())

# 定制后处理
def process_result(dify_output):
    # 添加定制逻辑
    return enhanced_output
""")
    
    print("\n组合 3: AutoGen 研究 + LangGraph 生产")
    print("场景：研究实验转生产部署")
    print("""
# 阶段 1: AutoGen 研究实验
# 验证多 Agent 协作逻辑

# 阶段 2: LangGraph 生产迁移
# 将 AutoGen 对话逻辑转为状态图

# AutoGen 概念迁移
researcher → LangGraph node A
analyst → LangGraph node B  
writer → LangGraph node C
GroupChat → StateGraph with edges
""")


def main():
    """运行所有选型示例"""
    print("=" * 60)
    print("选型指南示例 - Day 20")
    print("=" * 60)
    
    # 运行各场景选型示例
    selection_rag_application()
    selection_agent_application()
    selection_multi_agent()
    selection_enterprise_application()
    selection_prototype()
    
    # 选型决策流程
    selection_decision_flowchart()
    
    # 框架组合示例
    framework_combination_example()
    
    print("\n" + "=" * 60)
    print("选型指南总结")
    print("=" * 60)
    
    print("""
场景选型速查表：

| 场景          | 首选          | 备选          | 组合建议      |
|---------------|---------------|---------------|---------------|
| RAG 应用      | LlamaIndex    | LangChain     | -             |
| 单 Agent      | LangGraph     | LangChain     | -             |
| 多 Agent      | AutoGen       | LangGraph     | 研究→生产     |
| 企业集成      | Semantic Kernel| LangChain    | Azure 优先    |
| 快速原型      | Dify          | LangChain     | 原型→定制     |
| 生产部署      | LangGraph     | LangChain     | 状态管理      |

选型核心原则：
1. 场景适配 - 根据主要需求选择专业框架
2. 团队能力 - 匹配技术栈和熟悉度
3. 环境约束 - 考虑云环境和部署要求
4. 时间限制 - 快速验证 vs 长期定制
5. 扩展需求 - 未来功能的可扩展性
""")


if __name__ == "__main__":
    main()