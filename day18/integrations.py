"""
Day 18: LlamaIndex 外部工具集成示例

本文件演示外部集成功能：
- 向量数据库集成（Chroma/Pinecone）
- 外部 API 工具集成
- Wikipedia 搜索集成
- Web 搜索集成
- 多 LLM 后端配置
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载环境变量
load_dotenv()


def setup_settings():
    """配置全局设置"""
    llm = OpenAILike(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
        temperature=0.7,
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ 设置已配置")


# ==========================================
# 示例 1: Chroma 向量数据库集成
# ==========================================


def example_chroma_integration():
    """演示 Chroma 向量数据库集成"""
    print("\n" + "=" * 50)
    print("示例 1: Chroma 向量数据库集成")
    print("=" * 50)

    print("""
  Chroma 集成架构：

  ┌─────────────────────────────────────────────┐
  │              Chroma 集成流程                 │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   ┌─────────────────────────────────────┐  │
  │   │ Step 1: 创建 Chroma 客户端          │  │
  │   │                                     │  │
  │   │ db = chromadb.PersistentClient(    │  │
  │   │     path="./chroma_db"             │  │
  │   │ )                                  │  │
  │   └─────────────────────────────────────┘  │
  │                   │                         │
  │                   ▼                         │
  │   ┌─────────────────────────────────────┐  │
  │   │ Step 2: 创建 Collection             │  │
  │   │                                     │  │
  │   │ collection = db.get_or_create_collection( │
  │   │     "my_documents"                 │  │
  │   │ )                                  │  │
  │   └─────────────────────────────────────┘  │
  │                   │                         │
  │                   ▼                         │
  │   ┌─────────────────────────────────────┐  │
  │   │ Step 3: 创建向量存储适配器          │  │
  │   │                                     │  │
  │   │ vector_store = ChromaVectorStore(  │  │
  │   │     chroma_collection=collection   │  │
  │   │ )                                  │  │
  │   └─────────────────────────────────────┘  │
  │                   │                         │
  │                   ▼                         │
  │   ┌─────────────────────────────────────┐  │
  │   │ Step 4: 创建索引                    │  │
  │   │                                     │  │
  │   │ index = VectorStoreIndex.from_vector_store( │
  │   │     vector_store                   │  │
  │   │ )                                  │  │
  │   └─────────────────────────────────────┘  │
  │                                             │
  └─────────────────────────────────────────────┘

  安装依赖：
  pip install chromadb llama-index-vector-stores-chroma

  配置示例：
  ┌─────────────────────────────────────────────┐
  │ import chromadb                            │
  │ from llama_index.vector_stores.chroma import │
  │     ChromaVectorStore                      │
  │                                            │
  │ # 持久化客户端                              │
  │ db = chromadb.PersistentClient(            │
  │     path="./chroma_db"                     │
  │ )                                          │
  │                                            │
  │ # 创建/获取集合                             │
  │ chroma_collection = db.get_or_create_collection( │
  │     "my_collection"                        │
  │ )                                          │
  │                                            │
  │ # 向量存储适配器                            │
  │ vector_store = ChromaVectorStore(          │
  │     chroma_collection=chroma_collection    │
  │ )                                          │
  │                                            │
  │ # 创建索引                                  │
  │ storage_context = StorageContext.from_defaults( │
  │     vector_store=vector_store              │
  │ )                                          │
  │                                            │
  │ index = VectorStoreIndex.from_documents(   │
  │     documents,                              │
  │     storage_context=storage_context        │
  │ )                                          │
  └─────────────────────────────────────────────┘
    """)


# ==========================================
# 示例 2: Pinecone 向量数据库集成
# ==========================================


def example_pinecone_integration():
    """演示 Pinecone 向量数据库集成"""
    print("\n" + "=" * 50)
    print("示例 2: Pinecone 向量数据库集成")
    print("=" * 50)

    print("""
  Pinecone 集成架构：

  ┌─────────────────────────────────────────────┐
  │              Pinecone 特点                   │
  ├─────────────────────────────────────────────┤
  │                                             │
  │ 优势：                                      │
  │ - 云托管，无需本地运维                      │
  │ - 高性能，适合大规模                        │
  │ - 自动扩展                                  │
  │ - 企业级安全                                │
  │                                             │
  │ 缺点：                                      │
  │ - 需要付费（有免费额度）                    │
  │ - 数据存储在云端                            │
  │                                             │
  └─────────────────────────────────────────────┘

  安装依赖：
  pip install pinecone-client llama-index-vector-stores-pinecone

  配置示例：
  ┌─────────────────────────────────────────────┐
  │ from pinecone import Pinecone              │
  │ from llama_index.vector_stores.pinecone import │
  │     PineconeVectorStore                    │
  │                                            │
  │ # Pinecone 客户端                           │
  │ pc = Pinecone(                             │
  │     api_key="your-api-key"                 │
  │ )                                          │
  │                                            │
  │ # 创建索引（如果不存在）                     │
  │ pc.create_index(                           │
  │     name="my-index",                       │
  │     dimension=384,  # 与嵌入模型匹配        │
  │     metric="cosine",                       │
  │     spec=ServerlessSpec(                   │
  │         cloud="aws",                       │
  │         region="us-east-1"                 │
  │     )                                      │
  │ )                                          │
  │                                            │
  │ # 获取索引                                  │
  │ pinecone_index = pc.Index("my-index")      │
  │                                            │
  │ # 向量存储适配器                            │
  │ vector_store = PineconeVectorStore(        │
  │     pinecone_index=pinecone_index          │
  │ )                                          │
  └─────────────────────────────────────────────┘

  注意事项：
  - dimension 必须与嵌入模型输出维度匹配
  - 免费版有索引数量限制
  - 需要设置环境变量 PINECONE_API_KEY
    """)


# ==========================================
# 示例 3: 多 LLM 后端配置
# ==========================================


def example_multi_llm_backend():
    """演示多 LLM 后端配置"""
    print("\n" + "=" * 50)
    print("示例 3: 多 LLM 后端配置")
    print("=" * 50)

    print("""
  多 LLM 后端架构：

  ┌─────────────────────────────────────────────┐
  │              LLM 后端选择                    │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   ┌───────────┐                             │
  │   │ Query     │                             │
  │   └───────────┘                             │
  │         │                                   │
  │         ▼                                   │
  │   ┌─────────────────────────────────────┐  │
  │   │         LLM Router                  │  │
  │   │                                     │  │
  │   │   根据场景选择合适的 LLM            │  │
  │   └─────────────────────────────────────┘  │
  │         │                                   │
  │         ├────────────────────────────────┐ │
  │         │                                 │ │
  │         ▼                ▼                ▼ │
  │   ┌─────────┐   ┌─────────┐   ┌─────────┐ │
  │   │ OpenAI  │   │ DeepSeek│   │ Claude  │ │
  │   │         │   │         │   │         │ │
  │   └─────────┘   └─────────┘   └─────────┘ │
  │                                             │
  └─────────────────────────────────────────────┘

  LLM 配置示例：

  ┌─────────────────────────────────────────────┐
  │ # OpenAI                                   │
  │ from llama_index.llms.openai import OpenAI │
  │                                            │
  │ openai_llm = OpenAI(                       │
  │     model="gpt-4",                         │
  │     api_key="your-key"                     │
  │ )                                          │
  │                                            │
  │ # DeepSeek (OpenAI 兼容)                    │
  │ from llama_index.llms.openai_like import   │
  │     OpenAILike                             │
  │                                            │
  │ deepseek_llm = OpenAILike(                 │
  │     model="deepseek-chat",                 │
  │     api_key="your-key",                    │
  │     api_base="https://api.deepseek.com",   │
  │     is_chat_model=True                     │
  │ )                                          │
  │                                            │
  │ # Claude (Anthropic)                        │
  │ from llama_index.llms.anthropic import     │
  │     Anthropic                              │
  │                                            │
  │ claude_llm = Anthropic(                    │
  │     model="claude-3-opus",                 │
  │     api_key="your-key"                     │
  │ )                                          │
  │                                            │
  │ # 本地模型 (Ollama)                         │
  │ from llama_index.llms.ollama import Ollama │
  │                                            │
  │ local_llm = Ollama(                        │
  │     model="llama2",                        │
  │     url="http://localhost:11434"           │
  │ )                                          │
  └─────────────────────────────────────────────┘

  选择建议：

  ┌─────────────────┬─────────────────┬─────────────────┐
  │     LLM         │    适用场景     │    成本         │
  ├─────────────────┼─────────────────┼─────────────────┤
  │ GPT-4           │ 高质量复杂任务  │ 较高            │
  │ DeepSeek        │ 中文友好、性价比│ 中等            │
  │ Claude          │ 长文本、安全    │ 较高            │
  │ Ollama 本地     │ 隐私、无成本    │ 免费            │
  │ GPT-3.5         │ 快速简单任务    │ 低              │
  └─────────────────┴─────────────────┴─────────────────┘
    """)


# ==========================================
# 示例 4: Wikipedia 工具集成
# ==========================================


def example_wikipedia_integration():
    """演示 Wikipedia 工具集成"""
    print("\n" + "=" * 50)
    print("示例 4: Wikipedia 工具集成")
    print("=" * 50)

    print("""
  Wikipedia 工具集成：

  ┌─────────────────────────────────────────────┐
  │              Wikipedia 查询流程              │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   ┌─────────────────────────────────────┐  │
  │   │ Query Engine with Wikipedia Tool    │  │
  │   └─────────────────────────────────────┘  │
  │                   │                         │
  │                   ▼                         │
  │   ┌─────────────────────────────────────┐  │
  │   │ Wikipedia Search Tool               │  │
  │   │                                     │  │
  │   │ 1. 搜索 Wikipedia                   │  │
  │   │ 2. 获取相关文章                     │  │
  │   │ 3. 提取内容                         │  │
  │   └─────────────────────────────────────┘  │
  │                   │                         │
  │                   ▼                         │
  │   ┌─────────────────────────────────────┐  │
  │   │ Wikipedia Reader                    │  │
  │   │                                     │  │
  │   │ - 加载文章内容                       │  │
  │   │ - 创建本地索引                       │  │
  │   │ - 支持查询                          │  │
  │   └─────────────────────────────────────┘  │
  │                                             │
  └─────────────────────────────────────────────┘

  安装依赖：
  pip install wikipedia llama-index-tools-wikipedia

  使用示例：
  ┌─────────────────────────────────────────────┐
  │ from llama_index.tools.wikipedia import    │
  │     WikipediaToolSpec                       │
  │                                            │
  │ # 创建 Wikipedia 工具                        │
  │ wiki_tool = WikipediaToolSpec()            │
  │                                            │
  │ # 搜索 Wikipedia                            │
  │ results = wiki_tool.search("Python programming") │
  │                                            │
  │ # 加载特定文章                              │
  │ doc = wiki_tool.load_data(                 │
  │     pages=["Python_(programming_language)"] │
  │ )                                          │
  │                                            │
  │ # 创建索引并查询                            │
  │ index = VectorStoreIndex.from_documents(   │
  │     doc                                    │
  │ )                                          │
  │ query_engine = index.as_query_engine()     │
  │ response = query_engine.query(             │
  │     "What is Python used for?"             │
  │ )                                          │
  └─────────────────────────────────────────────┘

  应用场景：
  - 实时知识查询
  -百科知识问答
  - 动态信息检索
    """)


# ==========================================
# 示例 5: Web 搜索集成
# ==========================================


def example_web_search_integration():
    """演示 Web 搜索集成"""
    print("\n" + "=" * 50)
    print("示例 5: Web 搜索集成")
    print("=" * 50)

    print("""
  Web 搜索集成：

  ┌─────────────────────────────────────────────┐
  │              Web 搜索工具选项                │
  ├─────────────────────────────────────────────┤
  │                                             │
  │ 1. Google Search                            │
  │    ─────────────────────────────────────── │
  │    from llama_index.tools.google import     │
  │        GoogleSearchToolSpec                 │
  │                                             │
  │    需要：Google API Key 和 Search Engine ID │
  │                                             │
  │ 2. DuckDuckGo Search                        │
  │    ─────────────────────────────────────── │
  │    from llama_index.tools.duckduckgo import │
  │        DuckDuckGoSearchToolSpec             │
  │                                             │
  │    无需 API Key，免费                        │
  │                                             │
  │ 3. Bing Search                              │
  │    ─────────────────────────────────────── │
  │    from llama_index.tools.bing import       │
  │        BingSearchToolSpec                   │
  │                                             │
  │    需要：Bing API Key                        │
  │                                             │
  └─────────────────────────────────────────────┘

  DuckDuckGo 示例（免费）：
  ┌─────────────────────────────────────────────┐
  │ # 安装                                      │
  │ pip install duckduckgo-search               │
  │                                            │
  │ # 使用                                      │
  │ from duckduckgo_search import DDGS         │
  │                                            │
  │ def web_search(query: str, max_results: int = 5): │
  │     results = []                            │
  │     with DDGS() as ddgs:                   │
  │         for r in ddgs.text(query, max_results=max_results): │
  │             results.append(r)              │
  │     return results                          │
  │                                            │
  │ # 搜索示例                                  │
  │ results = web_search(                       │
  │     "latest AI news 2024",                  │
  │     max_results=5                           │
  │ )                                          │
  │                                            │
  │ for r in results:                           │
  │     print(r['title'])                       │
  │     print(r['href'])                        │
  │     print(r['body'][:100])                  │
  └─────────────────────────────────────────────┘

  集成到 Agent：
  ┌─────────────────────────────────────────────┐
  │ # 创建 Web 水果工具                          │
  │ from llama_index.core.tools import FunctionTool │
  │                                            │
  │ web_tool = FunctionTool.from_defaults(     │
  │     fn=web_search                           │
  │ )                                          │
  │                                            │
  │ # 创建 Agent                                │
  │ from llama_index.core.agent import ReActAgent │
  │                                            │
  │ agent = ReActAgent.from_tools(             │
  │     [web_tool],                            │
  │     llm=llm                                │
  │ )                                          │
  └─────────────────────────────────────────────┘
    """)


# ==========================================
# 示例 6: 自定义工具集成
# ==========================================


def example_custom_tool_integration():
    """演示自定义工具集成"""
    print("\n" + "=" * 50)
    print("示例 6: 自定义工具集成")
    print("=" * 50)

    print("""
  自定义工具开发：

  ┌─────────────────────────────────────────────┐
  │              自定义工具结构                  │
  ├─────────────────────────────────────────────┤
  │                                             │
  │ 1. 定义工具函数                              │
  │    ─────────────────────────────────────── │
  │    def my_tool(query: str) -> str:          │
  │        """工具描述"""                        │
  │        # 工具逻辑                            │
  │        return result                        │
  │                                             │
  │ 2. 创建 Tool 对象                            │
  │    ─────────────────────────────────────── │
  │    from llama_index.core.tools import       │
  │        FunctionTool                          │
  │                                             │
  │    tool = FunctionTool.from_defaults(       │
  │        fn=my_tool                            │
  │    )                                        │
  │                                             │
  │ 3. 集成到 Agent                              │
  │    ─────────────────────────────────────── │
  │    agent = ReActAgent.from_tools(           │
  │        [tool],                               │
  │        llm=llm                               │
  │    )                                        │
  │                                             │
  └─────────────────────────────────────────────┘

  示例：天气查询工具
  ┌─────────────────────────────────────────────┐
  │ import requests                             │
  │                                            │
  │ def get_weather(city: str) -> str:         │
  │     """获取城市天气信息                     │
  │                                            │
  │     Args:                                   │
  │         city: 城市名称                      │
  │                                            │
  │     Returns:                                │
  │         天气描述字符串                      │
  │     """                                    │
  │     # 实际使用时替换为真实 API              │
  │     url = f"https://api.weather.com/{city}" │
  │     response = requests.get(url)           │
  │     return response.json()                 │
  │                                            │
  │ # 创建工具                                  │
  │ weather_tool = FunctionTool.from_defaults( │
  │     fn=get_weather                          │
  │ )                                          │
  └─────────────────────────────────────────────┘

  示例：计算器工具
  ┌─────────────────────────────────────────────┐
  │ def calculator(expression: str) -> float:  │
  │     """计算数学表达式                      │
  │                                            │
  │     Args:                                   │
  │         expression: 数学表达式              │
  │                                            │
  │     Returns:                                │
  │         计算结果                            │
  │     """                                    │
  │     try:                                    │
  │         return eval(expression)            │
  │     except Exception as e:                 │
  │         return f"错误: {str(e)}"            │
  │                                            │
  │ calc_tool = FunctionTool.from_defaults(    │
  │     fn=calculator                           │
  │ )                                          │
  └─────────────────────────────────────────────┘

  工具最佳实践：
  - 清晰的函数文档字符串
  - 明确的参数类型
  - 错误处理
  - 返回值一致性
    """)


# ==========================================
# 示例 7: API 集成最佳实践
# ==========================================


def example_api_integration_best_practices():
    """演示 API 集成最佳实践"""
    print("\n" + "=" * 50)
    print("示例 7: API 集成最佳实践")
    print("=" * 50)

    print("""
  API 集成最佳实践：

  ┌─────────────────────────────────────────────┐
  │              API 集成要点                    │
  ├─────────────────────────────────────────────┤
  │                                             │
  │ 1. 错误处理                                  │
  │    ─────────────────────────────────────── │
  │    try:                                      │
  │        response = api_call()                 │
  │    except RateLimitError:                    │
  │        # 处理速率限制                        │
  │        wait_and_retry()                      │
  │    except APIError:                          │
  │        # 处理 API 错误                        │
  │        return fallback_response()            │
  │                                             │
  │ 2. 速率限制                                  │
  │    ─────────────────────────────────────── │
  │    - 了解 API 速率限制                       │
  │    - 使用合适的请求间隔                      │
  │    - 实现退避重试                            │
  │                                             │
  │ 3. 缓存策略                                  │
  │    ─────────────────────────────────────── │
  │    - 缓存常用查询结果                        │
  │    - 设置合理的过期时间                      │
  │    - 减少 API 调用                          │
  │                                             │
  │ 4. 安全配置                                  │
  │    ─────────────────────────────────────── │
  │    - 使用环境变量存储 Key                    │
  │    - 不要硬编码敏感信息                      │
  │    - 使用 HTTPS                              │
  │                                             │
  │ 5. 监控日志                                  │
  │    ─────────────────────────────────────── │
  │    - 记录 API 调用                           │
  │    - 监控响应时间                            │
  │    - 追踪错误率                              │
  │                                             │
  └─────────────────────────────────────────────┘

  示例：带错误处理的 API 工具
  ┌─────────────────────────────────────────────┐
  │ import time                                 │
  │ from functools import wraps                 │
  │                                            │
  │ def with_retry(max_retries=3, delay=1):     │
  │     """重试装饰器"""                         │
  │     def decorator(func):                   │
  │         @wraps(func)                        │
  │         def wrapper(*args, **kwargs):       │
  │             for i in range(max_retries):    │
  │                 try:                        │
  │                     return func(*args, **kwargs) │
  │                 except Exception as e:      │
  │                     if i == max_retries - 1: │
  │                         raise               │
  │                     time.sleep(delay * (i + 1)) │
  │             return None                     │
  │         return wrapper                      │
  │     return decorator                        │
  │                                            │
  │ @with_retry(max_retries=3)                  │
  │ def call_external_api(query):              │
  │     """带重试的 API 调用"""                  │
  │     response = requests.get(url)           │
  │     return response.json()                 │
  └─────────────────────────────────────────────┘
    """)


# ==========================================
# 示例 8: 多数据源集成
# ==========================================


def example_multi_source_integration():
    """演示多数据源集成"""
    print("\n" + "=" * 50)
    print("示例 8: 多数据源集成")
    print("=" * 50)

    print("""
  多数据源集成架构：

  ┌─────────────────────────────────────────────┐
  │              多数据源融合                    │
  ├─────────────────────────────────────────────┤
  │                                             │
  │   数据源类型：                               │
  │                                             │
  │   ┌─────────────┐                           │
  │   │ 本地文件    │                           │
  │   │ (.txt/.pdf) │                           │
  │   └─────────────┘                           │
  │         │                                   │
  │   ┌─────────────┐                           │
  │   │ 数据库      │                           │
  │   │ (SQL/NoSQL) │                           │
  │   └─────────────┘                           │
  │         │                                   │
  │   ┌─────────────┐                           │
  │   │ API 接口    │                           │
  │   │ (REST/Web)  │                           │
  │   └─────────────┘                           │
  │         │                                   │
  │   ┌─────────────┐                           │
  │   │ 云存储      │                           │
  │   │ (S3/GCS)    │                           │
  │   └─────────────┘                           │
  │         │                                   │
  │         ▼                                   │
  │   ┌─────────────────────────────────────┐  │
  │   │         统一索引                    │  │
  │   │                                     │  │
  │   │   VectorStoreIndex                  │  │
  │   │   (所有数据源融合)                   │  │
  │   │                                     │  │
  │   └─────────────────────────────────────┘  │
  │                                             │
  └─────────────────────────────────────────────┘

  多数据源加载示例：
  ┌─────────────────────────────────────────────┐
  │ # 本地文件                                  │
  │ from llama_index.core import SimpleDirectoryReader │
  │                                            │
  │ local_docs = SimpleDirectoryReader(        │
  │     input_dir="./local_docs"               │
  │ ).load_data()                              │
  │                                            │
  │ # 数据库                                    │
  │ from llama_index.readers.database import   │
  │     DatabaseReader                          │
  │                                            │
  │ db_docs = DatabaseReader(                  │
  │     sql_database=sql_db                    │
  │ ).load_data(query="SELECT * FROM articles") │
  │                                            │
  │ # Web API                                   │
  │ from llama_index.readers.web import        │
  │     SimpleWebPageReader                     │
  │                                            │
  │ web_docs = SimpleWebPageReader().load_data( │
  │     urls=["https://example.com/article1"]  │
  │ )                                          │
  │                                            │
  │ # 合合所有文档                              │
  │ all_docs = local_docs + db_docs + web_docs │
  │                                            │
  │ # 创建统一索引                              │
  │ index = VectorStoreIndex.from_documents(   │
  │     all_docs                                │
  │ )                                          │
  └─────────────────────────────────────────────┘

  元数据标记（区分来源）：
  ┌─────────────────────────────────────────────┐
  │ # 为不同来源添加元数据                      │
  │                                            │
  │ for doc in local_docs:                     │
  │     doc.metadata["source"] = "local"       │
  │                                            │
  │ for doc in db_docs:                        │
  │     doc.metadata["source"] = "database"    │
  │                                            │
  │ for doc in web_docs:                       │
  │     doc.metadata["source"] = "web"         │
  │                                            │
  │ # 查询时可以按来源过滤                      │
  │ from llama_index.core.vector_stores import  │
  │     MetadataFilters, ExactMatchFilter      │
  │                                            │
  │ filters = MetadataFilters(                 │
  │     filters=[                              │
  │         ExactMatchFilter(                  │
  │             key="source",                  │
  │             value="web"                    │
  │         )                                  │
  │     ]                                      │
  │ )                                          │
  │                                            │
  │ query_engine = index.as_query_engine(      │
  │     filters=filters                         │
  │ )                                          │
  └─────────────────────────────────────────────┘
    """)


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 18: LlamaIndex 外部工具集成示例")
    print("=" * 60)

    # 运行各示例
    example_chroma_integration()
    example_pinecone_integration()
    example_multi_llm_backend()
    example_wikipedia_integration()
    example_web_search_integration()
    example_custom_tool_integration()
    example_api_integration_best_practices()
    example_multi_source_integration()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()