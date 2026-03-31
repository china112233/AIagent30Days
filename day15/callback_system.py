"""
Day 15: 回调系统示例

本文件演示 LangChain 的回调系统：
- 自定义回调处理器
- Token 流追踪
- 执行日志记录
- 错误监控
- 追踪集成基础
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import (
    BaseCallbackHandler,
    AsyncCallbackHandler,
    CallbackManager,
)
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.documents import Document


# 加载环境变量
load_dotenv()


def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.7,
        streaming=True,  # 启用流式以支持 token 回调
    )


# ==========================================
# 示例 1: 基础回调处理器
# ==========================================


class BasicCallbackHandler(BaseCallbackHandler):
    """基础回调处理器 - 打印所有回调事件"""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """LLM 开始调用"""
        print("\n[LLM 开始]")
        print(f"  提示词数量: {len(prompts)}")
        print(f"  时间: {datetime.now().strftime('%H:%M:%S')}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 调用结束"""
        print("\n[LLM 结束]")
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            print(f"  Token 使用:")
            print(f"    - 提示词: {token_usage.get('prompt_tokens', 'N/A')}")
            print(f"    - 完成: {token_usage.get('completion_tokens', 'N/A')}")
            print(f"    - 总计: {token_usage.get('total_tokens', 'N/A')}")
        print(f"  时间: {datetime.now().strftime('%H:%M:%S')}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 调用出错"""
        print("\n[LLM 错误]")
        print(f"  错误: {str(error)}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """新 Token 生成"""
        print(token, end="", flush=True)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Chain 开始执行"""
        print("\n[Chain 开始]")
        print(f"  输入: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chain 执行结束"""
        print("\n[Chain 结束]")
        print(f"  输出: {outputs}")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Chain 执行出错"""
        print("\n[Chain 错误]")
        print(f"  错误: {str(error)}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Tool 开始执行"""
        print("\n[Tool 开始]")
        print(f"  工具名: {serialized.get('name', 'unknown')}")
        print(f"  输入: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Tool 执行结束"""
        print("\n[Tool 结束]")
        print(f"  输出: {output}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Tool 执行出错"""
        print("\n[Tool 错误]")
        print(f"  错误: {str(error)}")

    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, **kwargs: Any
    ) -> None:
        """Retriever 开始检索"""
        print("\n[Retriever 开始]")
        print(f"  查询: {query}")

    def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> None:
        """Retriever 检索结束"""
        print("\n[Retriever 结束]")
        print(f"  文档数量: {len(documents)}")


def example_basic_callback():
    """演示基础回调处理器"""
    print("\n" + "=" * 50)
    print("示例 1: 基础回调处理器")
    print("=" * 50)

    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请简单介绍：{topic}")

    chain = prompt | llm | StrOutputParser()

    # 使用回调执行
    print("\n执行链（带回调）...")
    result = chain.invoke(
        {"topic": "LangChain"},
        config={"callbacks": [BasicCallbackHandler()]},
    )

    print("\n" + "-" * 30)
    print(f"最终结果: {result[:100]}...")


# ==========================================
# 示例 2: Token 追踪回调
# ==========================================


class TokenTrackingHandler(BaseCallbackHandler):
    """Token 追踪回调处理器"""

    def __init__(self):
        self.tokens_received = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """记录提示词 Token"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tokens_received = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """统计生成的 Token"""
        self.tokens_received += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """记录 Token 使用统计"""
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            self.prompt_tokens = token_usage.get("prompt_tokens", 0)
            self.completion_tokens = token_usage.get("completion_tokens", 0)
            self.total_tokens = token_usage.get("total_tokens", 0)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "streaming_tokens": self.tokens_received,
        }


def example_token_tracking():
    """演示 Token 追踪"""
    print("\n" + "=" * 50)
    print("示例 2: Token 追踪")
    print("=" * 50)

    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请详细介绍：{topic}")

    chain = prompt | llm | StrOutputParser()

    handler = TokenTrackingHandler()

    print("\n执行链...")
    result = chain.invoke(
        {"topic": "Python 编程语言"},
        config={"callbacks": [handler]},
    )

    stats = handler.get_stats()
    print("\nToken 使用统计:")
    print(f"  提示词 Token: {stats['prompt_tokens']}")
    print(f"  完成 Token: {stats['completion_tokens']}")
    print(f"  总 Token: {stats['total_tokens']}")
    print(f"  流式 Token (近似): {stats['streaming_tokens']}")

    # 成本估算（假设价格）
    prompt_cost = stats["prompt_tokens"] * 0.001 / 1000  # $0.001/1K tokens
    completion_cost = stats["completion_tokens"] * 0.002 / 1000  # $0.002/1K tokens
    total_cost = prompt_cost + completion_cost

    print(f"\n估算成本: ${total_cost:.4f}")


# ==========================================
# 示例 3: 执行日志回调
# ==========================================


class ExecutionLoggerHandler(BaseCallbackHandler):
    """执行日志回调处理器"""

    def __init__(self, log_file: str = "execution_log.txt"):
        self.log_file = log_file
        self.start_time = None
        self.logs = []

    def _log(self, event: str, data: Dict[str, Any]):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        log_entry = f"[{timestamp}] {event}: {data}"
        self.logs.append(log_entry)
        print(f"  {log_entry}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.start_time = time.time()
        self._log("LLM_START", {"prompts_count": len(prompts)})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._log("LLM_END", {"elapsed_seconds": elapsed})

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self._log("CHAIN_START", {"inputs": inputs})

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self._log("CHAIN_END", {"outputs": str(outputs)[:100]})

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        self._log("CHAIN_ERROR", {"error": str(error)})

    def save_logs(self):
        """保存日志到文件"""
        with open(self.log_file, "w") as f:
            for log in self.logs:
                f.write(log + "\n")
        print(f"\n日志已保存到: {self.log_file}")


def example_execution_logger():
    """演示执行日志"""
    print("\n" + "=" * 50)
    print("示例 3: 执行日志")
    print("=" * 50)

    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请回答：{question}")

    chain = prompt | llm | StrOutputParser()

    handler = ExecutionLoggerHandler()

    print("\n执行链（记录日志）...")
    result = chain.invoke(
        {"question": "什么是 LCEL？"},
        config={"callbacks": [handler]},
    )

    print("\n执行日志记录:")
    handler.save_logs()


# ==========================================
# 示例 4: 错误监控回调
# ==========================================


class ErrorMonitoringHandler(BaseCallbackHandler):
    """错误监控回调处理器"""

    def __init__(self):
        self.errors = []
        self.error_count = 0

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """记录 LLM 错误"""
        self._record_error("LLM", error)

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """记录 Chain 错误"""
        self._record_error("CHAIN", error)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """记录 Tool 错误"""
        self._record_error("TOOL", error)

    def on_retriever_error(self, error: Exception, **kwargs: Any) -> None:
        """记录 Retriever 错误"""
        self._record_error("RETRIEVER", error)

    def _record_error(self, component: str, error: Exception):
        """记录错误"""
        self.error_count += 1
        error_info = {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        print(f"\n[错误 #{self.error_count}] {component}: {error}")

    def get_error_report(self) -> Dict[str, Any]:
        """获取错误报告"""
        return {
            "total_errors": self.error_count,
            "errors": self.errors,
        }


def example_error_monitoring():
    """演示错误监控"""
    print("\n" + "=" * 50)
    print("示例 4: 错误监控")
    print("=" * 50)

    handler = ErrorMonitoringHandler()

    # 正常执行
    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请回答：{question}")
    chain = prompt | llm | StrOutputParser()

    print("\n正常执行...")
    try:
        chain.invoke(
            {"question": "你好"},
            config={"callbacks": [handler]},
        )
    except Exception as e:
        print(f"执行失败: {e}")

    # 模拟错误场景
    print("\n模拟错误执行（无效输入）...")
    try:
        # 这个可能会成功，取决于模型如何处理空输入
        chain.invoke(
            {},  # 缺少 question 字段
            config={"callbacks": [handler]},
        )
    except Exception as e:
        print(f"执行失败（预期）: {e}")

    # 获取错误报告
    report = handler.get_error_report()
    print("\n错误报告:")
    print(f"  总错误数: {report['total_errors']}")
    for err in report["errors"]:
        print(f"  - {err['component']}: {err['error_message']}")


# ==========================================
# 示例 5: 性能监控回调
# ==========================================


class PerformanceMonitorHandler(BaseCallbackHandler):
    """性能监控回调处理器"""

    def __init__(self):
        self.metrics = {}
        self.current_start = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.current_start = time.time()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if self.current_start:
            elapsed = time.time() - self.current_start
            self.metrics["llm_latency"] = elapsed

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self.metrics["chain_start_time"] = time.time()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        if "chain_start_time" in self.metrics:
            total_time = time.time() - self.metrics["chain_start_time"]
            self.metrics["total_chain_time"] = total_time

    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            k: v for k, v in self.metrics.items()
            if k not in ["chain_start_time"]
        }


def example_performance_monitor():
    """演示性能监控"""
    print("\n" + "=" * 50)
    print("示例 5: 性能监控")
    print("=" * 50)

    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请回答：{question}")

    chain = prompt | llm | StrOutputParser()

    handler = PerformanceMonitorHandler()

    print("\n执行链...")
    result = chain.invoke(
        {"question": "请简单介绍 Python 语言"},
        config={"callbacks": [handler]},
    )

    metrics = handler.get_metrics()
    print("\n性能指标:")
    print(f"  LLM 延迟: {metrics.get('llm_latency', 0):.2f} 秒")
    print(f"  总 Chain 时间: {metrics.get('total_chain_time', 0):.2f} 秒")


# ==========================================
# 示例 6: 多回调组合
# ==========================================


def example_multiple_callbacks():
    """演示同时使用多个回调处理器"""
    print("\n" + "=" * 50)
    print("示例 6: 多回调组合")
    print("=" * 50)

    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请回答：{question}")

    chain = prompt | llm | StrOutputParser()

    # 创建多个回调处理器
    basic_handler = BasicCallbackHandler()
    token_handler = TokenTrackingHandler()
    perf_handler = PerformanceMonitorHandler()

    # 同时使用所有回调
    print("\n执行链（多回调）...")
    result = chain.invoke(
        {"question": "什么是 LangChain？"},
        config={"callbacks": [basic_handler, token_handler, perf_handler]},
    )

    print("\n" + "-" * 30)
    print(f"Token 统计: {token_handler.get_stats()}")
    print(f"性能指标: {perf_handler.get_metrics()}")


# ==========================================
# 示例 7: 异步回调处理器
# ==========================================


class AsyncCallbackHandlerExample(AsyncCallbackHandler):
    """异步回调处理器示例"""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("\n[异步 LLM 开始]")

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print("\n[异步 LLM 结束]")

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)


async def example_async_callback():
    """演示异步回调"""
    print("\n" + "=" * 50)
    print("示例 7: 异步回调")
    print("=" * 50)

    llm = create_llm()
    prompt = ChatPromptTemplate.from_template("请回答：{question}")

    chain = prompt | llm | StrOutputParser()

    handler = AsyncCallbackHandlerExample()

    print("\n异步执行链...")
    result = await chain.ainvoke(
        {"question": "你好"},
        config={"callbacks": [handler]},
    )

    print(f"\n结果: {result}")


# ==========================================
# 主程序
# ==========================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Day 15: 回调系统示例")
    print("=" * 60)

    example_basic_callback()
    example_token_tracking()
    example_execution_logger()
    example_error_monitoring()
    example_performance_monitor()
    example_multiple_callbacks()

    # 异步示例需要特殊处理
    print("\n运行异步示例...")
    import asyncio
    asyncio.run(example_async_callback())

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()