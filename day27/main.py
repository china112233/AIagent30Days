# day27/main.py
"""
Day 27: 性能优化 - 主程序

整合所有性能优化相关功能的入口程序
"""

import os
import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


async def main():
    """主函数"""
    print("=" * 70)
    print("Day 27: 性能优化")
    print("=" * 70)
    print("\n本程序演示了 Agent 性能优化的各个方面：")
    print("1. 延迟优化 - latency_optimization.py")
    print("2. 成本控制 - cost_control.py")
    print("3. 缓存策略 - caching_strategy.py")
    print("4. 并发处理 - concurrent_processing.py")
    print("\n运行各个模块的命令：")
    print("  python latency_optimization.py      # 延迟优化演示")
    print("  python cost_control.py               # 成本控制演示")
    print("  python caching_strategy.py           # 缓存策略演示")
    print("  python concurrent_processing.py      # 并发处理演示")
    print("\n性能优化关键点：")
    print("  • 流式输出降低 TTFT (首 Token 延迟)")
    print("  • KV Cache 缓存注意力计算")
    print("  • 多级缓存减少重复计算")
    print("  • 批处理提高吞吐量")
    print("  • 智能成本控制和预算管理")
    print("  • 连接池和请求队列管理")
    print("  • 多级限流防止过载")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())