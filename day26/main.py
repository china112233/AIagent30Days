# day26/main.py
"""
Day 26: Agent 部署 - 主程序

整合所有部署相关功能的入口程序
"""

import os
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def main():
    print("=" * 70)
    print("Day 26: Agent 部署")
    print("=" * 70)
    print("\n本程序演示了 Agent 部署的各个方面：")
    print("1. API 设计 - api_design.py")
    print("2. FastAPI Agent - fastapi_agent.py")
    print("3. 容器化配置 - docker_config.py")
    print("4. 服务编排 - service_orchestration.py")
    print("\n运行各个模块的命令：")
    print("  python api_design.py              # API 设计演示")
    print("  python fastapi_agent.py           # 启动 FastAPI Agent 服务")
    print("  python docker_config.py           # 容器化配置演示")
    print("  python service_orchestration.py   # 服务编排演示")
    print("\n启动服务的命令：")
    print("  uvicorn fastapi_agent:app --host 0.0.0.0 --port 8000")
    print("\nDocker 相关命令：")
    print("  docker build -t agent-api:latest .")
    print("  docker run -p 8000:8000 agent-api:latest")
    print("  docker-compose up -d")
    print("=" * 70)


if __name__ == "__main__":
    main()