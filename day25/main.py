# day25/main.py
"""
Day 25: 本地部署与隐私保护 - 主程序

整合所有部署和隐私保护相关功能的入口程序
"""

import os
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def main():
    print("=" * 70)
    print("Day 25: 本地部署与隐私保护")
    print("=" * 70)
    print("\n本程序演示了本地部署和隐私保护的各个方面：")
    print("1. 本地模型部署 - local_deployment.py")
    print("2. 隐私计算实践 - privacy_computing.py")
    print("3. 联邦学习 Agent - federated_learning_agent.py")
    print("\n运行各个模块的命令：")
    print("  python local_deployment.py           # 本地部署演示")
    print("  python privacy_computing.py          # 隐私计算演示")
    print("  python federated_learning_agent.py   # 联邦学习演示")
    print("=" * 70)


if __name__ == "__main__":
    main()