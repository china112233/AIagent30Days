# day24/main.py
"""
Day 24: 模型微调入门 - 主程序

整合所有微调相关功能的入口程序
"""

import os
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    print("=" * 70)
    print("Day 24: 模型微调入门")
    print("=" * 70)
    print("\n本程序演示了模型微调的各个方面：")
    print("1. 数据准备 - data_preparation.py")
    print("2. 基础微调 - fine_tuning_basics.py") 
    print("3. LoRA微调 - lora_finetuning.py")
    print("4. 模型评估 - evaluation.py")
    print("\n运行各个模块的命令：")
    print("  python data_preparation.py          # 数据准备演示")
    print("  python fine_tuning_basics.py        # 基础微调演示")
    print("  python lora_finetuning.py           # LoRA微调演示")
    print("  python evaluation.py                # 模型评估演示")
    print("=" * 70)

if __name__ == "__main__":
    main()