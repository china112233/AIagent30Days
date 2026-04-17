# day24/data_preparation.py
"""
Day 24: 数据准备示例

本文件演示如何准备微调所需的数据，包括：
1. 数据清洗 - 清理和标准化数据
2. 格式转换 - 转换为模型输入格式
3. 数据增强 - 扩充训练数据
4. 验证集划分 - 创建验证集

依赖安装：
pip install datasets pandas numpy
"""

import os
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class TrainingSample:
    """训练样本数据结构"""
    input_text: str
    output_text: str
    task_type: str = "general"
    difficulty: str = "medium"


# ============================================================
# 数据准备器
# ============================================================

class DataPreparator:
    """
    数据准备器
    负责数据清洗、格式化和预处理
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def load_raw_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载原始数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            原始数据列表
        """
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        return data
    
    def clean_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗数据
        
        Args:
            raw_data: 原始数据
            
        Returns:
            清洗后的数据
        """
        cleaned_data = []
        
        for item in raw_data:
            # 检查必要字段
            if not self._validate_item(item):
                continue
            
            # 清理文本
            cleaned_item = self._clean_item(item)
            
            # 检查长度
            if self._check_length(cleaned_item):
                cleaned_data.append(cleaned_item)
        
        return cleaned_data
    
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项"""
        # 检查必要字段是否存在
        required_fields = ['input', 'output']
        for field in required_fields:
            if field not in item or not item[field]:
                return False
        
        return True
    
    def _clean_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """清理单个数据项"""
        cleaned_item = item.copy()
        
        # 清理文本（去除多余空白字符）
        if 'input' in cleaned_item:
            cleaned_item['input'] = ' '.join(cleaned_item['input'].split())
        if 'output' in cleaned_item:
            cleaned_item['output'] = ' '.join(cleaned_item['output'].split())
        
        # 标准化字段名
        if 'prompt' in cleaned_item and 'input' not in cleaned_item:
            cleaned_item['input'] = cleaned_item.pop('prompt')
        if 'response' in cleaned_item and 'output' not in cleaned_item:
            cleaned_item['output'] = cleaned_item.pop('response')
        
        return cleaned_item
    
    def _check_length(self, item: Dict[str, Any], max_length: int = 2048) -> bool:
        """检查数据长度"""
        total_text = item.get('input', '') + ' ' + item.get('output', '')
        return len(total_text) <= max_length
    
    def format_for_training(self, cleaned_data: List[Dict[str, Any]], 
                           template_type: str = "alpaca") -> List[Dict[str, Any]]:
        """
        格式化为训练格式
        
        Args:
            cleaned_data: 清洗后的数据
            template_type: 模板类型
            
        Returns:
            格式化后的训练数据
        """
        formatted_data = []
        
        for item in cleaned_data:
            if template_type == "alpaca":
                formatted_item = self._format_alpaca(item)
            elif template_type == "chatml":
                formatted_item = self._format_chatml(item)
            elif template_type == "custom":
                formatted_item = self._format_custom(item)
            else:
                formatted_item = item  # 保持原始格式
            
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def _format_alpaca(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Alpaca格式"""
        alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        
        formatted_text = alpaca_template.format(
            instruction=item.get('input', ''),
            input=item.get('input_extra', ''),
            output=item.get('output', '')
        )
        
        return {
            'text': formatted_text,
            'input': item.get('input', ''),
            'output': item.get('output', ''),
            'template_type': 'alpaca'
        }
    
    def _format_chatml(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """ChatML格式"""
        chatml_template = """<|system|>
You are a helpful assistant.<|endoftext|>
<|user|>
{input}<|endoftext|>
<|assistant|>
{output}<|endoftext|>"""
        
        formatted_text = chatml_template.format(
            input=item.get('input', ''),
            output=item.get('output', '')
        )
        
        return {
            'text': formatted_text,
            'input': item.get('input', ''),
            'output': item.get('output', ''),
            'template_type': 'chatml'
        }
    
    def _format_custom(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """自定义格式"""
        custom_template = """[INST] {input} [/INST] {output}"""
        
        formatted_text = custom_template.format(
            input=item.get('input', ''),
            output=item.get('output', '')
        )
        
        return {
            'text': formatted_text,
            'input': item.get('input', ''),
            'output': item.get('output', ''),
            'template_type': 'custom'
        }
    
    def split_dataset(self, formatted_data: List[Dict[str, Any]], 
                     train_ratio: float = 0.8, val_ratio: float = 0.1) -> DatasetDict:
        """
        划分数据集
        
        Args:
            formatted_data: 格式化后的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            划分后的数据集
        """
        total_samples = len(formatted_data)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # 随机打乱数据
        shuffled_data = formatted_data.copy()
        random.shuffle(shuffled_data)
        
        # 划分数据集
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        # 创建Dataset对象
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        return dataset_dict
    
    def augment_data(self, data: List[Dict[str, Any]], 
                    augmentation_factor: int = 1) -> List[Dict[str, Any]]:
        """
        数据增强
        
        Args:
            data: 原始数据
            augmentation_factor: 增强倍数
            
        Returns:
            增强后的数据
        """
        if augmentation_factor <= 1:
            return data
        
        augmented_data = data.copy()
        
        for _ in range(augmentation_factor - 1):
            for item in data:
                # 简单的数据增强：随机改变输入输出顺序
                augmented_item = item.copy()
                
                # 可以添加更多增强策略
                # 例如：同义词替换、句子重组等
                
                augmented_data.append(augmented_item)
        
        return augmented_data
    
    def validate_quality(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            dataset_dict: 数据集
            
        Returns:
            质量报告
        """
        report = {}
        
        for split_name, dataset in dataset_dict.items():
            # 基本统计
            total_samples = len(dataset)
            
            # 长度统计
            lengths = [len(item.get('text', '')) for item in dataset]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            max_length = max(lengths) if lengths else 0
            min_length = min(lengths) if lengths else 0
            
            # 重复项检测
            texts = [item.get('text', '') for item in dataset]
            unique_texts = set(texts)
            duplicate_count = len(texts) - len(unique_texts)
            
            report[split_name] = {
                'total_samples': total_samples,
                'avg_length': avg_length,
                'max_length': max_length,
                'min_length': min_length,
                'duplicate_count': duplicate_count,
                'unique_ratio': len(unique_texts) / len(texts) if texts else 0
            }
        
        return report


# ============================================================
# 使用示例
# ============================================================

def demo_data_preparation():
    """演示数据准备流程"""
    print("=" * 60)
    print("Day 24: 数据准备演示")
    print("=" * 60)
    
    # 创建示例数据
    sample_data = [
        {"input": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。"},
        {"input": "Python中如何定义函数？", "output": "在Python中，使用def关键字定义函数，例如：def function_name(parameters): ..."},
        {"input": "解释神经网络的工作原理", "output": "神经网络由多个层组成，每层包含多个神经元，通过权重和激活函数处理输入数据。"},
        {"input": "如何优化SQL查询性能？", "output": "可以通过创建索引、优化查询语句、避免SELECT *等方式优化SQL查询性能。"},
        {"input": "什么是RESTful API？", "output": "RESTful API是一种基于HTTP协议的API设计风格，使用标准HTTP方法进行资源操作。"}
    ]
    
    # 初始化数据准备器
    preparator = DataPreparator()
    
    print("\n1. 原始数据:")
    for i, item in enumerate(sample_data, 1):
        print(f"  {i}. 输入: {item['input'][:50]}...")
        print(f"     输出: {item['output'][:50]}...")
    
    # 清洗数据
    print("\n2. 清洗数据...")
    cleaned_data = preparator.clean_data(sample_data)
    print(f"   清洗后数据量: {len(cleaned_data)}")
    
    # 格式化数据
    print("\n3. 格式化数据 (Alpaca格式)...")
    formatted_data = preparator.format_for_training(cleaned_data, "alpaca")
    print(f"   格式化后数据量: {len(formatted_data)}")
    
    # 显示格式化后的示例
    print(f"\n   格式化示例:")
    print(f"   {formatted_data[0]['text'][:100]}...")
    
    # 划分数据集
    print("\n4. 划分数据集...")
    dataset_dict = preparator.split_dataset(formatted_data)
    print(f"   训练集: {len(dataset_dict['train'])}")
    print(f"   验证集: {len(dataset_dict['validation'])}")
    print(f"   测试集: {len(dataset_dict['test'])}")
    
    # 验证数据质量
    print("\n5. 数据质量验证...")
    quality_report = preparator.validate_quality(dataset_dict)
    for split_name, metrics in quality_report.items():
        print(f"   {split_name}:")
        print(f"     样本数: {metrics['total_samples']}")
        print(f"     平均长度: {metrics['avg_length']:.2f}")
        print(f"     重复项: {metrics['duplicate_count']}")
        print(f"     唯一性: {metrics['unique_ratio']:.2f}")
    
    print("\n" + "=" * 60)
    print("数据准备演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_data_preparation()