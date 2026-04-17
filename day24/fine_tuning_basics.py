# day24/fine_tuning_basics.py
"""
Day 24: 微调基础示例

本文件演示微调的基本概念和流程，包括：
1. 模型加载 - 加载预训练模型
2. 训练配置 - 设置训练参数
3. 基础微调 - 实现简单微调流程
4. 模型保存 - 保存微调后的模型

依赖安装：
pip install transformers datasets torch accelerate
"""

import os
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 训练配置
# ============================================================

@dataclass
class FineTuningConfig:
    """微调配置"""
    # 模型配置
    model_name: str = "microsoft/DialoGPT-medium"  # 可替换为其他模型
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    
    # 训练配置
    output_dir: str = "./fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    prediction_loss_only: bool = True
    
    # 数据配置
    max_seq_length: int = 512
    train_file: str = "train.jsonl"
    eval_file: str = "eval.jsonl"


# ============================================================
# 数据处理器
# ============================================================

class DataProcessor:
    """
    数据处理器
    处理和编码训练数据
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def tokenize_function(self, examples):
        """
        分词函数
        
        Args:
            examples: 数据样本
            
        Returns:
            分词后的数据
        """
        # 对文本进行编码
        encoded = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 设置标签（用于语言模型训练）
        encoded["labels"] = encoded["input_ids"].clone()
        
        return encoded
    
    def prepare_dataset(self, texts: list) -> Dataset:
        """
        准备训练数据集
        
        Args:
            texts: 文本列表
            
        Returns:
            处理后的数据集
        """
        # 创建数据集
        dataset = Dataset.from_dict({"text": texts})
        
        # 应用分词
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset


# ============================================================
# 微调器
# ============================================================

class BasicFineTuner:
    """
    基础微调器
    实现基本的微调流程
    """
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        
        # 加载模型和分词器
        print(f"加载模型: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # 如果没有pad token，使用eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )
        
        # 设置pad token id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def prepare_training(self, train_dataset, eval_dataset=None):
        """
        准备训练环境
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
        """
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 因为是因果语言模型
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            prediction_loss_only=self.config.prediction_loss_only,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            report_to=None,  # 禁用wandb等报告
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    
    def train(self):
        """开始训练"""
        print("开始训练...")
        self.trainer.train()
        
        # 保存模型
        self.save_model()
    
    def save_model(self):
        """保存模型"""
        print(f"保存模型到: {self.config.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
    
    def evaluate(self, eval_dataset):
        """评估模型"""
        if hasattr(self, 'trainer'):
            results = self.trainer.evaluate(eval_dataset)
            return results
        else:
            print("训练器未初始化")
            return None


# ============================================================
# 使用示例
# ============================================================

def demo_basic_finetuning():
    """演示基础微调流程"""
    print("=" * 60)
    print("Day 24: 基础微调演示")
    print("=" * 60)
    
    # 创建示例数据
    sample_texts = [
        "今天天气很好，阳光明媚。",
        "我喜欢阅读书籍，特别是科幻小说。",
        "人工智能正在改变我们的生活方式。",
        "Python是一门非常流行的编程语言。",
        "深度学习在图像识别方面取得了巨大进展。",
        "自然语言处理技术不断发展。",
        "大数据分析帮助企业做出更好决策。",
        "云计算提供了灵活的计算资源。",
        "区块链技术具有去中心化的特点。",
        "物联网连接了各种智能设备。"
    ] * 10  # 重复以增加数据量
    
    # 配置
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-medium",  # 可替换为其他模型
        output_dir="./fine_tuned_demo",
        num_train_epochs=1,  # 演示用，减少训练轮数
        per_device_train_batch_size=2,
        learning_rate=5e-5
    )
    
    # 初始化微调器
    finetuner = BasicFineTuner(config)
    
    # 准备数据
    print("\n1. 准备数据...")
    processor = DataProcessor(finetuner.tokenizer, max_length=128)
    train_dataset = processor.prepare_dataset(sample_texts)
    print(f"   训练样本数: {len(train_dataset)}")
    
    # 准备训练
    print("\n2. 准备训练环境...")
    finetuner.prepare_training(train_dataset)
    
    # 开始训练
    print("\n3. 开始训练...")
    try:
        finetuner.train()
        print("   训练完成!")
    except Exception as e:
        print(f"   训练过程中出现错误: {e}")
        print("   这可能是由于模型大小或硬件限制导致的")
    
    # 演示推理
    print("\n4. 模型推理演示...")
    try:
        # 简单的推理测试
        input_text = "人工智能是"
        inputs = finetuner.tokenizer.encode(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = finetuner.model.generate(
                inputs, 
                max_length=len(inputs[0]) + 20,
                num_return_sequences=1,
                pad_token_id=finetuner.tokenizer.eos_token_id
            )
        
        generated_text = finetuner.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   输入: {input_text}")
        print(f"   生成: {generated_text[len(input_text):]}")
    except Exception as e:
        print(f"   推理过程中出现错误: {e}")
    
    print("\n" + "=" * 60)
    print("基础微调演示完成!")
    print("=" * 60)
    print("\n注意: 实际微调需要更大的数据集和更强的硬件支持")
    print("本演示主要用于展示微调的基本流程")


if __name__ == "__main__":
    demo_basic_finetuning()