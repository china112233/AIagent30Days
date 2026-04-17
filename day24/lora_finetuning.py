# day24/lora_finetuning.py
"""
Day 24: LoRA 微调示例

本文件演示如何使用LoRA（Low-Rank Adaptation）进行高效微调，包括：
1. LoRA配置 - 设置LoRA参数
2. 模型包装 - 将模型转换为LoRA模型
3. 训练流程 - 执行LoRA微调
4. 模型合并 - 合并LoRA权重

依赖安装：
pip install transformers datasets peft accelerate bitsandbytes trl
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
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
from peft.utils.other import fsdp_auto_wrap_policy
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# LoRA配置
# ============================================================

@dataclass
class LoRAConfig:
    """LoRA配置"""
    # 模型配置
    model_name: str = "microsoft/DialoGPT-medium"  # 可替换为其他模型
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    
    # LoRA参数
    lora_r: int = 16          # LoRA秩
    lora_alpha: int = 32      # LoRA缩放因子
    lora_dropout: float = 0.1 # LoRA dropout概率
    target_modules: list = None  # 目标模块
    
    # 训练配置
    output_dir: str = "./lora_fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    learning_rate: float = 5e-4  # LoRA通常使用较大的学习率
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    prediction_loss_only: bool = True
    
    # 数据配置
    max_seq_length: int = 512
    train_file: str = "train.jsonl"
    eval_file: str = "eval.jsonl"
    
    def __post_init__(self):
        # 设置默认目标模块
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


# ============================================================
# LoRA微调器
# ============================================================

class LoRAFineTuner:
    """
    LoRA微调器
    实现LoRA微调流程
    """
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        
        # 加载模型和分词器
        print(f"加载模型: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # 如果没有pad token，使用eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            load_in_8bit=True,  # 8bit量化以节省显存
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto"  # 自动分配到GPU/CPU
        )
        
        # 准备模型进行k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 设置pad token id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def setup_lora(self):
        """设置LoRA配置并包装模型"""
        print("设置LoRA配置...")
        
        # 创建LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
        )
        
        # 将模型转换为Peft模型
        self.model = get_peft_model(self.model, peft_config)
        
        # 打印可训练参数信息
        self.model.print_trainable_parameters()
    
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
            # 优化内存使用
            dataloader_pin_memory=False,
            remove_unused_columns=False,
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
        print("开始LoRA训练...")
        self.trainer.train()
        
        # 保存LoRA适配器
        self.save_lora_adapters()
    
    def save_lora_adapters(self):
        """保存LoRA适配器"""
        print(f"保存LoRA适配器到: {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
    
    def merge_and_save_model(self, merged_model_path: str = None):
        """
        合并LoRA权重并保存完整模型
        
        Args:
            merged_model_path: 合并后模型的保存路径
        """
        if merged_model_path is None:
            merged_model_path = f"{self.config.output_dir}_merged"
        
        print(f"合并LoRA权重并保存到: {merged_model_path}")
        
        # 合并模型
        merged_model = self.model.merge_and_unload()
        
        # 保存合并后的模型
        merged_model.save_pretrained(merged_model_path)
        self.tokenizer.save_pretrained(merged_model_path)
    
    def load_lora_adapters(self, adapter_path: str):
        """
        加载已保存的LoRA适配器
        
        Args:
            adapter_path: 适配器路径
        """
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
    
    def evaluate(self, eval_dataset):
        """评估模型"""
        if hasattr(self, 'trainer'):
            results = self.trainer.evaluate(eval_dataset)
            return results
        else:
            print("训练器未初始化")
            return None


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
# 使用示例
# ============================================================

def demo_lora_finetuning():
    """演示LoRA微调流程"""
    print("=" * 60)
    print("Day 24: LoRA微调演示")
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
    config = LoRAConfig(
        model_name="microsoft/DialoGPT-medium",  # 可替换为其他模型
        output_dir="./lora_demo",
        num_train_epochs=1,  # 演示用，减少训练轮数
        per_device_train_batch_size=2,
        learning_rate=5e-4,  # LoRA通常使用较大的学习率
        lora_r=8,           # 较小的r值用于演示
        lora_alpha=16,      # 较小的alpha值用于演示
    )
    
    # 初始化LoRA微调器
    lora_finetuner = LoRAFineTuner(config)
    
    # 设置LoRA
    lora_finetuner.setup_lora()
    
    # 准备数据
    print("\n1. 准备数据...")
    processor = DataProcessor(lora_finetuner.tokenizer, max_length=128)
    train_dataset = processor.prepare_dataset(sample_texts)
    print(f"   训练样本数: {len(train_dataset)}")
    
    # 准备训练
    print("\n2. 准备训练环境...")
    lora_finetuner.prepare_training(train_dataset)
    
    # 开始训练
    print("\n3. 开始LoRA训练...")
    try:
        lora_finetuner.train()
        print("   LoRA训练完成!")
    except Exception as e:
        print(f"   训练过程中出现错误: {e}")
        print("   这可能是由于模型大小或硬件限制导致的")
    
    # 演示推理
    print("\n4. 模型推理演示...")
    try:
        # 设置模型为评估模式
        lora_finetuner.model.eval()
        
        # 简单的推理测试
        input_text = "人工智能是"
        inputs = lora_finetuner.tokenizer.encode(input_text, return_tensors="pt").to(lora_finetuner.model.device)
        
        with torch.no_grad():
            outputs = lora_finetuner.model.generate(
                inputs, 
                max_length=len(inputs[0]) + 20,
                num_return_sequences=1,
                pad_token_id=lora_finetuner.tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = lora_finetuner.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   输入: {input_text}")
        print(f"   生成: {generated_text[len(input_text):]}")
    except Exception as e:
        print(f"   推理过程中出现错误: {e}")
    
    print("\n5. LoRA参数统计...")
    trainable_params = sum(p.numel() for p in lora_finetuner.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_finetuner.model.parameters())
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   总参数: {total_params:,}")
    print(f"   微调比例: {trainable_params/total_params*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("LoRA微调演示完成!")
    print("=" * 60)
    print("\n注意: LoRA大大减少了需要训练的参数量，提高了训练效率")
    print("本演示主要用于展示LoRA微调的基本流程")


if __name__ == "__main__":
    demo_lora_finetuning()