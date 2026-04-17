# day24/qlora_finetuning.py
"""
Day 24: QLoRA 微调示例

本文件演示如何使用QLoRA（Quantized Low-Rank Adaptation）进行极低资源微调，包括：
1. 模型量化 - 将模型量化为4bit以节省显存
2. QLoRA配置 - 设置QLoRA参数
3. 训练优化 - 内存优化技巧
4. 性能对比 - 比较LoRA与QLoRA

QLoRA优势：
- 4bit量化大幅减少显存占用（约降低75%）
- 可以在消费级GPU上微调大模型
- 保持接近全参数微调的性能

依赖安装：
pip install transformers datasets peft accelerate bitsandbytes trl
"""

import os
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# QLoRA配置
# ============================================================

@dataclass
class QLoRAConfig:
    """QLoRA配置"""
    # 模型配置
    model_name: str = "microsoft/DialoGPT-medium"  # 可替换为更大模型
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    
    # 量化配置
    load_in_4bit: bool = True           # 4bit量化
    bnb_4bit_quant_type: str = "nf4"    # 量化类型：nf4或fp4
    bnb_4bit_compute_dtype: str = "float16"  # 计算精度
    bnb_4bit_use_double_quant: bool = True   # 双量化进一步节省显存
    
    # LoRA参数
    lora_r: int = 16                    # LoRA秩
    lora_alpha: int = 32                # LoRA缩放因子  
    lora_dropout: float = 0.1           # LoRA dropout概率
    target_modules: list = None         # 目标模块
    
    # 训练配置
    output_dir: str = "./qlora_fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # QLoRA建议更大的累积步数
    warmup_steps: int = 100
    weight_decay: float = 0.01
    learning_rate: float = 2e-4         # QLoRA通常使用中等学习率
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    
    # 内存优化配置
    gradient_checkpointing: bool = True     # 梯度检查点节省显存
    optim: str = "paged_adamw_8bit"         # 分页优化器防止OOM
    max_grad_norm: float = 0.3              # 梯度裁剪
    
    # 数据配置
    max_seq_length: int = 512
    
    def __post_init__(self):
        # 设置默认目标模块
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ============================================================
# QLoRA微调器
# ============================================================

class QLoRAFineTuner:
    """
    QLoRA微调器
    实现4bit量化 + LoRA的高效微调流程
    """
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        
        print(f"加载模型: {config.model_name}")
        print(f"量化模式: 4bit ({config.bnb_4bit_quant_type})")
        
        # 配置4bit量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16 if config.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        
        # 加载量化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 准备k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 启用梯度检查点（节省显存）
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 记录显存使用
        self._log_memory_usage("模型加载后")
    
    def _log_memory_usage(self, stage: str):
        """记录显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[{stage}] 显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
    
    def setup_lora(self):
        """设置LoRA配置"""
        print("设置QLoRA配置...")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        # 包装模型
        self.model = get_peft_model(self.model, peft_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
        
        self._log_memory_usage("LoRA设置后")
    
    def prepare_dataset(self, texts: List[str]) -> Dataset:
        """准备训练数据集"""
        def tokenize(examples):
            encoded = self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_seq_length,
                return_tensors=None,
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )
        return tokenized_dataset
    
    def prepare_training(self, train_dataset, eval_dataset=None):
        """准备训练环境"""
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 训练参数（针对QLoRA优化）
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
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            report_to=None,
            # QLoRA内存优化设置
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=True,  # 混合精度训练
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    
    def train(self):
        """开始训练"""
        print("开始QLoRA训练...")
        self._log_memory_usage("训练开始前")
        
        self.trainer.train()
        
        self._log_memory_usage("训练完成后")
        self.save_model()
    
    def save_model(self):
        """保存QLoRA适配器"""
        print(f"保存QLoRA适配器到: {self.config.output_dir}")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """生成文本"""
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()


# ============================================================
# LoRA vs QLoRA 对比
# ============================================================

def compare_lora_vs_qlora():
    """对比LoRA和QLoRA的资源消耗"""
    print("=" * 60)
    print("LoRA vs QLoRA 资源对比")
    print("=" * 60)
    
    comparison_table = """
┌─────────────────────────────────────────────────────────────────┐
│                LoRA vs QLoRA 对比                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────┐              ┌───────────────┐             │
│   │    LoRA       │              │    QLoRA      │             │
│   ├───────────────┤              ├───────────────┤             │
│   │               │              │               │             │
│   │ 模型精度: 16bit│              │ 模型精度: 4bit │             │
│   │               │              │               │             │
│   │ 显存占用: 较高 │              │ 显存占用: 极低 │             │
│   │ (~16GB/7B模型) │              │ (~4GB/7B模型) │             │
│   │               │              │               │             │
│   │ 训练速度: 较快 │              │ 训练速度: 稍慢 │             │
│   │               │              │               │             │
│   │ 性能表现: 良好 │              │ 性能表现: 接近 │             │
│   │               │              │               │             │
│   │ 适用GPU: 24GB+ │              │ 适用GPU: 8GB+ │             │
│   │               │              │               │             │
│   └───────────────┘              └───────────────┘             │
│                                                                 │
│   显存节省比例: ~75%                                             │
│   性能损失: <5%（通常可接受）                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(comparison_table)
    
    # 具体参数对比
    params_comparison = """
┌─────────────────────────────────────────────────────────────────┐
│                参数设置对比                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   参数              │    LoRA       │    QLoRA                  │
│   ─────────────────────────────────────────────────────────────│
│   模型加载          │   fp16/bf16   │   4bit量化                │
│   学习率            │   5e-4        │   2e-4                    │
│   batch_size        │   4-8         │   1-4                     │
│   gradient_accum    │   2-4         │   4-16                    │
│   optimizer         │   adamw       │   paged_adamw_8bit        │
│   gradient_ckpt     │   可选        │   强烈建议                 │
│   max_grad_norm     │   1.0         │   0.3                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(params_comparison)


# ============================================================
# 内存优化技巧
# ============================================================

def show_memory_optimization_tips():
    """显示QLoRA内存优化技巧"""
    tips = """
┌─────────────────────────────────────────────────────────────────┐
│                QLoRA 内存优化技巧                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 模型量化                                                    │
│      • 使用4bit量化（nf4类型）                                   │
│      • 启用双量化（double quantization）                         │
│      • 计算 dtype 使用 float16                                  │
│                                                                 │
│   2. 梯度优化                                                    │
│      • 启用梯度检查点（gradient checkpointing）                  │
│      • 增大梯度累积步数（gradient accumulation）                 │
│      • 减小 batch size                                          │
│                                                                 │
│   3. 优化器选择                                                  │
│      • 使用 paged_adamw_8bit（分页优化器）                       │
│      • 防止梯度累积导致OOM                                       │
│                                                                 │
│   4. 其他技巧                                                    │
│      • 减小 max_seq_length                                      │
│      • 使用 Flash Attention（如果支持）                          │
│      • 清理不需要的缓存：torch.cuda.empty_cache()               │
│                                                                 │
│   示例配置：                                                     │
│   ┌───────────────────────────────────────────────────────────┐│
│   │ BitsAndBytesConfig(                                       ││
│   │     load_in_4bit=True,                                    ││
│   │     bnb_4bit_quant_type="nf4",                            ││
│   │     bnb_4bit_compute_dtype=torch.float16,                 ││
│   │     bnb_4bit_use_double_quant=True,                       ││
│   │ )                                                         ││
│   └───────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(tips)


# ============================================================
# 使用示例
# ============================================================

def demo_qlora_finetuning():
    """演示QLoRA微调流程"""
    print("=" * 60)
    print("Day 24: QLoRA微调演示")
    print("=" * 60)
    
    # 显示对比信息
    compare_lora_vs_qlora()
    
    # 显示优化技巧
    print("\n")
    show_memory_optimization_tips()
    
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
    ] * 10
    
    # QLoRA配置
    config = QLoRAConfig(
        model_name="microsoft/DialoGPT-medium",
        output_dir="./qlora_demo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lora_r=8,
        lora_alpha=16,
        max_seq_length=256,
    )
    
    print("\n" + "=" * 60)
    print("开始QLoRA微调实践演示")
    print("=" * 60)
    
    try:
        # 初始化QLoRA微调器
        qlora_finetuner = QLoRAFineTuner(config)
        
        # 设置LoRA
        qlora_finetuner.setup_lora()
        
        # 准备数据
        print("\n1. 准备数据...")
        train_dataset = qlora_finetuner.prepare_dataset(sample_texts)
        print(f"   训练样本数: {len(train_dataset)}")
        
        # 准备训练
        print("\n2. 准备训练环境...")
        qlora_finetuner.prepare_training(train_dataset)
        
        # 开始训练
        print("\n3. 开始QLoRA训练...")
        qlora_finetuner.train()
        print("   QLoRA训练完成!")
        
        # 推理测试
        print("\n4. 模型推理测试...")
        prompt = "人工智能是"
        generated = qlora_finetuner.generate(prompt, max_new_tokens=30)
        print(f"   输入: {prompt}")
        print(f"   生成: {generated}")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("这可能是由于硬件限制（QLoRA需要bitsandbytes支持）")
        print("\nQLoRA的主要价值:")
        print("  - 在消费级GPU（8GB显存）上微调7B参数模型")
        print("  - 显存占用仅为LoRA的约25%")
        print("  - 性能接近全参数微调")
    
    print("\n" + "=" * 60)
    print("QLoRA微调演示完成!")
    print("=" * 60)
    print("\nQLoRA让大模型微调不再需要昂贵硬件！")


# ============================================================
# 常见问题解决
# ============================================================

def show_common_issues():
    """显示QLoRA常见问题及解决方案"""
    issues = """
┌─────────────────────────────────────────────────────────────────┐
│                QLoRA 常见问题与解决方案                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   问题1: bitsandbytes安装失败                                   │
│   解决:                                                         │
│   • Windows: pip install bitsandbytes-windows                   │
│   • Linux: pip install bitsandbytes                             │
│   • 需要CUDA 11.8+                                              │
│                                                                 │
│   问题2: OOM（显存不足）                                        │
│   解决:                                                         │
│   • 减小 batch_size 和 max_seq_length                           │
│   • 增大 gradient_accumulation_steps                            │
│   • 确保 gradient_checkpointing=True                            │
│   • 使用 paged_adamw_8bit 优化器                                │
│                                                                 │
│   问题3: 训练速度慢                                              │
│   解决:                                                         │
│   • 这是正常的，量化会带来一定开销                               │
│   • 但显存节省是主要优势                                        │
│                                                                 │
│   问题4: 性能不如预期                                            │
│   解决:                                                         │
│   • 调整 lora_r（推荐16-64）                                    │
│   • 尝试不同的 learning_rate                                    │
│   • 确保数据质量                                                │
│                                                                 │
│   问题5: Windows兼容性问题                                      │
│   解决:                                                         │
│   • QLoRA在Linux上兼容性更好                                    │
│   • Windows可能需要特殊配置                                      │
│   • 或考虑使用云GPU（Colab等）                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    print(issues)


if __name__ == "__main__":
    demo_qlora_finetuning()
    print("\n")
    show_common_issues()