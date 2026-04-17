# day24/evaluation.py
"""
Day 24: 模型评估示例

本文件演示如何评估微调效果，包括：
1. 指标计算 - 计算准确率、F1等指标
2. 对比分析 - 比较微调前后性能
3. 可视化 - 绘制训练曲线
4. 推理测试 - 测试实际效果

依赖安装：
pip install transformers datasets torch scikit-learn matplotlib seaborn
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 评估指标计算器
# ============================================================

class MetricsCalculator:
    """
    指标计算器
    计算各种评估指标
    """
    
    @staticmethod
    def calculate_accuracy(y_true: List, y_pred: List) -> float:
        """计算准确率"""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_precision_recall_f1(y_true: List, y_pred: List, average: str = 'weighted') -> Tuple[float, float, float]:
        """计算精确率、召回率、F1分数"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
        return precision, recall, f1
    
    @staticmethod
    def calculate_perplexity(model, tokenizer, texts: List[str], max_length: int = 512) -> float:
        """计算困惑度"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = inputs["input_ids"].to(model.device)
                attention_mask = inputs["attention_mask"].to(model.device)
                
                # 获取模型输出
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # 累积损失和token数量
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        # 计算平均损失
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    @staticmethod
    def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
        """计算BLEU分数（简化版）"""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            scores = []
            for pred, ref in zip(predictions, references):
                score = sentence_bleu([ref.split()], pred.split())
                scores.append(score)
            return np.mean(scores)
        except ImportError:
            print("NLTK未安装，跳过BLEU计算")
            return 0.0
    
    @staticmethod
    def calculate_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数（简化版）"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                scores['rouge1'].append(score['rouge1'].fmeasure)
                scores['rouge2'].append(score['rouge2'].fmeasure)
                scores['rougeL'].append(score['rougeL'].fmeasure)
            
            avg_scores = {
                'rouge1': np.mean(scores['rouge1']),
                'rouge2': np.mean(scores['rouge2']),
                'rougeL': np.mean(scores['rougeL'])
            }
            return avg_scores
        except ImportError:
            print("rouge_score未安装，跳过ROUGE计算")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


# ============================================================
# 模型评估器
# ============================================================

class ModelEvaluator:
    """
    模型评估器
    评估微调前后模型性能
    """
    
    def __init__(self, base_model_name: str, fine_tuned_model_path: str = None):
        self.base_model_name = base_model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        
        # 加载基础模型
        print(f"加载基础模型: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # 如果有微调模型，也加载它
        self.fine_tuned_model = None
        self.fine_tuned_tokenizer = None
        if fine_tuned_model_path and os.path.exists(fine_tuned_model_path):
            print(f"加载微调模型: {fine_tuned_model_path}")
            self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
            self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
    
    def generate_responses(self, model, tokenizer, prompts: List[str], max_length: int = 100) -> List[str]:
        """生成模型响应"""
        responses = []
        
        for prompt in prompts:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + max_length, 512),
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取生成的部分（去掉输入的prompt）
            generated_part = response[len(prompt):].strip()
            responses.append(generated_part)
        
        return responses
    
    def evaluate_classification_task(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估分类任务"""
        if not test_data:
            return {}
        
        # 提取输入和标签
        inputs = [item['input'] for item in test_data]
        true_labels = [item['label'] for item in test_data]
        
        # 生成预测（这里简化为使用规则匹配）
        predicted_labels = []
        for inp in inputs:
            # 这里应该使用模型进行实际预测，为了演示使用简单规则
            if "积极" in inp or "好" in inp or "棒" in inp:
                predicted_labels.append("positive")
            elif "消极" in inp or "坏" in inp or "差" in inp:
                predicted_labels.append("negative")
            else:
                predicted_labels.append("neutral")
        
        # 计算指标
        accuracy = MetricsCalculator.calculate_accuracy(true_labels, predicted_labels)
        precision, recall, f1 = MetricsCalculator.calculate_precision_recall_f1(true_labels, predicted_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }
    
    def evaluate_generation_task(self, test_prompts: List[str], expected_outputs: List[str]) -> Dict[str, Any]:
        """评估生成任务"""
        if not test_prompts or not expected_outputs:
            return {}
        
        # 生成基础模型响应
        base_responses = self.generate_responses(
            self.base_model, self.base_tokenizer, test_prompts
        )
        
        # 生成微调模型响应（如果有）
        fine_tuned_responses = []
        if self.fine_tuned_model:
            fine_tuned_responses = self.generate_responses(
                self.fine_tuned_model, self.fine_tuned_tokenizer, test_prompts
            )
        
        # 计算各种指标
        results = {
            'prompts': test_prompts,
            'expected_outputs': expected_outputs,
            'base_responses': base_responses,
            'fine_tuned_responses': fine_tuned_responses if fine_tuned_responses else None
        }
        
        # 计算BLEU分数
        base_bleu = MetricsCalculator.calculate_bleu_score(base_responses, expected_outputs)
        results['base_bleu'] = base_bleu
        
        if fine_tuned_responses:
            fine_tuned_bleu = MetricsCalculator.calculate_bleu_score(fine_tuned_responses, expected_outputs)
            results['fine_tuned_bleu'] = fine_tuned_bleu
        
        # 计算ROUGE分数
        base_rouge = MetricsCalculator.calculate_rouge_score(base_responses, expected_outputs)
        results['base_rouge'] = base_rouge
        
        if fine_tuned_responses:
            fine_tuned_rouge = MetricsCalculator.calculate_rouge_score(fine_tuned_responses, expected_outputs)
            results['fine_tuned_rouge'] = fine_tuned_rouge
        
        return results
    
    def compare_models(self, test_texts: List[str]) -> Dict[str, Any]:
        """比较基础模型和微调模型"""
        if not self.fine_tuned_model:
            print("没有微调模型可供比较")
            return {}
        
        # 计算困惑度
        base_perplexity = MetricsCalculator.calculate_perplexity(
            self.base_model, self.base_tokenizer, test_texts
        )
        
        fine_tuned_perplexity = MetricsCalculator.calculate_perplexity(
            self.fine_tuned_model, self.fine_tuned_tokenizer, test_texts
        )
        
        improvement = ((base_perplexity - fine_tuned_perplexity) / base_perplexity) * 100
        
        return {
            'base_perplexity': base_perplexity,
            'fine_tuned_perplexity': fine_tuned_perplexity,
            'improvement_percentage': improvement
        }


# ============================================================
# 可视化工具
# ============================================================

class EvaluationVisualizer:
    """
    评估结果可视化工具
    """
    
    @staticmethod
    def plot_metrics_comparison(metrics_before: Dict, metrics_after: Dict, title: str = "模型性能对比"):
        """绘制指标对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # 准确率对比
        ax1 = axes[0, 0]
        models = ['基础模型', '微调模型']
        accuracies = [metrics_before.get('accuracy', 0), metrics_after.get('accuracy', 0)]
        ax1.bar(models, accuracies, color=['skyblue', 'lightcoral'])
        ax1.set_title('准确率对比')
        ax1.set_ylabel('准确率')
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # F1分数对比
        ax2 = axes[0, 1]
        f1_scores = [metrics_before.get('f1', 0), metrics_after.get('f1', 0)]
        ax2.bar(models, f1_scores, color=['skyblue', 'lightcoral'])
        ax2.set_title('F1分数对比')
        ax2.set_ylabel('F1分数')
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # BLEU分数对比
        ax3 = axes[1, 0]
        bleu_scores = [metrics_before.get('bleu', 0), metrics_after.get('bleu', 0)]
        ax3.bar(models, bleu_scores, color=['skyblue', 'lightcoral'])
        ax3.set_title('BLEU分数对比')
        ax3.set_ylabel('BLEU分数')
        for i, v in enumerate(bleu_scores):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 困惑度对比
        ax4 = axes[1, 1]
        perplexities = [metrics_before.get('perplexity', 0), metrics_after.get('perplexity', 0)]
        ax4.bar(models, perplexities, color=['skyblue', 'lightcoral'])
        ax4.set_title('困惑度对比 (越低越好)')
        ax4.set_ylabel('困惑度')
        for i, v in enumerate(perplexities):
            ax4.text(i, v + max(perplexities) * 0.01, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_curves(logs: List[Dict], metric: str = 'loss'):
        """绘制训练曲线"""
        steps = [log['step'] for log in logs if metric in log]
        values = [log[metric] for log in logs if metric in log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker='o', linestyle='-', linewidth=2)
        plt.title(f'{metric.capitalize()} 曲线')
        plt.xlabel('训练步数')
        plt.ylabel(metric.capitalize())
        plt.grid(True, alpha=0.3)
        plt.show()


# ============================================================
# 使用示例
# ============================================================

def demo_evaluation():
    """演示评估流程"""
    print("=" * 60)
    print("Day 24: 模型评估演示")
    print("=" * 60)
    
    # 创建示例数据
    sample_classification_data = [
        {"input": "这个产品质量很好，我很满意", "label": "positive"},
        {"input": "服务态度很差，不会再来了", "label": "negative"},
        {"input": "价格适中，功能一般", "label": "neutral"},
        {"input": "超出预期的好，强烈推荐", "label": "positive"},
        {"input": "完全不符合描述，很失望", "label": "negative"}
    ]
    
    sample_generation_prompts = [
        "人工智能的发展前景如何？",
        "Python编程语言有什么优势？",
        "机器学习的基本原理是什么？"
    ]
    
    sample_expected_outputs = [
        "人工智能将在未来几十年继续快速发展...",
        "Python因其简洁的语法和丰富的库而广受欢迎...",
        "机器学习通过算法从数据中学习模式..."
    ]
    
    sample_test_texts = [
        "今天天气很好，阳光明媚。",
        "人工智能正在改变我们的生活方式。",
        "Python是一门非常流行的编程语言。"
    ] * 5  # 重复以增加数据量
    
    # 初始化评估器
    evaluator = ModelEvaluator("microsoft/DialoGPT-medium")  # 使用较小的模型进行演示
    
    # 评估分类任务
    print("\n1. 分类任务评估...")
    classification_results = evaluator.evaluate_classification_task(sample_classification_data)
    print(f"   准确率: {classification_results.get('accuracy', 0):.3f}")
    print(f"   精确率: {classification_results.get('precision', 0):.3f}")
    print(f"   召回率: {classification_results.get('recall', 0):.3f}")
    print(f"   F1分数: {classification_results.get('f1', 0):.3f}")
    
    # 评估生成任务
    print("\n2. 生成任务评估...")
    generation_results = evaluator.evaluate_generation_task(
        sample_generation_prompts, 
        sample_expected_outputs
    )
    print(f"   基础模型BLEU: {generation_results.get('base_bleu', 0):.3f}")
    if generation_results.get('fine_tuned_bleu') is not None:
        print(f"   微调模型BLEU: {generation_results.get('fine_tuned_bleu', 0):.3f}")
    
    rouge_scores = generation_results.get('base_rouge', {})
    print(f"   基础模型ROUGE-1: {rouge_scores.get('rouge1', 0):.3f}")
    print(f"   基础模型ROUGE-2: {rouge_scores.get('rouge2', 0):.3f}")
    print(f"   基础模型ROUGE-L: {rouge_scores.get('rougeL', 0):.3f}")
    
    # 模型比较（仅演示，因为没有真正的微调模型）
    print("\n3. 模型比较...")
    comparison_results = evaluator.compare_models(sample_test_texts)
    if comparison_results:
        print(f"   基础模型困惑度: {comparison_results['base_perplexity']:.2f}")
        print(f"   微调模型困惑度: {comparison_results['fine_tuned_perplexity']:.2f}")
        print(f"   改善百分比: {comparison_results['improvement_percentage']:.2f}%")
    else:
        print("   没有微调模型进行比较")
    
    # 演示可视化（创建模拟数据）
    print("\n4. 可视化演示...")
    print("   （实际使用中，这里会显示性能对比图表）")
    
    print("\n" + "=" * 60)
    print("模型评估演示完成!")
    print("=" * 60)
    print("\n评估是微调过程中的重要环节，帮助我们了解模型性能提升情况")


if __name__ == "__main__":
    demo_evaluation()