# day27/cost_control.py
"""
Day 27: 性能优化 - 成本控制

演示智能成本管理系统：
1. Token 精确计数
2. 预算管理
3. 成本估算
4. 成本报告生成
"""

import time
import json
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


# ============================================================
# 数据模型定义
# ============================================================

@dataclass
class UsageRecord:
    """使用记录"""
    timestamp: datetime = field(default_factory=datetime.now)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    user_id: str = ""
    request_id: str = ""
    success: bool = True


@dataclass
class BudgetStatus:
    """预算状态"""
    total_budget: float = 0.0
    used: float = 0.0
    remaining: float = 0.0
    utilization_rate: float = 0.0
    alert_level: str = "normal"  # normal, warning, critical


# ============================================================
# Token 计数器
# ============================================================

class TokenCounter:
    """Token 精确计数"""

    # 简化的 Token 估算（实际应使用 tiktoken）
    CHAR_PER_TOKEN = {
        "chinese": 1.5,  # 中文约 1.5 字符/Token
        "english": 4,    # 英文约 4 字符/Token
        "mixed": 2.5     # 混合约 2.5 字符/Token
    }

    def __init__(self):
        self.stats = {
            "total_input": 0,
            "total_output": 0,
            "requests_count": 0
        }

    def estimate_tokens(self, text: str, lang_type: str = "mixed") -> int:
        """估算 Token 数量"""
        # 简化估算：实际应使用 tiktoken 库
        char_ratio = self.CHAR_PER_TOKEN.get(lang_type, 2.5)
        estimated = len(text) / char_ratio
        return int(estimated)

    def count_message_tokens(self, messages: List[Dict]) -> Dict:
        """计算消息列表的 Token 数"""
        input_tokens = 0
        output_tokens = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            tokens = self.estimate_tokens(content)

            if role in ["system", "user"]:
                input_tokens += tokens
            elif role == "assistant":
                output_tokens += tokens

            # 消息格式开销（约 4 tokens）
            input_tokens += 4

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

    def record_usage(self, input_tokens: int, output_tokens: int):
        """记录使用"""
        self.stats["total_input"] += input_tokens
        self.stats["total_output"] += output_tokens
        self.stats["requests_count"] += 1

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            "total_input_tokens": self.stats["total_input"],
            "total_output_tokens": self.stats["total_output"],
            "total_tokens": self.stats["total_input"] + self.stats["total_output"],
            "requests_count": self.stats["requests_count"],
            "avg_tokens_per_request": (
                self.stats["total_input"] + self.stats["total_output"]
            ) / max(1, self.stats["requests_count"])
        }


# ============================================================
# 预算管理器
# ============================================================

class BudgetManager:
    """预算管理"""

    def __init__(
        self,
        daily_budget: float = 100.0,
        weekly_budget: float = 700.0,
        monthly_budget: float = 3000.0,
        alert_threshold: float = 0.8
    ):
        self.budgets = {
            "daily": daily_budget,
            "weekly": weekly_budget,
            "monthly": monthly_budget
        }
        self.alert_threshold = alert_threshold

        self.usage: Dict[str, float] = defaultdict(float)
        self.usage_history: List[UsageRecord] = []

    def check_budget(self, estimated_cost: float, budget_type: str = "daily") -> Dict:
        """检查预算是否允许"""
        budget_limit = self.budgets[budget_type]
        current_usage = self.usage[budget_type]
        remaining = budget_limit - current_usage

        # 预算不足
        if remaining < estimated_cost:
            return {
                "allowed": False,
                "reason": "预算不足",
                "budget_type": budget_type,
                "budget_limit": budget_limit,
                "current_usage": current_usage,
                "remaining": remaining,
                "estimated": estimated_cost
            }

        # 预警检查
        projected_usage = current_usage + estimated_cost
        utilization_rate = projected_usage / budget_limit

        if utilization_rate > self.alert_threshold:
            alert_level = "critical" if utilization_rate > 0.95 else "warning"
            return {
                "allowed": True,
                "warning": f"预算使用已达 {utilization_rate:.1%}",
                "alert_level": alert_level,
                "remaining": remaining,
                "utilization_rate": utilization_rate
            }

        return {
            "allowed": True,
            "alert_level": "normal",
            "remaining": remaining,
            "utilization_rate": utilization_rate
        }

    def record_usage(self, actual_cost: float, budget_type: str = "daily"):
        """记录实际使用"""
        self.usage[budget_type] += actual_cost

        # 更新其他周期
        self.usage["weekly"] += actual_cost
        self.usage["monthly"] += actual_cost

    def add_usage_record(self, record: UsageRecord):
        """添加使用记录"""
        self.usage_history.append(record)
        self.record_usage(record.cost)

    def get_status(self, budget_type: str = "daily") -> BudgetStatus:
        """获取预算状态"""
        total_budget = self.budgets[budget_type]
        used = self.usage[budget_type]
        remaining = total_budget - used
        utilization_rate = used / total_budget

        if utilization_rate > 0.95:
            alert_level = "critical"
        elif utilization_rate > self.alert_threshold:
            alert_level = "warning"
        else:
            alert_level = "normal"

        return BudgetStatus(
            total_budget=total_budget,
            used=used,
            remaining=remaining,
            utilization_rate=utilization_rate,
            alert_level=alert_level
        )

    def reset_daily(self):
        """重置每日预算"""
        self.usage["daily"] = 0.0

    def reset_weekly(self):
        """重置每周预算"""
        self.usage["weekly"] = 0.0

    def reset_monthly(self):
        """重置每月预算"""
        self.usage["monthly"] = 0.0


# ============================================================
# 成本估算器
# ============================================================

class CostEstimator:
    """成本估算器"""

    # 各模型价格（每千 Token，美元）
    MODEL_PRICES = {
        "gpt-4": {
            "input": 0.03,
            "output": 0.06,
            "description": "GPT-4 (8K)"
        },
        "gpt-4-32k": {
            "input": 0.06,
            "output": 0.12,
            "description": "GPT-4 (32K)"
        },
        "gpt-3.5-turbo": {
            "input": 0.0015,
            "output": 0.002,
            "description": "GPT-3.5 Turbo"
        },
        "claude-3-opus": {
            "input": 0.015,
            "output": 0.075,
            "description": "Claude 3 Opus"
        },
        "claude-3-sonnet": {
            "input": 0.003,
            "output": 0.015,
            "description": "Claude 3 Sonnet"
        },
        "claude-3-haiku": {
            "input": 0.00025,
            "output": 0.00125,
            "description": "Claude 3 Haiku"
        },
        "deepseek": {
            "input": 0.001,
            "output": 0.002,
            "description": "DeepSeek"
        }
    }

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Dict:
        """估算请求成本"""
        prices = self.MODEL_PRICES.get(model)

        if not prices:
            return {
                "model": model,
                "error": "未知模型",
                "estimated_cost": 0.0
            }

        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]
        total_cost = input_cost + output_cost

        return {
            "model": model,
            "description": prices["description"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "price_per_1k_input": prices["input"],
            "price_per_1k_output": prices["output"]
        }

    def compare_models(self, input_tokens: int, output_tokens: int) -> List[Dict]:
        """比较不同模型的成本"""
        comparisons = []

        for model, prices in self.MODEL_PRICES.items():
            result = self.estimate_cost(model, input_tokens, output_tokens)
            comparisons.append(result)

        # 按成本排序
        comparisons.sort(key=lambda x: x["total_cost"])

        return comparisons

    def optimize_model_selection(
        self,
        task_type: str,
        complexity: str,
        budget_limit: float,
        input_tokens: int,
        output_tokens: int
    ) -> Dict:
        """根据任务选择最优模型"""

        # 根据复杂度推荐
        if complexity == "high":
            recommended = ["gpt-4", "claude-3-opus"]
        elif complexity == "medium":
            recommended = ["claude-3-sonnet", "gpt-3.5-turbo"]
        else:
            recommended = ["claude-3-haiku", "gpt-3.5-turbo", "deepseek"]

        # 选择预算范围内最合适的
        for model in recommended:
            cost_result = self.estimate_cost(model, input_tokens, output_tokens)
            if cost_result["total_cost"] <= budget_limit:
                return {
                    "recommended_model": model,
                    "reason": f"符合复杂度'{complexity}'且预算充足",
                    "cost_estimate": cost_result
                }

        # 都超出预算，选最便宜的
        cheapest_model = "claude-3-haiku"
        cost_result = self.estimate_cost(cheapest_model, input_tokens, output_tokens)

        return {
            "recommended_model": cheapest_model,
            "reason": "预算限制，选择最便宜模型",
            "cost_estimate": cost_result,
            "warning": "可能影响质量"
        }


# ============================================================
# 成本报告生成器
# ============================================================

class CostReporter:
    """成本报告生成"""

    def __init__(self, budget_manager: BudgetManager, token_counter: TokenCounter):
        self.budget_manager = budget_manager
        self.token_counter = token_counter

    def generate_daily_report(self) -> Dict:
        """生成每日成本报告"""
        status = self.budget_manager.get_status("daily")
        token_stats = self.token_counter.get_stats()

        # 模型使用分析
        model_breakdown = self._get_model_breakdown()

        # 时间分布
        hourly_distribution = self._get_hourly_distribution()

        # 生成建议
        recommendations = self._generate_recommendations(status, token_stats)

        return {
            "report_type": "daily",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "budget_status": {
                "total_budget": status.total_budget,
                "used": status.used,
                "remaining": status.remaining,
                "utilization_rate": f"{status.utilization_rate:.2%}",
                "alert_level": status.alert_level
            },
            "token_stats": token_stats,
            "model_breakdown": model_breakdown,
            "hourly_distribution": hourly_distribution,
            "cost_per_request": status.used / max(1, token_stats["requests_count"]),
            "recommendations": recommendations
        }

    def _get_model_breakdown(self) -> Dict:
        """获取模型使用分布"""
        breakdown = defaultdict(lambda: {"count": 0, "cost": 0.0})

        for record in self.budget_manager.usage_history:
            breakdown[record.model]["count"] += 1
            breakdown[record.model]["cost"] += record.cost

        return dict(breakdown)

    def _get_hourly_distribution(self) -> Dict:
        """获取小时分布"""
        hourly = defaultdict(lambda: {"count": 0, "cost": 0.0})

        for record in self.budget_manager.usage_history:
            hour = record.timestamp.strftime("%H")
            hourly[hour]["count"] += 1
            hourly[hour]["cost"] += record.cost

        return dict(hourly)

    def _generate_recommendations(self, status: BudgetStatus, token_stats: Dict) -> List[Dict]:
        """生成优化建议"""
        recommendations = []

        # 预算预警
        if status.alert_level == "warning":
            recommendations.append({
                "type": "budget",
                "priority": "high",
                "message": "预算使用已达 80%，建议降低高成本模型使用"
            })
        elif status.alert_level == "critical":
            recommendations.append({
                "type": "budget",
                "priority": "critical",
                "message": "预算即将用尽，建议暂停高成本操作"
            })

        # Token 效率
        avg_tokens = token_stats["avg_tokens_per_request"]
        if avg_tokens > 500:
            recommendations.append({
                "type": "efficiency",
                "priority": "medium",
                "message": f"平均 Token 数较高({avg_tokens:.0f})，建议压缩输入或优化输出"
            })

        return recommendations

    def generate_weekly_report(self) -> Dict:
        """生成每周成本报告"""
        status = self.budget_manager.get_status("weekly")

        return {
            "report_type": "weekly",
            "week": datetime.now().strftime("%Y-W%W"),
            "budget_status": {
                "total_budget": status.total_budget,
                "used": status.used,
                "remaining": status.remaining,
                "utilization_rate": f"{status.utilization_rate:.2%}"
            },
            "daily_avg_cost": status.used / 7,
            "trend": self._calculate_trend()
        }

    def _calculate_trend(self) -> Dict:
        """计算趋势"""
        # 简化：按天分组
        daily_costs = defaultdict(float)

        for record in self.budget_manager.usage_history:
            day = record.timestamp.strftime("%Y-%m-%d")
            daily_costs[day] += record.cost

        if len(daily_costs) < 2:
            return {"trend": "insufficient_data"}

        values = list(daily_costs.values())
        trend_direction = "increasing" if values[-1] > values[0] else "decreasing"

        return {
            "trend": trend_direction,
            "daily_costs": dict(daily_costs)
        }


# ============================================================
# 演示函数
# ============================================================

def demo_token_counting():
    """演示 Token 计数"""
    print("\n=== Token 计数演示 ===")

    counter = TokenCounter()

    # 测试文本
    texts = [
        ("你好，请介绍一下自己", "chinese"),
        ("Hello, please introduce yourself", "english"),
        ("你好，Hello，请介绍自己", "mixed")
    ]

    for text, lang in texts:
        tokens = counter.estimate_tokens(text, lang)
        print(f"文本: '{text}'")
        print(f"语言类型: {lang}")
        print(f"估算 Tokens: {tokens}")
        print()

    # 消息列表计数
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手"},
        {"role": "user", "content": "请介绍一下自己"},
        {"role": "assistant", "content": "你好！我是一个AI助手，可以帮助你解决问题"}
    ]

    result = counter.count_message_tokens(messages)
    print("消息列表统计:")
    print(f"  输入 Tokens: {result['input_tokens']}")
    print(f"  输出 Tokens: {result['output_tokens']}")
    print(f"  总 Tokens: {result['total_tokens']}")


def demo_budget_management():
    """演示预算管理"""
    print("\n=== 预算管理演示 ===")

    budget_manager = BudgetManager(
        daily_budget=10.0,
        alert_threshold=0.8
    )

    # 模拟请求
    requests = [
        {"cost": 0.5, "desc": "简单问答"},
        {"cost": 2.0, "desc": "复杂分析"},
        {"cost": 1.5, "desc": "文档总结"},
        {"cost": 3.0, "desc": "深度推理"},
        {"cost": 5.0, "desc": "批量处理"}  # 超预算测试
    ]

    for req in requests:
        result = budget_manager.check_budget(req["cost"])
        print(f"\n请求: {req['desc']} (成本: ${req['cost']:.2f})")

        if result["allowed"]:
            if "warning" in result:
                print(f"  ⚠️ {result['warning']}")
            print(f"  ✅ 允许执行，剩余预算: ${result['remaining']:.2f}")
            budget_manager.record_usage(req["cost"])
        else:
            print(f"  ❌ 拒绝执行: {result['reason']}")

    # 显示状态
    status = budget_manager.get_status()
    print(f"\n预算状态:")
    print(f"  已使用: ${status.used:.2f}")
    print(f"  剩余: ${status.remaining:.2f}")
    print(f"  使用率: {status.utilization_rate:.1%}")
    print(f"  告警级别: {status.alert_level}")


def demo_cost_estimation():
    """演示成本估算"""
    print("\n=== 成本估算演示 ===")

    estimator = CostEstimator()

    # 场景：1000 输入 tokens, 500 输出 tokens
    input_tokens = 1000
    output_tokens = 500

    # 比较不同模型
    comparisons = estimator.compare_models(input_tokens, output_tokens)

    print(f"\n场景: {input_tokens} 输入 Tokens, {output_tokens} 输出 Tokens")
    print("\n各模型成本对比:")
    print("-" * 60)

    for comp in comparisons:
        print(f"模型: {comp['model']} ({comp['description']})")
        print(f"  输入成本: ${comp['input_cost']:.4f}")
        print(f"  输出成本: ${comp['output_cost']:.4f}")
        print(f"  总成本: ${comp['total_cost']:.4f}")
        print()

    # 模型选择优化
    print("=== 模型选择优化 ===")

    for complexity in ["high", "medium", "low"]:
        budget_limit = 0.05 if complexity == "low" else 0.1
        result = estimator.optimize_model_selection(
            task_type="chat",
            complexity=complexity,
            budget_limit=budget_limit,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        print(f"\n复杂度 '{complexity}', 预算限制 ${budget_limit:.2f}:")
        print(f"  推荐模型: {result['recommended_model']}")
        print(f"  原因: {result['reason']}")
        print(f"  预估成本: ${result['cost_estimate']['total_cost']:.4f}")


def demo_cost_report():
    """演示成本报告"""
    print("\n=== 成本报告演示 ===")

    budget_manager = BudgetManager(daily_budget=100.0)
    token_counter = TokenCounter()

    # 添加模拟记录
    models = ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"]

    for i in range(20):
        model = models[i % 3]
        input_tokens = 100 + i * 10
        output_tokens = 50 + i * 5

        token_counter.record_usage(input_tokens, output_tokens)

        estimator = CostEstimator()
        cost_result = estimator.estimate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost_result["total_cost"],
            user_id=f"user_{i % 5}",
            request_id=f"req_{i}"
        )

        budget_manager.add_usage_record(record)

    # 生成报告
    reporter = CostReporter(budget_manager, token_counter)

    # 每日报告
    daily_report = reporter.generate_daily_report()

    print("\n每日成本报告:")
    print("-" * 40)
    print(f"日期: {daily_report['date']}")
    print(f"预算使用: {daily_report['budget_status']['utilization_rate']}")
    print(f"请求次数: {daily_report['token_stats']['requests_count']}")
    print(f"平均成本: ${daily_report['cost_per_request']:.4f}/请求")

    print("\n模型使用分布:")
    for model, stats in daily_report['model_breakdown'].items():
        print(f"  {model}: {stats['count']} 次, ${stats['cost']:.4f}")

    print("\n优化建议:")
    for rec in daily_report['recommendations']:
        print(f"  [{rec['priority']}] {rec['message']}")


def main():
    """主函数"""
    print("=" * 70)
    print("Day 27: 性能优化 - 成本控制演示")
    print("=" * 70)

    # 1. Token 计数演示
    demo_token_counting()

    # 2. 预算管理演示
    demo_budget_management()

    # 3. 成本估算演示
    demo_cost_estimation()

    # 4. 成本报告演示
    demo_cost_report()

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()