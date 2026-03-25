"""
结构化输出示例
演示 JSON 输出格式、Pydantic 模型约束、输出验证和修正

核心概念：
1. JSON Schema - 定义输出格式规范
2. Pydantic - Python 数据验证库
3. 输出验证 - 检查输出是否符合预期
4. 错误修正 - 处理格式错误和重试

最佳实践：
- 使用明确的格式描述
- 提供 JSON 示例
- 实现错误处理和重试
- 使用 Pydantic 进行验证
"""

import os
import json
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")


# ==================== Pydantic 模型定义 ====================

class Product(BaseModel):
    """商品信息模型"""
    name: str = Field(..., description="商品名称")
    price: float = Field(..., gt=0, description="价格，必须大于0")
    category: str = Field(..., description="商品分类")
    features: List[str] = Field(default_factory=list, description="商品特性列表")
    rating: Optional[float] = Field(None, ge=0, le=5, description="评分，0-5分")
    in_stock: bool = Field(True, description="是否有库存")

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('商品名称不能为空')
        return v.strip()


class Person(BaseModel):
    """人物信息模型"""
    name: str = Field(..., description="姓名")
    age: int = Field(..., ge=0, le=150, description="年龄")
    occupation: str = Field(..., description="职业")
    skills: List[str] = Field(default_factory=list, description="技能列表")
    contact: Optional[Dict[str, str]] = Field(None, description="联系方式")


class Article(BaseModel):
    """文章信息模型"""
    title: str = Field(..., description="标题")
    author: str = Field(..., description="作者")
    content: str = Field(..., description="内容")
    tags: List[str] = Field(default_factory=list, description="标签")
    word_count: int = Field(0, description="字数")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SentimentResult(BaseModel):
    """情感分析结果模型"""
    sentiment: str = Field(..., description="情感：正面/负面/中性")
    confidence: float = Field(..., ge=0, le=1, description="置信度，0-1")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    explanation: str = Field("", description="解释说明")

    @field_validator('sentiment')
    @classmethod
    def validate_sentiment(cls, v):
        valid = ['正面', '负面', '中性']
        if v not in valid:
            raise ValueError(f'情感必须是: {valid}')
        return v


class TaskPlan(BaseModel):
    """任务计划模型"""
    task_name: str = Field(..., description="任务名称")
    priority: str = Field("中", description="优先级：高/中/低")
    steps: List[str] = Field(default_factory=list, description="执行步骤")
    estimated_time: str = Field("", description="预计时间")
    dependencies: List[str] = Field(default_factory=list, description="依赖项")


# ==================== 基础 JSON 输出 ====================

def demo_basic_json_output():
    """演示基础 JSON 输出"""
    print("\n" + "=" * 60)
    print("示例1: 基础 JSON 输出")
    print("=" * 60)

    # 商品描述
    product_desc = """
这款智能手机配备6.7英寸AMOLED屏幕，搭载最新处理器，支持5G网络。
主摄像头为1亿像素，支持8K视频录制。电池容量5000mAh，支持65W快充。
目前售价3999元，评分4.5分，现货供应。
"""

    # 简单的 JSON 格式要求
    simple_prompt = f"""
请从以下商品描述中提取信息，并以 JSON 格式输出：

{product_desc}

输出 JSON 格式：
{{
    "name": "商品名称",
    "price": 价格数字,
    "category": "分类",
    "features": ["特性1", "特性2"]
}}
"""

    print("简单格式要求:")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": simple_prompt}],
        max_tokens=300
    )
    print(response.choices[0].message.content)

    # 详细的 JSON 格式要求
    detailed_prompt = f"""
请从以下商品描述中提取信息，并以 JSON 格式输出：

{product_desc}

要求：
1. 严格按照指定的 JSON 格式输出
2. 不要添加任何额外的说明文字
3. 所有字段都必须存在，如果没有则使用默认值
4. 价格必须是数字类型
5. features 是一个字符串数组

输出格式：
{{
    "name": "string - 商品完整名称",
    "price": "number - 价格，单位元",
    "category": "string - 商品分类",
    "features": ["string - 特性列表"],
    "rating": "number - 评分，0-5",
    "in_stock": "boolean - 是否有库存"
}}

只输出 JSON，不要其他内容：
"""

    print("\n详细格式要求:")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": detailed_prompt}],
        max_tokens=300
    )
    print(response.choices[0].message.content)


# ==================== 使用 JSON Schema ====================

def demo_json_schema():
    """演示使用 JSON Schema 定义格式"""
    print("\n" + "=" * 60)
    print("示例2: 使用 JSON Schema")
    print("=" * 60)

    # 定义 JSON Schema
    person_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "人物姓名"
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150,
                "description": "年龄"
            },
            "occupation": {
                "type": "string",
                "description": "职业"
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "技能列表"
            },
            "contact": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "phone": {"type": "string"}
                }
            }
        },
        "required": ["name", "age", "occupation"]
    }

    text = """
张明是一位35岁的软件工程师，精通Python、JavaScript和Go语言。
他有10年的开发经验，擅长后端系统设计。
联系方式：邮箱 zhangming@example.com，电话 13800138000。
"""

    prompt = f"""
请从以下文本中提取人物信息：

{text}

请严格按照以下 JSON Schema 输出：

{json.dumps(person_schema, ensure_ascii=False, indent=2)}

只输出 JSON 格式数据，不要添加任何说明文字。
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    result = response.choices[0].message.content
    print("原始输出:")
    print(result)

    # 尝试解析和验证
    try:
        # 提取 JSON 部分
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            data = json.loads(json_match.group())
            print("\n解析成功:")
            print(json.dumps(data, ensure_ascii=False, indent=2))

            # 使用 Pydantic 验证
            person = Person.model_validate(data)
            print("\n验证成功:")
            print(person.model_dump_json(indent=2))
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"\n解析/验证失败: {e}")


# ==================== Pydantic 验证 ====================

def demo_pydantic_validation():
    """演示 Pydantic 模型验证"""
    print("\n" + "=" * 60)
    print("示例3: Pydantic 模型验证")
    print("=" * 60)

    # 测试数据
    test_cases = [
        {
            "name": "iPhone 15",
            "price": 5999,
            "category": "手机",
            "features": ["5G", "AMOLED屏幕", "双摄像头"],
            "rating": 4.5,
            "in_stock": True
        },
        {
            "name": "",  # 空名称，应该验证失败
            "price": -100,  # 负价格，应该验证失败
            "category": "测试",
            "features": []
        },
        {
            "name": "测试商品",
            "price": 100,
            "category": "测试",
            "features": ["特性1"],
            "rating": 6.0  # 超出范围，应该验证失败
        }
    ]

    for i, data in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"输入数据: {json.dumps(data, ensure_ascii=False)}")

        try:
            product = Product.model_validate(data)
            print("验证成功:")
            print(product.model_dump_json(indent=2))
        except ValidationError as e:
            print(f"验证失败:")
            for error in e.errors():
                print(f"  - 字段 '{error['loc'][0]}': {error['msg']}")


# ==================== 结构化输出与重试 ====================

def get_structured_output(
    prompt: str,
    model_class: type,
    max_retries: int = 3
) -> tuple[Optional[BaseModel], Optional[str]]:
    """
    获取结构化输出，带重试机制

    Args:
        prompt: 提示词
        model_class: Pydantic 模型类
        max_retries: 最大重试次数

    Returns:
        (验证后的模型实例, 错误信息)
    """
    # 获取 Schema
    schema = model_class.model_json_schema()

    # 构建提示词
    full_prompt = f"""
{prompt}

请严格按照以下 JSON Schema 输出：

{json.dumps(schema, ensure_ascii=False, indent=2)}

要求：
1. 只输出 JSON 格式数据
2. 不要添加任何说明文字
3. 所有必填字段都必须存在
4. 数据类型必须正确

JSON 输出：
"""

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1000,
                temperature=0.3  # 降低温度以获得更稳定的输出
            )

            content = response.choices[0].message.content

            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError("未找到有效的 JSON 数据")

            data = json.loads(json_match.group())

            # 验证
            validated = model_class.model_validate(data)
            return validated, None

        except json.JSONDecodeError as e:
            last_error = f"JSON 解析失败: {str(e)}"
            full_prompt += f"\n\n上一次输出不是有效的 JSON，请修正后重新输出。错误: {last_error}"
        except ValidationError as e:
            last_error = f"数据验证失败: {str(e)}"
            full_prompt += f"\n\n上一次输出验证失败，请修正后重新输出。错误: {last_error}"
        except Exception as e:
            last_error = f"其他错误: {str(e)}"
            full_prompt += f"\n\n发生错误，请重新输出。错误: {last_error}"

    return None, last_error


def demo_structured_output_with_retry():
    """演示带重试的结构化输出"""
    print("\n" + "=" * 60)
    print("示例4: 带重试机制的结构化输出")
    print("=" * 60)

    # 情感分析任务
    text = """
这款手机的拍照效果真的太棒了！夜景模式简直惊艳，细节保留得很好。
但是电池续航有点拉胯，用不到一天就得充电。
总体来说是一款不错的手机，推荐给喜欢拍照的朋友。
"""

    prompt = f"""
请分析以下文本的情感：

"{text}"

提取以下信息：
1. 整体情感（正面/负面/中性）
2. 置信度（0-1）
3. 关键词列表
4. 简短的解释说明
"""

    result, error = get_structured_output(prompt, SentimentResult)

    if result:
        print("情感分析结果:")
        print(result.model_dump_json(indent=2))
    else:
        print(f"获取结构化输出失败: {error}")


# ==================== 批量结构化输出 ====================

def batch_structured_output(
    items: list,
    prompt_template: str,
    model_class: type
) -> list:
    """
    批量获取结构化输出

    Args:
        items: 待处理的文本列表
        prompt_template: 提示词模板（使用 {item} 作为占位符）
        model_class: Pydantic 模型类

    Returns:
        结果列表
    """
    results = []

    for i, item in enumerate(items, 1):
        print(f"\n处理第 {i}/{len(items)} 项...")

        prompt = prompt_template.format(item=item)
        result, error = get_structured_output(prompt, model_class)

        if result:
            results.append(result)
        else:
            print(f"处理失败: {error}")
            results.append(None)

    return results


def demo_batch_processing():
    """演示批量处理"""
    print("\n" + "=" * 60)
    print("示例5: 批量结构化输出")
    print("=" * 60)

    # 多个商品描述
    products = [
        "这款笔记本电脑搭载第13代酷睿处理器，16GB内存，512GB固态硬盘。屏幕为14英寸2K分辨率，售价5999元。",
        "无线蓝牙耳机，支持主动降噪，续航30小时。Type-C快充，IPX4防水。售价399元，评分4.2分。",
        "智能手表，支持心率监测、血氧检测、睡眠分析。1.4英寸AMOLED屏幕，防水50米。售价1299元。"
    ]

    prompt_template = """
请从以下商品描述中提取信息：

"{item}"

提取商品名称、价格、分类、主要特性。
"""

    results = batch_structured_output(products, prompt_template, Product)

    print("\n批量处理结果:")
    for i, result in enumerate(results, 1):
        print(f"\n商品 {i}:")
        if result:
            print(result.model_dump_json(indent=2))
        else:
            print("处理失败")


# ==================== 复杂嵌套结构 ====================

class OrderItem(BaseModel):
    """订单项"""
    product_name: str
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    subtotal: float = Field(..., ge=0)


class Order(BaseModel):
    """订单模型"""
    order_id: str
    customer_name: str
    items: List[OrderItem]
    total_amount: float = Field(..., ge=0)
    status: str = Field("待处理")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator('total_amount')
    @classmethod
    def validate_total(cls, v, info):
        # 验证总金额是否与明细一致
        if 'items' in info.data:
            calculated = sum(item.subtotal for item in info.data['items'])
            if abs(v - calculated) > 0.01:  # 允许小的浮点误差
                raise ValueError(f'总金额不一致，应为 {calculated}')
        return v


def demo_nested_structure():
    """演示复杂嵌套结构"""
    print("\n" + "=" * 60)
    print("示例6: 复杂嵌套结构")
    print("=" * 60)

    text = """
订单号：ORD-20240115-001
客户：张三

商品明细：
1. iPhone 15 Pro，数量：1，单价：8999元，小计：8999元
2. AirPods Pro，数量：2，单价：1899元，小计：3798元
3. 手机壳，数量：1，单价：99元，小计：99元

总金额：12896元
订单状态：已支付
"""

    prompt = f"""
请从以下订单文本中提取结构化信息：

{text}

输出格式要求：
- order_id: 订单号
- customer_name: 客户姓名
- items: 商品列表数组，每项包含 product_name, quantity, unit_price, subtotal
- total_amount: 总金额（数字）
- status: 订单状态
"""

    result, error = get_structured_output(prompt, Order)

    if result:
        print("订单解析结果:")
        print(result.model_dump_json(indent=2))
    else:
        print(f"解析失败: {error}")


# ==================== 输出格式修正 ====================

def fix_json_output(raw_output: str, model_class: type) -> tuple[Optional[BaseModel], str]:
    """
    尝试修正格式错误的 JSON 输出

    Args:
        raw_output: 原始输出
        model_class: Pydantic 模型类

    Returns:
        (修正后的模型实例, 修正说明)
    """
    fixes = []

    # 1. 提取 JSON 部分
    json_match = re.search(r'\{[\s\S]*\}', raw_output)
    if not json_match:
        # 尝试找数组
        json_match = re.search(r'\[[\s\S]*\]', raw_output)
        if not json_match:
            return None, "无法找到有效的 JSON 数据"

    json_str = json_match.group()

    # 2. 尝试直接解析
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 尝试修复常见的 JSON 错误
        fixes.append(f"JSON 解析错误: {str(e)}")

        # 修复未引用的键
        json_str = re.sub(r'(\w+)\s*:', r'"\1":', json_str)
        fixes.append("添加了缺失的引号")

        # 修复单引号为双引号
        json_str = json_str.replace("'", '"')
        fixes.append("将单引号替换为双引号")

        # 修复尾随逗号
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        fixes.append("移除了尾随逗号")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None, f"无法修正 JSON: {str(e)}"

    # 3. 验证数据
    try:
        validated = model_class.model_validate(data)
        return validated, f"成功修正，修复项: {'; '.join(fixes)}"
    except ValidationError as e:
        # 尝试修复验证错误
        for error in e.errors():
            field = error['loc'][0] if error['loc'] else 'unknown'
            if field in data:
                # 尝试类型转换
                try:
                    if error['type'] == 'int_type':
                        data[field] = int(float(data[field]))
                        fixes.append(f"转换字段 '{field}' 为整数")
                    elif error['type'] == 'float_type':
                        data[field] = float(data[field])
                        fixes.append(f"转换字段 '{field}' 为浮点数")
                    elif error['type'] == 'bool_type':
                        data[field] = bool(data[field])
                        fixes.append(f"转换字段 '{field}' 为布尔值")
                except (ValueError, TypeError):
                    pass

        # 再次验证
        try:
            validated = model_class.model_validate(data)
            return validated, f"成功修正，修复项: {'; '.join(fixes)}"
        except ValidationError:
            return None, f"验证失败: {str(e)}"


def demo_output_fixing():
    """演示输出格式修正"""
    print("\n" + "=" * 60)
    print("示例7: 输出格式修正")
    print("=" * 60)

    # 模拟一些可能有问题的输出
    test_outputs = [
        # 格式正确的输出
        '''
        {
            "name": "iPhone 15",
            "price": 5999,
            "category": "手机",
            "features": ["5G", "AMOLED屏幕"],
            "rating": 4.5,
            "in_stock": true
        }
        ''',
        # 有一些文本说明的输出
        '''
        这是商品信息：
        {
            "name": "MacBook Pro",
            "price": 14999,
            "category": "电脑",
            "features": ["M3芯片", "16GB内存"],
            "rating": 4.8,
            "in_stock": true
        }
        以上是提取的商品信息。
        ''',
        # 字段值需要修正的输出
        '''
        {
            "name": "AirPods",
            "price": "1299",
            "category": "耳机",
            "features": ["降噪", "无线充电"],
            "rating": "4.2",
            "in_stock": "yes"
        }
        '''
    ]

    for i, output in enumerate(test_outputs, 1):
        print(f"\n--- 测试 {i} ---")
        print(f"原始输出:\n{output[:200]}...")

        result, message = fix_json_output(output, Product)

        if result:
            print(f"\n修正成功: {message}")
            print(f"结果:\n{result.model_dump_json(indent=2)}")
        else:
            print(f"\n修正失败: {message}")


# ==================== 实战案例：简历解析 ====================

class Resume(BaseModel):
    """简历模型"""
    name: str = Field(..., description="姓名")
    age: int = Field(..., ge=18, le=65, description="年龄")
    education: str = Field(..., description="学历")
    experience_years: int = Field(..., ge=0, description="工作年限")
    skills: List[str] = Field(default_factory=list, description="技能列表")
    work_history: List[Dict[str, str]] = Field(default_factory=list, description="工作经历")
    expected_salary: Optional[str] = Field(None, description="期望薪资")


def demo_resume_parsing():
    """实战案例：简历解析"""
    print("\n" + "=" * 60)
    print("示例8: 实战案例 - 简历解析")
    print("=" * 60)

    resume_text = """
个人简历

姓名：李明
年龄：28岁
学历：硕士研究生

工作经历：
- 2020-2023：ABC科技有限公司，高级前端工程师
- 2018-2020：XYZ互联网公司，前端开发工程师

技能：
- 精通 React、Vue、TypeScript
- 熟悉 Node.js、Python
- 了解 Docker、Kubernetes

期望薪资：25-30K
"""

    prompt = f"""
请从以下简历文本中提取结构化信息：

{resume_text}

提取：姓名、年龄、学历、工作年限、技能列表、工作经历、期望薪资。
"""

    result, error = get_structured_output(prompt, Resume)

    if result:
        print("简历解析结果:")
        print(result.model_dump_json(indent=2))
    else:
        print(f"解析失败: {error}")


# ==================== 结构化输出工具类 ====================

class StructuredOutputGenerator:
    """结构化输出生成器"""

    def __init__(self, model_class: type, temperature: float = 0.3):
        """
        初始化

        Args:
            model_class: Pydantic 模型类
            temperature: 生成温度
        """
        self.model_class = model_class
        self.temperature = temperature

    def generate(self, prompt: str, max_retries: int = 3) -> tuple[Optional[BaseModel], List[str]]:
        """
        生成结构化输出

        Args:
            prompt: 提示词
            max_retries: 最大重试次数

        Returns:
            (验证后的模型, 修正历史)
        """
        schema = self.model_class.model_json_schema()
        history = []

        full_prompt = f"""
{prompt}

输出格式：
{json.dumps(schema, ensure_ascii=False, indent=2)}

只输出 JSON 格式数据：
"""

        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1000,
                temperature=self.temperature
            )

            content = response.choices[0].message.content
            result, fix_msg = fix_json_output(content, self.model_class)

            if result:
                history.append(f"尝试 {attempt + 1}: {fix_msg}")
                return result, history

            history.append(f"尝试 {attempt + 1}: {fix_msg}")
            full_prompt += f"\n\n上次输出有问题，请修正。问题: {fix_msg}"

        return None, history

    def generate_batch(self, prompts: list) -> list:
        """批量生成"""
        results = []
        for prompt in prompts:
            result, _ = self.generate(prompt)
            results.append(result)
        return results


def demo_generator_class():
    """演示结构化输出生成器"""
    print("\n" + "=" * 60)
    print("示例9: 使用 StructuredOutputGenerator 类")
    print("=" * 60)

    generator = StructuredOutputGenerator(TaskPlan, temperature=0.5)

    prompt = """
请为以下任务制定执行计划：

任务：学习 Python 数据分析

背景：
- 有一定的 Python 基础
- 想要学习数据分析相关技能
- 每天可以投入2小时学习

请输出任务计划，包括任务名称、优先级、执行步骤、预计时间。
"""

    result, history = generator.generate(prompt)

    print("生成历史:")
    for h in history:
        print(f"  - {h}")

    if result:
        print("\n任务计划:")
        print(result.model_dump_json(indent=2))
    else:
        print("\n生成失败")


# ==================== 交互式演示 ====================

def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("结构化输出 - 交互模式")
    print("=" * 60)
    print("选择要演示的功能：")
    print("1. 商品信息提取")
    print("2. 人物信息提取")
    print("3. 情感分析")
    print("4. 自定义模型验证")
    print("5. 退出")

    while True:
        choice = input("\n请选择 (1-5): ").strip()

        if choice == '5' or choice.lower() in ['quit', 'exit', 'q']:
            break

        if choice == '1':
            text = input("请输入商品描述: ").strip()
            if not text:
                continue
            prompt = f"请从以下文本中提取商品信息：\n{text}"
            result, error = get_structured_output(prompt, Product)
            if result:
                print(f"\n结果:\n{result.model_dump_json(indent=2)}")
            else:
                print(f"失败: {error}")

        elif choice == '2':
            text = input("请输入人物介绍: ").strip()
            if not text:
                continue
            prompt = f"请从以下文本中提取人物信息：\n{text}"
            result, error = get_structured_output(prompt, Person)
            if result:
                print(f"\n结果:\n{result.model_dump_json(indent=2)}")
            else:
                print(f"失败: {error}")

        elif choice == '3':
            text = input("请输入要分析的文本: ").strip()
            if not text:
                continue
            prompt = f"请分析以下文本的情感：\n{text}"
            result, error = get_structured_output(prompt, SentimentResult)
            if result:
                print(f"\n结果:\n{result.model_dump_json(indent=2)}")
            else:
                print(f"失败: {error}")

        elif choice == '4':
            print("输入 JSON 数据进行验证:")
            try:
                json_str = input("JSON: ").strip()
                data = json.loads(json_str)
                model_name = input("验证模型 (product/person/sentiment): ").strip().lower()

                model_map = {
                    'product': Product,
                    'person': Person,
                    'sentiment': SentimentResult
                }

                if model_name in model_map:
                    validated = model_map[model_name].model_validate(data)
                    print(f"\n验证成功:\n{validated.model_dump_json(indent=2)}")
                else:
                    print("未知模型")
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
            except ValidationError as e:
                print(f"验证错误: {e}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - 结构化输出")
    print("=" * 60)

    # 运行所有演示
    demo_basic_json_output()
    demo_json_schema()
    demo_pydantic_validation()
    demo_structured_output_with_retry()
    demo_batch_processing()
    demo_nested_structure()
    demo_output_fixing()
    demo_resume_parsing()
    demo_generator_class()

    # 交互模式
    print("\n" + "=" * 60)
    print("是否进入交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        interactive_demo()