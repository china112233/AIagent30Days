"""
输出检查与内容安全
演示如何检查和过滤 Agent 输出，保护敏感信息

核心功能：
1. 有害内容检测 - 识别暴力、仇恨、非法内容
2. PII 保护 - 检测和脱敏个人信息
3. 输出格式验证 - 确保输出符合预期格式
4. 内容审核策略 - 多层次内容安全检查
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
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


# ==================== 内容分类定义 ====================

class ContentCategory(Enum):
    """内容类别"""
    SAFE = "safe"                    # 安全内容
    HARMFUL = "harmful"              # 有害内容
    PII = "pii"                      # 包含个人信息
    SENSITIVE = "sensitive"          # 敏感信息
    INAPPROPRIATE = "inappropriate"  # 不当内容
    MISINFORMATION = "misinformation"  # 虚假信息


@dataclass
class OutputCheckResult:
    """输出检查结果"""
    is_safe: bool
    categories: List[ContentCategory]
    issues: List[Dict[str, Any]]
    original_output: str
    sanitized_output: Optional[str] = None
    confidence: float = 1.0
    requires_review: bool = False


# ==================== 有害内容检测器 ====================

class HarmfulContentDetector:
    """有害内容检测器"""

    # 有害内容模式
    HARMFUL_PATTERNS = {
        "violence": {
            "patterns": [
                r"杀死", r"谋杀", r"暴力", r"虐待",
                r"kill", r"murder", r"violence", r"abuse"
            ],
            "description": "暴力内容",
            "severity": "high"
        },
        "hate_speech": {
            "patterns": [
                r"种族歧视", r"性别歧视", r"仇恨",
                r"racist", r"sexist", r"hate"
            ],
            "description": "仇恨言论",
            "severity": "high"
        },
        "illegal": {
            "patterns": [
                r"毒品", r"走私", r"洗钱", r"赌博",
                r"drugs", r"smuggle", r"money laundering", r"gambling"
            ],
            "description": "非法活动",
            "severity": "critical"
        },
        "self_harm": {
            "patterns": [
                r"自杀", r"自残",
                r"suicide", r"self-harm"
            ],
            "description": "自残内容",
            "severity": "critical"
        },
        "sexual": {
            "patterns": [
                r"色情", r"成人内容",
                r"pornography", r"adult content"
            ],
            "description": "色情内容",
            "severity": "high"
        }
    }

    def __init__(self):
        self.compiled_patterns = {}
        for category, config in self.HARMFUL_PATTERNS.items():
            self.compiled_patterns[category] = [
                (re.compile(p, re.IGNORECASE), config["description"], config["severity"])
                for p in config["patterns"]
            ]

    def detect(self, text: str) -> List[Dict]:
        """检测有害内容"""
        findings = []

        for category, patterns in self.compiled_patterns.items():
            for pattern, description, severity in patterns:
                matches = pattern.findall(text)
                if matches:
                    findings.append({
                        "category": category,
                        "type": description,
                        "severity": severity,
                        "matches": matches[:3]  # 限制显示数量
                    })

        return findings

    def get_severity_level(self, findings: List[Dict]) -> str:
        """获取最高严重等级"""
        if not findings:
            return "none"

        severity_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        max_severity = "none"

        for finding in findings:
            if severity_order.get(finding["severity"], 0) > severity_order.get(max_severity, -1):
                max_severity = finding["severity"]

        return max_severity


# ==================== PII 检测与脱敏 ====================

class PIIProtector:
    """PII 检测与保护器"""

    # PII 类型定义
    PII_TYPES = {
        "phone": {
            "pattern": r'(?:\+?86)?1[3-9]\d{9}',
            "name": "手机号码",
            "mask_func": lambda m: m.group()[:3] + "****" + m.group()[-4:]
        },
        "email": {
            "pattern": r'[\w\.-]+@[\w\.-]+\.\w+',
            "name": "电子邮箱",
            "mask_func": lambda m: m.group().split('@')[0][:2] + "***@" + m.group().split('@')[1]
        },
        "id_card_cn": {
            "pattern": r'\d{17}[\dXx]',
            "name": "身份证号",
            "mask_func": lambda m: m.group()[:6] + "********" + m.group()[-4:]
        },
        "bank_card": {
            "pattern": r'\d{16,19}',
            "name": "银行卡号",
            "mask_func": lambda m: m.group()[:4] + "****" + m.group()[-4:]
        },
        "credit_card": {
            "pattern": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "name": "信用卡号",
            "mask_func": lambda m: "****-****-****-" + m.group()[-4:]
        },
        "ip_address": {
            "pattern": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "name": "IP地址",
            "mask_func": lambda m: "***.***.***." + m.group().split('.')[-1]
        },
        "passport": {
            "pattern": r'[A-Z]{1,2}\d{6,9}',
            "name": "护照号",
            "mask_func": lambda m: m.group()[0] + "******"
        }
    }

    def __init__(self):
        self.compiled_patterns = {}
        for pii_type, config in self.PII_TYPES.items():
            self.compiled_patterns[pii_type] = {
                "pattern": re.compile(config["pattern"]),
                "name": config["name"],
                "mask_func": config["mask_func"]
            }

    def detect(self, text: str) -> List[Dict]:
        """检测 PII"""
        findings = []

        for pii_type, config in self.compiled_patterns.items():
            matches = config["pattern"].findall(text)
            if matches:
                findings.append({
                    "type": pii_type,
                    "name": config["name"],
                    "count": len(matches),
                    "sample": matches[0] if matches else None
                })

        return findings

    def anonymize(self, text: str) -> Tuple[str, List[Dict]]:
        """脱敏处理"""
        findings = []
        anonymized = text

        for pii_type, config in self.compiled_patterns.items():
            pattern = config["pattern"]
            mask_func = config["mask_func"]

            matches = pattern.findall(text)
            if matches:
                findings.append({
                    "type": pii_type,
                    "name": config["name"],
                    "count": len(matches)
                })
                anonymized = pattern.sub(mask_func, anonymized)

        return anonymized, findings

    def redact(self, text: str, replacement: str = "[已删除]") -> Tuple[str, List[Dict]]:
        """完全删除 PII"""
        findings = []
        redacted = text

        for pii_type, config in self.compiled_patterns.items():
            pattern = config["pattern"]
            matches = pattern.findall(text)

            if matches:
                findings.append({
                    "type": pii_type,
                    "name": config["name"],
                    "count": len(matches)
                })
                redacted = pattern.sub(replacement, redacted)

        return redacted, findings


# ==================== 输出格式验证器 ====================

class OutputFormatValidator:
    """输出格式验证器"""

    @staticmethod
    def validate_json(text: str) -> Tuple[bool, Optional[Dict]]:
        """验证 JSON 格式"""
        try:
            # 尝试提取 JSON 块
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                data = json.loads(json_match.group(1))
                return True, data

            # 直接解析
            data = json.loads(text)
            return True, data
        except json.JSONDecodeError:
            return False, None

    @staticmethod
    def validate_markdown(text: str) -> bool:
        """验证 Markdown 格式"""
        # 检查基本 Markdown 元素
        has_headers = bool(re.search(r'^#+\s', text, re.MULTILINE))
        has_lists = bool(re.search(r'^[-*]\s', text, re.MULTILINE))
        has_code = bool(re.search(r'```', text))

        return has_headers or has_lists or has_code

    @staticmethod
    def validate_url(text: str) -> List[str]:
        """提取并验证 URL"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls

    @staticmethod
    def validate_code_blocks(text: str) -> List[Dict]:
        """提取并验证代码块"""
        pattern = r'```(\w+)?\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, text)

        code_blocks = []
        for lang, code in matches:
            code_blocks.append({
                "language": lang or "unknown",
                "code": code.strip(),
                "lines": len(code.strip().split('\n'))
            })

        return code_blocks


# ==================== 敏感信息检测器 ====================

class SensitiveInfoDetector:
    """敏感信息检测器"""

    # 敏感信息模式
    SENSITIVE_PATTERNS = {
        "api_key": {
            "pattern": r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?[\w-]{20,}["\']?',
            "name": "API密钥",
            "severity": "critical"
        },
        "password": {
            "pattern": r'(?:password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']{8,}["\']?',
            "name": "密码",
            "severity": "critical"
        },
        "token": {
            "pattern": r'(?:token|bearer)\s*[=:]\s*["\']?[\w.-]{20,}["\']?',
            "name": "访问令牌",
            "severity": "critical"
        },
        "secret": {
            "pattern": r'(?:secret|private[_-]?key)\s*[=:]\s*["\']?[\w-]{16,}["\']?',
            "name": "密钥/私钥",
            "severity": "critical"
        },
        "database_url": {
            "pattern": r'(?:mysql|postgres|mongodb|redis)://[^\s]+',
            "name": "数据库连接",
            "severity": "high"
        },
        "aws_key": {
            "pattern": r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}',
            "name": "AWS密钥",
            "severity": "critical"
        }
    }

    def __init__(self):
        self.compiled_patterns = {}
        for key, config in self.SENSITIVE_PATTERNS.items():
            self.compiled_patterns[key] = {
                "pattern": re.compile(config["pattern"], re.IGNORECASE),
                "name": config["name"],
                "severity": config["severity"]
            }

    def detect(self, text: str) -> List[Dict]:
        """检测敏感信息"""
        findings = []

        for key, config in self.compiled_patterns.items():
            matches = config["pattern"].findall(text)
            if matches:
                findings.append({
                    "type": key,
                    "name": config["name"],
                    "severity": config["severity"],
                    "count": len(matches),
                    "sample": matches[0][:20] + "..." if matches else None
                })

        return findings


# ==================== 综合输出检查器 ====================

class OutputChecker:
    """综合输出检查器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self.harmful_detector = HarmfulContentDetector()
        self.pii_protector = PIIProtector()
        self.format_validator = OutputFormatValidator()
        self.sensitive_detector = SensitiveInfoDetector()

        # 配置选项
        self.mask_pii = self.config.get("mask_pii", True)
        self.filter_harmful = self.config.get("filter_harmful", True)
        self.check_sensitive = self.config.get("check_sensitive", True)

    def check(self, output: str) -> OutputCheckResult:
        """综合检查输出"""
        all_issues = []
        categories = []
        requires_review = False

        # 1. 有害内容检查
        if self.filter_harmful:
            harmful_findings = self.harmful_detector.detect(output)
            if harmful_findings:
                categories.append(ContentCategory.HARMFUL)
                all_issues.extend([
                    {"type": "harmful", **finding}
                    for finding in harmful_findings
                ])
                # 高严重性需要人工审核
                if any(f["severity"] in ["critical", "high"] for f in harmful_findings):
                    requires_review = True

        # 2. PII 检查
        pii_findings = self.pii_protector.detect(output)
        if pii_findings:
            categories.append(ContentCategory.PII)
            all_issues.extend([
                {"type": "pii", **finding}
                for finding in pii_findings
            ])

        # 3. 敏感信息检查
        if self.check_sensitive:
            sensitive_findings = self.sensitive_detector.detect(output)
            if sensitive_findings:
                categories.append(ContentCategory.SENSITIVE)
                all_issues.extend([
                    {"type": "sensitive", **finding}
                    for finding in sensitive_findings
                ])
                requires_review = True

        # 判断是否安全
        is_safe = (
            ContentCategory.HARMFUL not in categories and
            ContentCategory.SENSITIVE not in categories and
            not any(
                issue.get("severity") in ["critical", "high"]
                for issue in all_issues
            )
        )

        # 生成清洗后的输出
        sanitized = output
        if pii_findings and self.mask_pii:
            sanitized, _ = self.pii_protector.anonymize(output)

        return OutputCheckResult(
            is_safe=is_safe,
            categories=list(set(categories)),
            issues=all_issues,
            original_output=output,
            sanitized_output=sanitized if sanitized != output else None,
            requires_review=requires_review
        )

    def sanitize(self, output: str) -> str:
        """清洗输出"""
        result = self.check(output)

        if result.sanitized_output:
            return result.sanitized_output
        return output

    def safe_output(self, output: str, on_unsafe: str = "block") -> str:
        """获取安全输出"""
        result = self.check(output)

        if result.is_safe:
            return result.sanitized_output or output

        if on_unsafe == "block":
            return "[内容已因安全原因被屏蔽]"
        elif on_unsafe == "mask":
            return self._mask_unsafe_content(output, result.issues)
        elif on_unsafe == "warn":
            return f"[警告：内容可能包含不安全信息]\n{output}"

        return output

    def _mask_unsafe_content(self, text: str, issues: List[Dict]) -> str:
        """屏蔽不安全内容"""
        masked = text

        for issue in issues:
            if issue["type"] == "pii":
                # PII 脱敏
                masked, _ = self.pii_protector.anonymize(masked)
            elif issue["type"] == "sensitive":
                # 敏感信息替换
                if "sample" in issue and issue["sample"]:
                    masked = masked.replace(issue["sample"], "[已删除]")

        return masked


# ==================== 内容审核策略 ====================

class ContentModerationPolicy:
    """内容审核策略"""

    def __init__(self):
        self.policies = {
            "strict": self._strict_policy,
            "balanced": self._balanced_policy,
            "permissive": self._permissive_policy
        }
        self.default_policy = "balanced"

    def _strict_policy(self, result: OutputCheckResult) -> Tuple[bool, str]:
        """严格策略"""
        if result.issues:
            return False, "内容不符合安全要求，已被拒绝"
        return True, "内容通过审核"

    def _balanced_policy(self, result: OutputCheckResult) -> Tuple[bool, str]:
        """平衡策略"""
        critical_issues = [i for i in result.issues if i.get("severity") == "critical"]
        if critical_issues:
            return False, "内容包含严重安全问题，已被拒绝"

        if result.requires_review:
            return True, "内容需要人工审核"

        return True, "内容通过审核"

    def _permissive_policy(self, result: OutputCheckResult) -> Tuple[bool, str]:
        """宽松策略"""
        critical_issues = [i for i in result.issues if i.get("severity") == "critical"]
        if critical_issues:
            return True, "警告：内容包含敏感信息"

        return True, "内容通过审核"

    def apply(self, result: OutputCheckResult, policy: str = None) -> Tuple[bool, str]:
        """应用审核策略"""
        policy_func = self.policies.get(policy or self.default_policy, self._balanced_policy)
        return policy_func(result)


# ==================== 演示函数 ====================

def demo_harmful_detection():
    """演示有害内容检测"""
    print("\n" + "=" * 60)
    print("示例1: 有害内容检测")
    print("=" * 60)

    detector = HarmfulContentDetector()

    test_outputs = [
        "这是一段正常的产品介绍文本。",
        "这个问题涉及暴力内容和仇恨言论。",
        "根据相关法律法规，此类毒品相关的信息不能提供。",
    ]

    for output in test_outputs:
        print(f"\n输出: {output}")
        findings = detector.detect(output)

        if findings:
            print(f"检测结果: 发现 {len(findings)} 个问题")
            for f in findings:
                print(f"  - {f['type']}: {f['matches']}")
        else:
            print("检测结果: 内容安全")


def demo_pii_protection():
    """演示 PII 保护"""
    print("\n" + "=" * 60)
    print("示例2: PII 保护")
    print("=" * 60)

    protector = PIIProtector()

    test_output = """
    用户信息：
    姓名：张三
    手机：13812345678
    邮箱：zhangsan@example.com
    身份证：110101199001011234
    银行卡：6222021234567890123
    """

    print("原始输出:")
    print(test_output)

    # 检测 PII
    findings = protector.detect(test_output)
    print("\n检测到的 PII:")
    for f in findings:
        print(f"  - {f['name']}: {f['count']} 个")

    # 脱敏处理
    anonymized, _ = protector.anonymize(test_output)
    print("\n脱敏后:")
    print(anonymized)


def demo_sensitive_info():
    """演示敏感信息检测"""
    print("\n" + "=" * 60)
    print("示例3: 敏感信息检测")
    print("=" * 60)

    detector = SensitiveInfoDetector()

    test_output = """
    配置文件：
    API_KEY=sk-1234567890abcdef1234567890abcdef
    DATABASE_URL=postgres://user:pass@localhost:5432/db
    AWS_KEY=AKIAIOSFODNN7EXAMPLE
    """

    print("检测内容:")
    print(test_output)

    findings = detector.detect(test_output)
    print("\n检测到的敏感信息:")
    for f in findings:
        print(f"  - {f['name']}: 严重性 {f['severity']}")


def demo_format_validation():
    """演示格式验证"""
    print("\n" + "=" * 60)
    print("示例4: 输出格式验证")
    print("=" * 60)

    validator = OutputFormatValidator()

    # JSON 验证
    json_output = '''
```json
{"name": "产品A", "price": 99.9, "category": "电子产品"}
```
'''
    is_json, data = validator.validate_json(json_output)
    print(f"JSON 验证: {'有效' if is_json else '无效'}")
    if data:
        print(f"解析数据: {data}")

    # 代码块提取
    code_output = """
```python
def hello():
    print("Hello World")
```
"""
    code_blocks = validator.validate_code_blocks(code_output)
    print(f"\n代码块数量: {len(code_blocks)}")
    for block in code_blocks:
        print(f"  语言: {block['language']}, 行数: {block['lines']}")


def demo_comprehensive_check():
    """演示综合检查"""
    print("\n" + "=" * 60)
    print("示例5: 综合输出检查")
    print("=" * 60)

    checker = OutputChecker()

    test_outputs = [
        "这是一段正常的产品介绍。",
        "联系方式：手机13812345678，邮箱test@example.com",
        "API密钥是sk-1234567890abcdef1234567890abcdef",
    ]

    for output in test_outputs:
        print(f"\n{'─' * 50}")
        print(f"检查输出: {output[:50]}...")

        result = checker.check(output)

        print(f"安全状态: {'安全' if result.is_safe else '不安全'}")
        print(f"内容类别: {[c.value for c in result.categories]}")
        print(f"需要审核: {'是' if result.requires_review else '否'}")

        if result.issues:
            print("发现问题:")
            for issue in result.issues:
                print(f"  - {issue['type']}: {issue.get('name', 'N/A')}")

        if result.sanitized_output:
            print(f"清洗后: {result.sanitized_output[:50]}...")


def demo_content_moderation():
    """演示内容审核策略"""
    print("\n" + "=" * 60)
    print("示例6: 内容审核策略")
    print("=" * 60)

    checker = OutputChecker()
    policy = ContentModerationPolicy()

    test_output = "用户手机号是13812345678，邮箱是test@example.com"

    result = checker.check(test_output)

    print(f"输出: {test_output}")
    print(f"检测问题: {len(result.issues)} 个")

    for policy_name in ["strict", "balanced", "permissive"]:
        approved, message = policy.apply(result, policy_name)
        print(f"\n{policy_name}策略: {'通过' if approved else '拒绝'}")
        print(f"消息: {message}")


def demo_safe_response():
    """演示安全响应处理"""
    print("\n" + "=" * 60)
    print("示例7: 安全响应处理流程")
    print("=" * 60)

    checker = OutputChecker()

    # 模拟一个包含敏感信息的响应
    raw_response = """
    用户查询结果：
    姓名：李四
    手机：13987654321
    邮箱：lisi@company.com

    建议：可以考虑购买我们的高级服务。
    """

    print("原始响应:")
    print(raw_response)

    # 检查并清洗
    safe_response = checker.safe_output(raw_response, on_unsafe="mask")
    print("\n安全处理后的响应:")
    print(safe_response)


def demo_llm_output_check():
    """演示 LLM 输出检查"""
    print("\n" + "=" * 60)
    print("示例8: LLM 输出安全检查")
    print("=" * 60)

    checker = OutputChecker()

    prompt = "请生成一段包含联系方式的客服回复示例"

    print(f"提示词: {prompt}")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    raw_output = response.choices[0].message.content
    print(f"\n原始输出:\n{raw_output}")

    # 安全检查
    result = checker.check(raw_output)
    print(f"\n安全状态: {'安全' if result.is_safe else '需要处理'}")

    if result.issues:
        print(f"检测到的问题: {len(result.issues)}")
        for issue in result.issues:
            print(f"  - {issue}")

    if result.sanitized_output:
        print(f"\n清洗后输出:\n{result.sanitized_output}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 13 - 输出检查与内容安全")
    print("=" * 60)

    # 运行所有演示
    demo_harmful_detection()
    demo_pii_protection()
    demo_sensitive_info()
    demo_format_validation()
    demo_comprehensive_check()
    demo_content_moderation()
    demo_safe_response()
    demo_llm_output_check()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)