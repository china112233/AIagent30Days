# day25/privacy_computing.py
"""
Day 25: 隐私计算实践示例

本文件演示隐私保护的计算方法，包括：
1. 差分隐私 - 数据发布时的噪声添加
2. 数据脱敏 - 敏感信息识别和替换
3. 安全计算 - 基础加密技术应用
4. 隐私审计 - 隐私风险评估

依赖安装：
pip install pydp differential-privacy numpy pandas pycryptodome
"""

import os
import re
import json
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================
# 隐私配置
# ============================================================

@dataclass
class PrivacyConfig:
    """隐私计算配置"""
    # 差分隐私参数
    epsilon: float = 1.0          # 隐私预算
    delta: float = 1e-5           # 失败概率
    sensitivity: float = 1.0      # 数据敏感度
    
    # 脱敏配置
    pii_patterns: Dict[str, str] = None
    replacement_template: str = "[<{field}>]"
    
    # 加密配置
    encryption_key: str = None    # 加密密钥
    hash_salt: str = "privacy_salt"
    
    def __post_init__(self):
        if self.pii_patterns is None:
            self.pii_patterns = {
                'phone': r'\b\d{11}\b',
                'email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
                'id_card': r'\b\d{17}[\dXx]\b',
                'credit_card': r'\b\d{16}\b',
                'bank_account': r'\b\d{10,20}\b',
                'name': r'姓名[:：]\s*[^\s]+',
                'address': r'地址[:：]\s*[^\s]+',
            }


# ============================================================
# 差分隐私
# ============================================================

class DifferentialPrivacy:
    """
    差分隐私实现
    通过添加噪声保护个体隐私
    """
    
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
    
    def add_laplace_noise(self, value: float) -> float:
        """
        Laplace 机制
        用于数值型数据的差分隐私保护
        
        Args:
            value: 原始数值
            
        Returns:
            加噪声后的数值
        """
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value: float, delta: float = 1e-5) -> float:
        """
        Gaussian 机制
        用于数值型数据的差分隐私保护
        
        Args:
            value: 原始数值
            delta: 失败概率
            
        Returns:
            加噪声后的数值
        """
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def randomize_count(self, count: int) -> int:
        """
        计数随机化
        用于统计计数的差分隐私保护
        
        Args:
            count: 原始计数
            
        Returns:
            加噪声后的计数
        """
        noisy_count = self.add_laplace_noise(count)
        # 计数必须为非负整数
        return max(0, int(round(noisy_count)))
    
    def randomize_mean(self, values: List[float]) -> float:
        """
        平均值随机化
        用于统计均值的差分隐私保护
        
        Args:
            values: 数值列表
            
        Returns:
            加噪声后的均值
        """
        true_mean = np.mean(values)
        # 均值的敏感度 = max_value / n
        n = len(values)
        if n > 0:
            self.sensitivity = max(values) - min(values) / n
        return self.add_laplace_noise(true_mean)
    
    def randomized_response(self, answer: bool) -> bool:
        """
        随机应答机制
        用于二值数据的差分隐私保护
        
        Args:
            answer: 原始回答（True/False）
            
        Returns:
            随机化后的回答
        """
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        if np.random.random() < p:
            return answer  # 返回真实答案
        else:
            return not answer  # 返回相反答案
    
    def privacy_loss(self) -> float:
        """计算隐私损失"""
        return self.epsilon
    
    def budget_remaining(self, total_budget: float, spent: float) -> float:
        """
        计算剩余隐私预算
        
        Args:
            total_budget: 总隐私预算
            spent: 已使用的隐私预算
            
        Returns:
            剩余隐私预算
        """
        return total_budget - spent


# ============================================================
# 数据脱敏
# ============================================================

class DataSanitizer:
    """
    数据脱敏器
    识别并替换敏感信息
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.detected_pii = []
    
    def sanitize_text(self, text: str) -> str:
        """
        文本脱敏
        识别并替换文本中的敏感信息
        
        Args:
            text: 原始文本
            
        Returns:
            脱敏后的文本
        """
        sanitized = text
        self.detected_pii = []
        
        for field, pattern in self.config.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    self.detected_pii.append({
                        'field': field,
                        'original': match,
                        'position': text.find(match)
                    })
                replacement = self.config.replacement_template.format(field=field)
                sanitized = re.sub(pattern, replacement, sanitized)
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        字典数据脱敏
        
        Args:
            data: 原始字典数据
            
        Returns:
            脱敏后的字典数据
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # 检查是否是敏感字段
                if any(pii_key in key.lower() for pii_key in ['name', 'phone', 'email', 'address', 'id']):
                    sanitized[key] = self.config.replacement_template.format(field=key)
                else:
                    sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def mask_partial(self, text: str, visible_ratio: float = 0.3) -> str:
        """
        部分掩码
        保留部分信息可见
        
        Args:
            text: 原始文本
            visible_ratio: 可见比例
            
        Returns:
            部分掩码后的文本
        """
        if len(text) <= 2:
            return text[0] + '*'
        
        visible_len = int(len(text) * visible_ratio)
        masked_len = len(text) - visible_len
        
        # 显示前部分，隐藏后部分
        return text[:visible_len] + '*' * masked_len
    
    def hash_pii(self, text: str, salt: str = None) -> str:
        """
        PII 哈希化
        对敏感信息进行不可逆哈希
        
        Args:
            text: 原始文本
            salt: 哈希盐值
            
        Returns:
            哈希后的字符串
        """
        if salt is None:
            salt = self.config.hash_salt
        
        combined = text + salt
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get_detected_pii(self) -> List[Dict[str, Any]]:
        """获取检测到的 PII 信息"""
        return self.detected_pii


# ============================================================
# 安全计算
# ============================================================

class SecureComputation:
    """
    安全计算实现
    基础加密和哈希技术
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
    
    def simple_encrypt(self, data: str) -> str:
        """
        简单加密（演示用）
        实际应用应使用专业加密库
        
        Args:
            data: 原始数据
            
        Returns:
            加密后的数据
        """
        try:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import pad
            
            key = self.config.encryption_key or "default_key_16b"
            key = key.encode()[:16].ljust(16, b'0')
            
            cipher = AES.new(key, AES.MODE_ECB)
            encrypted = cipher.encrypt(pad(data.encode(), AES.block_size))
            return encrypted.hex()
            
        except ImportError:
            # 简单替代方案
            return self._simple_xor_encrypt(data)
    
    def _simple_xor_encrypt(self, data: str) -> str:
        """简单 XOR 加密（仅演示）"""
        key = self.config.encryption_key or "simple_key"
        encrypted = []
        for i, char in enumerate(data):
            encrypted.append(ord(char) ^ ord(key[i % len(key)]))
        return bytes(encrypted).hex()
    
    def simple_decrypt(self, encrypted_data: str) -> str:
        """
        简单解密
        
        Args:
            encrypted_data: 加密数据
            
        Returns:
            解密后的数据
        """
        try:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import unpad
            
            key = self.config.encryption_key or "default_key_16b"
            key = key.encode()[:16].ljust(16, b'0')
            
            cipher = AES.new(key, AES.MODE_ECB)
            decrypted = unpad(cipher.decrypt(bytes.fromhex(encrypted_data)), AES.block_size)
            return decrypted.decode()
            
        except ImportError:
            return self._simple_xor_decrypt(encrypted_data)
    
    def _simple_xor_decrypt(self, encrypted_data: str) -> str:
        """简单 XOR 解密"""
        key = self.config.encryption_key or "simple_key"
        decrypted = []
        encrypted_bytes = bytes.fromhex(encrypted_data)
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(chr(byte ^ ord(key[i % len(key)])))
        return ''.join(decrypted)
    
    def compute_hash(self, data: str) -> str:
        """
        计算哈希值
        
        Args:
            data: 原始数据
            
        Returns:
            哈希值
        """
        return hashlib.sha256(data.encode()).hexdigest()
    
    def secure_compare(self, hash1: str, hash2: str) -> bool:
        """
        安全比较
        用于密码验证等场景
        
        Args:
            hash1: 第一个哈希
            hash2: 第二个哈希
            
        Returns:
            是否相等
        """
        # 防止时序攻击
        if len(hash1) != len(hash2):
            return False
        
        result = 0
        for a, b in zip(hash1, hash2):
            result |= ord(a) ^ ord(b)
        
        return result == 0


# ============================================================
# 隐私审计
# ============================================================

class PrivacyAuditor:
    """
    隐私风险评估器
    评估数据集和系统的隐私风险
    """
    
    def __init__(self):
        self.risk_factors = {
            'pii_presence': [],
            'quasi_identifiers': [],
            'sensitive_attributes': [],
            're_identification_risk': 0,
        }
    
    def assess_dataset(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估数据集隐私风险
        
        Args:
            sample_data: 数据样本
            
        Returns:
            风险评估报告
        """
        self.risk_factors = {
            'pii_presence': [],
            'quasi_identifiers': [],
            'sensitive_attributes': [],
            're_identification_risk': 0,
        }
        
        # 检测 PII 字段
        pii_keywords = ['name', 'phone', 'email', 'address', 'ssn', 'id', 'passport']
        quasi_keywords = ['age', 'gender', 'zip', 'city', 'occupation']
        sensitive_keywords = ['health', 'income', 'religion', 'political', 'race']
        
        for field in sample_data.keys():
            field_lower = field.lower()
            
            if any(kw in field_lower for kw in pii_keywords):
                self.risk_factors['pii_presence'].append(field)
            
            if any(kw in field_lower for kw in quasi_keywords):
                self.risk_factors['quasi_identifiers'].append(field)
            
            if any(kw in field_lower for kw in sensitive_keywords):
                self.risk_factors['sensitive_attributes'].append(field)
        
        # 计算重识别风险
        num_quasi = len(self.risk_factors['quasi_identifiers'])
        self.risk_factors['re_identification_risk'] = min(num_quasi * 20, 100)
        
        # 计算总风险分数
        risk_score = self._calculate_risk_score()
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'pii_fields': self.risk_factors['pii_presence'],
            'quasi_identifiers': self.risk_factors['quasi_identifiers'],
            'sensitive_attributes': self.risk_factors['sensitive_attributes'],
            're_identification_risk': self.risk_factors['re_identification_risk'],
            'recommendations': self._generate_recommendations(),
        }
    
    def _calculate_risk_score(self) -> float:
        """计算风险分数"""
        score = 0
        
        # PII 字段风险
        score += len(self.risk_factors['pii_presence']) * 30
        
        # 准标识符风险
        score += len(self.risk_factors['quasi_identifiers']) * 15
        
        # 敏感属性风险
        score += len(self.risk_factors['sensitive_attributes']) * 20
        
        # 重识别风险
        score += self.risk_factors['re_identification_risk']
        
        return min(score, 100)
    
    def _get_risk_level(self, score: float) -> str:
        """获取风险等级"""
        if score >= 70:
            return '高风险'
        elif score >= 40:
            return '中等风险'
        else:
            return '低风险'
    
    def _generate_recommendations(self) -> List[str]:
        """生成隐私保护建议"""
        recommendations = []
        
        if self.risk_factors['pii_presence']:
            recommendations.append('建议对 PII 字段进行脱敏或加密处理')
        
        if self.risk_factors['quasi_identifiers']:
            recommendations.append('建议使用 k-匿名化处理准标识符')
        
        if self.risk_factors['sensitive_attributes']:
            recommendations.append('建议使用 l-多样性保护敏感属性')
        
        if self.risk_factors['re_identification_risk'] > 50:
            recommendations.append('重识别风险较高，建议加强隐私保护措施')
        
        if not recommendations:
            recommendations.append('当前数据集隐私风险较低，但仍需定期审计')
        
        return recommendations
    
    def assess_text(self, text: str) -> Dict[str, Any]:
        """
        评估文本隐私风险
        
        Args:
            text: 文本内容
            
        Returns:
            风险评估报告
        """
        config = PrivacyConfig()
        sanitizer = DataSanitizer(config)
        
        sanitized = sanitizer.sanitize_text(text)
        detected = sanitizer.get_detected_pii()
        
        return {
            'risk_score': len(detected) * 25,
            'risk_level': self._get_risk_level(len(detected) * 25),
            'detected_pii': detected,
            'sanitized_text': sanitized,
            'recommendations': ['建议对检测到的 PII 进行脱敏处理'] if detected else []
        }


# ============================================================
# 隐私保护管道
# ============================================================

class PrivacyPreservingPipeline:
    """
    隐私保护数据处理管道
    整合所有隐私保护技术
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.dp = DifferentialPrivacy(config.epsilon, config.sensitivity)
        self.sanitizer = DataSanitizer(config)
        self.computation = SecureComputation(config)
        self.auditor = PrivacyAuditor()
    
    def process(self, data: Dict[str, Any], operations: List[str] = None) -> Dict[str, Any]:
        """
        处理数据
        
        Args:
            data: 输入数据
            operations: 要执行的操作列表
            
        Returns:
            处理后的数据
        """
        if operations is None:
            operations = ['audit', 'sanitize', 'encrypt']
        
        result = {'original': data.copy(), 'processed': {}, 'report': {}}
        
        # 隐私审计
        if 'audit' in operations:
            audit_report = self.auditor.assess_dataset(data)
            result['report']['audit'] = audit_report
        
        # 数据脱敏
        if 'sanitize' in operations:
            sanitized = self.sanitizer.sanitize_dict(data)
            result['processed']['sanitized'] = sanitized
            result['report']['pii_detected'] = self.sanitizer.get_detected_pii()
        
        # 加密存储
        if 'encrypt' in operations:
            to_encrypt = result['processed'].get('sanitized', data)
            encrypted = {}
            for key, value in to_encrypt.items():
                if isinstance(value, str) and value:
                    encrypted[key] = self.computation.simple_encrypt(value)
                else:
                    encrypted[key] = value
            result['processed']['encrypted'] = encrypted
        
        # 差分隐私统计
        if 'dp_statistics' in operations:
            # 对数值型数据应用差分隐私
            dp_values = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    dp_values[key] = self.dp.add_laplace_noise(value)
            result['processed']['dp_statistics'] = dp_values
        
        return result


# ============================================================
# 使用示例
# ============================================================

def demo_privacy_computing():
    """演示隐私计算流程"""
    print("=" * 60)
    print("Day 25: 隐私计算演示")
    print("=" * 60)
    
    # 1. 差分隐私演示
    print("\n1. 差分隐私演示:")
    dp = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0)
    
    original_count = 100
    noisy_count = dp.randomize_count(original_count)
    print(f"   原始计数: {original_count}")
    print(f"   加噪声后: {noisy_count}")
    print(f"   隐私损失: ε = {dp.privacy_loss()}")
    
    # 多次测试均值
    values = [10, 20, 30, 40, 50]
    true_mean = np.mean(values)
    noisy_mean = dp.randomize_mean(values)
    print(f"   原始均值: {true_mean}")
    print(f"   加噪声后: {noisy_mean:.2f}")
    
    # 随机应答
    answer = True
    randomized = dp.randomized_response(answer)
    print(f"   原始回答: {answer}")
    print(f"   随机应答: {randomized}")
    
    # 2. 数据脱敏演示
    print("\n2. 数据脱敏演示:")
    config = PrivacyConfig()
    sanitizer = DataSanitizer(config)
    
    sensitive_text = "用户张三的手机号是13812345678，邮箱是zhangsan@example.com，身份证号是12345678901234567X"
    sanitized = sanitizer.sanitize_text(sensitive_text)
    print(f"   原始文本: {sensitive_text}")
    print(f"   脱敏后: {sanitized}")
    
    detected = sanitizer.get_detected_pii()
    print(f"   检测到的 PII: {len(detected)} 处")
    for item in detected:
        print(f"     - {item['field']}: {item['original'][:3]}...")
    
    # 部分掩码
    phone = "13812345678"
    masked = sanitizer.mask_partial(phone, visible_ratio=0.3)
    print(f"   部分掩码: {phone} -> {masked}")
    
    # 哈希化
    hashed = sanitizer.hash_pii(phone)
    print(f"   哈希化: {phone} -> {hashed}")
    
    # 3. 安全计算演示
    print("\n3. 安全计算演示:")
    computation = SecureComputation(config)
    
    secret_data = "敏感信息需要加密保护"
    encrypted = computation.simple_encrypt(secret_data)
    decrypted = computation.simple_decrypt(encrypted)
    print(f"   原始数据: {secret_data}")
    print(f"   加密后: {encrypted[:32]}...")
    print(f"   解密后: {decrypted}")
    
    # 哈希验证
    hash1 = computation.compute_hash("password123")
    hash2 = computation.compute_hash("password123")
    is_match = computation.secure_compare(hash1, hash2)
    print(f"   哈希验证: {is_match}")
    
    # 4. 隐私审计演示
    print("\n4. 隐私审计演示:")
    auditor = PrivacyAuditor()
    
    sample_data = {
        'name': '张三',
        'phone': '13812345678',
        'email': 'zhangsan@example.com',
        'age': 35,
        'gender': 'male',
        'city': '北京',
        'income': 50000,
        'health_status': '良好'
    }
    
    audit_report = auditor.assess_dataset(sample_data)
    print(f"   风险分数: {audit_report['risk_score']}")
    print(f"   风险等级: {audit_report['risk_level']}")
    print(f"   PII 字段: {audit_report['pii_fields']}")
    print(f"   准标识符: {audit_report['quasi_identifiers']}")
    print(f"   敏感属性: {audit_report['sensitive_attributes']}")
    print(f"   建议: {audit_report['recommendations']}")
    
    # 5. 隐私保护管道演示
    print("\n5. 隐私保护管道演示:")
    pipeline = PrivacyPreservingPipeline(config)
    
    result = pipeline.process(sample_data)
    print(f"   原始数据: {result['original']}")
    print(f"   脱敏数据: {result['processed']['sanitized']}")
    print(f"   审计报告: 风险分数={result['report']['audit']['risk_score']}")
    
    print("\n" + "=" * 60)
    print("隐私计算演示完成!")
    print("=" * 60)
    print("\n关键点:")
    print("  • 差分隐私通过添加噪声保护个体")
    print("  • 数据脱敏识别并替换敏感信息")
    print("  • 安全计算使用加密和哈希技术")
    print("  • 隐私审计评估和监控风险")


if __name__ == "__main__":
    demo_privacy_computing()