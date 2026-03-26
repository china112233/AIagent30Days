"""
权限控制系统
演示如何实现 Agent 的权限管理和操作控制

核心功能：
1. 角色权限定义 - 定义不同角色的权限范围
2. 操作权限检查 - 检查操作是否被允许
3. 资源访问控制 - 控制对资源的访问
4. 审计日志记录 - 记录所有操作和决策
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from collections import defaultdict
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


# ==================== 权限定义 ====================

class Permission(Enum):
    """权限枚举"""
    # 读取权限
    READ_PUBLIC = auto()       # 读取公开数据
    READ_PRIVATE = auto()      # 读取私有数据
    READ_SYSTEM = auto()       # 读取系统配置

    # 写入权限
    WRITE_OWN = auto()         # 写入自己的数据
    WRITE_OTHERS = auto()      # 写入他人的数据
    WRITE_SYSTEM = auto()      # 写入系统配置

    # 执行权限
    EXECUTE_SAFE = auto()      # 执行安全操作
    EXECUTE_UNSAFE = auto()    # 执行危险操作
    EXECUTE_SYSTEM = auto()    # 执行系统命令

    # 管理权限
    MANAGE_USERS = auto()      # 管理用户
    MANAGE_ROLES = auto()      # 管理角色
    MANAGE_SYSTEM = auto()     # 管理系统

    # 特殊权限
    BYPASS_FILTER = auto()     # 绕过内容过滤
    ACCESS_SECRETS = auto()    # 访问敏感信息
    UNLIMITED = auto()         # 无限制


class Role(Enum):
    """角色枚举"""
    GUEST = "guest"           # 访客
    USER = "user"             # 普通用户
    POWER_USER = "power_user" # 高级用户
    ADMIN = "admin"           # 管理员
    SUPER_ADMIN = "super_admin"  # 超级管理员


# 角色权限映射
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.GUEST: {
        Permission.READ_PUBLIC,
    },
    Role.USER: {
        Permission.READ_PUBLIC,
        Permission.READ_PRIVATE,
        Permission.WRITE_OWN,
        Permission.EXECUTE_SAFE,
    },
    Role.POWER_USER: {
        Permission.READ_PUBLIC,
        Permission.READ_PRIVATE,
        Permission.WRITE_OWN,
        Permission.EXECUTE_SAFE,
        Permission.EXECUTE_UNSAFE,
    },
    Role.ADMIN: {
        Permission.READ_PUBLIC,
        Permission.READ_PRIVATE,
        Permission.READ_SYSTEM,
        Permission.WRITE_OWN,
        Permission.WRITE_OTHERS,
        Permission.WRITE_SYSTEM,
        Permission.EXECUTE_SAFE,
        Permission.EXECUTE_UNSAFE,
        Permission.MANAGE_USERS,
    },
    Role.SUPER_ADMIN: {
        Permission.UNLIMITED,  # 拥有所有权限
    }
}


# ==================== 操作定义 ====================

class OperationType(Enum):
    """操作类型"""
    # 数据操作
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"

    # 代码执行
    EXECUTE_CODE = "execute_code"
    EXECUTE_SHELL = "execute_shell"

    # 文件操作
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"

    # 网络操作
    HTTP_GET = "http_get"
    HTTP_POST = "http_post"

    # 系统操作
    SYSTEM_CONFIG = "system_config"
    USER_MANAGEMENT = "user_management"


# 操作权限需求映射
OPERATION_PERMISSIONS: Dict[OperationType, Set[Permission]] = {
    OperationType.READ_DATA: {Permission.READ_PUBLIC, Permission.READ_PRIVATE},
    OperationType.WRITE_DATA: {Permission.WRITE_OWN, Permission.WRITE_OTHERS},
    OperationType.DELETE_DATA: {Permission.WRITE_OWN, Permission.WRITE_OTHERS},
    OperationType.EXECUTE_CODE: {Permission.EXECUTE_SAFE},
    OperationType.EXECUTE_SHELL: {Permission.EXECUTE_SYSTEM},
    OperationType.READ_FILE: {Permission.READ_PUBLIC, Permission.READ_PRIVATE},
    OperationType.WRITE_FILE: {Permission.WRITE_OWN},
    OperationType.DELETE_FILE: {Permission.WRITE_OWN},
    OperationType.HTTP_GET: {Permission.EXECUTE_SAFE},
    OperationType.HTTP_POST: {Permission.EXECUTE_UNSAFE},
    OperationType.SYSTEM_CONFIG: {Permission.WRITE_SYSTEM},
    OperationType.USER_MANAGEMENT: {Permission.MANAGE_USERS},
}


# ==================== 数据类定义 ====================

@dataclass
class User:
    """用户类"""
    user_id: str
    username: str
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """检查是否拥有指定权限"""
        if Permission.UNLIMITED in self.permissions:
            return True
        return permission in self.permissions


@dataclass
class PermissionCheckResult:
    """权限检查结果"""
    allowed: bool
    operation: OperationType
    user_role: Role
    missing_permissions: List[Permission]
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    timestamp: datetime
    user_id: str
    operation: OperationType
    resource: str
    allowed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


# ==================== 权限检查器 ====================

class PermissionChecker:
    """权限检查器"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.audit_logs: List[AuditLog] = []
        self.custom_rules: List[Callable] = []

    def register_user(self, user: User) -> None:
        """注册用户"""
        # 获取角色基础权限
        base_permissions = ROLE_PERMISSIONS.get(user.role, set())
        # 合并自定义权限
        user.permissions = base_permissions.union(user.permissions)
        self.users[user.user_id] = user

    def create_user(self, user_id: str, username: str, role: Role,
                   extra_permissions: Set[Permission] = None) -> User:
        """创建用户"""
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            permissions=extra_permissions or set()
        )
        self.register_user(user)
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)

    def check_permission(self, user_id: str, operation: OperationType,
                        resource: str = None) -> PermissionCheckResult:
        """检查权限"""
        user = self.get_user(user_id)
        if not user:
            return PermissionCheckResult(
                allowed=False,
                operation=operation,
                user_role=Role.GUEST,
                missing_permissions=[],
                reason="用户不存在"
            )

        required_permissions = OPERATION_PERMISSIONS.get(operation, set())
        missing = []

        # 检查每个所需权限
        for perm in required_permissions:
            if not user.has_permission(perm):
                missing.append(perm)

        # 检查自定义规则
        custom_allowed, custom_reason = self._check_custom_rules(user, operation, resource)

        allowed = len(missing) == 0 and custom_allowed
        reason = "权限检查通过" if allowed else (custom_reason or "缺少必要权限")

        # 记录审计日志
        self._log_audit(user, operation, resource, allowed, reason)

        return PermissionCheckResult(
            allowed=allowed,
            operation=operation,
            user_role=user.role,
            missing_permissions=missing,
            reason=reason
        )

    def add_custom_rule(self, rule: Callable) -> None:
        """添加自定义规则"""
        self.custom_rules.append(rule)

    def _check_custom_rules(self, user: User, operation: OperationType,
                           resource: str) -> tuple:
        """检查自定义规则"""
        for rule in self.custom_rules:
            result, reason = rule(user, operation, resource)
            if not result:
                return False, reason
        return True, ""

    def _log_audit(self, user: User, operation: OperationType,
                   resource: str, allowed: bool, reason: str) -> None:
        """记录审计日志"""
        log = AuditLog(
            log_id=f"log_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            user_id=user.user_id,
            operation=operation,
            resource=resource or "N/A",
            allowed=allowed,
            reason=reason
        )
        self.audit_logs.append(log)

    def get_audit_logs(self, user_id: str = None, limit: int = 100) -> List[AuditLog]:
        """获取审计日志"""
        logs = self.audit_logs
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        return logs[-limit:]


# ==================== 资源访问控制 ====================

class ResourceAccessControl:
    """资源访问控制"""

    def __init__(self):
        self.resources: Dict[str, Dict] = {}
        self.access_rules: List[Callable] = []

    def register_resource(self, resource_id: str, owner_id: str,
                         resource_type: str, metadata: Dict = None) -> None:
        """注册资源"""
        self.resources[resource_id] = {
            "owner_id": owner_id,
            "type": resource_type,
            "metadata": metadata or {},
            "created_at": datetime.now()
        }

    def check_access(self, user: User, resource_id: str,
                    action: str) -> tuple:
        """检查资源访问权限"""
        resource = self.resources.get(resource_id)
        if not resource:
            return False, "资源不存在"

        # 拥有者拥有完全访问权
        if resource["owner_id"] == user.user_id:
            return True, "资源拥有者"

        # 检查角色权限
        if action == "read" and Permission.READ_PRIVATE not in user.permissions:
            return False, "无权读取他人资源"
        if action == "write" and Permission.WRITE_OTHERS not in user.permissions:
            return False, "无权修改他人资源"

        # 检查自定义规则
        for rule in self.access_rules:
            result, reason = rule(user, resource, action)
            if not result:
                return False, reason

        return True, "访问授权"

    def add_access_rule(self, rule: Callable) -> None:
        """添加访问规则"""
        self.access_rules.append(rule)


# ==================== 速率限制器 ====================

class RateLimiter:
    """速率限制器"""

    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.limits: Dict[str, Dict[str, int]] = {}

    def set_limit(self, key: str, max_requests: int, window_seconds: int) -> None:
        """设置限制"""
        self.limits[key] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds
        }

    def check(self, key: str) -> tuple:
        """检查是否超限"""
        if key not in self.limits:
            return True, "无限制"

        now = time.time()
        limit_config = self.limits[key]
        window = limit_config["window_seconds"]
        max_requests = limit_config["max_requests"]

        # 清理过期请求
        self.requests[key] = [
            t for t in self.requests[key]
            if now - t < window
        ]

        # 检查限制
        if len(self.requests[key]) >= max_requests:
            return False, f"超过速率限制 ({max_requests}/{window}秒)"

        # 记录请求
        self.requests[key].append(now)
        return True, "请求允许"


# ==================== 操作确认器 ====================

class OperationConfirmer:
    """危险操作确认器"""

    DANGEROUS_OPERATIONS = {
        OperationType.DELETE_DATA,
        OperationType.DELETE_FILE,
        OperationType.EXECUTE_SHELL,
        OperationType.SYSTEM_CONFIG,
    }

    def __init__(self):
        self.pending_confirmations: Dict[str, Dict] = {}

    def needs_confirmation(self, operation: OperationType) -> bool:
        """检查是否需要确认"""
        return operation in self.DANGEROUS_OPERATIONS

    def request_confirmation(self, user_id: str, operation: OperationType,
                           details: Dict) -> str:
        """请求确认"""
        confirmation_id = f"confirm_{int(time.time() * 1000)}"
        self.pending_confirmations[confirmation_id] = {
            "user_id": user_id,
            "operation": operation,
            "details": details,
            "created_at": datetime.now(),
            "status": "pending"
        }
        return confirmation_id

    def confirm(self, confirmation_id: str) -> tuple:
        """确认操作"""
        if confirmation_id not in self.pending_confirmations:
            return False, "确认请求不存在"

        confirmation = self.pending_confirmations[confirmation_id]
        confirmation["status"] = "confirmed"
        return True, "操作已确认"

    def reject(self, confirmation_id: str) -> tuple:
        """拒绝操作"""
        if confirmation_id not in self.pending_confirmations:
            return False, "确认请求不存在"

        confirmation = self.pending_confirmations[confirmation_id]
        confirmation["status"] = "rejected"
        return True, "操作已拒绝"


# ==================== 安全 Agent ====================

class SecureAgent:
    """安全 Agent 封装"""

    def __init__(self):
        self.permission_checker = PermissionChecker()
        self.resource_control = ResourceAccessControl()
        self.rate_limiter = RateLimiter()
        self.operation_confirmer = OperationConfirmer()

        # 设置默认速率限制
        self.rate_limiter.set_limit("default", 100, 3600)  # 每小时100次
        self.rate_limiter.set_limit("api_calls", 60, 60)   # 每分钟60次

    def execute(self, user_id: str, operation: OperationType,
               resource: str = None, params: Dict = None) -> Dict:
        """执行操作"""
        # 1. 速率限制检查
        rate_ok, rate_msg = self.rate_limiter.check(f"{user_id}_{operation.value}")
        if not rate_ok:
            return {"success": False, "error": rate_msg}

        # 2. 权限检查
        perm_result = self.permission_checker.check_permission(user_id, operation, resource)
        if not perm_result.allowed:
            return {"success": False, "error": perm_result.reason}

        # 3. 危险操作确认
        if self.operation_confirmer.needs_confirmation(operation):
            confirmation_id = self.operation_confirmer.request_confirmation(
                user_id, operation, params or {}
            )
            return {
                "success": False,
                "needs_confirmation": True,
                "confirmation_id": confirmation_id,
                "message": "此操作需要确认"
            }

        # 4. 执行操作
        return self._do_execute(operation, params)

    def confirm_and_execute(self, confirmation_id: str) -> Dict:
        """确认并执行"""
        confirmed, msg = self.operation_confirmer.confirm(confirmation_id)
        if not confirmed:
            return {"success": False, "error": msg}

        # 获取确认详情并执行
        # 这里简化处理，实际应从 pending_confirmations 获取
        return {"success": True, "message": "操作已执行"}

    def _do_execute(self, operation: OperationType, params: Dict) -> Dict:
        """实际执行操作"""
        # 这里是操作的实际执行逻辑
        return {"success": True, "operation": operation.value, "result": "执行成功"}


# ==================== 权限装饰器 ====================

def require_permission(permission: Permission):
    """权限装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 假设第一个参数是 user_id
            user_id = kwargs.get('user_id') or (args[0] if args else None)
            if not user_id:
                raise PermissionError("未提供用户ID")

            user = self.permission_checker.get_user(user_id)
            if not user or not user.has_permission(permission):
                raise PermissionError(f"缺少权限: {permission.name}")

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def require_role(role: Role):
    """角色装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            user_id = kwargs.get('user_id') or (args[0] if args else None)
            if not user_id:
                raise PermissionError("未提供用户ID")

            user = self.permission_checker.get_user(user_id)
            if not user or user.role.value != role.value:
                role_order = [Role.GUEST, Role.USER, Role.POWER_USER, Role.ADMIN, Role.SUPER_ADMIN]
                if role_order.index(user.role) < role_order.index(role):
                    raise PermissionError(f"需要角色: {role.value}")

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ==================== 演示函数 ====================

def demo_role_permissions():
    """演示角色权限"""
    print("\n" + "=" * 60)
    print("示例1: 角色权限定义")
    print("=" * 60)

    print("\n各角色拥有的权限:")
    for role in Role:
        permissions = ROLE_PERMISSIONS.get(role, set())
        perm_names = [p.name for p in permissions]
        print(f"\n{role.value}:")
        print(f"  权限数量: {len(permissions)}")
        if perm_names:
            print(f"  权限列表: {', '.join(perm_names[:5])}{'...' if len(perm_names) > 5 else ''}")


def demo_permission_check():
    """演示权限检查"""
    print("\n" + "=" * 60)
    print("示例2: 权限检查")
    print("=" * 60)

    checker = PermissionChecker()

    # 创建用户
    guest = checker.create_user("u1", "访客用户", Role.GUEST)
    user = checker.create_user("u2", "普通用户", Role.USER)
    admin = checker.create_user("u3", "管理员", Role.ADMIN)

    # 测试不同用户的权限
    test_operations = [
        (OperationType.READ_DATA, "读取数据"),
        (OperationType.WRITE_DATA, "写入数据"),
        (OperationType.EXECUTE_CODE, "执行代码"),
        (OperationType.SYSTEM_CONFIG, "系统配置"),
    ]

    for operation, desc in test_operations:
        print(f"\n操作: {desc}")
        for user_id, role_name in [("u1", "访客"), ("u2", "用户"), ("u3", "管理员")]:
            result = checker.check_permission(user_id, operation)
            status = "✓ 允许" if result.allowed else "✗ 拒绝"
            print(f"  {role_name}: {status} - {result.reason}")


def demo_audit_logs():
    """演示审计日志"""
    print("\n" + "=" * 60)
    print("示例3: 审计日志记录")
    print("=" * 60)

    checker = PermissionChecker()

    # 创建用户并执行操作
    user = checker.create_user("u1", "测试用户", Role.USER)

    operations = [
        (OperationType.READ_DATA, "读取用户数据"),
        (OperationType.WRITE_DATA, "写入用户数据"),
        (OperationType.EXECUTE_SHELL, "执行Shell命令"),
    ]

    for operation, resource in operations:
        checker.check_permission("u1", operation, resource)

    # 查看日志
    logs = checker.get_audit_logs("u1")
    print("\n审计日志:")
    for log in logs:
        status = "允许" if log.allowed else "拒绝"
        print(f"  [{log.timestamp.strftime('%H:%M:%S')}] {log.operation.value} - {status}")
        print(f"    原因: {log.reason}")


def demo_rate_limiting():
    """演示速率限制"""
    print("\n" + "=" * 60)
    print("示例4: 速率限制")
    print("=" * 60)

    limiter = RateLimiter()
    limiter.set_limit("api_test", 5, 10)  # 10秒内最多5次

    print("\n模拟请求 (限制: 10秒内最多5次):")
    for i in range(7):
        allowed, msg = limiter.check("api_test")
        status = "✓" if allowed else "✗"
        print(f"  请求 {i+1}: {status} {msg}")
        time.sleep(0.1)  # 短暂延迟


def demo_operation_confirmation():
    """演示操作确认"""
    print("\n" + "=" * 60)
    print("示例5: 危险操作确认")
    print("=" * 60)

    confirmer = OperationConfirmer()

    # 检查哪些操作需要确认
    print("\n需要确认的操作:")
    for operation in OperationType:
        if confirmer.needs_confirmation(operation):
            print(f"  - {operation.value}")

    # 模拟确认流程
    print("\n模拟删除数据操作:")
    print("1. 请求执行删除操作...")
    confirmation_id = confirmer.request_confirmation(
        "u1", OperationType.DELETE_DATA, {"target": "user_data"}
    )
    print(f"   需要确认，ID: {confirmation_id}")

    print("2. 用户确认操作...")
    success, msg = confirmer.confirm(confirmation_id)
    print(f"   结果: {msg}")

    print("3. 执行操作...")
    print("   操作已执行")


def demo_resource_access():
    """演示资源访问控制"""
    print("\n" + "=" * 60)
    print("示例6: 资源访问控制")
    print("=" * 60)

    checker = PermissionChecker()
    resource_control = ResourceAccessControl()

    # 创建用户
    owner = checker.create_user("u1", "资源拥有者", Role.USER)
    other_user = checker.create_user("u2", "其他用户", Role.USER)
    admin = checker.create_user("u3", "管理员", Role.ADMIN)

    # 注册资源
    resource_control.register_resource("doc_001", "u1", "document")

    print("\n资源访问测试 (资源拥有者: u1):")

    test_cases = [
        ("u1", "read", "资源拥有者读取"),
        ("u2", "read", "其他用户读取"),
        ("u1", "write", "资源拥有者写入"),
        ("u2", "write", "其他用户写入"),
    ]

    for user_id, action, desc in test_cases:
        user = checker.get_user(user_id)
        allowed, reason = resource_control.check_access(user, "doc_001", action)
        status = "✓" if allowed else "✗"
        print(f"  {desc}: {status} - {reason}")


def demo_secure_agent():
    """演示安全 Agent"""
    print("\n" + "=" * 60)
    print("示例7: 安全 Agent 执行流程")
    print("=" * 60)

    agent = SecureAgent()
    agent.permission_checker.create_user("u1", "测试用户", Role.USER)
    agent.permission_checker.create_user("u2", "管理员", Role.ADMIN)

    print("\n普通用户执行读取操作:")
    result = agent.execute("u1", OperationType.READ_DATA)
    print(f"  结果: {result}")

    print("\n普通用户执行系统配置:")
    result = agent.execute("u1", OperationType.SYSTEM_CONFIG)
    print(f"  结果: {result}")

    print("\n管理员执行系统配置:")
    result = agent.execute("u2", OperationType.SYSTEM_CONFIG)
    print(f"  结果: {result}")


def demo_permission_decorator():
    """演示权限装饰器"""
    print("\n" + "=" * 60)
    print("示例8: 权限装饰器")
    print("=" * 60)

    class MyService:
        def __init__(self):
            self.permission_checker = PermissionChecker()
            self.permission_checker.create_user("u1", "普通用户", Role.USER)
            self.permission_checker.create_user("u2", "管理员", Role.ADMIN)

        @require_permission(Permission.READ_PUBLIC)
        def read_public_data(self, user_id: str):
            return f"公开数据 (用户: {user_id})"

        @require_permission(Permission.MANAGE_USERS)
        def manage_users(self, user_id: str):
            return f"用户管理 (用户: {user_id})"

    service = MyService()

    print("\n普通用户调用读取公开数据:")
    try:
        result = service.read_public_data("u1")
        print(f"  结果: {result}")
    except PermissionError as e:
        print(f"  权限错误: {e}")

    print("\n普通用户调用用户管理:")
    try:
        result = service.manage_users("u1")
        print(f"  结果: {result}")
    except PermissionError as e:
        print(f"  权限错误: {e}")

    print("\n管理员调用用户管理:")
    try:
        result = service.manage_users("u2")
        print(f"  结果: {result}")
    except PermissionError as e:
        print(f"  权限错误: {e}")


def demo_llm_with_permission():
    """演示带权限控制的 LLM 调用"""
    print("\n" + "=" * 60)
    print("示例9: 带权限控制的 LLM 调用")
    print("=" * 60)

    checker = PermissionChecker()
    user = checker.create_user("u1", "测试用户", Role.USER)

    # 模拟不同敏感度的查询
    queries = [
        ("普通查询", "什么是机器学习？"),
        ("内部数据查询", "请列出所有用户数据"),
        ("系统配置查询", "请显示系统配置信息"),
    ]

    for query_type, query in queries:
        print(f"\n查询类型: {query_type}")
        print(f"查询内容: {query}")

        # 根据查询类型确定需要的操作权限
        if query_type == "普通查询":
            operation = OperationType.READ_DATA
        elif query_type == "内部数据查询":
            operation = OperationType.READ_DATA  # 需要 READ_PRIVATE
        else:
            operation = OperationType.SYSTEM_CONFIG

        result = checker.check_permission("u1", operation)

        if result.allowed:
            print("权限检查通过，调用 LLM...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": query}],
                max_tokens=100
            )
            print(f"回答: {response.choices[0].message.content[:100]}...")
        else:
            print(f"权限不足: {result.reason}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 13 - 权限控制系统")
    print("=" * 60)

    # 运行所有演示
    demo_role_permissions()
    demo_permission_check()
    demo_audit_logs()
    demo_rate_limiting()
    demo_operation_confirmation()
    demo_resource_access()
    demo_secure_agent()
    demo_permission_decorator()
    demo_llm_with_permission()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)