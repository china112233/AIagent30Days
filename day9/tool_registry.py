"""
工具注册表模块

管理工具的注册、查找和调用。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from .tool_definition import Tool, ToolParameter, ToolResult
import json


class ToolRegistryError(Exception):
    """工具注册表错误"""
    pass


class ToolNotFoundError(ToolRegistryError):
    """工具未找到"""
    pass


class ToolAlreadyExistsError(ToolRegistryError):
    """工具已存在"""
    pass


@dataclass
class ToolCategory:
    """工具分类"""
    name: str
    description: str
    tools: List[str] = field(default_factory=list)


class ToolRegistry:
    """
    工具注册表
    
    管理工具的注册、查找、列出和调用。
    """
    
    def __init__(self, name: str = "default"):
        """
        初始化工具注册表
        
        Args:
            name: 注册表名称
        """
        self.name = name
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, ToolCategory] = {}
        self._aliases: Dict[str, str] = {}  # 工具别名
    
    def register(
        self, 
        tool: Tool, 
        category: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> None:
        """
        注册工具
        
        Args:
            tool: 要注册的工具
            category: 工具分类
            aliases: 工具别名列表
            overwrite: 是否覆盖已存在的工具
            
        Raises:
            ToolAlreadyExistsError: 工具已存在且不允许覆盖
        """
        # 检查是否已存在
        if tool.name in self._tools and not overwrite:
            raise ToolAlreadyExistsError(f"工具 '{tool.name}' 已存在")
        
        # 注册工具
        self._tools[tool.name] = tool
        
        # 添加到分类
        if category:
            if category not in self._categories:
                self._categories[category] = ToolCategory(
                    name=category,
                    description=f"{category} 相关工具"
                )
            self._categories[category].tools.append(tool.name)
        
        # 注册别名
        if aliases:
            for alias in aliases:
                self._aliases[alias] = tool.name
        
        # 从工具标签自动分类
        for tag in tool.tags:
            if tag not in self._categories:
                self._categories[tag] = ToolCategory(
                    name=tag,
                    description=f"{tag} 相关工具"
                )
            if tool.name not in self._categories[tag].tools:
                self._categories[tag].tools.append(tool.name)
    
    def unregister(self, name: str) -> bool:
        """
        注销工具
        
        Args:
            name: 工具名称或别名
            
        Returns:
            是否成功注销
        """
        # 解析别名
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._tools:
            return False
        
        # 从分类中移除
        for category in self._categories.values():
            if actual_name in category.tools:
                category.tools.remove(actual_name)
        
        # 移除别名
        aliases_to_remove = [k for k, v in self._aliases.items() if v == actual_name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        # 移除工具
        del self._tools[actual_name]
        return True
    
    def get_tool(self, name: str) -> Tool:
        """
        获取工具
        
        Args:
            name: 工具名称或别名
            
        Returns:
            工具对象
            
        Raises:
            ToolNotFoundError: 工具未找到
        """
        # 解析别名
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._tools:
            raise ToolNotFoundError(f"工具 '{name}' 未找到")
        
        return self._tools[actual_name]
    
    def has_tool(self, name: str) -> bool:
        """
        检查工具是否存在
        
        Args:
            name: 工具名称或别名
            
        Returns:
            是否存在
        """
        actual_name = self._aliases.get(name, name)
        return actual_name in self._tools
    
    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """
        列出工具
        
        Args:
            category: 按分类过滤
            
        Returns:
            工具列表
        """
        if category:
            if category not in self._categories:
                return []
            return [self._tools[name] for name in self._categories[category].tools if name in self._tools]
        
        return list(self._tools.values())
    
    def list_categories(self) -> List[str]:
        """列出所有分类"""
        return list(self._categories.keys())
    
    def get_tools_by_tag(self, tag: str) -> List[Tool]:
        """
        按标签获取工具
        
        Args:
            tag: 标签名称
            
        Returns:
            工具列表
        """
        return [tool for tool in self._tools.values() if tag in tool.tags]
    
    def search_tools(self, query: str) -> List[Tool]:
        """
        搜索工具
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的工具列表
        """
        query = query.lower()
        results = []
        
        for tool in self._tools.values():
            # 在名称和描述中搜索
            if query in tool.name.lower() or query in tool.description.lower():
                results.append(tool)
                continue
            
            # 在标签中搜索
            for tag in tool.tags:
                if query in tag.lower():
                    results.append(tool)
                    break
        
        return results
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        执行工具
        
        Args:
            name: 工具名称或别名
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        tool = self.get_tool(name)
        return tool.execute(**kwargs)
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        转换为 OpenAI Function Calling 格式
        
        Returns:
            OpenAI 工具列表
        """
        return [tool.to_openai_function() for tool in self._tools.values()]
    
    def to_json_schema(self) -> Dict[str, Any]:
        """
        转换为 JSON Schema
        
        Returns:
            JSON Schema 对象
        """
        return {
            "name": self.name,
            "tools": [tool.to_json_schema() for tool in self._tools.values()],
            "categories": {
                name: {"description": cat.description, "tools": cat.tools}
                for name, cat in self._categories.items()
            }
        }
    
    def get_tool_documentation(self) -> str:
        """
        获取工具文档
        
        Returns:
            Markdown 格式的工具文档
        """
        doc = f"# 工具注册表: {self.name}\n\n"
        doc += f"共 {len(self._tools)} 个工具\n\n"
        
        # 按分类组织
        if self._categories:
            for category_name, category in self._categories.items():
                doc += f"## {category_name}\n\n"
                for tool_name in category.tools:
                    if tool_name in self._tools:
                        tool = self._tools[tool_name]
                        doc += f"### {tool.name}\n\n"
                        doc += f"{tool.description}\n\n"
                        doc += "**参数:**\n\n"
                        for param in tool.parameters:
                            required = "必需" if param.required else "可选"
                            doc += f"- `{param.name}` ({param.type}, {required}): {param.description}\n"
                        doc += "\n"
        else:
            for tool in self._tools.values():
                doc += f"## {tool.name}\n\n"
                doc += f"{tool.description}\n\n"
        
        return doc
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return self.has_tool(name)
    
    def __getitem__(self, name: str) -> Tool:
        return self.get_tool(name)


class ToolRegistryManager:
    """
    工具注册表管理器
    
    管理多个工具注册表实例。
    """
    
    _instance = None
    _registries: Dict[str, ToolRegistry] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_registry(cls, name: str = "default") -> ToolRegistry:
        """
        获取注册表
        
        Args:
            name: 注册表名称
            
        Returns:
            工具注册表
        """
        if name not in cls._registries:
            cls._registries[name] = ToolRegistry(name)
        return cls._registries[name]
    
    @classmethod
    def list_registries(cls) -> List[str]:
        """列出所有注册表"""
        return list(cls._registries.keys())


# ============ 演示 ============

def demo():
    """演示工具注册表的使用"""
    print("=" * 60)
    print("工具注册表演示")
    print("=" * 60)
    
    from tool_definition import get_weather, calculate, search_web
    
    # 创建注册表
    registry = ToolRegistry(name="demo")
    
    # 注册工具
    print("\n1. 注册工具...")
    registry.register(get_weather, category="信息查询")
    registry.register(calculate, aliases=["calc", "math"])
    registry.register(search_web, category="信息查询")
    
    print(f"已注册 {len(registry)} 个工具")
    
    # 列出工具
    print("\n2. 列出所有工具:")
    for tool in registry.list_tools():
        print(f"  - {tool.name}: {tool.description}")
    
    # 按分类列出
    print("\n3. 按分类列出工具:")
    for category in registry.list_categories():
        tools = registry.list_tools(category)
        print(f"  {category}: {[t.name for t in tools]}")
    
    # 搜索工具
    print("\n4. 搜索工具 'weather':")
    results = registry.search_tools("weather")
    for tool in results:
        print(f"  - {tool.name}")
    
    # 使用别名获取
    print("\n5. 使用别名获取工具:")
    tool = registry.get_tool("calc")  # calculate 的别名
    print(f"  calc -> {tool.name}")
    
    # 执行工具
    print("\n6. 执行工具:")
    result = registry.execute("get_weather", city="北京")
    print(f"  get_weather(city='北京'): {result.to_dict()}")
    
    result = registry.execute("calc", expression="2 + 3 * 4")
    print(f"  calc(expression='2 + 3 * 4'): {result.to_dict()}")
    
    # OpenAI 格式
    print("\n7. OpenAI Function Calling 格式:")
    openai_tools = registry.to_openai_tools()
    print(json.dumps(openai_tools[0], indent=2, ensure_ascii=False))
    
    # 工具文档
    print("\n8. 工具文档:")
    print(registry.get_tool_documentation())


if __name__ == "__main__":
    demo()