"""
天气查询代理
一个实用的天气查询工具，支持多城市对比
"""

import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# 模拟天气数据库（实际应用中应调用真实API）
WEATHER_DATA = {
    "北京": {
        "temp": 12, "weather": "晴", "humidity": 45,
        "wind": "北风3级", "air_quality": "良",
        "suggestion": "天气较好，适合户外活动"
    },
    "上海": {
        "temp": 16, "weather": "多云", "humidity": 65,
        "wind": "东风2级", "air_quality": "轻度污染",
        "suggestion": "建议佩戴口罩"
    },
    "广州": {
        "temp": 24, "weather": "晴", "humidity": 70,
        "wind": "南风2级", "air_quality": "优",
        "suggestion": "天气宜人，适合外出"
    },
    "深圳": {
        "temp": 25, "weather": "晴", "humidity": 75,
        "wind": "东南风2级", "air_quality": "优",
        "suggestion": "天气很好，注意防晒"
    },
    "杭州": {
        "temp": 14, "weather": "小雨", "humidity": 80,
        "wind": "东风1级", "air_quality": "良",
        "suggestion": "建议携带雨具"
    },
    "成都": {
        "temp": 14, "weather": "阴", "humidity": 75,
        "wind": "微风", "air_quality": "良",
        "suggestion": "天气凉爽，适合散步"
    },
    "武汉": {
        "temp": 15, "weather": "多云", "humidity": 60,
        "wind": "北风2级", "air_quality": "良",
        "suggestion": "天气一般，注意保暖"
    },
    "西安": {
        "temp": 10, "weather": "晴", "humidity": 40,
        "wind": "北风3级", "air_quality": "轻度污染",
        "suggestion": "天气干燥，注意补水"
    }
}


def get_weather(city: str) -> str:
    """
    获取城市天气详情
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息字符串
    """
    city = city.replace("市", "").replace("省", "")
    
    if city not in WEATHER_DATA:
        supported = "、".join(WEATHER_DATA.keys())
        return f"抱歉，暂不支持查询 {city} 的天气。\n支持的城市：{supported}"
    
    data = WEATHER_DATA[city]
    
    return f"""
📍 {city}天气详情
━━━━━━━━━━━━━━━━
🌡️  温度：{data['temp']}°C
☁️  天气：{data['weather']}
💧 湿度：{data['humidity']}%
🌬️  风力：{data['wind']}
🍃 空气质量：{data['air_quality']}
💡 建议：{data['suggestion']}
━━━━━━━━━━━━━━━━"""


def compare_weather(cities: list) -> str:
    """
    对比多个城市的天气
    
    Args:
        cities: 城市名称列表
    
    Returns:
        对比结果
    """
    results = []
    for city in cities:
        city = city.replace("市", "").replace("省", "")
        if city in WEATHER_DATA:
            data = WEATHER_DATA[city]
            results.append({
                "city": city,
                "temp": data["temp"],
                "weather": data["weather"],
                "air_quality": data["air_quality"]
            })
    
    if not results:
        return "没有找到有效城市的天气数据"
    
    # 构建对比表格
    output = "\n📊 城市天气对比\n"
    output += "┌────────┬────────┬────────┬──────────┐\n"
    output += "│  城市  │ 温度   │ 天气   │ 空气质量 │\n"
    output += "├────────┼────────┼────────┼──────────┤\n"
    
    for r in results:
        output += f"│ {r['city']:^6} │ {r['temp']:>4}°C │ {r['weather']:^6} │ {r['air_quality']:^8} │\n"
    
    output += "└────────┴────────┴────────┴──────────┘\n"
    
    # 温度排序
    sorted_by_temp = sorted(results, key=lambda x: x["temp"], reverse=True)
    output += f"\n🌡️  温度最高：{sorted_by_temp[0]['city']} ({sorted_by_temp[0]['temp']}°C)\n"
    output += f"❄️  温度最低：{sorted_by_temp[-1]['city']} ({sorted_by_temp[-1]['temp']}°C)\n"
    
    return output


def get_weather_alert(city: str) -> str:
    """
    获取天气预警信息
    
    Args:
        city: 城市名称
    
    Returns:
        预警信息
    """
    city = city.replace("市", "").replace("省", "")
    
    if city not in WEATHER_DATA:
        return f"未找到 {city} 的天气预警信息"
    
    data = WEATHER_DATA[city]
    alerts = []
    
    # 检查各种预警条件
    if data["temp"] < 5:
        alerts.append("🥶 低温预警：气温较低，注意防寒保暖")
    elif data["temp"] > 30:
        alerts.append("🥵 高温预警：气温较高，注意防暑降温")
    
    if data["humidity"] > 80:
        alerts.append("💧 高湿预警：湿度较高，注意防潮")
    
    if "雨" in data["weather"]:
        alerts.append("🌧️  降雨提醒：建议携带雨具")
    
    if "污染" in data["air_quality"]:
        alerts.append("😷 空气污染提醒：建议佩戴口罩")
    
    if not alerts:
        return f"✅ {city}目前没有天气预警"
    
    return f"⚠️ {city}天气预警\n" + "\n".join(alerts)


# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的详细天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_weather",
            "description": "对比多个城市的天气情况，支持2-5个城市",
            "parameters": {
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的城市列表"
                    }
                },
                "required": ["cities"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_alert",
            "description": "获取城市的天气预警信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {
    "get_weather": get_weather,
    "compare_weather": compare_weather,
    "get_weather_alert": get_weather_alert
}


class WeatherAgent:
    """天气查询代理"""
    
    def __init__(self):
        self.messages = [{
            "role": "system",
            "content": """你是一个专业的天气助手。

你可以：
- 查询单个城市的详细天气
- 对比多个城市的天气
- 提供天气预警信息

支持的城市：北京、上海、广州、深圳、杭州、成都、武汉、西安

根据用户需求选择合适的工具，用友好的方式展示天气信息。"""
        }]
    
    def chat(self, user_message: str) -> str:
        """处理用户消息"""
        self.messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # 处理工具调用
        while message.tool_calls:
            self.messages.append(message)
            
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                result = available_functions[func_name](**func_args)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages
            )
            message = response.choices[0].message
        
        self.messages.append(message)
        return message.content


def demo():
    """演示天气代理"""
    print("=" * 50)
    print("🌤️  天气查询代理")
    print("=" * 50)
    
    agent = WeatherAgent()
    
    queries = [
        "北京今天天气怎么样？",
        "帮我对比一下北京、上海、广州的天气",
        "杭州有天气预警吗？",
        "我想去旅游，深圳和成都哪个天气更好？"
    ]
    
    for query in queries:
        print(f"\n用户: {query}")
        response = agent.chat(query)
        print(f"助手: {response}")


def interactive():
    """交互模式"""
    print("\n" + "=" * 50)
    print("🌤️  天气助手 - 交互模式")
    print("支持查询：北京、上海、广州、深圳、杭州、成都、武汉、西安")
    print("输入 'quit' 退出")
    print("=" * 50)
    
    agent = WeatherAgent()
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！祝您天气愉快！")
            break
        
        if not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\n助手: {response}")


if __name__ == "__main__":
    demo()
    
    print("\n启动交互模式？(y/n): ", end="")
    if input().lower() == 'y':
        interactive()