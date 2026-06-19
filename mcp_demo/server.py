# server.py
import logging
import random
import requests
from datetime import datetime
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

# 创建服务器实例（3.x 版本）
mcp = FastMCP("MyDemoServer")

# -------------------- Tools --------------------
@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个整数相加"""
    logger.info(f"add({a}, {b})")
    return a + b

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的当前天气（使用 wttr.in）"""
    logger.info(f"get_weather({city})")
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        resp = requests.get(url, timeout=5)
        return f"{city} 天气：{resp.text.strip()}"
    except Exception as e:
        return f"获取天气失败：{e}"

@mcp.tool()
def get_joke() -> str:
    """获取一个随机笑话"""
    jokes = [
        "为什么程序员总是混淆万圣节和圣诞节？因为 Oct 31 = Dec 25。",
        "一个 SQL 查询走进酒吧，看到两张桌子，问：我能 JOIN 你们吗？",
        "Python 程序员为什么喜欢蚊子？因为它们会 import 吸血。"
    ]
    return random.choice(jokes)

@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """计算 BMI 并给出健康建议"""
    if height_m <= 0:
        return "身高必须大于 0"
    bmi = weight_kg / (height_m ** 2)
    category = (
        "偏瘦" if bmi < 18.5 else
        "正常" if bmi < 24 else
        "超重" if bmi < 28 else
        "肥胖"
    )
    return f"BMI = {bmi:.1f}，属于「{category}」"

# -------------------- Resources --------------------
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """个性化问候资源"""
    return f"你好，{name}！欢迎使用 MCP 服务。"

@mcp.resource("system://status")
def get_system_status() -> str:
    """系统状态资源"""
    return f"服务器时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，运行正常。"

# -------------------- Prompts --------------------
@mcp.prompt()
def report_prompt(topic: str, detail: str = "简要") -> str:
    """生成报告提示词模板"""
    return f"""请撰写一份关于「{topic}」的{detail}报告。
要求：
1. 结构清晰，分章节
2. 数据准确
3. 结论明确
"""

# -------------------- 启动 --------------------
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)