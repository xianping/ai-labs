# client.py
import asyncio
from fastmcp import Client

async def main():
    url = "http://127.0.0.1:8000/mcp"

    async with Client(url) as client:
        # 1. 列出工具
        tools = await client.list_tools()
        print("🛠️ 可用工具：")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # 2. 调用工具
        print("\n📞 调用工具：")
        result = await client.call_tool("add", {"a": 10, "b": 5})
        print(f"  add(10,5) = {result}")

        result = await client.call_tool("get_weather", {"city": "Beijing"})
        print(f"  get_weather(Beijing) = {result}")

        result = await client.call_tool("get_joke", {})
        print(f"  get_joke() = {result}")

        result = await client.call_tool("calculate_bmi", {"weight_kg": 70, "height_m": 1.75})
        print(f"  calculate_bmi(70,1.75) = {result}")

        # 3. 读取资源
        print("\n📄 读取资源：")
        greeting = await client.read_resource("greeting://Alice")
        print(f"  greeting://Alice -> {greeting}")

        status = await client.read_resource("system://status")
        print(f"  system://status -> {status}")

        # 4. 获取提示词
        print("\n💬 获取提示词：")
        prompt = await client.get_prompt("report_prompt", {"topic": "人工智能", "detail": "详细"})
        print(f"  report_prompt(topic='人工智能', detail='详细') ->\n{prompt}")

if __name__ == "__main__":
    asyncio.run(main())