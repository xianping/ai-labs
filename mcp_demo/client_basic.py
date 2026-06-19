# client.py
import asyncio
from fastmcp import Client

async def main():
    # 务必使用正确的端点（你的服务器日志已确认监听 /mcp）
    url = "http://127.0.0.1:8000/mcp"
    
    try:
        async with Client(url) as client:
            # 列出工具（测试连接）
            tools = await client.list_tools()
            print("✅ 连接成功！可用工具:", [t.name for t in tools])
            
            # 调用工具
            result = await client.call_tool("add", {"a": 5, "b": 3})
            print("add(5,3) =", result)
    except Exception as e:
        print("❌ 连接失败:", e)

if __name__ == "__main__":
    asyncio.run(main())