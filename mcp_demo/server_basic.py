# minimal_server.py
import logging
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)

# 创建服务器实例（3.x 不需要 json_response 参数）
mcp = FastMCP("MinimalDemo")

# 定义一个简单的工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个数字相加"""
    return a + b

if __name__ == "__main__":
    # 3.x 版本：host 和 port 直接在 run() 中传递
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)