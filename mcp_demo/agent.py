# agent.py (适配 DeepSeek + fastmcp 3.x)
import asyncio
import json
import os
from openai import AsyncOpenAI
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv(encoding='utf-8')

class MCPAgent:
    def __init__(
        self,
        mcp_url: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-v4-flash"
    ):
        self.mcp_url = mcp_url
        self.model = model
        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.mcp_client = None
        self.tools = []
        self.tool_map = {}

    async def connect(self):
        self.mcp_client = Client(self.mcp_url)
        await self.mcp_client.__aenter__()
        tools_list = await self.mcp_client.list_tools()
        self.tools = tools_list
        self.tool_map = {t.name: t for t in tools_list}
        print(f"🔌 已连接到 MCP，加载了 {len(self.tools)} 个工具")

    async def disconnect(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)

    def _build_openai_functions(self):
        """将 MCP 工具转换为 OpenAI/DeepSeek 函数定义（兼容 fastmcp 3.x）"""
        functions = []
        for tool in self.tools:
            # ---------- 修复开始 ----------
            # 优先获取 input_schema（新版本）
            if hasattr(tool, 'input_schema'):
                params = tool.input_schema
            elif hasattr(tool, 'inputSchema'):
                params = tool.inputSchema
            elif hasattr(tool, 'parameters'):
                params = tool.parameters
            else:
                # 后备：尝试从 model_dump 提取
                raw = tool.model_dump() if hasattr(tool, 'model_dump') else {}
                params = raw.get('inputSchema', {}) or raw.get('parameters', {})

            # 如果 params 是 Pydantic 模型，转为 dict
            if hasattr(params, 'model_dump'):
                params = params.model_dump()
            elif hasattr(params, 'dict'):
                params = params.dict()

            properties = params.get("properties", {}) if isinstance(params, dict) else {}
            required = params.get("required", []) if isinstance(params, dict) else []
            # ---------- 修复结束 ----------

            functions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })
        return functions

    async def run(self, user_query: str) -> str:
        messages = [{"role": "user", "content": user_query}]
        functions = self._build_openai_functions()

        if not functions:
            resp = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return resp.choices[0].message.content

        while True:
            resp = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=functions,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                return msg.content or "（无回答）"

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"🤖 Agent 调用工具: {func_name}({func_args})")
                result = await self.mcp_client.call_tool(func_name, func_args)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

async def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️ 请设置环境变量 DEEPSEEK_API_KEY")
        return

    agent = MCPAgent(
        mcp_url="http://127.0.0.1:8000/mcp",
        api_key=api_key,
        model="deepseek-chat"  # 或 "deepseek-v4-flash"
    )

    await agent.connect()
    try:
        queries = [
            "请帮我计算 25 和 37 的和",
            "北京今天天气怎么样？",
            "给我讲个笑话",
            "我身高 1.8 米，体重 80 公斤，请计算我的 BMI",
        ]
        for q in queries:
            print(f"\n👤 用户: {q}")
            answer = await agent.run(q)
            print(f"🤖 Agent: {answer}")
    finally:
        await agent.disconnect()

if __name__ == "__main__":
    asyncio.run(main())