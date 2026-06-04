import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv(encoding='utf-8')

# 1. 初始化模型
model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

search_tool = DuckDuckGoSearchRun()
# 显式允许并行调用（生产环境默认开启，我们不限制它）
model_with_tools = model.bind_tools([search_tool])

# 异步包装工具执行器
async def run_single_tool(tool_call):
    """并发执行单个工具调用的工作线程/协程"""
    query_str = tool_call["args"]["query"]
    print(f"🌐 [🔍 异步启动] 正在检索关键词: '{query_str}' (ID: {tool_call['id']})")
    
    # 在线程池中异步运行同步的 DuckDuckGo 工具
    # 如果工具本身支持 ainvoke，可以直接 await search_tool.ainvoke(...)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, search_tool.invoke, query_str)
    
    # 严格封装并返回 ToolMessage，带上原配的 id 和 name
    return ToolMessage(
        content=result,
        tool_call_id=tool_call["id"],
        name=tool_call["name"]
    )

async def main():
    query = "2026年最近有什么最新的科技新闻或重大突破？"
    print(f"👤 用户提问: {query}\n")
    
    messages = [HumanMessage(content=query)]
    
    # 第一阶段：让大模型生成并行的 tool_calls 任务数组
    response = await model_with_tools.ainvoke(messages)
    messages.append(response)
    
    if response.tool_calls:
        print(f"🤖 大模型思考后决定：并发调用 {len(response.tool_calls)} 个搜索任务！\n")
        
        # 🚀 【生产核心】：构建异步任务并发池
        tasks = [run_single_tool(call) for call in response.tool_calls]
        
        # 使用 asyncio.gather 同时并发执行所有搜索，I/O 性能达到极致
        tool_messages = await asyncio.gather(*tasks)
        
        # 将所有返回的 ToolMessage 批量追加到消息队列中
        # 此时队列中会完美包含每一个 tool_call_id 的答卷，彻底满足网关校验
        messages.extend(tool_messages)
        
        print("\n📥 所有并发网络数据已整齐回填，正在发起第二阶段最终聚合...")
        
        # 第二阶段：把完美的队列发回，大模型会将两个搜索结果融合，输出终极研报
        # final_response = await model_with_tools.ainvoke(messages)
        final_response = await model.ainvoke(messages)

        print("\n✨ [AI 行业研究员最终联网报告]：\n")
        print(final_response.content)
    else:
        print("🤖 直接回答：", response.content)

if __name__ == "__main__":
    # 启动异步事件循环
    asyncio.run(main())