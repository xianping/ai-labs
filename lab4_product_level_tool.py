import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv(encoding='utf-8')

class ProductionResearchAgent:
    def __init__(self, max_loops: int = 3):
        # 原则三：清晰划分模型的生命周期
        # 1. 专门用于路由和决策的实例（带工具）
        self.base_model = ChatOpenAI(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.router_model = self.base_model.bind_tools([self.search_tool])
        
        # 原则二：生产级防死循环阈值
        self.max_loops = max_loops

    async def _execute_tools_async(self, tool_calls) -> list:
        """原则一：高并发异步执行器，彻底压榨网络 I/O 性能"""
        tasks = []
        loop = asyncio.get_event_loop()
        
        for tool_call in tool_calls:
            query_str = tool_call["args"]["query"]
            print(f"🌐 [I/O 并发] 正在检索: '{query_str}'")
            
            # 异步线程池包裹同步 I/O
            task = loop.run_in_executor(None, self.search_tool.invoke, query_str)
            tasks.append((tool_call, task))
            
        tool_messages = []
        for tool_call, task in tasks:
            result = await task
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            ))
        return tool_messages

    async def run(self, user_query: str) -> str:
        """Agent 核心分布式状态机主循环"""
        # 初始化全局会话状态（State）
        messages = [HumanMessage(content=user_query)]
        loop_count = 0
        
        while loop_count < self.max_loops:
            loop_count += 1
            print(f"\n🔄 --- 状态机第 {loop_count} 轮迭代 ---")
            
            # 1. 决策阶段：使用带工具的模型，看它是否还需要数据
            response = await self.router_model.ainvoke(messages)
            
            # 2. 如果模型不需要调用工具了，说明“收敛完成”
            if not response.tool_calls:
                print("🏁 状态机判定：数据已补齐，进入最终文本收敛。")
                # 显式使用基础清净模型进行收尾，防止 DeepSeek 网关截断 content
                final_response = await self.base_model.ainvoke(messages)
                return final_response.content
                
            # 3. 如果需要调用工具，将其原生响应塞入历史，进入执行阶段
            messages.append(response)
            
            # 4. 执行阶段：并发网络检索
            tool_messages = await self._execute_tools_async(response.tool_calls)
            
            # 5. 将结果回填状态队列，状态机进入下一轮循环
            messages.extend(tool_messages)
            
        raise TimeoutError("Agent 触发防死循环保护，未能按时收敛。")

# === 生产级调用入口 ===
async def main():
    agent = ProductionResearchAgent(max_loops=3)
    question = "2026年最近有什么最新的科技新闻或重大突破？"
    
    try:
        report = await agent.run(question)
        print("\n✨ [工业级框架产出的终极联网研报]：\n")
        print(report)
    except Exception as e:
        print(f"❌ 生产链路异常: {e}")

if __name__ == "__main__":
    asyncio.run(main())