import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv(encoding='utf-8')

class ProductionMultiTurnAgent:
    """
    test one agent with tool calling, and short memory
    """
    def __init__(self, window_size: int = 4, max_loops: int = 3):
        # 🤝 严格尊崇工业级标准范式定义底层无状态模型
        self.base_model = ChatOpenAI(
            model="deepseek-v4-flash", 
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0
        )
        
        # 从 community 包初始化联网组件
        self.search_tool = DuckDuckGoSearchRun()
        
        # 绑定工具用于路由决策阶段（显式允许并行调用）
        self.router_model = self.base_model.bind_tools([self.search_tool])
        
        # 核心防 Cache 匹配与记忆脱敏提示词：防止模型废话偷渡
        self.system_prompt_template = (
            "你是一个极其高冷且专业的科技行业研究员。\n"
            "【三大铁律】：\n"
            "1. 只基于最新的实时搜索资料和历史记忆回答，绝不讲任何多余的客套话或无用申明。\n"
            "2. 绝对不准在回复中复述用户的个人身份背景或名字，保持报告的绝对客观。\n"
            "3. 启动标记: {timestamp}"
        )
        
        # 维护全局的用户会话短期记忆（只存真实的对话文本，不存中间工具的碎数据）
        self.chat_history = []
        self.window_size = window_size
        self.max_loops = max_loops

    async def _execute_tools_async(self, tool_calls) -> list:
        """异步非阻塞并发执行所有模型拆解出的网络任务"""
        tasks = []
        loop = asyncio.get_event_loop()
        
        for tool_call in tool_calls:
            query_str = tool_call["args"]["query"]
            print(f"  🌐 [I/O 并发发起] 检索关键词: '{query_str}'")
            
            # 使用线程池包裹同步的网络请求
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

    async def chat(self, user_input: str) -> str:
        """多轮对话+多轮路由状态机核心函数"""
        # 1. 组装最基础的本次调用链上下文
        timestamp_prompt = self.system_prompt_template.format(timestamp=time.time())
        messages = [SystemMessage(content=timestamp_prompt)]
        
        # 2. 注入历史会话短期记忆（滑动窗口截取最近 N 条）
        active_memory = self.chat_history[-self.window_size:]
        messages.extend(active_memory)
        
        # 3. 压入本次最新的提问
        messages.append(HumanMessage(content=user_input))
        
        # 建立当前轮次的局部状态闭环（防止工具消息污染 self.chat_history）
        current_turn_messages = list(messages)
        
        loop_count = 0
        executed_queries = set()  # 用于方案三：关键词去重防死循环

        while loop_count < self.max_loops:
            loop_count += 1
            print(f"🔍 [Agent 思考中] 正在对当前上下文做决策 (循环 {loop_count}/{self.max_loops})...")
            
            # 方案一：动态 Prompt 诱导机制 (随着轮次增加，施加收敛压力)
            if loop_count == self.max_loops - 1:
                print("⏳ [系统干预] 检测到长轮次路由，正在向大模型注入收敛紧迫感...")
                current_turn_messages.append(SystemMessage(
                    content="【核心警告】：系统即将耗尽网络带宽。请不要再生成新的 tool_calls，请直接结合现有全部资料，产出最终报告。"
                ))
            
            # 决策：大模型判断是否需要额外数据
            response = await self.router_model.ainvoke(current_turn_messages)
            
            # 收敛退出点
            if not response.tool_calls:
                print("🏁 状态机判定：模型主动终止工具轮询，进入最终文本组织...")
                final_response = await self.base_model.ainvoke(current_turn_messages)
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=final_response.content))
                return final_response.content
                
            # 方案三：代码层拦截拦截死循环（去重检测）
            current_query = response.tool_calls[0]["args"].get("query", "")
            if current_query in executed_queries:
                print(f"⚠️ [拦截死循环] 检测到重复搜索关键词: '{current_query}'，强行阻断并引导收敛。")
                current_turn_messages.append(SystemMessage(
                    content="【系统提示】：你刚才已经搜索过完全相同的关键词，且没有找到更多新内容。请不要重复搜索，请立刻总结现有信息并回答。"
                ))
                continue # 跳过本次网络请求，直接让模型基于警告在下一轮做收敛判断
                
            # 记录执行过的关键词
            executed_queries.add(current_query)
            
            # 正常执行流程...
            current_turn_messages.append(response)
            tool_messages = await self._execute_tools_async(response.tool_calls)
            current_turn_messages.extend(tool_messages)
            
        raise TimeoutError("Agent 无法收敛，触发死循环保护。")

# === 工业级连续对话测试入口 ===
async def main():
    agent = ProductionMultiTurnAgent(window_size=4, max_loops=3)
    
    # --- 第一轮提问：测试多关键词并行联网与基础回答 ---
    q1 = "帮我查一下2026年最近有什么最新的低空经济或飞行汽车的核心痛点？"
    print(f"\n💬 用户 Question 1: {q1}")
    report1 = await agent.chat(q1)
    print(f"\n✨ [AI 行业研究员第一轮报告]：\n{report1}\n")
    print("-" * 50)
    
    await asyncio.sleep(2) # 喘息两秒
    
    # --- 第二轮提问：测试历史短期记忆留存 + 新的二级联网检索能力 ---
    q2 = "针对你刚才提到的第一个痛点，目前国内有什么代表性的企业在尝试解决它吗？"
    print(f"\n💬 用户 Question 2: {q2}")
    report2 = await agent.chat(q2)
    print(f"\n✨ [AI 行业研究员第二轮报告]：\n{report2}\n")

if __name__ == "__main__":
    asyncio.run(main())