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
        while loop_count < self.max_loops:
            loop_count += 1
            print(f"🔍 [Agent 思考中] 正在对当前上下文做决策 (循环 {loop_count})...")
            #############
            # 【工业级防御】：如果已经到了最后一轮，强行打破僵局，不再让模型做路由决策
            if loop_count == self.max_loops:
                print("⚠️ [安全限制] 已达到最大搜寻次数上限，强行终止工具轮询，进行平滑降级答复...")
                final_response = await self.base_model.ainvoke(current_turn_messages)
                
                # 沉淀记忆
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=final_response.content))
                return final_response.content

            ##########
            # 决策：大模型判断是否需要额外数据
            response = await self.router_model.ainvoke(current_turn_messages)
            
            # 如果判定不需要工具，说明本轮信息已完整，收敛输出
            if not response.tool_calls:
                print("🏁 状态机判定：当前轮次数据收敛完毕。正在组织文本报告...")
                
                # 显式使用基础 model 实例进行收尾，彻底杜绝内容截断，拉满 content 吞吐量
                final_response = await self.base_model.ainvoke(current_turn_messages)
                
                # 【最重要的一步】：将真实的 HumanInput 和 AI 的最终文本回复，持久化沉淀进全局历史中
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=final_response.content))
                
                return final_response.content
                
            # 如果判定需要工具：将包含 tool_calls 的原生 AIMessage 塞入当前轮次的局部队列
            current_turn_messages.append(response)
            print(f"🤖 大模型判定：触发并行工具调用，共分配 {len(response.tool_calls)} 个并发网络任务。")
            
            # 高并发异步执行网络检索
            tool_messages = await self._execute_tools_async(response.tool_calls)
            
            # 严格回填闭环数据，塞入当前轮次队列，供大模型在下一轮 while 循环中阅读
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