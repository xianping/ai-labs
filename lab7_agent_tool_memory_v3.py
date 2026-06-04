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

    def _is_similar_query(self, new_query: str, executed_queries: set, threshold: float = 0.7) -> bool:
        """
        判断当前 query 是否与已执行的 query 存在高相似度（Jaccard 相似度）
        """
        if not new_query:
            return False
            
        # 1. 基础清洗：转小写、去标点、按空格拆分成词集
        def get_words(text: str):
            clean_text = text.lower().replace("的", "").replace("了", "").replace("年", "")
            return set([w for w in clean_text.split() if w.strip()])
            
        new_words = get_words(new_query)
        if not new_words:
            return False
            
        for old_query in executed_queries:
            old_words = get_words(old_query)
            if not old_words:
                continue
                
            # 计算交集与并集比例
            intersection = new_words.intersection(old_words)
            union = new_words.union(old_words)
            similarity = len(intersection) / len(union)
            
            # 如果重合度高于阈值，或者一方是另一方的子集
            if similarity >= threshold or new_words.issubset(old_words) or old_words.issubset(new_words):
                return True
                
        return False

    async def chat(self, user_input: str) -> str:
        """多轮对话+多轮路由状态机核心函数（优化版）"""
        # 1. 组装基础上下文
        timestamp_prompt = self.system_prompt_template.format(timestamp=time.time())
        messages = [SystemMessage(content=timestamp_prompt)]
        
        # 2. 注入历史会话短期记忆
        active_memory = self.chat_history[-self.window_size:]
        messages.extend(active_memory)
        
        # 3. 压入本次最新的提问
        messages.append(HumanMessage(content=user_input))
        current_turn_messages = list(messages)
        
        loop_count = 0
        executed_queries = set()  # 存放历史搜索过的原始 query 字符串

        while loop_count < self.max_loops:
            loop_count += 1
            print(f"🔍 [Agent 思考中] 正在对当前上下文做决策 (循环 {loop_count}/{self.max_loops})...")
            
            # ================= 优化点 1：把强力收敛提前到倒数第一轮 =================
            if loop_count == self.max_loops:
                print("⏳ [临界干预] 达到最大循环轮次！强行剥夺工具调用权限，逼迫模型产出最终报告...")
                current_turn_messages.append(SystemMessage(
                    content="【绝对命令】：网络连接已永久切断！你无法再调用任何工具。请立刻整合前面几轮搜索到的所有线索，直接为用户产出最终的专业研究报告，拒绝客套话。"
                ))
                # 核心改变：直接用 base_model（没绑定工具的模型）接管，让它没有机会生成 tool_calls
                final_response = await self.base_model.ainvoke(current_turn_messages)
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=final_response.content))
                return final_response.content
            
            # 方案一：倒数第二轮施加压力
            if loop_count == self.max_loops - 1:
                print("⏳ [系统提示] 接近尾声，正在向大模型注入收敛紧迫感...")
                current_turn_messages.append(SystemMessage(
                    content="【重要提示】：由于带宽限制，这是你最后一轮搜索机会。请确保本次调用的工具能拿到最终所需的数据，下一轮必须交出最终报告。"
                ))
            
            # 决策：大模型判断是否需要额外数据
            response = await self.router_model.ainvoke(current_turn_messages)
            
            # 收敛退出点 1：模型主动终止工具轮询
            if not response.tool_calls:
                print("🏁 状态机判定：模型主动终止工具轮询，进入最终文本组织...")
                final_response = await self.base_model.ainvoke(current_turn_messages)
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=final_response.content))
                return final_response.content
                
            # 方案三：模糊重复检索拦截
            is_loop_detected = False
            for tool_call in response.tool_calls:
                current_query = tool_call["args"].get("query", "")
                if self._is_similar_query(current_query, executed_queries, threshold=0.7):
                    print(f"⚠️ [拦截死循环] 检测到相似重复搜索: '{current_query}'，强行掐断路由。")
                    is_loop_detected = True
                    break
            
            # 收敛退出点 2：被代码层判定为复读，强制收敛
            if is_loop_detected:
                current_turn_messages.append(SystemMessage(
                    content="【系统干预】：检测到你在重复检索类似信息。请立刻利用已知的所有线索，直接回答用户。"
                ))
                print("🏁 强行拦截：强制模型进入最终文本组织状态...")
                final_response = await self.base_model.ainvoke(current_turn_messages)
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=final_response.content))
                return final_response.content

            # 正常流程：记录本次执行的所有 query
            for tool_call in response.tool_calls:
                q_str = tool_call["args"].get("query", "")
                if q_str:
                    executed_queries.add(q_str)
            
            # 正常执行工具调用
            current_turn_messages.append(response)
            tool_messages = await self._execute_tools_async(response.tool_calls)
            current_turn_messages.extend(tool_messages)
            
        # 如果走出了 while 循环还没 return，说明逻辑设计或 max_loops 的条件控制出了漏洞
        raise TimeoutError("Agent 意外跳出状态机，触发死循环保护。")

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