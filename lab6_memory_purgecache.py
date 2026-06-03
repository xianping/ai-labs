import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(encoding='utf-8')

class ExplicitWindowMemoryAgent:
    def __init__(self, window_size: int = 2):
        # 🤝 严格尊崇用户给出的工业级标准范式定义模型
        self.model = ChatOpenAI(
            model="deepseek-v4-flash", 
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0  # 设为 0，排除模型的随机发散干扰
        )
        
        # 核心防 Cache 机制：通过加入动态时间戳，强迫 DeepSeek 网关刷新物理缓存，不做前缀匹配
        self.system_prompt = f"你是一个 IT 导师。当前会话安全启动时间戳: {time.time()}"
        self.history = []
        self.window_size = window_size

    def chat(self, user_input: str) -> str:
        # 1. 组装消息队列，头部永远是系统提示词
        current_messages = [SystemMessage(content=self.system_prompt)]
        
        # 2. 严格根据窗口截取历史
        active_memory = self.history[-self.window_size:]
        current_messages.extend(active_memory)
        
        # 3. 放入最新提问
        current_messages.append(HumanMessage(content=user_input))
        
        print(f"\n📊 [监控] 本轮实际发给 DeepSeek 的消息链路结构:")
        for msg in current_messages:
            # 打印出真正发给服务器的文本，你会看到第一句在后续请求中被彻底踢出了
            print(f" -> [{type(msg).__name__}]: {msg.content[:30]}...") 
            
        # 4. 请求模型
        response = self.model.invoke(current_messages)
        
        # 5. 沉淀记忆
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))
        
        return response.content

if __name__ == "__main__":
    # window_size=2 意味着只能记住“上一句话和上一句回答”
    agent = ExplicitWindowMemoryAgent(window_size=2)
    
    print("\n🤖 AI 回复:", agent.chat("你好，我是放下夫，一个有着10多年经验的大数据工程师。"))
    time.sleep(1) # 稍微停顿，给网关中间件喘息和刷新的时间
    
    print("\n🤖 AI 回复:", agent.chat("我目前正在学习大模型的 Agent 架构开发。"))
    time.sleep(1)
    
    # 💡 此时发过去的消息只有：System, Human("我正在学习..."), AI("那太棒了..."), 以及下面这句提问。
    # 里面已经没有任何关于“放下夫”和“大数据”的字眼了！
    print("\n🤖 AI 回复:", agent.chat("请问我最开始告诉你，我的名字叫什么？我是做什么职业的？"))