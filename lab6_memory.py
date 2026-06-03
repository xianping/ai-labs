import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv(encoding='utf-8')

class WindowMemoryAgent:
    def __init__(self, window_size: int = 4):
        self.model = ChatOpenAI(model="deepseek-v4-flash", 
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1",
                    temperature=0)
        
        self.system_prompt = "你是一个贴心的 IT 团队架构师导师，请用专业且温和的语气回答问题。"
        # 用于存储真实历史消息的内存双端队列（也可以用原生 list 模拟）
        self.history = []
        # 限制最多保留最近几条消息（4 条相当于 2 轮一问一答）
        self.window_size = window_size

    def chat(self, user_input: str) -> str:
        # 1. 构造当前请求的消息队列
        current_messages = [HumanMessage(content=self.system_prompt)]
        
        # 2. 注入滑动窗口内的历史记忆
        # 取最近的 N 条历史消息
        active_memory = self.history[-self.window_size:]
        current_messages.extend(active_memory)
        
        # 3. 塞入本次最新的提问
        current_messages.append(HumanMessage(content=user_input))
        
        print(f"📊 [调试-内存监控] 当前实际喂给大模型的历史记忆条数: {len(active_memory)}")
        
        # 4. 请求大模型
        response = self.model.invoke(current_messages)
        
        # 5. 【核心】：将本次交互的双向数据，沉淀进历史记忆库
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response.content))
        
        return response.content

if __name__ == "__main__":
    agent = WindowMemoryAgent(window_size=4)
    
    # 模拟连续对话
    print("AI:", agent.chat("你好，我是放下夫，一个有着10多年经验的大数据工程师。"))
    print("AI:", agent.chat("我目前正在学习大模型的 Agent 架构开发。"))
    print("AI:", agent.chat("你还记得我的名字以及我是做什么的吗？"))
    print("AI:", agent.chat("今天北京天气怎么样？"))
    print("AI:", agent.chat("我们最开始聊的我的职业是什么来着？")) # 💡 注意：这时候由于窗口滑动，最早的记忆已经被洗掉了！