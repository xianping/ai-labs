import os
from typing import List, Dict, Any
# 🌟 采纳你的指正：严谨加载环境变量
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# ==========================================
# 0. 初始化基础客户端
# ==========================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.7,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# ==========================================
# 1. 托管 Agent 状态的数据仓 (State Memory)
# ==========================================
class LangChainSummaryMemoryAgent:
    def __init__(self):
        self.summary = "当前没有任何历史背景。"
        self.recent_messages: List[BaseMessage] = [] # 存储 LangChain 的 Message 对象
        self.max_buffer_turns = 3

    def _compress_memory_internal(self):
        """
        后台隐形任务：使用 LangChain 调用大模型进行记忆压缩
        """
        print("\n⚠️ [LangChain Memory Engine] 检测到短期对话过长，正在启动后台记忆压缩...")
        
        # 将 LangChain 的 Message 对象数组转化为纯文本串供总结使用
        history_text = ""
        for msg in self.recent_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            history_text += f"{role}: {msg.content}\n"

        # 组装压缩 Prompt
        compress_prompt = ChatPromptTemplate.from_template("""
        你是一个记忆管理专家。请把下方的[长期记忆存量]和[最新对话增量]进行无缝合并，生成一段新的、精简的全局记忆摘要。
        
        <当前长期记忆存量>
        {summary}
        </当前长期记忆存量>
        
        <最新对话增量>
        {history_text}
        </最新对话增量>
        
        <约束条件>
        1. 摘要必须极其精炼，删掉寒暄和废话，仅保留核心事实（如：用户的姓名、提到的技能、聊过的主题）。
        2. 字数严格控制在 150 字以内。
        </约束条件>
        """)

        # 运行压缩链
        compress_chain = compress_prompt | llm
        response = compress_chain.invoke({
            "summary": self.summary,
            "history_text": history_text
        })
        
        self.summary = response.content.strip()
        self.recent_messages = [] # 清空未压缩的短期队列
        print(f"✅ [LangChain Memory Engine] 压缩完成！新长期记忆摘要：\n👉 {self.summary}\n")

    def chat(self, user_input: str) -> str:
        """
        使用 LCEL 动态链运行对话
        """
        # 🌟 核心学习点：ChatPromptTemplate 可以通过 MessagesPlaceholder 动态插入一个 Message 数组
        # 这比我们手写拼接 messages 列表优雅且安全得多！
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能助理。请结合[核心历史背景]来回答当前用户的问题。
            
            <核心历史背景>
            {current_summary}
            </核心历史背景>"""),
            MessagesPlaceholder(variable_name="chat_history"), # 动态占位符：近期未压缩的短期记忆
            ("user", "{input}")
        ])

        # 组装 LCEL 声明式链
        chain = prompt_template | llm

        # 运行 Chain
        response = chain.invoke({
            "current_summary": self.summary,
            "chat_history": self.recent_messages,
            "input": user_input
        })

        reply = response.content

        # 沉淀到短期记忆队列中（使用 LangChain 的标准消息对象）
        self.recent_messages.append(HumanMessage(content=user_input))
        self.recent_messages.append(AIMessage(content=reply))

        # 判定是否触发压缩（一问一答算2条消息）
        if len(self.recent_messages) >= self.max_buffer_turns * 2:
            self._compress_memory_internal()

        return reply

# ==========================================
# 3. 运行测试：完全对齐手写版本的测试流程
# ==========================================
if __name__ == "__main__":
    agent = LangChainSummaryMemoryAgent()
    
    print("--- 👤 第一轮对话 ---")
    q1 = "你好，我叫王五，我目前在自学 AI Agent 开发，今天刚学完了 Context Engineering。"
    print(f"User: {q1}")
    print(f"Agent: {agent.chat(q1)}\n")
    
    print("--- 👤 第二轮对话 ---")
    q2 = "我以前有 3 年的全栈开发经验，精通 Python 和 Vue。我很看好 Agent 的未来。"
    print(f"User: {q2}")
    print(f"Agent: {agent.chat(q2)}\n")
    
    print("--- 👤 第三轮对话 ---")
    q3 = "我目前在 Windows 电脑上用 VS Code 敲代码，用的是 DeepSeek 的 API。"
    print(f"User: {q3}")
    print(f"Agent: {agent.chat(q3)}\n")
    
    print("--- 👤 第四轮对话（验证失忆情况） ---")
    q4 = "既然你认识我了，请问我叫什么名字？我有几年什么开发经验？我现在正在学什么？"
    print(f"User: {q4}")
    print(f"Agent: {agent.chat(q4)}\n")