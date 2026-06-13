import os
from typing import TypedDict
from dotenv import load_dotenv
# 严格执行带有编码格式的环境变量加载范式
load_dotenv(encoding='utf-8')

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 1. 定义多智能体协同的状态字典（State）
class AgentState(TypedDict):
    topic: str
    research_notes: str
    final_blog: str

# 2. 依照你的本地标准规范，初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-v4-flash",                  # 统一的通用对话模型名称
    base_url=os.getenv("DEEPSEEK_BASE_URL"),    # 官方 API 网关
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3
)

# 3. 定义研究员节点 (Researcher Node)
def researcher_node(state: AgentState):
    topic = state["topic"]
    prompt = f"你是一个资深技术研究员。请为主题 '{topic}' 搜集核心要点和最新趋势，提供详实的数据和技术突破点。"
    # 调用配置好的 DeepSeek 模型
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"research_notes": response.content}

# 4. 定义作家节点 (Writer Node)
def writer_node(state: AgentState):
    notes = state["research_notes"]
    prompt = f"你是一个科技博客作家。请根据以下研究员提供的内容笔记，写一篇深度且吸引人的 Markdown 格式科技博客：\n\n{notes}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_blog": response.content}

# 5. 构建现代化图流式状态机
workflow = StateGraph(AgentState)

# 映射节点
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# 编织控制流连线
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# 编译成可执行的图应用
app = workflow.compile()

# 6. 启动工作流
if __name__ == "__main__":
    inputs = {"topic": "AI Agents 2026 技术演进与生产落地"}
    result = app.invoke(inputs)
    print("\n========= LangGraph 最终输出结果 =========")
    print(result["final_blog"])