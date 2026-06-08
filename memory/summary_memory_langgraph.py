import os
from typing import List, Dict, Any, Literal
# 🌟 始终坚持最严谨的环境变量加载
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# 引入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# ==========================================
# 0. 初始化基础大模型客户端
# ==========================================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.7,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# ==========================================
# 1. 定义工业级图状态 (Graph State)
# ==========================================
# 在 LangGraph 中，所有的节点都共享并操作这一个全局状态
class AgentState(TypedDict):
    input: str                       # 用户当前输入的文本
    current_summary: str             # 长期记忆：压缩后的摘要
    chat_history: List[BaseMessage]  # 短期记忆：未压缩的聊天记录列表
    agent_reply: str                 # 大模型本次的回答

# ==========================================
# 2. 定义图节点 (Nodes) —— 具体的业务执行单元
# ==========================================

def chat_node(state: AgentState) -> Dict[str, Any]:
    """
    节点1：负责带上长期摘要和短期历史，与用户进行聊天
    """
    print("🤖 [Node: Chat] 正在结合历史上下文生成回答...")
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助理。请结合[核心历史背景]来回答当前用户的问题。
        
        <核心历史背景>
        {current_summary}
        </核心历史背景>"""),
        MessagesPlaceholder(variable_name="chat_history"), # 动态平铺近期对话
        ("user", "{input}")
    ])
    
    chain = prompt_template | llm
    response = chain.invoke({
        "current_summary": state["current_summary"],
        "chat_history": state["chat_history"],
        "input": state["input"]
    })
    
    # 将最新的一轮对话，追加到短期历史队列中
    updated_history = list(state["chat_history"])
    updated_history.append(HumanMessage(content=state["input"]))
    updated_history.append(AIMessage(content=response.content))
    
    # 返回更新后的状态（LangGraph 会自动做字典合并覆盖）
    return {
        "agent_reply": response.content,
        "chat_history": updated_history
    }


def compress_node(state: AgentState) -> Dict[str, Any]:
    """
    节点2：负责在后台无感压缩记忆
    """
    print("⚠️ [Node: Compress] 检测到短期对话过长，启动后台记忆蒸馏...")
    
    # 将短期记忆转化为文本
    history_text = ""
    for msg in state["chat_history"]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        history_text += f"{role}: {msg.content}\n"
        
    compress_prompt = ChatPromptTemplate.from_template("""
    你是一个记忆 management 专家。请把下方的[长期记忆存量]和[最新对话增量]进行无缝合并，生成一段新的、精简的全局记忆摘要。
    
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
    
    chain = compress_prompt | llm
    response = chain.invoke({
        "summary": state["current_summary"],
        "history_text": history_text
    })
    
    # 压缩完成后，更新长期摘要，同时【清空】短期记忆
    return {
        "current_summary": response.content.strip(),
        "chat_history": [] # 彻底清空短期缓存，释放上下文
    }

# ==========================================
# 3. 定义条件路由边 (Conditional Edge Router)
# ==========================================
def should_compress_router(state: AgentState) -> Literal["to_compress", "to_end"]:
    """
    路由决策器：判断接下来应该去压缩节点，还是直接结束流程
    """
    # 满3轮对话（一问一答共6条message）触发压缩
    if len(state["chat_history"]) >= 6:
        return "to_compress"
    return "to_end"

# ==========================================
# 4. 构建与编译 LangGraph 拓扑图
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("chat_node", chat_node)
workflow.add_node("compress_node", compress_node)

# 配置连线关系（数据流向）
workflow.add_edge(START, "chat_node") # 起点进入聊天节点

# 🌟 从聊天节点出来时，交由路由函数决定去向
workflow.add_conditional_edges(
    "chat_node", 
    should_compress_router,
    {
        "to_compress": "compress_node", # 如果路由说要压缩，流向压缩节点
        "to_end": END                   # 如果路由说不用，直接结束本次会话
    }
)

# 压缩节点执行完后，也直接结束
workflow.add_edge("compress_node", END)

# 编译拓扑图，生成可运行的 app 实例
app = workflow.compile()

# ==========================================
# 5. 模拟工业级生产调用
# ==========================================
if __name__ == "__main__":
    # 初始化全局持久化状态
    current_state = {
        "input": "",
        "current_summary": "当前没有任何历史背景。",
        "chat_history": [],
        "agent_reply": ""
    }
    
    turns = [
        "你好，我叫王五，我目前在自学 AI Agent 开发，今天刚学完了 Context Engineering。",
        "我以前有 3 年的全栈开发经验，精通 Python 和 Vue。我很看好 Agent 的未来。",
        "我目前在 Windows 电脑上用 VS Code 敲代码，用的是 DeepSeek 的 API。",
        "既然你认识我了，请问我叫什么名字？我有几年什么开发经验？我现在正在学什么？"
    ]
    
    print("--- 🚀 LangGraph 工业级状态引擎启动 ---")
    
    for i, user_input in enumerate(turns):
        print(f"\n--- 👤 第 {i+1} 轮对话 ---")
        # 接收用户新输入，更新进状态字典
        current_state["input"] = user_input
        print(f"User: {user_input}")
        
        # 丢进 LangGraph 状态图引擎中流转
        output_state = app.invoke(current_state)
        
        # 打印大模型的回答
        print(f"Agent: {output_state['agent_reply']}")
        
        # 🌟 核心点：将图运行完吐出的全新 State 覆盖本地 State，实现多轮状态持久化
        current_state = output_state
        
        # 悄悄打印一下当前幕后的长期记忆状态，方便你作为架构师进行观测
        print(f"💡 [幕后看盘 - 长期记忆仓]: {current_state['current_summary']}")
        print(f"💡 [幕后看盘 - 短期消息数]: {len(current_state['chat_history'])} 条消息")