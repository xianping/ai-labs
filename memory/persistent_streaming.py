import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# 核心导入：内存级持久化检查点管理器
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 严格的环境变量加载范式
# ==========================================
load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

if not DEEPSEEK_API_KEY:
    raise ValueError("❌ 错误：请在 .env 文件中配置 DEEPSEEK_API_KEY！")

# ==========================================
# 2. 状态结构定义与大模型初始化
# ==========================================
class AgentState(TypedDict):
    # 使用 add_messages 机制，使新消息自动 append 到消息列表中
    messages: Annotated[List[BaseMessage], add_messages]

# 初始化你指定的 deepseek-v4-flash 模型
llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.3
)

# ==========================================
# 3. 定义图节点（Nodes）
# ==========================================
def assistant_node(state: AgentState):
    """助手节点：负责接收当前状态中的消息列表，并调用大模型生成回复"""
    print("\n[Node 🎬] Assistant 节点正在处理...")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ==========================================
# 4. 构建带 Persistence 的拓扑图
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# 【核心关卡】实例化持久化检查点
# 在工业级生产中可以替换为 SqliteSaver 或 MongoDBSaver
memory_checkpointer = MemorySaver()

# 编译图时，将 checkpointer 作为核心固件注入
graph = builder.compile(checkpointer=memory_checkpointer)

# ==========================================
# 5. 动手演练：验证多轮记忆（Persistence）与流式更新（Streaming）
# ==========================================
if __name__ == "__main__":
    print("🚀 LangGraph 状态机编译成功！开始进行持久化与流式测试...\n")

    # 💡 概念要点：必须提供 thread_id 才能让 checkpointer 识别同一根“对话线”
    thread_config = {"configurable": {"thread_id": "user_session_999"}}

    # --- 第一轮对话 ---
    user_msg_1 = HumanMessage(content="你好！我是小明，我是一名资深的 Python 后端工程师。")
    print("--- 📥 发送第一轮消息 ---")
    
    # 演示 Streaming：以 updates 模式流式输出图中节点的变更状态
    for event in graph.stream({"messages": [user_msg_1]}, config=thread_config, stream_mode="updates"):
        for node_name, node_update in event.items():
            print(f"📡 实时流更新来自 [{node_name}]:")
            # 打印该节点塞进 State 的最新消息
            for msg in node_update.get("messages", []):
                if isinstance(msg, AIMessage):
                    print(f"🤖 助手原始回答: {msg.content}")

    print("\n" + "="*50 + "\n")

    # --- 第二轮对话（测试 Agent 的记忆保留能力） ---
    # 我们完全没有在输入中提“小明”或者“Python”，看它能否利用 Persistence 记住
    user_msg_2 = HumanMessage(content="你还记得我叫什么名字，以及我的职业是什么吗？")
    print("--- 📥 发送第二轮消息（依赖持久化线索） ---")
    
    for event in graph.stream({"messages": [user_msg_2]}, config=thread_config, stream_mode="updates"):
        for node_name, node_update in event.items():
            print(f"📡 实时流更新来自 [{node_name}]:")
            for msg in node_update.get("messages", []):
                if isinstance(msg, AIMessage):
                    print(f"🤖 助手记忆读取回答: {msg.content}")

    print("\n" + "="*50 + "\n")

    # --- 第三轮对话（换一个 thread_id，验证状态隔离） ---
    different_thread_config = {"configurable": {"thread_id": "user_session_888"}}
    user_msg_3 = HumanMessage(content="你知道我是谁吗？")
    print("--- 📥 发送第三轮消息（换了新的隔离线程 id） ---")
    
    for event in graph.stream({"messages": [user_msg_3]}, config=different_thread_config, stream_mode="updates"):
        for node_name, node_update in event.items():
            print(f"📡 实时流更新来自 [{node_name}]:")
            for msg in node_update.get("messages", []):
                if isinstance(msg, AIMessage):
                    print(f"🤖 独立线程回答: {msg.content}")