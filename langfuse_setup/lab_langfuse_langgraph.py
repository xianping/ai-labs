import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 💡 核心导入：4.9.0 时代驱动 LangGraph 的正统回调处理器
from langfuse.callback import CallbackHandler

# 1. 加载高级环境变量
load_dotenv()

# 2. 定义 LangGraph 拓扑状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. 初始化大模型（确保你的环境变量里有 DEEPSEEK_API_KEY 和对应的 BASE_URL）
# 这里我们使用兼容 OpenAI 规范的客户端加载 DeepSeek
model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7
)

# 4. 定义图节点（纯净的业务逻辑，不需要加任何 @observe 装饰器！）
def researcher_node(state: AgentState):
    print("      研究员节点正在思考...")
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def critic_node(state: AgentState):
    print("      ⚙️ 审计员节点正在审查...")
    # 模拟审计员在历史消息后追加意见
    audit_prompt = state["messages"] + [HumanMessage(content="请对上述回答进行安全合规性审计，指出是否有越狱风险，并给出最终结论。")]
    response = model.invoke(audit_prompt)
    return {"messages": [response]}

# 5. 构建 LangGraph 拓扑网状图
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)

# 编排工作流：开始 -> 研究员 -> 审计员 -> 结束
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_edge("critic", END)

# 编译状态机
app = workflow.compile()

# ==========================================
# 🚀 核心追踪逻辑：4.9.0 标准运行时拦截
# ==========================================
if __name__ == "__main__":
    print("===== 🚀 开始执行 LangGraph 自动化流水线 =====")
    
    # 1. 初始化 4.9.0 专属回调处理器（它会自动读取你的 .env 中的 LANGFUSE 配置）
    langfuse_handler = CallbackHandler()
    
    # 2. 模拟用户输入
    initial_input = {
        "messages": [HumanMessage(content="如何评价大模型全链路追踪技术？")]
    }
    
    # 3. 核心卡点：通过 config 显式注入 callbacks
    # 这样整个图里所有的 Node 调用、大模型 Token 消耗、耗时都会被自动捕获
    config = {
        "callbacks": [langfuse_handler],
        "configurable": {"thread_id": "agent_session_001"}
    }
    
    # 4. 触发流式运行
    final_state = app.invoke(initial_input, config=config)
    
    print("\n===== 🏁 执行完毕，最终图输出结果 =====")
    print(final_state["messages"][-1].content)
    
    # 5. 确保 4.9.0 异步任务在上报完成后再关闭程序
    langfuse_handler.flush()
    print("\n🎉 链路数据已成功无感异步上报至 Langfuse 后台！")