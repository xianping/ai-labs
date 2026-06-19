import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
# ==========================================
# 🚀 拥抱最新版：直接从 langfuse 导入被包裹的 OpenAI 客户端
# ==========================================
from langfuse.openai import OpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langfuse.decorators import langfuse_context
# 1. 加载高级环境变量（自动读取你的 LANGFUSE_PUBLIC_KEY, SECRET_KEY 和 HOST）
load_dotenv()

# 2. 初始化最新版 Langfuse 托管的客户端
# 它会自动在底层代理所有的 trace 上报，再也不需要满世界传 callback 了！
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# 3. 定义 LangGraph 拓扑状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. 定义图节点（用最新版原生 SDK 替换掉 langchain 包装，更透明可控）
def researcher_node(state: AgentState):
    print("      🔍 研究员节点正在思考...")
    
    # 提取最后一条用户消息
    user_content = state["messages"][-1].content
    
    # 使用新版规范：指定 trace 节点名称
    response = client.chat.completions.create(
        name="researcher-node-llm", # 网页端直接显示的 Step 名字
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": user_content}],
        temperature=0.7
    )
    
    ai_output = response.choices[0].message.content
    return {"messages": [HumanMessage(content=ai_output)]}

def critic_node(state: AgentState):
    print("      ⚙️ 审计员节点正在审查...")
    
    previous_reply = state["messages"][-1].content
    
    # 同样使用新版规范：无感追踪审计流
    response = client.chat.completions.create(
        name="critic-node-audit",
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "请对回答进行安全合规性审计，指出风险。精简限定150字以内"},
            {"role": "user", "content": previous_reply}
        ],
        temperature=0.1
    )
    
    audit_output = response.choices[0].message.content
    return {"messages": [HumanMessage(content=audit_output)]}

# 5. 构建 LangGraph 拓扑状态机
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_edge("critic", END)

app = workflow.compile()

# ==========================================
# 🚀 运行时调用
# ==========================================
if __name__ == "__main__":
    print("===== 🚀 开始执行 LangGraph + Langfuse 最新原生追踪 =====")
    
    initial_input = {
        "messages": [HumanMessage(content="如何评价大模型全链路追踪技术？精简限定150字以内")]
    }
    
    # 彻底告别 config={"callbacks": [...]} 的累赘！
    final_state = app.invoke(initial_input)
    
    print("\n===== 🏁 执行完毕，最终图输出结果 =====")
    print(final_state["messages"][-1].content)
    
    # # 2. 🔥 核心卡点：强行把异步队列里的数据清空并安全发射
    # print("\n⏳ 正在强制同步数据至本地 Langfuse...")
    # langfuse_context.flush()

    print("\n🎉 链路数据已通过最新版 Native Wrap 机制异步同步至本地 Langfuse 后台！")