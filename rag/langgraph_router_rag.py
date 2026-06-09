import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
# 导入 Pydantic 用于严格的结构化输出控制
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# 1. 严格加载环境变量
load_dotenv(encoding="utf-8")

# 2. 初始化核心思考组件
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    openai_api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.1
)

# 3. 初始化你的轻量化本地 HF 向量引擎（利用你刚才下载好的缓存，秒启动）
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = InMemoryVectorStore(embeddings)

# 4. 灌入公司长期记忆规章
vector_store.add_texts([
    "项目CodeNexus的负责人是张三，当前项目进度为80%，预计下个月底上线。",
    "公司关于远程办公的规定：员工每周一和周五必须在办公室，周二至周四可申请居家。",
    "关于报销：差旅住宿费每日上限为北京/上海 500 元，其他城市 350 元。"
])

# ---------------------------------------------------------------------------
# 🧠 路由决策的结构化输出定义
# ---------------------------------------------------------------------------
class RouteDecision(BaseModel):
    """让大模型严格按照 JSON 格式返回路由抉择"""
    reason: str = Field(description="做出该路由选择的简短原因")
    target: str = Field(description="必须是 'retrieve'（需要查私有知识/记忆）或 'chat'（属于日常闲聊/通用常识）")

# ---------------------------------------------------------------------------
# 🗺️ LangGraph 状态机定义
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    route_target: str              # 新增：记录路由决策结果
    retrieved_context: List[str]
    reply: str

# 节点 1：智能路由决策节点 (Router Node)
def router_node(state: AgentState):
    print(f"\n🧠 [LangGraph - Node: Router] 正在分析用户意图: '{state['question']}'")
    
    # 强制让大模型以结构化 JSON 的形式回答，避免概率性胡言乱语
    #NOTE: here will generate ERROR in Deepseek, As deepseek doesn't support the Json schema
    structured_llm = llm.with_structured_output(RouteDecision)
    
    router_prompt = """你是一个高精度的路由网关裁判。
分析用户的输入，判断这个问题是否需要查阅“公司规章、报销标准、项目进度、团队负责人”等企业内部私有长期记忆知识。
- 如果需要查询企业私有记忆，target 字段必须填 'retrieve'。
- 如果仅仅是日常问候、闲聊、或者完全不需要企业背景的通用常识问题（如写代码、晚餐推荐、科学常识），target 字段必须填 'chat'。"""

    decision = structured_llm.invoke([
        SystemMessage(content=router_prompt),
        HumanMessage(content=state["question"])
    ])
    
    print(f"   -> 🧠 路由判断理由: {decision.reason}")
    print(f"   -> 🚀 决定分流至: {decision.target}")
    
    return {"route_target": decision.target}

# 节点 2：检索节点
def retrieve_knowledge_node(state: AgentState):
    print(f"🔍 [LangGraph - Node: Retrieve] 检测到合规请求，启动本地 HF 向量检索...")
    docs = vector_store.similarity_search(state["question"], k=2)
    context_list = [doc.page_content for doc in docs]
    return {"retrieved_context": context_list}

# 节点 3：生成节点
def generate_answer_node(state: AgentState):
    print("🤖 [LangGraph - Node: Generate] 驱动 DeepSeek 生成最终回复...")
    
    # 如果检索上下文为空（说明跳过了检索节点），context_str 就是空的
    contexts = state.get("retrieved_context", [])
    if contexts:
        context_str = "\n".join([f"- {c}" for c in contexts])
        system_prompt = f"请严格基于以下长期记忆上下文回答用户问题：\n<long_term_memory>\n{context_str}\n</long_term_memory>"
    else:
        system_prompt = "你是一个全能的 AI 助理，请直接热情地回答用户的问题。"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"])
    ]
    
    response = llm.invoke(messages)
    return {"reply": response.content}

# ---------------------------------------------------------------------------
# 🏗️ 构建“带条件边”的 LangGraph 拓扑图
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

# 挂载三个节点
builder.add_node("router_node", router_node)
builder.add_node("retrieve_node", retrieve_knowledge_node)
builder.add_node("generate_node", generate_answer_node)

# 定义路由边的跳转条件（根据状态里的 route_target 决定下一步去哪）
def route_decision_edge(state: AgentState):
    if state["route_target"] == "retrieve":
        return "retrieve_node"     # 走检索分支
    else:
        return "generate_node"     # 绕过检索，直接去生成

# 构建有向控制图连线
builder.add_edge(START, "router_node")

# 🌟 核心硬核点：添加条件边
# 参数1：从哪个节点发出
# 参数2：条件决策函数（返回字符串）
# 参数3：路由映射表（把函数返回的字符串映射到图中的真实节点名）
builder.add_conditional_edges(
    "router_node",
    route_decision_edge,
    {
        "retrieve_node": "retrieve_node",
        "generate_node": "generate_node"
    }
)

# 补齐后续的连线
builder.add_edge("retrieve_node", "generate_node")
builder.add_edge("generate_node", END)

# 编译图应用
rag_router_app = builder.compile()

# ---------------------------------------------------------------------------
# 🚀 极限压力测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    # 🧪 测试场景 1：需要触发 RAG 检索的问题
    print("\n--- 🌟 压力测试 1：企业内部私有知识查询 ---")
    res1 = rag_router_app.invoke({"question": "张三最近在负责哪个项目？进度到哪了？"})
    print(f"🤖 回复 1:\n{res1['reply']}")
    print("="*60)
    
    # 🧪 测试场景 2：完全不需要触发 RAG 的闲聊常识
    print("\n--- 🌟 压力测试 2：通用生活健康常识（应自动跳过检索） ---")
    res2 = rag_router_app.invoke({"question": "今晚吃点什么既健康又不容易长胖？推荐3个菜。"})
    print(f"🤖 回复 2:\n{res2['reply']}")
    print("="*60)