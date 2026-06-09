import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
# 导入 Pydantic 输出解析器
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# 1. 严格加载环境变量
load_dotenv(encoding="utf-8")

# 2. 初始化核心思考组件
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    openai_api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.1
)

# 3. 初始化本地轻量化 HF 向量引擎
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = InMemoryVectorStore(embeddings)

# 4. 灌入公司长期记忆规章
vector_store.add_texts([
    "项目CodeNexus的负责人是张三，当前项目进度为80%，预计下个月底上线。",
    "公司关于远程办公的规定：员工每周一和周五必须在办公室，周二至周四可申请居家。",
    "关于报销：差旅住宿费每日上限为北京/上海 500 元，其他城市 350 元。"
])

# ---------------------------------------------------------------------------
# 🧠 路由决策的结构化输出与解析器定义
# ---------------------------------------------------------------------------
class RouteDecision(BaseModel):
    reason: str = Field(description="做出该路由选择的简短原因")
    target: str = Field(description="必须是 'retrieve'（需要查私有知识）或 'chat'（属于日常闲聊/通用常识）")

# 初始化 LangChain 原生解析器
parser = PydanticOutputParser(pydantic_object=RouteDecision)

# ---------------------------------------------------------------------------
# 🗺️ LangGraph 状态机定义
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    route_target: str              
    retrieved_context: List[str]
    reply: str

# 节点 1：智能路由决策节点 (Router Node)
def router_node(state: AgentState):
    print(f"\n🧠 [LangGraph - Node: Router] 正在分析用户意图: '{state['question']}'")
    
    # 动态注入解析器的格式化指令，让提示词里自动包含严格的 JSON 约束规范
    format_instructions = parser.get_format_instructions()
    
    router_prompt = f"""你是一个高精度的路由网关裁判。
分析用户的输入，判断这个问题是否需要查阅“公司规章、报销标准、项目进度、团队负责人”等企业内部私有长期记忆知识。
- 如果需要查询企业私有记忆，target 字段必须填 'retrieve'。
- 如果仅仅是日常问候、闲聊、或者完全不需要企业背景的通用常识问题（如写代码、晚餐推荐），target 字段必须填 'chat'。

{format_instructions}
"""

    response = llm.invoke([
        SystemMessage(content=router_prompt),
        HumanMessage(content=state["question"])
    ])
    
    try:
        # 代码层解析大模型吐出的纯文本 JSON，100% 稳健
        decision = parser.parse(response.content)
        print(f"   -> 🧠 路由判断理由: {decision.reason}")
        print(f"   -> 🚀 决定分流至: {decision.target}")
        return {"route_target": decision.target}
    except Exception as e:
        # 降级兜底方案：如果模型没有完美按 JSON 返回，默认去向量库捞一把，保证核心业务不挂
        print(f"   ⚠️ 结构化解析失败，启动 RAG 兜底策略: {e}")
        return {"route_target": "retrieve"}

# 节点 2：检索节点
def retrieve_knowledge_node(state: AgentState):
    print(f"🔍 [LangGraph - Node: Retrieve] 检测到合规请求，启动本地 HF 向量检索...")
    docs = vector_store.similarity_search(state["question"], k=2)
    context_list = [doc.page_content for doc in docs]
    return {"retrieved_context": context_list}

# 节点 3：生成节点
def generate_answer_node(state: AgentState):
    print("🤖 [LangGraph - Node: Generate] 驱动 DeepSeek 生成最终回复...")
    
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

builder.add_node("router_node", router_node)
builder.add_node("retrieve_node", retrieve_knowledge_node)
builder.add_node("generate_node", generate_answer_node)

def route_decision_edge(state: AgentState):
    if state["route_target"] == "retrieve":
        return "retrieve_node"     
    else:
        return "generate_node"     

builder.add_edge(START, "router_node")

builder.add_conditional_edges(
    "router_node",
    route_decision_edge,
    {
        "retrieve_node": "retrieve_node",
        "generate_node": "generate_node"
    }
)

builder.add_edge("retrieve_node", "generate_node")
builder.add_edge("generate_node", END)

rag_router_app = builder.compile()

# ---------------------------------------------------------------------------
# 🚀 极限压力测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    print("\n--- 🌟 压力测试 1：企业内部私有知识查询 ---")
    res1 = rag_router_app.invoke({"question": "张三最近在负责哪个项目？进度到哪了？"})
    print(f"🤖 回复 1:\n{res1['reply']}")
    print("="*60)
    
    print("\n--- 🌟 压力测试 2：通用生活健康常识（应自动跳过检索） ---")
    res2 = rag_router_app.invoke({"question": "今晚吃点什么既健康又不容易长胖？推荐3个菜。"})
    print(f"🤖 回复 2:\n{res2['reply']}")
    print("="*60)