import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
# 导入 HuggingFace 专属的本地轻量化嵌入类
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END
"""
pip install langchain-huggingface sentence-transformers
using hugging face small local embedding model.
"""
# 1. 严格加载环境变量
load_dotenv(encoding="utf-8")

# 2. 思考/生成组件（继续压榨云端 DeepSeek API 的超高性价比，不吃本地资源）
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    openai_api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    temperature=0.2
)

# 3. 核心替换：初始化本地轻量化 HuggingFace Embedding 模型
# 💡 这里我们选用 BAAI 的 bge-small-zh-v1.5，体积仅不到 100MB，专为中文语义检索优化
print("⏳ 正在初始化本地 HuggingFace 轻量化向量模型 (首次运行若无缓存会自动下载，仅约 90MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 初始化 LangChain 内存向量数据库（绑定本地 HF 向量引擎）
vector_store = InMemoryVectorStore(embeddings)

# ---------------------------------------------------------------------------
# 🗄️ 模拟数据灌入（长期记忆初始化）
# ---------------------------------------------------------------------------
print("📦 [LangChain] 正在通过本地轻量化 HF 模型序列化并挂载长期记忆资产...")
vector_store.add_texts([
    "项目CodeNexus的负责人是张三，当前项目进度为80%，预计下个月底上线。",
    "公司关于远程办公的规定：员工每周一和周五必须在办公室，周二至周四可申请居家。",
    "关于报销：差旅住宿费每日上限为北京/上海 500 元，其他城市 350 元。"
])
print("✅ 长期记忆本地向量化挂载成功！")

# ---------------------------------------------------------------------------
# 🗺️ LangGraph 状态机定义
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str                  # 用户输入的问题
    retrieved_context: List[str]   # RAG 检索出来的上下文片段
    reply: str                     # LLM 最终生成的回答

# 节点 1：检索节点 (Retrieve Node)
def retrieve_knowledge_node(state: AgentState):
    print(f"\n🔍 [LangGraph - Node: Retrieve] 正在利用本地 HF 向量检索与 '{state['question']}' 相关的记忆...")
    
    # 本地进行余弦相似度计算，秒级响应
    docs = vector_store.similarity_search(state["question"], k=2)
    context_list = [doc.page_content for doc in docs]
    
    return {"retrieved_context": context_list}

# 节点 2：生成节点 (Generate Node)
def generate_answer_node(state: AgentState):
    print("🤖 [LangGraph - Node: Generate] 正在合并上下文，驱动云端 DeepSeek 思考回答...")
    
    context_str = "\n".join([f"- {c}" for c in state["retrieved_context"]])
    
    system_prompt = f"""你是一个拥有企业长期规章记忆的 AI 助理。
请严格基于以下长期记忆上下文回答用户问题，如果无法从中得出结论，请礼貌告知。

<long_term_memory>
{context_str}
</long_term_memory>
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"])
    ]
    
    response = llm.invoke(messages)
    return {"reply": response.content}

# ---------------------------------------------------------------------------
# 🏗️ 构建 LangGraph 拓扑图
# ---------------------------------------------------------------------------
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("retrieve_node", retrieve_knowledge_node)
builder.add_node("generate_node", generate_answer_node)

# 构建控制流边
builder.add_edge(START, "retrieve_node")
builder.add_edge("retrieve_node", "generate_node")
builder.add_edge("generate_node", END)

# 编译图
rag_app = builder.compile()

# ---------------------------------------------------------------------------
# 🚀 运行与流水线验证
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    user_query = "我想问一下，我们下周二能直接申请在家里办公吗？"
    
    print(f"🚀 [Trigger] 用户发起提问: {user_query}")
    
    # 启动 LangGraph 引擎
    inputs = {"question": user_query}
    result = rag_app.invoke(inputs)
    
    print("\n==================== 🎉 最终执行结果 ====================")
    print(f"🤖 Agent 最终回复:\n{result['reply']}")