import os
import json
from typing import List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime

# LangGraph & LangChain Components
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB 原生组件
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# 联网工具
from ddgs import DDGS
from dotenv import load_dotenv

# 1. 初始化环境变量
load_dotenv(encoding='utf-8')
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

# =====================================================================
# 2. 初始化本地 ChromaDB 与 Embedding
# =====================================================================
print("📦 正在加载本地 Hugging Face 语义嵌入模型...")
hf_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)
chroma_client = chromadb.PersistentClient(path="./.chroma_data")

class ChromaHFEmbeddingFunction(EmbeddingFunction):
    def __init__(self, hf_embeddings: HuggingFaceEmbeddings):
        self.hf = hf_embeddings
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.hf.embed_documents(input)
        return [[float(x) for x in emb] for emb in embeddings]

hf_ef_wrapper = ChromaHFEmbeddingFunction(hf_embedding)
collection = chroma_client.get_or_create_collection(
    name="ai_intelligence_v3",
    embedding_function=hf_ef_wrapper
)

# =====================================================================
# 3. 初始化多模型（使用 DeepSeek 官方端点）
# =====================================================================
llm_flash = ChatDeepSeek(
    model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
)

llm_pro = ChatDeepSeek(
    model=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-reasoner"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3
)

# =====================================================================
# 4. 结构化输出模型：澄清与追问接口 (Clarification Interface)
# =====================================================================
class CriticReview(BaseModel):
    is_sufficient: bool = Field(description="当前收集到的素材是否足以完美回答课题。")
    reason: str = Field(description="做出该判断的详细理由（通俗指出哪里不够好、不solid）。")
    missing_dimensions: List[str] = Field(description="明确缺失的知识维度。例如：['OpenAI最新模型价格', 'Anthropic路线图对比']")
    suggested_keywords: str = Field(description="针对缺失维度，精确提炼的下一轮检索关键词（不带标点）。")
    recommended_source: Literal["general_search", "tech_depth"] = Field(
        description="战略性更换工具指导：最新发布/价格用 general_search；底层架构/深度路线用 tech_depth。"
    )

# =====================================================================
# 5. 状态定义与 Token 账单
# =====================================================================
class AgentState(TypedDict):
    query: str
    research_notes: List[str]
    loop_count: int
    final_report: str
    # 核心进化：澄清上下文路由
    is_sufficient: bool
    critic_reason: str
    missing_dimensions: List[str]
    suggested_keywords: str
    recommended_source: str
    # Token 统计
    total_prompt_tokens: int
    total_completion_tokens: int

TOKEN_BUDGET_LIMIT = 60000

def track_tokens(state: AgentState, response_metadata: Dict[str, Any]) -> Dict[str, int]:
    token_usage = response_metadata.get("token_usage", {})
    p_tokens = token_usage.get("prompt_tokens", 0)
    c_tokens = token_usage.get("completion_tokens", 0)
    new_p = state.get("total_prompt_tokens", 0) + p_tokens
    new_c = state.get("total_completion_tokens", 0) + c_tokens
    print(f"   [本节点消耗] Prompt: {p_tokens} | Completion: {c_tokens}")
    print(f"   [项目总账单] 累计: {new_p + new_c} / {TOKEN_BUDGET_LIMIT}")
    return {"total_prompt_tokens": new_p, "total_completion_tokens": new_c}

# =====================================================================
# 6. 图节点定义 (Nodes)
# =====================================================================

def researcher_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n======== 👩‍💻 研究员节点启动 (第 {state['loop_count'] + 1} 轮) ========")
    query = state["query"]
    
    # 提取上一轮来自审计官的澄清指南
    keywords = state.get("suggested_keywords", "").strip()
    source_tool = state.get("recommended_source", "general_search")
    missing_dims = state.get("missing_dimensions", [])

    # 如果是第一轮，初始化关键词
    if not keywords:
        print("💡 初始轮次：正在生成初始检索蓝图...")
        res = llm_pro.invoke([
            SystemMessage(content="你是一个顶级科技情报专家。请输出针对该课题最精准的初始检索关键词，不要带标点或Markdown。"),
            HumanMessage(content=f"课题: {query}")
        ])
        keywords = res.content.strip()
        source_tool = "general_search"
    else:
        print(f"🎯 收到审计官澄清要求！专门针对缺失维度: {missing_dims}")
        print(f"🔄 决定更换/优化数据源，使用工具类型: 【{source_tool}】")

    print(f"🔍 本轮检索关键词: 【{keywords}】")
    fetched_texts = []

    # 动态工具箱切换机制
    if source_tool == "tech_depth":
        # 模拟深度技术渠道（补充专业 prompt 特征词以过滤噪音）
        keywords += " architecture technical whitepaper deep dive"
    
    try:
        with DDGS() as ddgs:
            # 针对不同工具类型可以调整搜索范围或条数
            max_res = 4 if source_tool == "tech_depth" else 3
            results = list(ddgs.text(keywords, max_results=max_res))
            for r in results:
                prefix = "[深度技术分析]" if source_tool == "tech_depth" else "[全网最新资讯]"
                fetched_texts.append(f"{prefix} 标题: {r['title']}\n内容: {r['body']}\n链接: {r.get('href', 'None')}")
    except Exception as e:
        print(f"⚠️ 联网检索跳过: {e}")

    # 本地 RAG 激活
    try:
        rag_results = collection.query(query_texts=[keywords if keywords else query], n_results=1)
        if rag_results and rag_results['documents'] and rag_results['documents'][0]:
            for doc in rag_results['documents'][0]:
                fetched_texts.append(f"[本地历史沉淀资产] {doc}")
                print("📦 激活本地持久化关联资产！")
    except Exception as e:
        print(f"⚠️ 本地 Chroma 检索跳过: {e}")

    if not fetched_texts:
        fetched_texts.append(f"[系统提示] 针对关键词 '{keywords}' 本轮未捕获到新信息。")

    return {
        "research_notes": list(state["research_notes"]) + fetched_texts,
        "loop_count": state["loop_count"] + 1
    }


def critic_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n======== 👨‍⚖️ 审计官沟通与澄清节点启动 ========")
    query = state["query"]
    notes = state["research_notes"]
    loop_count = state["loop_count"]
    
    formatted_notes = "\n\n".join([f"--- 素材片段 {i+1} ---\n{note}" for i, note in enumerate(notes)])
    
    # 动态期望值机制：随着轮数增加，引导审计官收敛，不要无限挑刺
    flexibility_guide = ""
    if loop_count >= 2:
        flexibility_guide = "\n⚠️ 注意：当前已经是多轮对话澄清。如果全网确实缺乏某些前沿细节，不要死缠烂打。允许你在 reason 中写明‘数据受限但可交付’并将 is_sufficient 置为 true。"

    system_prompt = (
        f"你是一名极其苛刻但也具备良好沟通常识的科技情报审计官（Critic）。\n"
        f"你需要审查目前的素材是否足以撰写关于 OpenAI 和 Anthropic 的深度研报。\n"
        f"研报标准包含：最新发布、技术路线、商业收益、核心局限、未来预测。\n"
        f"如果不 solid，请明确告诉研究员【哪个维度不够好】、【哪个数据不 solid】，并指明下一轮他应该【去哪里、用什么关键词】查找。{flexibility_guide}\n"
        f"输出必须严格符合给定的 JSON Schema 格式。"
    )
    
    structured_flash = llm_flash.with_structured_output(CriticReview)
    
    try:
        review: CriticReview = structured_flash.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"研报课题: {query}\n\n当前研究员交付的全部素材:\n{formatted_notes}")
        ])
        
        metadata = getattr(review, 'response_metadata', getattr(review, '_response_metadata', {}))
        token_updates = track_tokens(state, metadata) if metadata else {"total_prompt_tokens": state["total_prompt_tokens"], "total_completion_tokens": state["total_completion_tokens"]}
        
        print(f"📊 审计沟通结果 -> 是否通关: 【{review.is_sufficient}】")
        print(f"📝 审计官批注反馈: {review.reason}")
        if not review.is_sufficient:
            print(f"🧭 指派缺失维度: {review.missing_dimensions}")
            print(f"🎯 推荐下一轮工具: {review.recommended_source}")

        return {
            "is_sufficient": review.is_sufficient,
            "critic_reason": review.reason,
            "missing_dimensions": review.missing_dimensions,
            "suggested_keywords": review.suggested_keywords,
            "recommended_source": review.recommended_source,
            **token_updates
        }
    except Exception as e:
        print(f"⚠️ 结构化审计异常，启动宽容放行降级: {e}")
        return {"is_sufficient": True, "critic_reason": "降级放行"}


def reporter_node(state: AgentState) -> Dict[str, Any]:
    print("\n======== 📝 资深首席分析师报告生成节点启动 ========")
    query = state["query"]
    notes = state["research_notes"]
    critic_reason = state.get("critic_reason", "")
    is_sufficient = state.get("is_sufficient", True)

    formatted_notes = "\n\n".join(notes)
    
    # 如果最后仍然数据不全，带着局限性去写
    appendix_prompt = ""
    if not is_sufficient:
        appendix_prompt = f"\n⚠️ 注意：由于全网公开技术资产受限，审计官指出以下维度未能完美对齐：{state.get('missing_dimensions', [])}。请在报告末尾开辟【研究局限性与免责声明】章节，将这些未澄清的盲点转化为专业的风险提示。"

    system_prompt = (
        "你位居全球顶级 AI 行业首席资深分析师。\n"
        "请基于多轮人机协同攻坚获取的丰富素材，撰写一份结构化、排版完美的 Markdown 深度行业研究研报。\n"
        "报告必须深度覆盖以下 5 大纵深维度：\n"
        "1. 最新核心技术发布矩阵\n"
        "2. 技术路线演进与底层核心特点（如强化学习、Agent架构等对比）\n"
        "3. 商业落地、API推理成本与商业ROI收益分析\n"
        "4. 核心技术局限性与现阶段痛点\n"
        "5. 未来 2-3 年技术演进路线预测与对抗格局{appendix_prompt}"
    )
    
    res = llm_pro.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"研报课题: {query}\n\n历经多轮澄清并过审的完整原始素材:\n{formatted_notes}\n\n审计官最终批注: {critic_reason}")
    ])
    
    token_updates = track_tokens(state, res.response_metadata)
    final_report_md = res.content
    
    # 长期记忆沉淀
    print("💾 正在将高价值研报沉淀回本地向量库...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection.add(
            documents=[final_report_md],
            metadatas=[{"query": query, "type": "industry_report", "saved_at": timestamp}],
            ids=[f"report_{timestamp}"]
        )
        print("✅ 长期记忆本地持久化成功！")
    except Exception as e:
        print(f"⚠️ 写入本地向量库失败: {e}")
        
    return {
        "final_report": final_report_md,
        **token_updates
    }

# =====================================================================
# 7. 路由决策（动态收敛）
# =====================================================================
def should_continue(state: AgentState) -> Literal["researcher", "reporter"]:
    is_sufficient = state["is_sufficient"]
    loop_count = state["loop_count"]
    total_tokens = state.get("total_prompt_tokens", 0) + state.get("total_completion_tokens", 0)
    
    if is_sufficient:
        print("🎉 达成共识！审计官与研究员完成认知对齐，下发生成最终报告。")
        return "reporter"
    
    if total_tokens >= TOKEN_BUDGET_LIMIT:
        print(f"🚨 触发 Token 预算熔断（当前 {total_tokens}），带着现有资产强行进入生成流程。")
        return "reporter"
    
    if loop_count >= 3:
        print("🚨 达到最大澄清轮次（3轮），强制转入报告拼装模式。")
        return "reporter"
    
    print(f"🔄 路由决策: 双方仍在沟通中，进入下一轮定向爆破。")
    return "researcher"

# =====================================================================
# 8. 构建 LangGraph 工作流
# =====================================================================
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("reporter", reporter_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "researcher": "researcher",
        "reporter": "reporter"
    }
)
workflow.add_edge("reporter", END)
app = workflow.compile()

# =====================================================================
# 9. 执行重构后的测试
# =====================================================================
if __name__ == "__main__":
    test_query = "深入调研 2026 年 OpenAI 与 Anthropic 两大巨头最新发布的核心模型与技术成果，横向对比其技术路线、落地收益、现存核心技术痛点并对未来格局做出预测。"
    
    print(f"🚀 协同澄清流研报系统启动！\n课题: 【{test_query}】")
    
    initial_state: AgentState = {
        "query": test_query,
        "research_notes": [],
        "loop_count": 0,
        "final_report": "",
        "is_sufficient": False,
        "critic_reason": "",
        "missing_dimensions": [],
        "suggested_keywords": "",
        "recommended_source": "general_search",
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0
    }
    
    final_output = app.invoke(initial_state)
    
    print("\n" + "="*60)
    print("🏆 最终交付的顶级行业研究报告 :")
    print("="*60)
    print(final_output["final_report"])