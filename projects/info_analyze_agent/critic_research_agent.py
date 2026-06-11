import os
import json
from typing import List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime

# 导入 LangGraph 核心组件
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 导入 LangChain 相关组件
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# 导入外部工具 已经move到ddgs package
# from duckduckgo_search import DDGS
from ddgs import DDGS

import chromadb
import chromadb.utils.embedding_functions as ef
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

# 1. 显式指定 UTF-8 编码加载环境
load_dotenv(encoding='utf-8')

# 确保 OpenAI 兼容的环境变量（DeepSeek 使用）
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

# =====================================================================
# 1. 初始化本地 HuggingFace Embedding 模型
# =====================================================================
print("📦 正在加载本地 Hugging Face 语义嵌入模型...")
hf_embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)

# 初始化 Chroma 持久化客户端
chroma_client = chromadb.PersistentClient(path="./.chroma_data")

# 包装器：兼容 LangChain 嵌入函数与 ChromaDB 原生接口
# 包装器：严格兼容 ChromaDB 原生接口
class ChromaHFEmbeddingFunction(EmbeddingFunction):
    def __init__(self, hf_embeddings: HuggingFaceEmbeddings):
        self.hf = hf_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB 无论是对 documents 还是对 query_texts 编码，都会调用这个方法"""
        # 确保输入是列表
        if isinstance(input, str):
            input = [input]
        
        # embed_documents 返回的是 List[List[float]]
        embeddings = self.hf.embed_documents(input)
        
        # 显式转换为原生 float，防止某些 numpy 类型导致序列化问题
        return [[float(x) for x in emb] for emb in embeddings]

# 创建包装器实例
hf_ef_wrapper = ChromaHFEmbeddingFunction(hf_embedding)

# 创建或获取集合，绑定本地 HuggingFace 嵌入函数
collection = chroma_client.get_or_create_collection(
    name="ai_intelligence_v2",
    embedding_function=hf_ef_wrapper
)

# =====================================================================
# 2. 多模型配置（关闭思考模式，确保结构化输出正常）
# =====================================================================
# 通用模型参数：禁用 thinking mode（必须）
# COMMON_MODEL_KWARGS = {
#     "thinking": {"type": "disabled"}
# }

# 审计官：使用轻量快速模型，关闭思考模式
llm_flash = ChatDeepSeek(
    model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-chat"),  # 可使用 deepseek-chat 或 deepseek-v4-flash
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
    # ,
    # model_kwargs=COMMON_MODEL_KWARGS
)

# 研究员 & 分析师：使用深度推理模型，同样关闭思考模式
llm_pro = ChatDeepSeek(
    model=os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-reasoner"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3
    # ,
    # model_kwargs=COMMON_MODEL_KWARGS
)

# =====================================================================
# 3. 结构化输出的 Pydantic 模型
# =====================================================================
class CriticReview(BaseModel):
    is_sufficient: bool = Field(description="当前搜集到的原始数据是否足以完美且有深度地回答用户问题。")
    reason: str = Field(description="做出该判断的详细理由，若为False必须指出缺失维度和下一轮搜索指令。")

# =====================================================================
# 4. 状态定义与全局 Token 账单
# =====================================================================
class AgentState(TypedDict):
    query: str
    research_notes: List[str]
    critic_feedback: str
    loop_count: int
    final_report: str
    total_prompt_tokens: int
    total_completion_tokens: int

TOKEN_BUDGET_LIMIT = 50000

def track_tokens(state: AgentState, response_metadata: Dict[str, Any]) -> Dict[str, int]:
    """安全统计 Token 消耗，兼容元数据缺失的情况"""
    token_usage = response_metadata.get("token_usage", {})
    p_tokens = token_usage.get("prompt_tokens", 0)
    c_tokens = token_usage.get("completion_tokens", 0)
    
    new_p = state.get("total_prompt_tokens", 0) + p_tokens
    new_c = state.get("total_completion_tokens", 0) + c_tokens
    
    print(f"   [本节点消耗] Prompt: {p_tokens} | Completion: {c_tokens}")
    print(f"   [项目总账单] Prompt累计: {new_p} | Completion累计: {new_c} (上限: {TOKEN_BUDGET_LIMIT})")
    
    return {
        "total_prompt_tokens": new_p,
        "total_completion_tokens": new_c
    }

# =====================================================================
# 5. 图节点定义
# =====================================================================

def researcher_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n======== 👩‍💻 研究员节点启动 (第 {state['loop_count'] + 1} 轮) ========")
    query = state["query"]
    feedback = state["critic_feedback"]
    
    if feedback:
        print("💡 收到审计官打回意见，正在重新设计搜索方向...")
        search_prompt = f"基于核心课题: '{query}'。\n务必注意审计官的修正意见: '{feedback}'。\n请为搜索引擎提炼出下一步最精准的关键词。"
    else:
        search_prompt = f"核心课题: '{query}'。\n请提取适合搜索引擎的初始核心关键词。"
        
    res = llm_pro.invoke([
        SystemMessage(content="你是一个顶尖的科技情报检索专家。请直接输出适合搜索引擎的简短关键词，不要带任何标点符号或Markdown。"),
        HumanMessage(content=search_prompt)
    ])
    
    token_updates = track_tokens(state, res.response_metadata)
    search_keywords = res.content.strip()
    print(f"🔍 Pro 模型生成的优化关键词: 【{search_keywords}】")
    
    # 联网检索
    fetched_texts = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_keywords, max_results=3))
            for r in results:
                fetched_texts.append(f"[公网最新资讯] 标题: {r['title']}\n内容: {r['body']}\n链接: {r.get('href', 'None')}")
    except Exception as e:
        print(f"⚠️ DuckDuckGo 公网检索跳过: {e}")

    # 本地 RAG 检索
    try:
        rag_results = collection.query(query_texts=[query], n_results=2)
        if rag_results and rag_results['documents'] and rag_results['documents'][0]:
            for doc in rag_results['documents'][0]:
                fetched_texts.append(f"[本地沉淀资产] {doc}")
                print("📦 从本地 Chroma 库匹配到关联历史资产！")
    except Exception as e:
        print(f"⚠️ 本地 Chroma 检索跳过: {e}")

    if not fetched_texts:
        fetched_texts.append(f"[系统提示] 针对关键词 '{search_keywords}' 未能捕获到新信息。")

    updated_notes = list(state["research_notes"]) + fetched_texts
    
    return {
        "research_notes": updated_notes,
        "loop_count": state["loop_count"] + 1,
        **token_updates
    }


def critic_node(state: AgentState) -> Dict[str, Any]:
    print("\n======== 👨‍⚖️ 审计官节点启动 ========")
    query = state["query"]
    notes = state["research_notes"]
    
    formatted_notes = "\n\n".join([f"--- 资料片段 {i+1} ---\n{note}" for i, note in enumerate(notes)])
    
    system_prompt = (
        "你是一名极其苛刻的科技情报审计官（Critic）。\n"
        "检查当前搜集到的所有资料是否足够详尽。如果不满足技术深度、或者缺乏时效性，必须无情地判定为 False，并给出后续检索关键词提示。\n"
        "你的输出必须严格符合给定的 JSON Schema 格式。"
    )
    user_content = f"研究课题: {query}\n\n已收集到的全部资料如下:\n{formatted_notes}"
    
    # 使用结构化输出（已关闭思考模式，不再报错）
    structured_flash = llm_flash.with_structured_output(CriticReview)
    
    try:
        review: CriticReview = structured_flash.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        
        # 尝试获取响应元数据（LangChain 结构化输出可能附加在 .response_metadata 或通过最后一条消息获得）
        metadata = {}
        if hasattr(review, 'response_metadata'):
            metadata = review.response_metadata
        elif hasattr(review, '_response_metadata'):
            metadata = review._response_metadata
        else:
            # 降级：主动调用一次空请求获取 token 用量？避免复杂，暂标记为0并警告
            print("   ⚠️ 无法从结构化输出中提取 Token 元数据，本次审计不计费（不影响后续熔断）")
        
        # token_updates = track_tokens(state, metadata) if metadata else {"total_prompt_tokens": 0, "total_completion_tokens": 0}
        if metadata:
            token_updates = track_tokens(state, metadata)
        else:
            print("   ⚠️ 无法获取 Token 元数据，本次审计不计费")
            token_updates = {
                "total_prompt_tokens": state.get("total_prompt_tokens", 0),
                "total_completion_tokens": state.get("total_completion_tokens", 0)
    }
        print(f"📊 审计结论 -> 是否详尽: 【{review.is_sufficient}】")
        print(f"📝 审计批注: {review.reason}")
        
        return {
            "critic_feedback": review.reason if not review.is_sufficient else "",
            **token_updates
        }
    except Exception as e:
        print(f"⚠️ 审计官结构化输出异常: {e}")
        # 降级方案：尝试普通 JSON 输出
        try:
            simple_response = llm_flash.invoke([
                SystemMessage(content=system_prompt + "\n请直接输出一个 JSON 对象，格式为 {\"is_sufficient\": true/false, \"reason\": \"...\"}，不要其他文字。"),
                HumanMessage(content=user_content)
            ])
            # 尝试解析 JSON
            json_text = simple_response.content.strip()
            # 提取大括号内的内容（防止模型输出额外说明）
            start = json_text.find('{')
            end = json_text.rfind('}') + 1
            if start != -1 and end > start:
                json_text = json_text[start:end]
                data = json.loads(json_text)
                is_suff = bool(data.get("is_sufficient", True))
                reason = data.get("reason", "")
                token_updates = track_tokens(state, simple_response.response_metadata)
                return {
                    "critic_feedback": reason if not is_suff else "",
                    **token_updates
                }
            else:
                raise ValueError("未找到有效 JSON")
        except Exception as json_err:
            print(f"⚠️ 降级解析也失败: {json_err}，默认放行")
            return {"critic_feedback": ""}


def reporter_node(state: AgentState) -> Dict[str, Any]:
    print("\n======== 📝 报告生成与长期记忆沉淀节点启动 ========")
    query = state["query"]
    notes = state["research_notes"]
    
    formatted_notes = "\n\n".join(notes)
    system_prompt = "你位居顶尖 AI 行业首席分析师。请基于通过严格审计的丰富原始素材，撰写一份排版完美的 Markdown 深度行业研究报告。"
    
    res = llm_pro.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"课题: {query}\n\n经审计过审的素材:\n{formatted_notes}")
    ])
    
    token_updates = track_tokens(state, res.response_metadata)
    final_report_md = res.content
    
    # 知识回写本地向量库
    print("💾 正在通过本地 HuggingFace Embedding 将研报写入 ChromaDB...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection.add(
            documents=[final_report_md],
            metadatas=[{"query": query, "type": "final_report", "saved_at": timestamp}],
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
# 6. 路由决策（含 Token 熔断）
# =====================================================================
def should_continue(state: AgentState) -> Literal["researcher", "reporter"]:
    feedback = state["critic_feedback"]
    loop_count = state["loop_count"]
    total_tokens = state.get("total_prompt_tokens", 0) + state.get("total_completion_tokens", 0)
    
    if not feedback:
        print("🎉 审计官高度认可，批准下发生成最终报告！")
        return "reporter"
    
    if total_tokens >= TOKEN_BUDGET_LIMIT:
        print(f"🚨 财务红色警告：当前总 Token 消耗【{total_tokens}】已超出安全预算【{TOKEN_BUDGET_LIMIT}】！执行硬熔断收敛。")
        return "reporter"
    
    if loop_count >= 3:
        print("🚨 循环次数到头，强行收敛去写报告。")
        return "reporter"
    
    print(f"🔄 路由决策: 审计未通过，打回研究员节点继续深挖。")
    return "researcher"

# =====================================================================
# 7. 构建 LangGraph 工作流
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
# 8. 本地测试
# =====================================================================
if __name__ == "__main__":
    test_query = "调研 2026 年最新发布的 DeepSeek-R1 模型在强化学习领域的核心创新，并与其之前的旧版本进行横向对比"
    
    print(f"🚀 核心流测试启动！研究课题: 【{test_query}】")
    
    initial_state: AgentState = {
        "query": test_query,
        "research_notes": [],
        "critic_feedback": "",
        "loop_count": 0,
        "final_report": "",
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0
    }
    
    final_output = app.invoke(initial_state)
    
    print("\n" + "="*55)
    print("🏆 最终交付的研究报告 :")
    print("="*55)
    print(final_output["final_report"])
    
    print("\n" + "="*55)
    print("📈 本次任务最终本地计费总账单 :")
    print(f"总计 Prompt Tokens 消耗: {final_output['total_prompt_tokens']}")
    print(f"总计 Completion Tokens 消耗: {final_output['total_completion_tokens']}")
    print(f"合计总 Token 消耗: {final_output['total_prompt_tokens'] + final_output['total_completion_tokens']}")
    print("="*55)