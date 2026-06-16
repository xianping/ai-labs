import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI  # 新增导入

from dotenv import load_dotenv

# 1. 严格执行带有编码格式的环境变量加载范式
load_dotenv(encoding='utf-8')

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
# 引入你提到的官方标准化高层 Agent 包
from langchain.agents import create_agent

# ==========================================
# 2. 定义智能体之间流转的 Pydantic 数据模型 (结构化契约)
# ==========================================

class ResearchReport(BaseModel):
    """研究员交付的标准结构化报告"""
    key_trends: list[str] = Field(description="解构出的核心技术突破点和演进趋势列表")
    technical_depth: str = Field(description="关于底层架构、技术痛点的深度技术分析细节")
    confidence_score: float = Field(description="研究员对自己搜集到的数据可信度的打分 (0.0-1.0)")

class BlogOutput(BaseModel):
    """作家最终生成的结构化交付物"""
    markdown_content: str = Field(description="排版优美、包含二级标题和结构化清单的完整 Markdown 博客正文")
    estimated_read_time: int = Field(description="预计读者阅读所需的分钟数")

# ==========================================
# 3. 定义 LangGraph 拓扑图的状态机存储结构 (State)
# ==========================================
from typing import TypedDict, Optional

class AgentState(TypedDict):
    topic: str
    # 状态机里直接存储 Pydantic 强类型对象，彻底告别字符串拼接带来的混乱
    research_data: Optional[ResearchReport]
    final_blog_data: Optional[BlogOutput]

# ==========================================
# 4. 利用官方推荐的 create_agent 机制，封装并初始化 DeepSeek 智能体
# ==========================================

# 统一映射到你的本地 DeepSeek 官方配置网关
# 提示：create_agent 支持直接传入符合模型的配置
model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
    # extra_body={"thinking": False}   # 关键：关闭思考模式
    extra_body={"thinking": {"type": "disabled"}}   # 关键修正

)


# 实例化【强类型结构化研究员】
# 传入官方定义格式，通过 response_format 强制让 DeepSeek 返回 ResearchReport 实例
researcher_agent = create_agent(
    model=model, 
    tools=[],  # 后续可直接在此塞入你的 duckgo search 或 ChromaDB 检索工具
    response_format=ResearchReport,
)

# 实例化【强类型结构化作家】
writer_agent = create_agent(
    model=model,
    tools=[],
    response_format=BlogOutput,
)

# ==========================================
# 5. 极其干净的 LangGraph 节点（Nodes）实现
# ==========================================

def researcher_node(state: AgentState):
    """研究员节点：调用官方高层 agent，抽取强类型数据"""
    topic = state["topic"]
    
    prompt_content = (
        f"你是一个顶尖的技术研究员。请对主题 '{topic}' 进行深度解构。\n"
        f"请严格按照指定的格式输出关键趋势、深度的底层技术架构分析，并对可信度评估打分。"
    )
    
    # 按照官方统一的统一调用格式输入 messages
    result = researcher_agent.invoke({
        "messages": [{"role": "user", "content": prompt_content}]
    })
    
    # 官方 create_agent 会自动将解析好的 Pydantic 对象塞进 "structured_response" 键
    structured_report: ResearchReport = result["structured_response"]
    
    # 将强类型对象写回图的状态中
    return {"research_data": structured_report}


def writer_node(state: AgentState):
    """作家节点：消费研究员的强类型成果，输出最终结构化博客"""
    report: ResearchReport = state["research_data"]
    
    prompt_content = (
        f"你是一个科技媒体专栏作家。请根据以下由研究员提供的高质量结构化报告，"
        f"撰写一篇引人入胜的 Markdown 格式技术博客。\n\n"
        f"核心趋势: {report.key_trends}\n"
        f"底层技术分析: {report.technical_depth}\n"
        f"数据可信度参考: {report.confidence_score}\n\n"
        f"请同时返回整篇 Markdown 文章内容并预估读者的阅读时间。"
    )
    
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": prompt_content}]
    })
    
    structured_blog: BlogOutput = result["structured_response"]
    
    return {"final_blog_data": structured_blog}

# ==========================================
# 6. 拓扑图的编织与编译
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# ==========================================
# 7. 运行验证
# ==========================================
if __name__ == "__main__":
    inputs = {"topic": "AI Agents 2026 技术演进与生产落地"}
    
    print("🚀 正在使用 LangChain 官方最新的强类型高层 create_agent 架构运行 DeepSeek...")
    result = app.invoke(inputs)
    
    final_blog: BlogOutput = result["final_blog_data"]
    
    print("\n========= 📥 强类型多智能体流转成功 =========")
    print(f"⏱️ 预计阅读时间：{final_blog.estimated_read_time} 分钟")
    print("\n========= 📝 最终生成的 Markdown 博客正文 =========")
    print(final_blog.markdown_content)