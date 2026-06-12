import os
import json
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_deepseek import ChatDeepSeek

from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv(encoding='utf-8')
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("DEEPSEEK_BASE_URl")
# 使用 2026 年的主流大模型
llm_flash = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
llm_pro = ChatDeepSeek(model="deepseek-reasoner", temperature=0.3)

# =====================================================================
# 1. 辩论达成共识后的结构化输出模型
# =====================================================================
class ClarificationConsensus(BaseModel):
    is_sufficient: bool = Field(description="经过双向澄清后，双方是否达成共识认为当前素材已足够写报告。")
    debate_summary: str = Field(description="简述刚才研究员与审计官争论、澄清的核心焦点。")
    next_action_memo: str = Field(description="双方妥协/对齐后的下一步行动备忘录。如果数据不够，说明接下来怎么找、找什么。")
    suggested_keywords: str = Field(description="双方对齐后提炼的下一轮精确检索词（无标点）。")
    recommended_source: Literal["general_search", "tech_depth"] = Field(description="达成共识后指派的工具渠道。")

# =====================================================================
# 2. 状态定义
# =====================================================================
class AgentState(TypedDict):
    query: str
    research_notes: List[str]
    loop_count: int
    final_report: str
    # 双向交谈后的沉淀变量
    is_sufficient: bool
    consensus_memo: str       # 双方澄清后的备忘录（下一次检索的指挥棒）
    suggested_keywords: str
    recommended_source: str
    # Token 统计
    total_prompt_tokens: int
    total_completion_tokens: int

# =====================================================================
# 3. 节点定义
# =====================================================================

def researcher_node(state: AgentState) -> Dict[str, Any]:
    print(f"\n======== 👩‍💻 研究员节点：执行检索 (第 {state['loop_count'] + 1} 轮) ========")
    query = state["query"]
    memo = state.get("consensus_memo", "")
    keywords = state.get("suggested_keywords", "").strip()
    source_tool = state.get("recommended_source", "general_search")

    # 初始轮次处理
    if not keywords:
        print("💡 初始轮次：正在构建初始检索蓝图...")
        res = llm_flash.invoke([HumanMessage(content=f"请为课题提炼简短的搜索引擎关键词: {query}")])
        keywords = res.content.strip()

    if memo:
        print(f"📋 拿着与审计官【澄清会谈】后的备忘录去攻坚: {memo}")

    print(f"🔍 正在通过 【{source_tool}】 检索关键词: 【{keywords}】")
    fetched_texts = []
    
    # 执行工具检索
    try:
        if source_tool == "tech_depth":
            keywords += " filetype:pdf technical deep dive"
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords, max_results=3))
            for r in results:
                fetched_texts.append(f"[{source_tool.upper()}] 标题: {r['title']}\n内容: {r['body']}")
    except Exception as e:
        print(f"⚠️ 联网检索跳过: {e}")

    return {
        "research_notes": list(state["research_notes"]) + fetched_texts,
        "loop_count": state["loop_count"] + 1
    }


def debate_room_node(state: AgentState) -> Dict[str, Any]:
    """
    🔥 核心重构：双向辩论与澄清室
    这里不再是单向通知，而是让 Researcher 角色和 Critic 角色现场进行对话交谈。
    """
    print(f"\n======== 🤝 进入双向澄清辩论室 (第 {state['loop_count']} 轮会谈) ========")
    query = state["query"]
    notes = state["research_notes"]
    
    formatted_notes = "\n\n".join([f"[-]: {n}" for n in notes])
    
    # ─── 模拟人与人合作的双向拉扯会话 ───
    print("🤖 审计官(Critic) 正在审查并率先发难...")
    # 1. Critic 发难
    critic_prompt = f"针对课题 '{query}'，研究员找了以下素材:\n{formatted_notes}\n作为苛刻的审计官，指出你最不满意的1-2个章节或不solid的数据硬伤。"
    critic_msg = llm_flash.invoke([SystemMessage(content="你是严苛的科技情报审计官。"), HumanMessage(content=critic_prompt)]).content
    print(f"👨‍⚖️ [Critic 发难]: {critic_msg}")

    print("\n🤖 研究员(Researcher) 收到意见，正在审视并主动追问/反驳...")
    # 2. Researcher 追问或反驳
    researcher_prompt = (
        f"针对你的课题 '{query}'，你辛辛苦苦找了素材。但审计官提出了以下挑剔意见:\n'{critic_msg}'\n"
        f"请站在资深研究员的角度，与他进行‘双向澄清交流’。你可以反驳、解释为什么某些数据找不到（比如2026年最新发布还没公开架构），"
        f"或者明确追问他：‘你要求的数据到底要solid到什么程度？允许我换成什么工具去哪里找补？’。请写出你的沟通反问。"
    )
    researcher_msg = llm_flash.invoke([SystemMessage(content="你是拥有独立思考能力的资深研究员。"), HumanMessage(content=researcher_prompt)]).content
    print(f"👩‍💻 [Researcher 主动追问/反驳]: {researcher_msg}")

    print("\n🤖 辩论主持人在调解双方论点，并达成最终澄清共识...")
    # 3. 最终由主持人（或高阶模型）综合两者的对话，达成最终共识与下一轮行动方案
    moderator_system = (
        "你负责主持研究员和审计官的‘澄清对齐会谈’。你要听取两者的辩论，促成他们达成共识。\n"
        "如果两者的对话中显示有些数据全网确实不存在，你要劝说审计官妥协，降级放行；\n"
        "如果研究员理解了缺失维度，你要为研究员提炼精准的妥协方案与更精准的搜索关键词。\n"
        "你的输出必须严格符合给定的 JSON Schema 格式。"
    )
    
    debate_history = (
        f"【研报课题】: {query}\n\n"
        f"【当前素材】:\n{formatted_notes}\n\n"
        f"【第一轮交谈】\n"
        f"审计官(Critic) 挑刺: {critic_msg}\n\n"
        f"研究员(Researcher) 追问与澄清要求: {researcher_msg}"
    )

    structured_moderator = llm_flash.with_structured_output(ClarificationConsensus)
    
    try:
        consensus: ClarificationConsensus = structured_moderator.invoke([
            SystemMessage(content=moderator_system),
            HumanMessage(content=debate_history)
        ])
        
        print(f"\n📊 澄清会谈成果 -> 是否达成通关共识: 【{consensus.is_sufficient}】")
        print(f"📝 辩论对齐纪要: {consensus.debate_summary}")
        print(f"📌 下一步行动指令: {consensus.next_action_memo}")
        
        return {
            "is_sufficient": consensus.is_sufficient,
            "consensus_memo": consensus.next_action_memo,
            "suggested_keywords": consensus.suggested_keywords,
            "recommended_source": consensus.recommended_source
        }
    except Exception as e:
        print(f"⚠️ 共识输出异常，默认切换为强制放行: {e}")
        return {"is_sufficient": True, "consensus_memo": "降级通过"}


def reporter_node(state: AgentState) -> Dict[str, Any]:
    print("\n======== 📝 资深首席分析师：撰写研究报告 ========")
    query = state["query"]
    notes = state["research_notes"]
    memo = state.get("consensus_memo", "")

    system_prompt = (
        "你位居全球顶级 AI 行业首席资深分析师。\n"
        "请基于研究员和审计官在【双向澄清会谈】后达成的共识资产，撰写一份结构化、排版完美的 Markdown 深度行业研究研报。\n"
        "报告必须深度覆盖以下维度：\n"
        "1. 最新核心技术发布矩阵\n"
        "2. 技术路线演进（重点横向对比技术路线）\n"
        "3. 商业落地与API推理成本ROI收益分析\n"
        "4. 核心技术局限性\n"
        "5. 未来预测"
    )
    
    res = llm_pro.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"课题: {query}\n\n素材:\n{chr(10).join(notes)}\n\n双方交谈备忘录: {memo}")
    ])
    
    return {"final_report": res.content}

# =====================================================================
# 4. 路由与工作流构建
# =====================================================================
def should_continue(state: AgentState) -> Literal["researcher", "reporter"]:
    if state["is_sufficient"] or state["loop_count"] >= 3:
        return "reporter"
    return "researcher"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("debate_room", debate_room_node)
workflow.add_node("reporter", reporter_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "debate_room")
workflow.add_conditional_edges("debate_room", should_continue, {"researcher": "researcher", "reporter": "reporter"})
workflow.add_edge("reporter", END)
app = workflow.compile()

# =====================================================================
# 5. 测试运行
# =====================================================================
if __name__ == "__main__":
    test_query = "深入调研 2026 年 OpenAI 与 Anthropic 最新发布的技术成果，对比技术路线、收益与局限，并预测未来趋势。"
    initial_state = {
        "query": test_query, "research_notes": [], "loop_count": 0, "final_report": "",
        "is_sufficient": False, "consensus_memo": "", "suggested_keywords": "", "recommended_source": "general_search",
        "total_prompt_tokens": 0, "total_completion_tokens": 0
    }
    final_output = app.invoke(initial_state)
    print("\n🏆 最终交付的深度报告 :\n", final_output["final_report"])