# graph.py
from typing import Literal
from langgraph.graph import StateGraph, START, END
from state import AgentState

# 导入异构节点工厂
from agents.researcher import make_execution_researcher, make_researcher_clarify
from agents.critic import make_critic
from agents.moderator import make_moderator

def router_after_meeting(state: AgentState) -> Literal["agent_critic", "execution_researcher", "reporter_node"]:
    """
    🎛️ 整个多智能体系统的控制中枢大路由
    根据 Moderator 的会议纪要，决定工作流应该往哪走
    """
    # 场景 A：会还没开完，双方意见没统一，继续在会议室吵。让 Critic 出来针对研究员刚才的反驳做正面回应
    if not state["is_meeting_adjourned"]:
        return "agent_critic"
        
    # 场景 B：虽然散会了，但是出去干活 -> 回来开会的大轮次已经到了上限（防止不收敛的大熔断）
    if state["global_loop_count"] >= 3:
        print("\n🚨 [大循环熔断] 已滚动迭代3次大轮次，由于防止不收敛机制被激活，强行送往最终报告交付节点！")
        return "reporter_node"
        
    # 场景 C：大家达成了一致，散会！研究员认领了 To-Do 清单，拿着新关键词和工具出去干活
    return "execution_researcher"


def reporter_node(state: AgentState) -> dict:
    """纯交付函数节点（不需要做成独立 Agent），负责对通过审计的资产进行格式化扩写"""
    print("\n================ 📝 首席分析师：撰写并交付最终研报 ================")
    assets_ctx = "\n".join(state["research_assets"])
    summary_ctx = state.get("confirmed_todo", {}).get("summary", "无")
    
    report = f"""
# 🚀 2026年 OpenAI 与 Anthropic 深度技术发布行业研究报告
**交付时间**：2026年6月
**研报编制背景（会议纪要总结）**：{summary_ctx}

## 1. 历经多轮异构审计辩论、固化的核心确定性情报资产
{assets_ctx}

## 2. 结论与技术演进未来预测
本报告数据已通过 Multi-Agent 深度异构拉扯审计，剔除了全网自媒体营销噪音，具备极高商业落地参考 ROI 价值。
    """
    return {"final_report": report}


def create_research_graph(llm):
    """组装并编译 LangGraph 拓扑网"""
    workflow = StateGraph(AgentState)
    
    # 注册节点：利用闭包工厂动态注入底座 LLM
    workflow.add_node("execution_researcher", make_execution_researcher(llm))
    workflow.add_node("agent_critic", make_critic(llm))
    workflow.add_node("agent_researcher_clarify", make_researcher_clarify(llm))
    workflow.add_node("agent_moderator", make_moderator(llm))
    workflow.add_node("reporter_node", reporter_node)
    
    # 构建基础控制连线
    workflow.add_edge(START, "execution_researcher")
    workflow.add_edge("execution_researcher", "agent_critic")       # 抓完数据，直接进会议室，审计官率先发难
    workflow.add_edge("agent_critic", "agent_researcher_clarify")   # 研究员现场反驳挑刺，或追问线索
    workflow.add_edge("agent_researcher_clarify", "agent_moderator")  # 秘书旁听群聊，产出本轮结构化会议纪要
    
    # 挂载核心条件路由
    workflow.add_conditional_edges(
        "agent_moderator",
        router_after_meeting,
        {
            "agent_critic": "agent_critic",                 # 会没开完，留在内部小循环继续交谈
            "execution_researcher": "execution_researcher", # 散会了，大循环滚动，研究员出去执行 To-Do 任务
            "reporter_node": "reporter_node"               # 触发熔断或完美对齐，直接去出报告
        }
    )
    workflow.add_edge("reporter_node", END)
    
    return workflow.compile()