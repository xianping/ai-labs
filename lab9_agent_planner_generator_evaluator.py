import os
import re
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ==========================================
# 0. 环境初始化与客户端配置（安全加载范式）
# ==========================================
load_dotenv(encoding='utf-8')  # 严格指定编码，防止 Windows 环境下乱码

if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("❌ 错误: 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

# 初始化底层大模型驱动（客户端完全可控，低 Temperature 确保确定性）
llm = ChatOpenAI(
    model="deepseek-v4-flash",             # 使用高性价比、低延迟的 Flash 模型进行工作流编排
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.2                        # 生产环境控制随机性
)

# ==========================================
# 1. 工业级强类型全局状态（State）设计
# ==========================================
class ResearchState(TypedDict):
    query: str                # 用户输入的原始复杂研究课题
    plan: str                 # Planner 制定或修正后的行动计划
    compiled_data: List[str]  # Generator 搜集并累加的高价值技术情报事实
    critic_feedback: str      # Evaluator 审计未通过时的冷酷批注意见
    is_approved: bool         # Evaluator 给出的确定性布尔值卡点标记
    loop_count: int           # 熔断器计数：防止发生极端死循环
    final_report: str         # Reporter 最终汇编并排版完美的 Markdown 报告

# ==========================================
# 2. 工具库：全纯原生 Python 防御性通用标签提取器
# ==========================================
def extract_tag_content(text: str, tag_name: str, fallback: str = "") -> str:
    """
    通过原生正则精准剥离 XML 标签内容，提供企业级兜底，彻底解决格式漂移带来的崩溃风险
    """
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else fallback

# ==========================================
# 3. 多智能体核心节点（Nodes）实现
# ==========================================

def planner_node(state: ResearchState) -> dict:
    """
    【控制面】Planner 节点：负责大局观规划与动态修正
    """
    print(f"\n[➔ Node 激活] : 🧭 Planner (当前迭代轮次: {state.get('loop_count', 0)})")
    
    query = state["query"]
    feedback = state.get("critic_feedback", "【首次执行，暂无历史审计反馈】")
    
    prompt = f"""你是一个顶级的工业级 AI 系统规划师（Planner）。你的职责是针对用户的研究课题，制定一套严密、分步清晰的行动计划（Plan）。

[用户原始课题]
<query>{query}</query>

[上轮审计官打回的批注意见]
<feedback>{feedback}</feedback>

[约束条件]
1. 你的输出必须完全包裹在 <planner_output>...</planner_output> 标签中。
2. 内部必须首先输出 <reasoning> 详细分析当前课题和上轮审计反馈，思考应该如何调整或加强规划方向 </reasoning>。
3. 接着输出 <plan> 列出具体的 3 个核心执行的深度研究维度或技术要点 </plan> * 严禁带有任何多余的解释。
"""
    response = llm.invoke(prompt).content
    
    # 提取规划 facts
    plan_content = extract_tag_content(response, "plan", fallback="1. 基础原理解析\n2. 落地挑战研究\n3. 行业最佳实践")
    reasoning = extract_tag_content(response, "reasoning", fallback="正常规划推导")
    
    print(f"   [Planner 内部思考]: {reasoning[:60]}...")
    print(f"   [Planner 制定计划]:\n{plan_content}")
    
    return {"plan": plan_content}


def generator_node(state: ResearchState) -> dict:
    """
    【数据面】Generator 节点：负责死磕技术事实，拼命挖掘深度的干货
    """
    print(f"\n[➔ Node 激活] : 🔨 Generator (开始深挖数据事实...)")
    
    query = state["query"]
    plan = state["plan"]
    
    prompt = f"""你是一个顶级的工业级技术情报挖掘专家（Generator）。你的职责是根据规划师制定的行动计划，执行深度的专业事实推导和技术解构。

[研究核心]
<query>{query}</query>

[当前行动计划]
<plan>{plan}</plan>

[约束条件]
1. 你的输出必须完全包裹在 <generator_output>...</generator_output> 标签中。
2. 内部必须首先输出 <reasoning> 思考如何避开宽泛的套话，推导具有深度行业说服力的底层技术事实与架构指标 </reasoning>。
3. 接着输出 <data> 针对行动计划中的点，给出条理极度清晰、充斥底层架构逻辑的干货事实集合 </data>。
"""
    response = llm.invoke(prompt).content
    
    data_content = extract_tag_content(response, "data", fallback="未捕获到有效数据片段。")
    
    # 工业级防御：读取现有数据列表，进行安全的累加叠加，避免依赖框架底层高阶 API 的隐式 reducer
    current_compiled_data = state.get("compiled_data", []) or []
    updated_compiled_data = current_compiled_data + [data_content]
    
    # 计数器在执行面每调用一次便自增，作为熔断的铁轨
    current_loops = state.get("loop_count", 0)
    
    print(f"   [Generator 产出情报]: 成功追加了 {len(data_content)} 字符的核心事实资产。")
    return {
        "compiled_data": updated_compiled_data,
        "loop_count": current_loops + 1
    }


def evaluator_node(state: ResearchState) -> dict:
    """
    【审计面】Evaluator 节点：冷酷无情的裁判官，基于 Rubric 硬切边界
    """
    print(f"\n[➔ Node 激活] : ⚖️ Evaluator (冷酷审计启动...)")
    
    query = state["query"]
    plan = state["plan"]
    # 聚合当前所有已搜集的情报资产供审计官全面审视
    all_data_str = "\n\n".join(state["compiled_data"])
    
    prompt = f"""你是一个严苛的首席架构审计官（Evaluator）。你遵循 Anthropic 的“量规锚定”机理，专门负责审计 Generator 产生的数据是否足以完美回应用户课题。

[核心课题]
<query>{query}</query>
[既定计划]
<plan>{plan}</plan>
[当前挖掘到的全部情报]
<data>{all_data_str}</data>

[评分量规 (Rubric)]
- YES 合格标准：信息绝对不能停留在表面概念，必须深入到算法内核、内存拓扑或工业瓶颈，且全面覆盖了 Plan 的每个点。
- NO 驳回标准：如果信息未提及“真实高并发/长上下文最坏情况下的内存暴涨或故障防御兜底机制”，或者论述依然偏向理论化，一律冷酷打回！

[约束条件]
1. 你的输出必须完全包裹在 <eval_output>...</eval_output> 标签中。
2. 内部首先输出 <reasoning> 对照 Rubric 检查表，理智、挑剔地找出数据的亮点与致命缺陷 </reasoning>。
3. 接着输出结构化决策：
   <decision>必须为 YES 或 NO</decision>
   <critique>如果是 NO，给出极其尖锐的、具体的下一轮增补搜集指导意见；如果是 YES，请写“完全通过”</critique>
"""
    response = llm.invoke(prompt).content
    
    decision = extract_tag_content(response, "decision", fallback="NO")
    critique = extract_tag_content(response, "critique", fallback="未能解析出规范批评，强制要求深化技术细节。")
    reasoning = extract_tag_content(response, "reasoning", fallback="正常审计分析。")
    
    is_approved = (decision == "YES")
    
    print(f"   [Evaluator 审计推导]: {reasoning[:80]}...")
    print(f"   [Evaluator 最终裁决]: 批准通过? -> {is_approved} | 指导批注: {critique}")
    
    return {
        "is_approved": is_approved,
        "critic_feedback": critique
    }


def reporter_node(state: ResearchState) -> dict:
    """
    【收敛面】Reporter 节点：将历史所有的有效沉淀，汇编成终极的排版工业级技术研报
    """
    print(f"\n[➔ Node 激活] : 📝 Reporter (进入终极技术研报编排...)")
    
    query = state["query"]
    all_data_str = "\n\n".join(state["compiled_data"])
    loop_count = state["loop_count"]
    
    prompt = f"""你是一个享誉业内的顶级首席技术报告官（Reporter）。你的职责是将所有通过严格审计的情报数据聚合提炼，撰写一份排版完美的工业级 Markdown 深度技术分析报告。

[研究原始课题]
<query>{query}</query>

[通过严格审计的素材集合]
<compiled_data>{all_data_str}</compiled_data>

[全局熔断状态]
<loop_count>{loop_count}</loop_count>

[约束条件]
1. 必须使用极其清晰的 Markdown 标题、多级列表、技术加粗以及代码/伪代码块来增强专业感。
2. 如果 loop_count >= 3，说明系统触发了熔断保护，请在报告的最后添加一个独立的【研究局限性与潜在盲区提示】模块，用批判性思维警示可能未完全对齐的死角。
"""
    report_text = llm.invoke(prompt).content
    return {"final_report": report_text}

# ==========================================
# 4. 动态图路由边（Router Edge）控制面硬编码
# ==========================================
def router_edge(state: ResearchState) -> str:
    """
    控制流核心中枢：通过纯 Python 逻辑彻底锁死退出条件，防止大模型发生幻觉
    """
    if state["is_approved"]:
        print("\n[🚦 Edge Router] -> 审计官绿灯放行！即将进入最终报告生成。")
        return "finalize"
    elif state["loop_count"] >= 3:
        print(f"\n[🚦 Edge Router] -> 🚨 触发安全熔断器（已达最大迭代上限: {state['loop_count']}/3）。强行收敛，进入残卷汇编！")
        return "finalize"
    else:
        print(f"\n[🚦 Edge Router] -> ❌ 审计不通过。将打回并把反馈意见递交给 Planner 修正。当前循环: {state['loop_count']}/3")
        return "continue"

# ==========================================
# 5. 状态机拓扑构建与编译 (LangGraph 现代规范)
# ==========================================
workflow = StateGraph(ResearchState)

# 注册所有解耦的专业智能体节点
workflow.add_node("planner", planner_node)
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("reporter", reporter_node)

# 构建底层静态管道连线
workflow.add_edge(START, "planner")      # 起点指向规划师
workflow.add_edge("planner", "generator") # 规划师指向干活的
workflow.add_edge("generator", "evaluator") # 干活的向裁判交卷

# 插入带路由图的控制面动态条件边
workflow.add_conditional_edges(
    "evaluator",
    router_edge,
    {
        "continue": "planner",  # 打回重修：带上反馈意见流回 Planner 节点动态修补计划
        "finalize": "reporter"  # 审计通过或熔断：流向报告编写官
    }
)
workflow.add_edge("reporter", END)       # 终点结束

# 编译生成状态机网关
app = workflow.compile()

# ==========================================
# 6. 本地可执行测试客户端
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("     Enterprise Multi-Agent Engine Started (No Langfuse Mode)     ")
    print("=" * 60)
    
    test_query = (
        "深入分析 2026 年最新大模型架构中，混合专家模型（MoE）的细粒度专家路由算法（Fine-grained Expert Routing）"
        "相较于传统 Top-2 路由的核心优势，并推导其在极端高并发端侧部署场景下的 KV Cache 内存压榨极限与硬件级防御手段。"
    )
    
    initial_state = {
        "query": test_query,
        "plan": "",
        "compiled_data": [],
        "critic_feedback": "",
        "is_approved": False,
        "loop_count": 0,
        "final_report": ""
    }
    
    final_output = app.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("🎯 【多智能体协同攻坚战结束】最终生成的深度技术分析研报如下：")
    print("=" * 60 + "\n")
    print(final_output["final_report"])