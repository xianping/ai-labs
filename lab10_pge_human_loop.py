import os
import re
import asyncio
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# =====================================================================
# 1. 环境初始化与安全加载范式
# =====================================================================
load_dotenv(encoding='utf-8')

if not os.getenv("DEEPSEEK_API_KEY"):
    raise ValueError("❌ 错误: 未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查配置！")

# 初始化高并发大模型客户端（开启异步支持）
llm = ChatOpenAI(
    model="deepseek-v4-flash",             # 高性价比、低延迟的生产级编排/工作模型
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.2,                       # 控制随机性，确保结构输出稳定
    max_retries=3                          # 工业级网络防御重试
)

# =====================================================================
# 2. 工业级强类型全局状态（State）设计
# =====================================================================
class ResearchState(TypedDict):
    query: str                # 用户输入的原始课题
    tasks: List[str]          # Planner 拆解出的并行子任务矩阵列表
    compiled_data: List[str]  # 各并发 Worker 增量累加的情报事实库
    critic_feedback: str      # Evaluator 机器审计官给出的批注说明
    is_approved: bool         # 最终通过放行标记
    loop_count: int           # 循环计数器（熔断核心指标）
    human_instruction: str    # 人工介入时的定制化方向性指令
    final_report: str         # 终极排版报告交付件

# =====================================================================
# 3. 核心工具库：防御性 XML 标签提取器
# =====================================================================
def extract_tag_content(text: str, tag_name: str, fallback: str = "") -> str:
    """利用原生正则对大模型吐出的 XML 标签进行秒级强解析"""
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else fallback

def extract_all_tags(text: str, tag_name: str) -> List[str]:
    """批量捞取并行的同名技术标签块"""
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]

# 模拟工业级异步外部搜索工具（实际生产中可无缝替换为 HTTP 客户端如 httpx / aiohttp 接入真实全网搜索 API）
async def async_mock_search_tool(task: str) -> str:
    """模拟高并发网络 I/O 吞吐延迟"""
    await asyncio.sleep(0.5)  
    return f"【高并发全网检索事实】针对技术栈「{task}」抓取到的最新底层参数、时序开销与拓扑边界。"

# =====================================================================
# 4. 多智能体异步并发节点（Nodes）实现
# =====================================================================

async def planner_node(state: ResearchState) -> dict:
    """
    【控制面】Planner 节点：负责将复杂课题进行多维度的解耦拆解，吐出并发任务矩阵
    """
    print(f"\n[➔ Node 激活] : 🧭 Planner (当前迭代轮次: {state.get('loop_count', 0)})")
    
    query = state["query"]
    feedback = state.get("critic_feedback", "【首次执行，暂无历史审计反馈】")
    human_instruction = state.get("human_instruction", "")
    
    # 动态融合机器审计资产与人类最高专家的直接指令
    if human_instruction:
        feedback = f"{feedback} | 🚨 [人类专家最高批示]: {human_instruction}"

    prompt = f"""你是一个顶级的工业级 AI 系统规划师（Planner）。你的职责是针对复杂研究课题进行解耦，拆解出 3 个**完全独立、可并行计算**的细分技术子任务。

[用户原始课题]
<query>{query}</query>

[当前的审计反馈与专家修正意见]
<feedback>{feedback}</feedback>

[约束条件]
1. 你的输出必须完全包裹在 <planner_output>...</planner_output> 标签中。
2. 内部首先输出 <reasoning> 思考如何将课题拆解为互不干扰、适合并行并发挖掘的 3 个技术维度 </reasoning>。
3. 接着输出任务清单，每个任务必须单独用 <task>...</task> 标签包裹，严禁带有编号或多余文本。
"""
    response = (await llm.ainvoke(prompt)).content
    
    tasks = extract_all_tags(response, "task")
    reasoning = extract_tag_content(response, "reasoning", "解耦并行任务")
    
    # 鲁棒性防御机制：如果模型未按格式吐出任务，进行强制静态对齐兜底
    if not tasks:
        tasks = [f"技术架构维度 A: {query[:15]}", "底层实现瓶颈 B", "防御与固具加固手段 C"]
        
    print(f"   [Planner 内部思考]: {reasoning[:60]}...")
    print(f"   [Planner 成功拆解出 {len(tasks)} 个可供并行计算的子任务]")
    for i, t in enumerate(tasks, 1):
        print(f"     ├── 并行子通道 {i}: {t}")
        
    return {"tasks": tasks}


async def concurrent_subtask_worker(task: str, query: str) -> str:
    """
    【最小工作单元】负责单个子任务的『异步搜索 + LLM 结构化提炼』组合拳
    """
    # 1. 异步并行发起网络检索（榨干机器 I/O 性能）
    search_result = await async_mock_search_tool(task)
    
    # 2. 调度模型基于并发检索事实进行原子层面的高密度解构
    prompt = f"""你是一个底层的技术情报挖掘单元。请针对以下指定的子任务进行极限深挖。

[核心课题] {query}
[当前子任务] {task}
[外部检索素材] {search_result}

[约束条件]
请直接吐出高技术密度的分析事实，包含底层参数、架构推导，包裹在 <sub_data>...</sub_data> 标签中。
"""
    response = (await llm.ainvoke(prompt)).content
    return extract_tag_content(response, "sub_data", f"子任务【{task}】执行超时或未获取到有效数据。")


async def parallel_generator_node(state: ResearchState) -> dict:
    """
    【性能王牌】Generator 节点：采用经典 Map-Reduce 拓扑，利用 asyncio.gather 瞬间实现多路全网并发
    """
    print(f"\n[➔ Node 激活] : 🔨 Parallel Generator (启动 Map-Reduce 异步并发矩阵...)")
    
    query = state["query"]
    tasks = state["tasks"]
    
    # 核心并发锁：将所有子任务打包进 asyncio.gather，同时触发多路连接
    print(f"   [高并发调度中] 正在同时向搜索引擎与模型节点发起 {len(tasks)} 路异步协程...")
    async_tasks = [concurrent_subtask_worker(task, query) for task in tasks]
    
    # 瞬间扇出（Fan-out），等待所有结果并发回收（Fan-in）
    results = await asyncio.gather(*async_tasks)
    
    # 状态的无损增量累加（Incremental Update）
    current_compiled_data = state.get("compiled_data", []) or []
    new_data_block = "\n\n".join(results)
    updated_compiled_data = current_compiled_data + [new_data_block]
    
    current_loops = state.get("loop_count", 0)
    
    print(f"   [Map-Reduce 汇聚完成]: 成功秒级回收 {len(results)} 路并行资产，事实库当前积攒 {len(updated_compiled_data)} 个大批次数据。")
    return {
        "compiled_data": updated_compiled_data,
        "loop_count": current_loops + 1
    }


async def evaluator_node(state: ResearchState) -> dict:
    """
    【审计面】Evaluator 节点：硬红线准则机器卡点
    """
    print(f"\n[➔ Node 激活] : ⚖️ Evaluator (硬量规机器审计中...)")
    
    query = state["query"]
    all_data_str = "\n\n".join(state["compiled_data"])
    
    prompt = f"""你是一个严苛的首席架构审计官（Evaluator）。你负责判定当前挖掘到的情报事实是否足以完美交付。

[核心课题] <query>{query}</query>
[当前挖掘到的全部情报] <data>{all_data_str}</data>

[评分量规 (Rubric)]
- YES：技术细节极度饱满，深刻涉及算法底层、内存指标或硬件级防御拓扑。
- NO：论述包含大面上的套话、缺乏真实的端侧最坏场景内存计算或缺乏具体的量化指标。

[约束条件]
1. 必须完全输出在 <eval_output>...</eval_output> 中。
2. 内部必须包含 <decision>YES 或 NO</decision> 以及 <critique>具体的缺陷说明或补充要求</critique>。
"""
    response = (await llm.ainvoke(prompt)).content
    
    decision = extract_tag_content(response, "decision", "NO")
    critique = extract_tag_content(response, "critique", "机器未能提取到有效评估。")
    
    is_approved = (decision == "YES")
    print(f"   [Evaluator 机器决议]: 批准放行? -> {is_approved} | 机器缺陷批注: {critique}")
    
    return {
        "is_approved": is_approved,
        "critic_feedback": critique
    }


async def human_in_the_loop_node(state: ResearchState) -> dict:
    """
    【工业安全加固】Human-in-the-Loop (HITL) 人工审查流：无感拦截卡点，引入人类专家的批判性特权
    """
    print(f"\n" + "="*60)
    print("🚨 [Human-in-the-Loop 安全护栏网关激活] 核心专家介入抽审...")
    print(f"   当前机器审计决策: {'🟢 合格放行' if state['is_approved'] else '🔴 打回重修'}")
    print(f"   机器审计官理由: {state.get('critic_feedback')}")
    print("="*60)
    
    print("\n[人类专家拥有最高仲裁裁决权，请选择]:")
    print(" 1. 行使特权，直接放行 (在控制台输入 'YES' -> 将直接覆盖机器决议，推进至研报收敛阶段)")
    print(" 2. 终止任务，紧急退场 (在控制台输入 'EXIT' -> 强行触发断路器安全熔断)")
    print(" 3. 补充定制化修订意见 (直接输入您的修改话术，系统将秒级捕获并回传给控制面)")
    
    # 真实的线程阻塞输入，模拟上层生产环境中 Webhook 或者是人工审批流的拦截行为
    user_input = input("\n✍️ 首席架构师高级指令 (直接回车默认完全信任并维持机器审计决议) -> ").strip()
    
    if not user_input:
        print("   [专家选择]: 信任原有状态，自动流转。")
        return {"human_instruction": ""}
        
    if user_input.upper() == "YES":
        print("   [专家选择]: 🟢 强行行使至高特权！状态重置为【完全通过】。")
        return {
            "is_approved": True, 
            "critic_feedback": "人类专家行使最高豁免权，强行通过放行。",
            "human_instruction": ""
        }
        
    if user_input.upper() == "EXIT":
        print("   [专家选择]: 🛑 强行下发紧急止损熔断令！")
        return {
            "loop_count": 999,  # 瞬间灌满计数器，使得路由边强行走向收敛，不再请求大模型
            "critic_feedback": "人类专家执行了人工熔断中止，系统安全退场。",
            "human_instruction": "终止"
        }
    
    print(f"   [专家选择]: 🔴 驳回！已捕获专家最高修订指示: '{user_input}'")
    return {
        "is_approved": False,
        "human_instruction": user_input
    }


async def reporter_node(state: ResearchState) -> dict:
    """
    【收敛面】Reporter 节点：完美汇总分布式资产并包装交付
    """
    print(f"\n[➔ Node 激活] : 📝 Reporter (最终工业研报高保真提炼中...)")
    
    query = state["query"]
    all_data_str = "\n\n".join(state["compiled_data"])
    loop_count = state["loop_count"]
    
    prompt = f"""你是一个顶级首席技术报告官（Reporter）。请将以下多路并行并发挖掘到的技术事实融汇贯通，写出一份极具行业说服力的 Markdown 顶级技术研报。

[核心课题] {query}
[分布式并发挖掘到的情报事实] {all_data_str}

[约束条件]
1. 必须使用完美的 Markdown 排版（包含多级标题、加粗加固、技术清单表格、以及精确的代码块占位）。
2. 如果满足熔断状态（loop_count 异常或人类执行了 EXIT），必须在报告末尾追加一个【工业级未尽死角与防御提示】模块进行辩证解耦。
"""
    report_text = (await llm.ainvoke(prompt)).content
    return {"final_report": report_text}

# =====================================================================
# 5. 确定性动态控制路由边（Router Edge）
# =====================================================================
def router_edge(state: ResearchState) -> str:
    """基于状态判定实现硬连线卡点，从根本上斩断黑盒无限死循环"""
    if state["is_approved"]:
        print("\n[🚦 Edge Router] -> 🟢 审计全线通过！正在进入终期技术研报编排...")
        return "finalize"
    elif state["loop_count"] >= 3:
        print(f"\n[🚦 Edge Router] -> 🚨 熔断断路器拉闸！（当前迭代次数已达安全阈值上限: {state['loop_count']}/3）。系统强行降级收敛，进入收尾交付！")
        return "finalize"
    else:
        print(f"\n[🚦 Edge Router] -> ❌ 资产未达到基线。控制面将状态打回并由控制面流回 Planner。当前循环进度: {state['loop_count']}/3")
        return "continue"

# =====================================================================
# 6. 状态机拓扑构建与编译（LangGraph 异步标准架构）
# =====================================================================
workflow = StateGraph(ResearchState)

# 注册所有解耦的专业节点
workflow.add_node("planner", planner_node)
workflow.add_node("parallel_generator", parallel_generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("human_review", human_in_the_loop_node)  # 注入人工介入关口
workflow.add_node("reporter", reporter_node)

# 编排刚性依赖轨道
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "parallel_generator")
workflow.add_edge("parallel_generator", "evaluator")
workflow.add_edge("evaluator", "human_review")            # 任何机器决议出来后，必须立刻被人类截停审查

# 从人工审查网关节点发起有条件分支流转
workflow.add_conditional_edges(
    "human_review",
    router_edge,
    {
        "continue": "planner",
        "finalize": "reporter"
    }
)
workflow.add_edge("reporter", END)

# 挂载轻量级内存检查点（为后续的断电恢复、多时序人工回溯 Human-in-the-loop 准备好数据库接口）
memory_checkpointer = MemorySaver()
app = workflow.compile(checkpointer=memory_checkpointer)

# =====================================================================
# 7. 本地测试交互网关
# =====================================================================
async def main():
    print("=" * 70)
    print("🚀  Enterprise Multi-Agent Engine V2 (Parallel Execution & HITL Mode) ")
    print("=" * 70)
    
    test_query = (
        "探讨混合专家模型（MoE）的细粒度专家路由算法（Fine-grained Expert Routing）"
        "相较于传统 Top-2 路由的核心优势，并推导其在端侧部署场景下的 KV Cache 内存压榨极限。"
    )
    
    initial_state = {
        "query": test_query,
        "tasks": [],
        "compiled_data": [],
        "critic_feedback": "",
        "is_approved": False,
        "loop_count": 0,
        "human_instruction": "",
        "final_report": ""
    }
    
    # LangGraph 规范：必须配置 thread_id 才能完整启用检查点追溯与内存挂起状态
    config = {"configurable": {"thread_id": "production_run_thread_002"}}
    
    # 异步触发引擎
    final_output = await app.ainvoke(initial_state, config=config)
    
    print("\n" + "=" * 70)
    print("🎯 【多智能体高并发攻坚战结束】最终交付的高保真技术报告如下：")
    print("=" * 70 + "\n")
    print(final_output["final_report"])

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        # 兼容 Windows 本地事件循环防报错机制
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())