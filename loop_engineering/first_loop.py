from typing import Annotated, TypedDict
from typing_extensions import Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage

# ==========================================
# 1. 定义状态 (State) -> Loop Engineering 的基础
# ==========================================
class AgentState(TypedDict):
    task: str           # 用户初始任务
    code: str           # 当前生成的代码
    error: str          # 执行错误信息 (如果有)
    iterations: int     # 当前循环次数
    success: bool       # 是否成功

# 模拟一个轻量级的大模型和执行沙箱
def fake_llm_generate(prompt: str) -> str:
    # 模拟初次生成有 bug 的代码，第二次生成正确的代码
    if "修复" in prompt:
        return "print('Hello, Big Data World!')"
    return "print(Hello, World!)" # 缺少引号，会导致 NameError

def execute_sandbox(code: str) -> str:
    """模拟沙箱执行环境 (Harness 工程提供的安全隔离环境)"""
    try:
        # 危险！实际生产中切勿使用 raw exec，需在安全沙箱中运行
        # 这里仅作逻辑模拟
        if "Hello, World!" in code and "print('Hello" not in code:
            raise NameError("name 'Hello' is not defined")
        return "" # 执行成功，无错误
    except Exception as e:
        return str(e)

# ==========================================
# 2. 定义节点 (Nodes)
# ==========================================
def coder_node(state: AgentState) -> Dict:
    """负责生成/修复代码的节点"""
    iters = state.get("iterations", 0) + 1
    if state.get("error"):
        prompt = f"任务: {state['task']}\n之前代码报错: {state['error']}\n请修复此代码。"
    else:
        prompt = f"任务: {state['task']}\n请编写 Python 代码实现。"
    
    # 调用 LLM
    generated_code = fake_llm_generate(prompt)
    return {"code": generated_code, "iterations": iters}

def executor_node(state: AgentState) -> Dict:
    """负责在 Harness 环境中执行并捕获结果的节点"""
    error_msg = execute_sandbox(state["code"])
    if error_msg:
        return {"error": error_msg, "success": False}
    return {"error": "", "success": True}

# ==========================================
# 3. 定义条件路由 (Conditional Edges) -> Loop Engineering 的灵魂
# ==========================================
def should_continue(state: AgentState) -> str:
    """控制运行时循环的核心逻辑"""
    if state["success"]:
        return END
    if state["iterations"] >= 3:
        print("[-] 触发断路器：达到最大重试次数，被迫终止。")
        return END
    print(f"[!] 代码执行报错: {state['error']}。正在进入下一次循环修复...")
    return "coder"

# ==========================================
# 4. 构建图 (Graph Assembly)
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("coder", coder_node)
workflow.add_node("executor", executor_node)

# 构建线条
workflow.add_edge(START, "coder")
workflow.add_edge("coder", "executor")

# 动态条件循环
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {END: END, "coder": "coder"}
)

# 编译图
agent_app = workflow.compile()

# 运行示例
if __name__ == "__main__":
    initial_state = {"task": "打印一句话", "iterations": 0, "success": False, "code": "", "error": ""}
    output = agent_app.invoke(initial_state)
    print(f"\n[+] 最终运行结果：\n代码: {output['code']}\n是否成功: {output['success']}")