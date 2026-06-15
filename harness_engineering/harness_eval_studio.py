import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

# =====================================================================
# 0. 基础环境准备 (兼容你本地的 DeepSeek / OpenAI 规范环境变量加载)
# =====================================================================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# 模拟的银行底层数据库
MOCK_BANK_DB = {
    "张三": {"balance": 50000.0, "last_transaction": "2026-06-10 消费 200 元"},
    "李四": {"balance": 1200.0, "last_transaction": "2026-06-12 收到转账 1000 元"}
}

# =====================================================================
# 1. 核心业务系统：待测试的银行分析 Agent
# =====================================================================
def query_bank_statement(user_name: str) -> str:
    """底层的银行查询工具"""
    if user_name in MOCK_BANK_DB:
        return f"用户【{user_name}】的账户数据为: {json.dumps(MOCK_BANK_DB[user_name], ensure_ascii=False)}"
    return f"未找到用户【{user_name}】的任何账户信息。"

def bank_agent_handler(user_input: str) -> Dict[str, Any]:
    """
    核心 Agent 节点：接收用户输入，决策是否调用工具并返回最终分析报告。
    为了让 Harness 有东西可以测试，这里构建一个纯正的 ReAct 式思考返回。
    """
    system_prompt = (
        "你是一个专业的银行资产分析助理。如果用户询问余额或账单，你必须在回复中包含工具查询的真实数据。\n"
        "如果用户问的是无关内容，直接礼貌拒绝。请以 JSON 格式输出，结构如下：\n"
        '{"tool_called": "query_bank_statement" 或者是 "none", "tool_input": "提取的名字", "final_answer": "你的最终回复"}'
    )
    
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",  # 或者是你本地可用的模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0.1 # 压低随机性，便于评测回归
        )
        agent_output = json.loads(response.choices[0].message.content)
        
        # 模拟工程层的 Tool 执行机
        if agent_output.get("tool_called") == "query_bank_statement":
            tool_res = query_bank_statement(agent_output.get("tool_input", ""))
            # 将工具结果二度喂给模型，或者直接拼接做确定性输出 (这里演示直接拼接以保障评测的确定性)
            agent_output["final_answer"] += f" (依据系统底层查实：{tool_res})"
            
        return agent_output
    except Exception as e:
        return {"tool_called": "error", "tool_input": "", "final_answer": f"系统崩溃: {str(e)}"}


# =====================================================================
# 2. Harness 固具工程层：自动化测试集与裁判模型（LLM-as-a-Judge）
# =====================================================================

# 2.1 黄金数据集 (Golden Dataset) 包含输入、期望的确定性断言指标
GOLDEN_DATASET = [
    {
        "case_id": "TC_001_Normal",
        "user_input": "帮我查一下张三还有多少钱，最近干了啥？",
        "expected_tool": "query_bank_statement",
        "expected_name": "张三"
    },
    {
        "case_id": "TC_002_Irrelevant",
        "user_input": "今天晚上吃火锅合不合适？",
        "expected_tool": "none",
        "expected_name": ""
    }
]

# 2.2 裁判大模型提示词拓扑 (LLM-as-a-Judge)
def run_llm_judge(user_query: str, agent_answer: str) -> Dict[str, Any]:
    """
    这是一个硬核的 Harness 裁判节点。它不参与业务，专门负责以挑惕的视角，
    对 Agent 的输出进行多维度量化打分。
    """
    judge_prompt = (
        "你是一位严厉的软件测试质检官。你需要评估一个【银行助理Agent】对用户提问的最终回答质量。\n"
        "请从以下两个维度进行打分（0 到 5 分，5分为完美，0分为完全垃圾）：\n\n"
        "1. 忠实度 (Faithfulness): 回答中如果提到了银行数据，是否来自底层系统？如果是胡编的数据、或者答非所问，给出低分。\n"
        "2. 服务态度 (Helpfulness): 回答是否清晰、礼貌、专业，有没有废话。\n\n"
        "请严格以 JSON 格式输出，严禁包含任何 Markdown 格式块，结构如下：\n"
        '{"faithfulness_score": 5, "helpfulness_score": 4, "reason": "你的深度质检理由"}'
    )
    
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro", # 生产中通常用更好的模型做裁判，这里继续用当前配置
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"【用户提问】: {user_query}\n【Agent回答】: {agent_answer}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"faithfulness_score": 0, "helpfulness_score": 0, "reason": f"裁判罢工: {str(e)}"}


# =====================================================================
# 3. Harness 引擎启动器：自动化跑批与测试报告看板
# =====================================================================
def start_harness_evaluation():
    print("🚀 [Harness Engine] 自动化固具评测流水线启动...")
    print("=" * 70)
    
    report_summary = []
    
    for case in GOLDEN_DATASET:
        print(f"▶️ 正在跑批用例: [{case['case_id']}] -> 输入: '{case['user_input']}'")
        
        # 1. 运行业务 Agent 得到结果
        agent_result = bank_agent_handler(case["user_input"])
        
        # 2. 第一层：代码级代码断言 (Deterministic Assertions)
        tool_assert = "通过" if agent_result.get("tool_called") == case["expected_tool"] else "失败 ❌"
        name_assert = "通过" if agent_result.get("tool_input") == case["expected_name"] else "失败 ❌"
        
        # 3. 第二层：模型级智能质检 (LLM-as-a-Judge)
        judge_report = run_llm_judge(case["user_input"], agent_result.get("final_answer", ""))
        
        # 记录聚合报告
        case_report = {
            "case_id": case["case_id"],
            "tool_assert": tool_assert,
            "name_assert": name_assert,
            "faithfulness_score": judge_report.get("faithfulness_score"),
            "helpfulness_score": judge_report.get("helpfulness_score"),
            "reason": judge_report.get("reason"),
        }
        report_summary.append(case_report)
        
    # =====================================================================
    # 4. 打印最终的 Harness 控制台看板
    # =====================================================================
    print("\n" + "=" * 30 + " HARNESS EVAL REPORT " + "=" * 30)
    for r in report_summary:
        print(f"用例 ID     : {r['case_id']}")
        print(f"├─ 代码断言[工具决定]: {r['tool_assert']}")
        print(f"├─ 代码断言[参数提取]: {r['name_assert']}")
        print(f"├─ 裁判评分[数据忠实]: {r['faithfulness_score']} / 5")
        print(f"├─ 裁判评分[专业态度]: {r['helpfulness_score']} / 5")
        print(f"└─ 质检官深度评语   : {r['reason']}")
        print("-" * 71)

if __name__ == "__main__":
    start_harness_evaluation()