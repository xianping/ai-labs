import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# =====================================================================
# 0. 严谨的环境变量与密钥加载范式（注入编码支持）
# =====================================================================
load_dotenv(encoding='utf-8')

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
)

# =====================================================================
# 1. 评测目标数据定义：模拟 2 个不同的 Agent 在面对高危客诉时的表现
# =====================================================================
# 业务背景：用户投诉“充值了 100 元，但账户未到账，要求立刻退款，否则起诉”
# 官方标准业务知识（Ground Truth）：由于银行通道延迟，到账可能延迟 10 分钟。若超过 24 小时未到账方可人工退款，当前无法立即退款。

TEST_CASES = [
    {
        "case_id": "CASE_001_The_Liar",
        "user_query": "我刚充了100块钱怎么还没到账？！你们是不是诈骗公司？赶紧给我退钱，不退我立马去工商局告你们！",
        "agent_response": "亲爱的用户您好！真诚地向您道歉，给您带来这么不好的体验真的很抱歉呜呜。您放心，小爱绝对不会骗您的！我已经为您点击了【全额退款】按钮，100元将在3秒钟内原路退回您的银行卡，请您消消气，祝您生活愉快开心每一天哦！",
        "note": "这个 Agent 态度极其卑微完美，但是业务全错（违反公司规定瞎承诺退款，属于严重的合规幻觉）"
    },
    {
        "case_id": "CASE_002_The_Professional",
        "user_query": "我刚充了100块钱怎么还没到账？！你们是不是诈骗公司？赶紧给我退钱，不退我立马去工商局告你们！",
        "agent_response": "您好，非常理解您焦急的心情。请您放心，平台绝不会侵占您的资金。经查目前由于银行接口偶发性延迟，部分充值会有 5-10 分钟的滞后。请您耐心等待 10 分钟，若 24 小时后仍未到账，我们将全力协助您办理人工退款流程。感谢您的理解。",
        "note": "这个 Agent 语气冷静客观，且完美执行了官方合规知识点"
    }
]

# =====================================================================
# 2. Harness 评测层：构建高级 Pydantic 质检出参结构
# =====================================================================
class AdvancedQualityReport(BaseModel):
    # 强制让思维链推理字段排在最前面，利用 LLM 的自回归特性，先推理、再打分
    reasoning_process: str = Field(
        ..., 
        description="针对回答进行的逐句深度质检审计流。必须明确指出哪些句子符合或违反了 Rubric 规则。"
    )
    compliance_score: int = Field(
        ..., 
        description="业务合规与事实准确度得分，严格按照 1-5 分量规执行。"
    )
    empathy_score: int = Field(
        ..., 
        description="用户情绪安抚与话术专业度得分，严格按照 1-5 分量规执行。"
    )

# =====================================================================
# 3. 核心固具：多维度 Rubric 锚定裁判引擎
# =====================================================================
def run_advanced_rubric_judge(query: str, response: str) -> AdvancedQualityReport:
    """
    工业级 Rubric 锚定裁判器：通过极度细化的阶梯式条规，彻底干掉老好人偏见。
    """
    
    # 工业界的核心技巧：把具体的打分标准（Rubric）写得像法律条文一样冰冷和确定
    rubric_compliance = (
        "【维度 1：业务合规与事实准确度 (Compliance) 打分量规】\n"
        "5分 - 完美无瑕：回答完全符合官方事实（告知延迟10分钟，且告知24小时后才可退款），未做出任何越权承诺。\n"
        "3分 - 避重就轻：没有说错知识点，但没有明确告知用户‘延迟10分钟’或‘24小时限制’，导致用户依然困惑。\n"
        "1分 - 合规灾难：回答包含了严重的幻觉、谎言，或违反公司规定私自承诺‘立刻退款’。一旦出现，直接判1分，绝不姑息！\n"
    )
    
    rubric_empathy = (
        "【维度 2：情绪安抚与话术专业度 (Empathy) 打分量规】\n"
        "5分 - 专业克制：既表达了对用户焦急情绪的理解，又保持了企业级助理的专业与沉稳，无情绪化废话。\n"
        "3分 - 过于机械：公式化套话（如‘亲，非常抱歉’），缺乏对用户‘要告工商局’这一特定高危恐慌情绪的正面回应。\n"
        "1分 - 情绪失控：使用极端谄媚的词汇（如‘呜呜’、‘跪求原谅’）严重损害企业形象，或者与用户对骂。\n"
    )

    system_instruction = (
        "你是一位冷酷的企业资产质量控制审计官。你需要评估【在线客服Agent】对高危客诉用户的最终回复质量。\n"
        "请严格遵循以下两条军规进行评估：\n"
        f"{rubric_compliance}\n"
        f"{rubric_empathy}\n"
        "【特别审计指令】\n"
        "1. 严禁做‘老好人’！必须把‘态度好’和‘业务对’完全解耦。如果态度极好但业务承诺违反了公司规定，其合规分必须判定为 1 分！\n"
        "2. 你的推理过程（reasoning_process）必须先于分数生成。在理由中，请以审计官的视角指出‘该回答在第几行、哪句话违反了哪条规定’。"
         "3. 请以 **JSON 格式** 输出，必须包含三个字段：reasoning_process (string), compliance_score (int 1-5), empathy_score (int 1-5)。"   # ← 这里明确包含 "JSON"
    )

    user_content = f"【用户高危客诉】: {query}\n\n【待审计的 Agent 回答】: {response}"

    try:
        # 使用 response_format 实施 Pydantic 强结构化卡点
        res = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # 评测基建必须使用 0.0，确保每次跑批结果稳定、具备可回归性
        )
        
        raw_json = json.loads(res.choices[0].message.content)
        # 转化为 Pydantic 对象返回
        return AdvancedQualityReport(**raw_json)
    except Exception as e:
        return AdvancedQualityReport(
            reasoning_process=f"高级审计引擎调用崩溃: {str(e)}",
            compliance_score=1,
            empathy_score=1
        )

# =====================================================================
# 4. Harness 跑批控制器与质量雷达看板
# =====================================================================
def execute_harness_pipeline():
    print("\n" + "="*25 + " 🚀 HARNESS ADVANCED EVAL START " + "="*25)
    print("目标课题：课题A - 基于多维度 Rubric 锚定干掉 Judge 模型的‘老好人偏见’\n")

    for case in TEST_CASES:
        print(f"▶️ [正在审计用例] ID: {case['case_id']} ({case['note']})")
        print(f"   ├─ 用户提问: {case['user_query']}")
        print(f"   └─ Agent输出: {case['agent_response']}")
        print("   正在拉取高级审计官进行盲审...")
        
        # 运行裁判固具
        report = run_advanced_rubric_judge(case["user_query"], case["agent_response"])
        
        print("\n" + "   " + "-"*20 + " ⚖️ 审计官判决书 " + "-"*20)
        
        # 1. 先在 f-string 外部处理好换行符的替换逻辑
        cleaned_reasoning = report.reasoning_process.replace('\n', '\n    │    ')

        # 2. 在 f-string 中直接引用清洗后的变量（此时大括号内部没有反斜杠，安全通过编译）
        print(f"    │ 🧠 深度审计流 (Reasoning): \n    │    {cleaned_reasoning}")
        # print(f"   │ 🧠 深度审计流 (Reasoning): \n   │   {report.reasoning_process.replace('\n', '\n   │   ')}")
        print(f"   │")
        
        # 工业界高亮展示致命的合规判决
        comp_icon = "❌ (严重违规)" if report.compliance_score <= 2 else "✅ (符合标准)"
        emp_icon = "⚠️ (话术损害)" if report.empathy_score <= 2 else "✅ (体面专业)"
        
        print(f"   │ 📊 业务合规分 (Compliance): {report.compliance_score} / 5 -> {comp_icon}")
        print(f"   │ 🎭 情绪安抚分 (Empathy):    {report.empathy_score} / 5 -> {emp_icon}")
        print("   " + "="*57 + "\n")

if __name__ == "__main__":
    execute_harness_pipeline()