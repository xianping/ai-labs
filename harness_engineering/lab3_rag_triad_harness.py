import os
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# =====================================================================
# 0. 严谨的环境变量与密钥加载范式（注入编码支持）
# =====================================================================
load_dotenv(encoding='utf-8')
# ==========================================
# 1. 现代化大模型网关初始化 (锁定 deepseek-v4-flash)
# ==========================================
def get_audit_llm():
    # 保持你纠正的最高级环境变量加载范式，严禁硬编码 API KEY
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    
    if not api_key:
        raise ValueError("❌ 错误：环境变量 DEEPSEEK_API_KEY 未设置！")
        
    return ChatOpenAI(
        model="deepseek-v4-flash", # 严格切换至最新高性能模型
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.0,            # 评测必须极度严谨，控制面不允许任何随机性
        max_tokens=1024
    )

# ==========================================
# 2. Pydantic 审计报告结构体定义
# ==========================================
class AuditReport(BaseModel):
    reasoning: str = Field(..., description="深度审计流（Thinking/Reasoning），必须详述扣分或给分的严密逻辑支撑。")
    score: int = Field(..., description="量化评分，必须严格限定在 1 到 5 分之间（1分最差，5分完美）。")

# ==========================================
# 3. 核心审计固具实现 (Harness Metrics)
# ==========================================

def audit_context_relevance(query: str, contexts: List[str]) -> AuditReport:
    """度量 1: 上下文相关性 (Query <-> Context)"""
    llm = get_audit_llm()
    structured_llm = llm.with_structured_output(AuditReport, method="json_mode")
    
    joined_context = "\n".join([f"[Chunk {i}]: {c}" for i, c in enumerate(contexts)])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你现在是工业级 RAG 系统运维审计专家。你的任务是评估【检索上下文】与【用户提问】之间的相关性。
        
【评级边界 (Rubrics)】:
5分: 召回的文本切片完美包含了解答提问所需的全部核心事实，没有明显的噪声垃圾。
3分: 召回的文本切片中包含了部分相关线索，但掺杂了大量无关的上下文噪声，或者需要模型进行重度推理才能拼凑出答案。
1分: 召回的文本切片跟用户提问风马牛不相及，完全无法提供任何解答依据。

【严密指令】:
1. 先在 `reasoning` 中一步步输出你的深度审计流（寻找 Chunk 中的事实点，识别无效噪声），最后给出 1-5 的整数评分。
2. 结果必须以 **JSON 格式** 输出，必须包含2个字段：reasoning (string) . score (int)\n        
         """),
        ("human", f"用户提问:\n{query}\n\n检索上下文:\n{joined_context}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({})


def audit_groundedness(contexts: List[str], response: str) -> AuditReport:
    """度量 2: 忠实度 / 扎实度 (Context <-> Response)"""
    llm = get_audit_llm()
    structured_llm = llm.with_structured_output(AuditReport, method="json_mode")
    
    joined_context = "\n".join(contexts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你现在是严苛的知识库合规审计法官。你的任务是审查【最终答案】是否严格基于【检索上下文】生成。你必须揪出任何无中生有的幻觉。
        
【评级边界 (Rubrics)】:
5分: 答案中的每一个核心事实、数据、结论，都能在上下文中找到一比一的文本原像或直接的事实支撑。
3分: 答案的大体方向与上下文相符，但大模型自己脑补了部分细节、数据，或者做出了超出上下文范围的推论。
1分: 答案严重脱离上下文，充斥着大模型自身的先验知识或凭空捏造的幻觉事实（哪怕答案本身听起来很专业）。

【反幻觉卡点原则】:
如果答案中出现了上下文中从未提及的专有名词、时间节点、或具体数字，一律直接降至 1-2 分。
先在 `reasoning` 中列出答案中的断言（Claims），并与上下文进行事实核对，最后输出评分。
 结果必须以 **JSON 格式** 输出，必须包含2个字段：reasoning (string) . score (int)\n        
        
         """),
        ("human", f"检索上下文:\n{joined_context}\n\n最终答案:\n{response}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({})


def audit_answer_relevance(query: str, response: str) -> AuditReport:
    """度量 3: 答案相关性 (Query <-> Response)"""
    llm = get_audit_llm()
    structured_llm = llm.with_structured_output(AuditReport, method="json_mode")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你现在是用户体验与产品质量审计专家。你的任务是评估【最终答案】是否正面解答了【用户提问】。这里不关心事实是否正确，只关心是否答非所问。
        
【评级边界 (Rubrics)】:
5分: 答案直接、清晰地回应了用户的核心提问，结构严谨，无兜圈子、顾左右而言他的废话。
3分: 答案提及了提问中的主题，但绕过了核心痛点，或者给出了极其模棱两可、大而无当的回应。
1分: 答案完全偏离主题，或者在处理特殊拒绝（拒绝回答、死机兜底）时体验极差，没有提供任何有用价值。

先在 `reasoning` 中拆解用户真实意图，评估答案的响应切合度，最后输出评分。
结果必须以 **JSON 格式** 输出，必须包含2个字段：reasoning (string) . score (int)\n        
"""),
        ("human", f"用户提问:\n{query}\n\n最终答案:\n{response}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({})

# ==========================================
# 4. 自动化测试跑道 (Test Harness Runway)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始运行 RAG 三元组度量自动化合规审计...")
    
    # 模拟一个典型的“幻觉爆雷”生产场景
    test_case = {
        "query": "公司2026年Q2研发投入的增长率是多少？",
        "contexts": [
            "根据财务报表，公司2025年Q2研发投入为 1.2 亿元。2026年Q2由于战略调整，研发投入总额达到 1.5 亿元，各项云资源采购开支同比有所下降。",
            "公司计划在2026年下半年加大对 AI 大模型和智能体架构（LangGraph）的工程化落地投入。"
        ],
        # 错误示范：大模型虽然算对了增长率（(1.5-1.2)/1.2 = 25%），但它自己脑补了 2.1 亿元和降薪等幻觉事实
        "response": "公司2026年Q2研发投入的增长率是 25%。Q2研发总投入达到了 1.5 亿元（去年同期为 1.2 亿元）。同时，得益于管理层全面降薪和行政开支压缩，整体研发预算得到了高效保障，预计Q3会突破 2.1 亿元。"
    }
    
    # 1. 审计 上下文相关性
    report_ctx = audit_context_relevance(test_case["query"], test_case["contexts"])
    print("\n" + "="*50)
    print(f"📊 【度量 1：上下文相关性】 评分: {report_ctx.score} 分")
    print(f"🧠 深度审计流:\n{report_ctx.reasoning}")
    
    # 2. 审计 忠实度（抓幻觉）
    report_gnd = audit_groundedness(test_case["contexts"], test_case["response"])
    print("\n" + "="*50)
    print(f"🚨 【度量 2：忠实度 / 反幻觉】 评分: {report_gnd.score} 分")
    print(f"🧠 深度审计流:\n{report_gnd.reasoning}")
    
    # 3. 审计 答案相关性
    report_rel = audit_answer_relevance(test_case["query"], test_case["response"])
    print("\n" + "="*50)
    print(f"🎯 【度量 3：答案相关性】 评分: {report_rel.score} 分")
    print(f"🧠 深度审计流:\n{report_rel.reasoning}")
    print("="*50 + "\n🚀 审计流程全部收敛。")