import os
from typing import List, Optional
from pydantic import BaseModel, Field
# 🌟 重点学习：引入 LangChain 的核心组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")
# ==========================================
# 0. 基础配置与环境初始化
# ==========================================
# LangChain 会默认读取名为 OPENAI_API_KEY 和 OPENAI_API_BASE 的环境变量。
# 为了完全适配 DeepSeek，我们显式传入配置。
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 初始化 LangChain 的 ChatOpenAI 客户端并绑定 DeepSeek 端点
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.0,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ==========================================
# 1. 定义数据契约 (Data Contract)
# ==========================================
# 保持原生的 Pydantic 模型不变，LangChain 原生完美支持 Pydantic
class ResumeSchema(BaseModel):
    reasoning: str = Field(description="在提取各项数据时的矛盾点分析与工作年限推导过程")
    name: str = Field(description="候选人姓名")
    category: str = Field(description="岗位分类，例如技术、市场、全栈等")
    skills: List[str] = Field(description="技能列表数组")
    experience_years: Optional[int] = Field(
        default=None, 
        description="总工作年限，如果是整数则填入；若信息模糊、严重前后矛盾或未提及，必须为 null"
    )
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度，范围 0.0 到 1.0")

# ==========================================
# 2. 核心魔法：使用 LangChain 的 with_structured_output
# ==========================================
# 这是 LangChain 的灵魂方法。它在后台干了三件事：
# 1. 自动把 ResumeSchema 转换成 JSON Schema 提示词并塞进系统提示中
# 2. 自动给 API 开启 response_format={"type": "json_object"} (或者 Tool Calling Mode)
# 3. 自动帮你做 json.loads() 并实力化为 ResumeSchema 对象，不需要你写任何 try-except 解析代码
structured_llm = llm.with_structured_output(ResumeSchema)

# ==========================================
# 3. 构建 LangChain 提示词模版 (Prompt Template)
# ==========================================
# LangChain 推荐用 ChatPromptTemplate 来管理多角色对话，支持 {variable} 动态变量替换
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个极其严谨的 HR 简历数据提取专家。
你的任务是从用户提供的非结构化简历文本中提取信息。
必须完全基于 <Input_Resume> 标签内的原始文本提取，严禁脑补。"""),
    
    ("user", "请解析以下简历：\n<Input_Resume>\n{resume_text}\n</Input_Resume>")
])

# ==========================================
# 4. 组装 LCEL 链并运行 (LangChain Expression Language)
# ==========================================
# 使用管道符 `|` 将提示词和结构化模型串联起来，形成一条流水线 (Chain)
resume_chain = prompt_template | structured_llm

if __name__ == "__main__":
    test_resume = "我叫王五。写过两年 Java 爬虫，后来转做全栈，前端精通 Vue 和 React，掌握 K8s 部署, 又做了3年。后来又做agent开发"
    
    print("--- 开始调用 LangChain 链 ---")
    
    # 运行 Chain，传入变量。LangChain 会在后台把结果直接实例化成我们定义的 Pydantic 类对象
    try:
        result: ResumeSchema = resume_chain.invoke({"resume_text": test_resume})
        
        print("--- 🎉 恭喜！LangChain 链完美运行完成 ---")
        # 此时 result 直接就是 ResumeSchema 类型，享受全套 IDE 补全提示
        print(f"姓名: {result.name}")
        print(f"推导年限: {result.experience_years}") # 观察这里是否和原生 Pydantic 版本一致输出 None/null
        print(f"技能树: {result.skills}")
        print(f"内部推理链: {result.reasoning}")
        
    except Exception as e:
        print(f"运行失败: {e}")