import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")
# ==========================================
# 0. 基础配置与环境初始化
# ==========================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.0,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ==========================================
# 1. 定义数据契约 (Data Contract)
# ==========================================
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
# 2. 核心修正：指定 method="json_mode" 适配 DeepSeek
# ==========================================
structured_llm = llm.with_structured_output(ResumeSchema, method="json_mode")

# ==========================================
# 3. 动态获取 Schema 并注入到 Prompt 模板中
# ==========================================
# 因为开启了 json_mode，我们必须把预期的 JSON 格式亲手喂给 Prompt
json_schema_str = json.dumps(ResumeSchema.model_json_schema(), ensure_ascii=False, indent=2)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个极其严谨的 HR 简历数据提取专家。
你的任务是从用户提供的非结构化简历文本中提取信息。

必须完全基于 <Input_Resume> 标签内的原始文本提取，严禁脑补。
你必须返回一个纯净的 JSON 对象，且结构必须严格符合下方的 JSON Schema：
{schema_placeholder}"""),
    
    ("user", "请解析以下简历：\n<Input_Resume>\n{resume_text}\n</Input_Resume>")
])

# 🌟 使用 partial 预先将 Schema 字符串填入 System Prompt，免得后面 invoke 时重复传参
prompt_template = prompt_template.partial(schema_placeholder=json_schema_str)

# ==========================================
# 4. 组装 LCEL 链并运行
# ==========================================
resume_chain = prompt_template | structured_llm

if __name__ == "__main__":
    test_resume = "我叫王五。写过两年 Java 爬虫，后来转做全栈，前端精通 Vue 和 React，掌握 K8s 部署, 又做了3年。后来又做agent开发"
    
    print("--- 开始调用修复后的 LangChain 链 ---")
    
    try:
        result: ResumeSchema = resume_chain.invoke({"resume_text": test_resume})
        
        print("--- 🎉 恭喜！LangChain 链在 DeepSeek 环境跑通 ---")
        print(f"姓名: {result.name}")
        print(f"推导年限: {result.experience_years}") 
        print(f"技能树: {result.skills}")
        print(f"内部推理链: {result.reasoning}")
        
    except Exception as e:
        print(f"运行失败: {e}")