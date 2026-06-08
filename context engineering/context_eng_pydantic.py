import os
import json
import openai
from typing import List, Optional
# 🌟 重点学习：引入 Pydantic 的 BaseModel 和 Field
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")

client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# ==========================================
# 1. 定义数据契约 (Data Contract)
# ==========================================
# 这一步至关重要！我们用 Python 类严格定义 Agent 必须返回什么，不仅约束类型，还能约束取值范围
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
# 2. 核心函数：带有强类型校验的 Agent
# ==========================================
def parse_resume_agent_v2(raw_resume_text: str) -> Optional[ResumeSchema]:
    """
    【工业级重构】结合 Context Engineering 和 Pydantic 的简历解析 Agent
    """
    
    # 🌟 重点学习：我们直接把 Pydantic 模型的 schema 注入到 Prompt 中
    # 这样模型能百分百理解我们要的 JSON 结构和每个字段的真实含义
    json_schema_str = json.dumps(ResumeSchema.model_json_schema(), ensure_ascii=False, indent=2)

    system_prompt = f"""
    你是一个极其严谨的 HR 简历数据提取专家。
    你的任务是从用户提供的非结构化简历文本中提取信息，并严格按照下方的 JSON Schema 结构返回。

    <Json_Schema>
    {json_schema_str}
    </Json_Schema>

    <Constraints>
    1. 必须完全基于 <Input_Resume> 标签内的原始文本提取，严禁脑补。
    2. 返回的 JSON 必须完美符合上述 <Json_Schema> 的定义。
    3. 严禁包含任何 Markdown 格式包裹或解释性文字。
    </Constraints>
    """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": f"<Input_Resume>\n{raw_resume_text}\n</Input_Resume>"}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_content = response.choices[0].message.content
        print(f"\n[LLM Raw Response]:\n{raw_content}\n")
        
        # 🌟 重点学习：双重关卡校验
        # 关卡 1: 标准 JSON 解析
        raw_json = json.loads(raw_content)
        
        # 关卡 2: Pydantic 强类型与业务数据规则校验
        # 如果大模型少返回了字段、或者把 int 填成了 string，或者 confidence_score 超出了 0-1 范围，这里会立刻拦截报错
        validated_data = ResumeSchema(**raw_json)
        return validated_data

    except json.JSONDecodeError:
        print("❌ 关卡 1 失败：模型吐出的不是合法 JSON")
        return None
    except ValidationError as ve:
        print(f"❌ 关卡 2 失败：JSON 语法合法，但未能通过 Pydantic 业务契约校验！\n详细错误：{ve}")
        return None
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        return None

# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    test_resume = "我叫王五。写过两年 Java 爬虫，后来转做全栈，前端精通 Vue 和 React，掌握 K8s 部署, 又做了3年。后来又做agent开发"
    
    print("--- 开始调用 V2 强类型 Agent ---")
    result = parse_resume_agent_v2(test_resume)
    
    if result:
        print("--- 🎉 恭喜！数据通过所有校验关卡 ---")
        # 此时 result 是一个真正的 Python 对象，具备全套 IDE 代码自动补全提示！
        print(f"姓名: {result.name}")
        print(f"推导年限: {result.experience_years} 年")
        print(f"技能树: {result.skills}")
        print(f"推理链记录: {result.reasoning}")