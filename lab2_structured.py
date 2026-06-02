import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 引入关键组件：Pydantic 文本解析器
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv(encoding='utf-8')

# 1. 依然定义我们的强类型数据结构
class SubTask(BaseModel):
    task_name: str = Field(description="子任务名称")
    estimated_hours: float = Field(description="预估工时")

class ProjectPlan(BaseModel):
    project_name: str = Field(description="项目名称")
    architecture_suggestion: str = Field(description="核心架构设计建议")
    breakdown_tasks: List[SubTask] = Field(description="拆解出来的子任务列表")

# 2. 【核心变化】：创建一个针对 ProjectPlan 的解析器
parser = PydanticOutputParser(pydantic_object=ProjectPlan)

# 3. 初始化模型（这次不使用 with_structured_output，保持普通模型状态）
model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1
)

# 4. 编写提示词
# 注意：我们在系统提示词中引入了 {format_instructions}，让解析器在运行时动态注入格式规范
prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "你是一个资深的大数据架构师。请针对用户的业务需求，进行架构设计并拆解子任务。\n"
        "【特别要求】：你必须严格按照以下的格式返回数据，不要返回任何多余的 Markdown 标记或废话，只需要返回 JSON 块。\n"
        "{format_instructions}"
    ),
    ("user", "业务需求：{requirement}")
])

# 5. 【核心变化】：组合成标准 LCEL 链
# 管道流：提示词 -> 模型 -> 解析器（最终解析成 ProjectPlan 对象）
chain = prompt | model | parser

if __name__ == "__main__":
    demo_requirement = "设计一个支持秒级10万并发的广告点击实时计费系统。"
    print("🚀 使用 PydanticOutputParser 的稳健链正在运行...")
    
    try:
        # 运行链时，必须将 parser 生成的格式化指令注入到提示词中
        plan = chain.invoke({
            "requirement": demo_requirement,
            "format_instructions": parser.get_format_instructions()
        })
        
        print("\n✨ [架构设计建议]：", plan.architecture_suggestion)
        print("\n🔍 [任务拆解明细]：")
        for task in plan.breakdown_tasks:
            print(f"📌 任务: {task.task_name} | 预估耗时: {task.estimated_hours} 小时")
            
    except Exception as e:
        print(f"❌ 运行依然出错: {e}")