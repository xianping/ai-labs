import os
from dotenv import load_dotenv
# 严格执行带有编码格式的环境变量加载范式
load_dotenv(encoding='utf-8')

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# 1. 依照你的本地标准规范，初始化 DeepSeek 模型 (供 CrewAI 底层无缝调用)
llm = ChatOpenAI(
    model="deepseek-v4-flash",                  # 统一的通用对话模型名称
    base_url=os.getenv("DEEPSEEK_BASE_URL"),    # 官方 API 网关
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3
)

# 2. 定义团队成员角色 (Agents) - 注入 Backstory 引导 DeepSeek 扮演特定人格
researcher = Agent(
    role='资深技术研究员',
    goal='挖掘关于 {topic} 的前沿技术架构和核心演进趋势',
    backstory="""你在一家顶级科技研究院工作，擅长从繁杂的技术资产中筛选出最具行业颠覆性的核心突破点。
    你提供的信息必须严谨、准确且富有深度。""",
    verbose=True,
    llm=llm  # 注入你的标准 DeepSeek 实例
)

writer = Agent(
    role='科技博客作家',
    goal='将晦涩的技术研究报告转化为通俗易懂、排版优美的科技博客文章',
    backstory="""你是一位拥有百万粉丝的科技博主，文笔专业且富有表现力。
    你擅长捕捉研究报告中的高光数据，用完美的 Markdown 结构将其呈现给读者。""",
    verbose=True,
    llm=llm  # 注入同一个 DeepSeek 实例实现协同
)

# 3. 定义任务流水线 (Tasks) - 显式声明预期输出
research_task = Task(
    description='深入研究 {topic} 的最新进展，总结出至少 3 个工业级核心突破点。',
    expected_output='一份包含核心观点、技术细节和结构清晰的研究简报。',
    agent=researcher
)

write_task = Task(
    description='根据研究员交付的高质量研究简报，重塑并撰写一篇引人入胜的科技博客。',
    expected_output='一篇包含清晰二级标题（H2）、代码块或对比清单的完整 Markdown 博客文章。',
    agent=writer
)

# 4. 组建虚拟团队 (Crew) 
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential  # 顺序过程：研究员完成的输出会自动转化为作家的输入上下文
)

# 5. 启动虚拟团队业务
if __name__ == "__main__":
    result = tech_crew.kickoff(inputs={'topic': 'AI Agents 2026 技术演进与生产落地'})
    print("\n========= CrewAI 最终输出结果 =========")
    print(result)