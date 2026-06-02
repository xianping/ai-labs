import os
from dotenv import load_dotenv
# 注意：这里改用 langchain_openai 的包
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 显式指定 UTF-8 编码加载环境，防止 Windows 乱码
load_dotenv(encoding='utf-8')

# 2. 初始化 DeepSeek 模型
# DeepSeek 极其良心，其 API 完美兼容 OpenAI 格式，只需换个 base_url 即可
model = ChatOpenAI(
    model="deepseek-v4-flash",              # DeepSeek-V3 / R1 统一的通用对话模型名称
    base_url="https://api.deepseek.com/v1", # 官方 API 网关
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3
)

# 3. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的科技行业研究员。请用专业、客观、数据驱动的语言回答用户的问题。"),
    ("user", "请帮我简单列出【{topic}】行业目前最受关注的3个核心痛点。")
])

# 4. 组合成链 (LCEL)
# chain = prompt | model | StrOutputParser()

chain = prompt | model 

if __name__ == "__main__":
    print("🚀 正在向 DeepSeek 服务器发起请求...")
    try:
        result = chain.invoke({"topic": "全固态电池"})
        print("\n✨ [行业研究员回复]：\n")
        print(result)
    except Exception as e:
        print(f"❌ 运行出错，错误信息：{e}")