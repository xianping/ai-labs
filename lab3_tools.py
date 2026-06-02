import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv(encoding='utf-8')

# 1. 初始化大模型
model = ChatOpenAI(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# 2. 从 community 包中初始化联网搜索工具
search_tool = DuckDuckGoSearchRun()

# 3. 让大模型感知到这个工具的存在（绑定工具）
model_with_tools = model.bind_tools([search_tool])

if __name__ == "__main__":
    query = "2026年最近有什么最新的科技新闻或重大突破？"
    print(f"👤 用户提问: {query}\n")
    
    messages = [HumanMessage(content=query)]
    
    # 【修正 1】: 显式关闭并行工具调用，强制要求模型单兵作战，防止网关产生多余的 ID 预期
    model_with_tools = model.bind_tools([search_tool], parallel_tool_calls=False)
    
    print("🚀 正在让大模型决策是否需要联网...")
    response = model_with_tools.invoke(messages)
    
    # 必须把大模型完整的、毫无裁剪的原生响应对象塞进队列
    messages.append(response)
    
    if response.tool_calls:
        print(f"🤖 大模型思考后决定：【需要联网】！")
        
        # 【修正 2】: 极其严谨地遍历模型产生的所有 tool_calls（哪怕只有1个），确保 100% 闭环
        for tool_call in response.tool_calls:
            print(f"🔑 正在处理 Tool Call ID: {tool_call['id']} | 工具: {tool_call['name']}")
            
            # 本地执行搜索
            search_result = search_tool.invoke(tool_call["args"]["query"])
            
            # 严格构造对应的 ToolMessage，必须把 tool_call_id 和 name 原封不动传回
            tool_message = ToolMessage(
                content=search_result, 
                tool_call_id=tool_call["id"],
                name=tool_call["name"] # 显式带上工具名，增强部分网关的兼容性
            )
            messages.append(tool_message)
        
        print("\n🌐 联网数据已严丝合缝装配完毕，正在发起第二阶段最终握手...")
        
        # 此时的 messages 队列完美符合：[Human, AI(带calls), Tool(带对应id)]
        final_response = model.invoke(messages)
        
        print("\n✨ [AI 行业研究员最终联网报告]：\n")
        print(final_response.content)
    else:
        print("🤖 直接回答：", response.content)