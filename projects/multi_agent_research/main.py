# main.py
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from graph import create_research_graph
from state import AgentState

def main():
    # 1. 显式加载环境变量
    load_dotenv()
    
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("⚠️ 警告: 未在环境或.env文件中检测到 DEEPSEEK_API_KEY，将切换为 mock 运行模式。")
        
    # 2. 初始化 2026 年主流大模型底座
    llm_engine = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", "mock_key"),
        temperature=0.2
    )
    
    # 3. 织网：构建多智能体图应用
    app = create_research_graph(llm_engine)
    
    # 4. 定义企业级研报课题
    task_query = "深入调研 2026 年 OpenAI 与 Anthropic 最新发布的核心模型技术成果与商业落地收益横向对比。"
    
    print(f"🚀 [系统启动] 多智能体异构激辩协同系统正式运行！")
    print(f"🎯 课题: 【{task_query}】\n" + "="*50)
    
    # 5. 初始化状态机大脑
    initial_state: AgentState = {
        "query": task_query,
        "research_assets": [],
        "temp_fetched_notes": [],
        "meeting_history": [],
        "meeting_loop_count": 0,
        "global_loop_count": 0,
        "is_meeting_adjourned": False,
        "confirmed_todo": {},
        "final_report": ""
    }
    
    # 6. 执行流
    final_output = app.invoke(initial_state)
    
    print("\n" + "🏆"*20)
    print("✨ 最终过审交付的高价值深度行业研究报告如下：")
    print("🏆"*20)
    print(final_output["final_report"])

if __name__ == "__main__":
    main()