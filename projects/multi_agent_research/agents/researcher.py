# agents/researcher.py
from langchain_core.messages import SystemMessage, HumanMessage
from config_loader import load_agent_config

def make_execution_researcher(llm):
    """工厂闭包：返回负责‘公网抓取/干活’的纯执行研究员节点"""
    config = load_agent_config("researcher")
    
    def node(state):
        print(f"\n========= 👩‍💻 1.1 执行研究员：启动数据抓取 (大循环第 {state['global_loop_count'] + 1} 轮) =========")
        
        todo = state.get("confirmed_todo", {})
        keywords = todo.get("next_search_keywords", "")
        channel = todo.get("search_channel", "general_search")
        
        # 初始第一轮，没有秘书指派的关键词，动态派生
        if not keywords:
            init_prompt = config["prompts"]["initial_keywords"].format(query=state["query"])
            res = llm.invoke([
                SystemMessage(content=config["persona"]),
                HumanMessage(content=init_prompt)
            ])
            keywords = res.content.strip()
            
        print(f"📋 认领到的 To-Do 任务清单: {todo.get('todo_list', ['初次全网广度调研'])}")
        print(f"🔍 正在使用【{channel}】渠道检索关键词: 【{keywords}】")
        
        # 工业级工具降级模拟数据（在真实业务中此处对接 DDGS/Google API）
        mock_info = f"[{channel.upper()} 情报资产] 捕获到 2026 年 OpenAI (包含新模型发布与API定价) 和 Anthropic (Claude新动向) 的一手技术路线。当前关键词：{keywords}"
        
        # 核心清空机制：只要大循环去干新活了，代表上一场会的恩怨结束，清空 meeting_history
        return {
            "temp_fetched_notes": [mock_info],
            "global_loop_count": state["global_loop_count"] + 1,
            "meeting_history": [],     
            "meeting_loop_count": 0,
            "is_meeting_adjourned": False
        }
    return node

def make_researcher_clarify(llm):
    """工厂闭包：返回在会议室内进行‘双向反驳与澄清追问’的辩论研究员节点"""
    config = load_agent_config("researcher")
    
    def node(state):
        print("\n👩‍💻 [Agent: Researcher] 正在消化审计意见，准备在会场发起反驳或要求澄清...")
        
        # 提取 Critic 刚刚在群聊里说的最后一句话
        critic_opinion = state["meeting_history"][-1] if state["meeting_history"] else "暂无"
        history_context = "\n".join(state["meeting_history"])
        
        prompt = config["prompts"]["debate_clarify"].format(
            query=state["query"],
            critic_opinion=critic_opinion,
            meeting_history=history_context
        )
        
        res = llm.invoke([
            SystemMessage(content=config["persona"]),
            HumanMessage(content=prompt)
        ])
        
        new_speak = f"Researcher (研究员): {res.content.strip()}"
        print(f"💬 {new_speak}")
        
        return {
            "meeting_history": list(state["meeting_history"]) + [new_speak],
            "meeting_loop_count": state["meeting_loop_count"] + 1
        }
    return node