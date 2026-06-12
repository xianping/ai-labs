# agents/critic.py
from langchain_core.messages import SystemMessage, HumanMessage
from config_loader import load_agent_config

def make_critic(llm):
    """工厂闭包：返回具备‘质检挑刺’和‘正面回应追问’双重能力的审计官节点"""
    config = load_agent_config("critic")
    
    def node(state):
        print("\n👨‍⚖️ [Agent: Critic] 正在审阅当前临时资产并做出正面回应...")
        
        temp_notes_ctx = "\n\n".join(state["temp_fetched_notes"]) if state["temp_fetched_notes"] else "尚未提交任何新素材"
        history_context = "\n".join(state["meeting_history"]) if state["meeting_history"] else "会议刚刚开始，请大刀阔斧指出硬伤。"
        
        prompt = config["prompts"]["audit_and_response"].format(
            query=state["query"],
            temp_notes=temp_notes_ctx,
            meeting_history=history_context
        )
        
        res = llm.invoke([
            SystemMessage(content=config["persona"]),
            HumanMessage(content=prompt)
        ])
        
        new_speak = f"Critic (审计官): {res.content.strip()}"
        print(f"💬 {new_speak}")
        
        return {
            "meeting_history": list(state["meeting_history"]) + [new_speak]
        }
    return node