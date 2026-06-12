# agents/moderator.py
from langchain_core.messages import SystemMessage, HumanMessage
from config_loader import load_agent_config
from schema.meeting import MeetingOutputSchema

def make_moderator(llm):
    """工厂闭包：返回承担控场、极限施压、资产固化、下发结构化纪要职责的主持人节点"""
    config = load_agent_config("moderator")
    
    def node(state):
        print(f"\n🤝 [Agent: Moderator] 正在旁听并梳理会议（当前会议内部拉扯第 {state['meeting_loop_count']} 轮）...")
        
        # 灵魂机制：内部小循环极限施压控制
        pressure_instruction = "目前会谈效率正常，请客观中立地评估双方是否达成一致、可以散会。"
        if state['meeting_loop_count'] >= 2:
            pressure_instruction = (
                "🚨【限时强行收敛收兵命令】：两个 Agent 已经在会议室严重超时扯皮！作为主持人，你必须展现绝对威严，"
                "本轮无条件、强行将 is_aligned 设为 True 宣告散会！把当前的临时素材全部作为 confirmed_points 固化下来强行通过，"
                "并在 summary 中写下：‘[极限施压] 触发会议拉扯轮次熔断，强行调停散会！立刻释放研究员！’"
            )
            
        system_prompt = config["persona"]
        user_content = config["prompts"]["meeting_analyze"].format(
            query=state["query"],
            meeting_history="\n".join(state["meeting_history"]),
            meeting_loop_count=state["meeting_loop_count"],
            pressure_instruction=pressure_instruction
        )
        
        # 强行绑定 Pydantic Schema 进行结构化输出
        structured_llm = llm.with_structured_output(MeetingOutputSchema)
        consensus = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        
        print(f"📊 [Moderator 秘书决议] 是否散会达成一致: 【{consensus.is_aligned}】")
        print(f"📌 固化资产数量: {len(consensus.confirmed_points)} | 下发 To-Do 数量: {len(consensus.todo_list)}")
        if not consensus.is_aligned:
            print(f"🧭 指派下一轮检索通道: 【{consensus.search_channel}】 | 推荐词: 【{consensus.next_search_keywords}】")
        else:
            print(f"📝 散会总评: {consensus.summary}")
        
        # 资产隔离与合并策略：只有在散会时（对齐了），才把这次的临时素材 merge 到确定性资产中
        current_assets = list(state.get("research_assets", []))
        if consensus.is_aligned and state["temp_fetched_notes"]:
            current_assets.extend(state["temp_fetched_notes"])
            
        return {
            "is_meeting_adjourned": consensus.is_aligned,
            "confirmed_todo": consensus.model_dump(),
            "research_assets": current_assets
        }
    return node