# state.py
from typing import List, Dict, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    query: str                       # 原始课题
    research_assets: List[str]       # 已经固化、通过审计的“确定性资产 (Confirmed)”
    temp_fetched_notes: List[str]    # 研报员刚拿回来、还没开会通过的“临时素材”
    
    # 会议室状态
    meeting_history: List[str]       # 当前这场会里，Critic 和 Researcher 的发言记录
    meeting_loop_count: int          # 当前会议小循环的轮次（拉扯了几次）
    global_loop_count: int           # 大循环轮次（去干活 $\rightarrow$ 来开会的次数，上限3次）
    
    # 秘书产出的标准纪要
    is_meeting_adjourned: bool       # 会议是否结束
    confirmed_todo: Dict[str, Any]   # 包含最新的 todo_list 和 confirmed_points
    final_report: str                # 最终交付物