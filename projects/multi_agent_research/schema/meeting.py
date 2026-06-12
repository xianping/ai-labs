# schema/meeting.py
from pydantic import BaseModel, Field
from typing import List, Literal

class MeetingOutputSchema(BaseModel):
    is_aligned: bool = Field(
        description="Critic和Researcher是否达成一致。若为True，临时素材将固化进资产，散会；若为False，继续留在会议室拉扯。"
    )
    focus_check_passed: bool = Field(
        description="评估刚才两人的辩论是否偏离了最初的课题。若偏离，需在摘要中提出警告。"
    )
    confirmed_points: List[str] = Field(
        description="本次会议双方已经100%确认、不许再反悔的技术点或数据。"
    )
    todo_list: List[str] = Field(
        description="明确开给Researcher的下一步工作重点和抓取任务清单。"
    )
    next_search_keywords: str = Field(
        description="根据Todo提炼的下一轮精确检索词。若散会则留空。"
    )
    search_channel: Literal["general_search", "tech_depth"] = Field(
        description="指派下一轮干活的工具渠道。"
    )
    summary: str = Field(
        description="会议纪要摘要。如果会议轮次快到了，请写下：'【施压收敛】因时间紧迫，主持人强行调停，本轮强行达成如下折中一致...'"
    )