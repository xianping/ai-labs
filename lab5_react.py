import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(encoding='utf-8')

# 1. 模拟两个底层大数据/后台工具
def get_order_status(order_id: str) -> str:
    """模拟查询订单状态的系统组件"""
    if order_id == "ORD100": return "状态: 已出库, 承运商: 顺丰速运, 单号: SF12345"
    return "错误: 未找到该订单号"

def get_delivery_speed(carrier: str) -> str:
    """模拟查询物流时效的组件"""
    if "顺丰" in carrier: return "顺丰速运预计在发货后 24 小时内送达指定区域。"
    return "未知承运商时效"

# 2. 核心：精心设计的 ReAct 提示词（提示词工程的核心）
# 强迫大模型必须严格按照固定格式交替输出，不能乱吐字符
REACT_PROMPT = """你是一个智能后台客服助手。你必须通过思考、调用工具来解决用户的问题。

你可以使用的工具有：
1. get_order_status: 输入订单号，返回订单状态。
2. get_delivery_speed: 输入承运商名称，返回预计时效。

你必须严格按照以下格式进行回答，一次只输出一个周期的 Thought 和 Action，绝对不要一口气把后面的内容编出来：

Thought: 思考你当前知道什么，还需要知道什么。
Action: 工具名称[工具输入参数]
Observation: 这一步你不需要输出，这是由系统调用工具后输入给你的。

当你知道最终答案时，请以以下格式结尾：
Final Answer: 你的最终答复内容。

现在开始。
用户问题: {question}
"""

model = ChatOpenAI(model="deepseek-v4-flash", 
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1",
                    temperature=0)

def run_react_agent(question: str):
    # 将初始 Prompt 拼好
    context = REACT_PROMPT.format(question=question)
    
    # 设定硬防护，防止模型逻辑死循环
    for loop in range(1, 5):
        print(f"\n⚡️ --- ReAct 内部循环第 {loop} 轮 ---")
        
        # 让模型思考下一步
        response = model.invoke(context)
        output = response.content
        print(output) # 打印模型的思考过程
        
        # 如果模型判定已经得出最终答案
        if "Final Answer:" in output:
            print("\n🏁 代理执行完毕！")
            break
            
        # 3. 使用正则表达式提取 Action 
        # 解析格式如: Action: get_order_status[ORD100]
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", output)
        if action_match:
            tool_name = action_match.group(1)
            tool_input = action_match.group(2).strip("'\"") # 移除可能的引号
            
            # 4. 执行行动（Observation 阶段）
            print(f"⚙️ [系统拦截] 正在本地执行工具 {tool_name}, 参数: {tool_input}")
            if tool_name == "get_order_status":
                obs = get_order_status(tool_input)
            elif tool_name == "get_delivery_speed":
                obs = get_delivery_speed(tool_input)
            else:
                obs = "错误: 未知工具"
                
            print(f"📥 [系统反馈] Observation: {obs}")
            
            # 5. 上下文追加：把模型的输出和系统的观察结果，全部追加到上下文，喂回给模型
            context += f"\n{output}\nObservation: {obs}\n"
        else:
            print("❌ 无法解析模型的 Action 格式，平滑退出。")
            context += f"\n{output}\nObservation: 格式错误，请重新输出 Action 格式。\n"

if __name__ == "__main__":
    # 这个问题需要跨越两个工具链条才能回答
    run_react_agent("我的订单号是 ORD100，我大概什么时候能收到货？")