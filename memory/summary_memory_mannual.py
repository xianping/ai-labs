import os
import openai
# 🌟 采纳你的指正：引入并初始化 dotenv，确保 Windows 下环境变量 100% 正确加载
from dotenv import load_dotenv
load_dotenv(encoding="utf-8")

# 初始化 DeepSeek 客户端
client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

class SummaryMemoryAgent:
    def __init__(self):
        # 核心记忆仓
        self.summary = "当前没有任何历史背景。" # 长期压缩记忆
        self.recent_messages = []               # 短期未压缩的对话（滑动窗口）
        self.max_buffer_turns = 3               # 触发记忆压缩的临界点（为了测试，设为3轮）
        
    def _compress_memory(self):
        """
        内部私有函数：当短期记忆满了，调用大模型在后台进行“记忆蒸馏与压缩”
        """
        print("\n⚠️ [Memory Engine] 检测到短期对话过长，正在启动后台记忆压缩...")
        
        # 拼接用于压缩记忆的 Prompt
        history_text = ""
        for msg in self.recent_messages:
            history_text += f"{msg['role']}: {msg['content']}\n"
            
        compress_prompt = f"""
        你是一个记忆管理专家。请把下方的[长期记忆存量]和[最新对话增量]进行无缝合并，生成一段新的、精简的全局记忆摘要。
        
        <当前长期记忆存量>
        {self.summary}
        </当前长期记忆存量>
        
        <最新对话增量>
        {history_text}
        </最新对话增量>
        
        <约束条件>
        1. 摘要必须极其精炼，删掉寒暄和废话，仅保留核心事实（如：用户的姓名、提到的技能、聊过的主题）。
        2. 字数严格控制在 150 字以内。
        </约束条件>
        """
        
        try:
            # 使用更便宜快捷的 v4-flash 来做这种后台管理任务
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": compress_prompt}],
                temperature=0.0 # 记忆压缩不需要创造力，只要绝对的准确
            )
            
            # 更新长期记忆，清空已被压缩的短期记忆队列
            self.summary = response.choices[0].message.content.strip()
            self.recent_messages = []
            print(f"✅ [Memory Engine] 压缩完成！新长期记忆摘要：\n👉 {self.summary}\n")
            
        except Exception as e:
            print(f"❌ 记忆压缩失败: {e}")

    def chat(self, user_input: str) -> str:
        """
        与用户对话的主函数
        """
        # 1. 构建本次请求的动态上下文 (Dynamic Context Injection)
        # 我们把长期压缩记忆作为系统提示(System Prompt)的一部分注入进去
        system_prompt = f"""你是一个智能助理。请结合[核心历史背景]来回答当前用户的问题。
        
        <核心历史背景>
        {self.summary}
        </核心历史背景>
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 2. 追加近期还没被压缩的短期记忆
        messages.extend(self.recent_messages)
        
        # 3. 追加用户当下的输入
        messages.append({"role": "user", "content": user_input})
        
        # 4. 请求大模型
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                temperature=0.7 # 对话可以稍微有一些温度
            )
            
            reply = response.choices[0].message.content
            
            # 5. 【记忆更新策略】将这一轮真正的对话沉淀到短期记忆队列中
            self.recent_messages.append({"role": "user", "content": user_input})
            self.recent_messages.append({"role": "assistant", "content": reply})
            
            # 6. 【判定机制】如果短期记忆超长，立刻触发后台压缩
            if len(self.recent_messages) >= self.max_buffer_turns * 2: # 一问一答算2条记录
                self._compress_memory()
                
            return reply
            
        except Exception as e:
            return f"出错了: {e}"

# ==========================================
# 运行测试：模拟一个连续的多轮对话
# ==========================================
if __name__ == "__main__":
    agent = SummaryMemoryAgent()
    
    # 第 1 轮：交代背景
    print("--- 👤 第一轮对话 ---")
    q1 = "你好，我叫王五，我目前在自学 AI Agent 开发，今天刚学完了 Context Engineering。"
    print(f"User: {q1}")
    print(f"Agent: {agent.chat(q1)}\n")
    
    # 第 2 轮：补充技术栈
    print("--- 👤 第二轮对话 ---")
    q2 = "我以前有 3 年的全栈开发经验，精通 Python 和 Vue。我很看好 Agent 的未来。"
    print(f"User: {q2}")
    print(f"Agent: {agent.chat(q2)}\n")
    
    # 第 3 轮：触发压缩轮（由于超长，答完这轮后后台会静默执行 _compress_memory）
    print("--- 👤 第三轮对话 ---")
    q3 = "我目前在 Windows 电脑上用 VS Code 敲代码，用的是 DeepSeek 的 API。"
    print(f"User: {q3}")
    print(f"Agent: {agent.chat(q3)}\n")
    
    # 第 4 轮：终极测试（测试 Agent 有没有丧失最初的记忆）
    print("--- 👤 第四轮对话（验证失忆情况） ---")
    # 注意：此时最初的第一、二轮对话的原始 Message 已经被完全清空了！
    # 如果它还能叫出你的名字、记得你的技术栈，说明 Summary Memory 机制彻底生效了！
    q4 = "既然你认识我了，请问我叫什么名字？我有几年什么开发经验？我现在正在学什么？"
    print(f"User: {q4}")
    print(f"Agent: {agent.chat(q4)}\n")