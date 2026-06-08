import os
import json
import openai  # 或者是 deepseek 官方 SDK，由于接口兼容，直接用 openai 库最方便
from dotenv import load_dotenv

# ==========================================
# 0. 基础配置与环境初始化
# ==========================================
# 请确保你在 Windows 系统中配置了环境变量，或者直接在下面填入你的 API Key
# os.environ["DEEPSEEK_API_KEY"] = "你的_DEEPSEEK_API_KEY"
load_dotenv(encoding="utf-8")

client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",  # 请根据实际官方或中转端点调整
)

# ==========================================
# 1. 准备候选的 Few-shot 示例库 (Example Pool)
# ==========================================
# 在实际工程中，这些示例通常保存在本地 JSON 文件或数据库中。
# 大模型是“自回归”的，针对不同类型的输入，喂给它相似的正确范例，能极大地提高准确率。
FEW_SHOT_POOL = {
    "tech": {
        "user_input": "张三，精通Python开发，3年Django后端经验，熟练使用MySQL和Docker。",
        "ideal_output": {
            "name": "张三",
            "category": "技术",
            "skills": ["Python", "Django", "MySQL", "Docker"],
            "experience_years": 3,
            "confidence_score": 0.95,
        },
    },
    "marketing": {
        "user_input": "李四，5年互联网大厂市场运营，擅长信息流投放和活动策划，预算管理能力强。",
        "ideal_output": {
            "name": "李四",
            "category": "市场",
            "skills": ["市场运营", "信息流投放", "活动策划", "预算管理"],
            "experience_years": 5,
            "confidence_score": 0.9,
        },
    },
}


# ==========================================
# 2. 核心函数：动态上下文工程与大模型调用
# ==========================================
def parse_resume_agent(raw_resume_text: str, job_type: str = "tech") -> dict:
    """
    通过动态 Context 工程解析简历的 Agent 函数
    :param raw_resume_text: 用户输入的原始简历文本
    :param job_type: 岗位类型 ('tech' 或 'marketing')，用于动态选择 Few-shot 示例
    """

    # 🌟 重点学习 1: 结构化 System Prompt 设计
    # 使用 XML 标签（如 <Role>, <Constraints>）能让大模型更清晰地划分注意力，极大地减少幻觉
    system_prompt = """
    <Role>
    你是一个资深的 HR 简历解析助手。你的任务是从用户提供的非结构化简历文本中，提取出关键信息。
    </Role>

    <Constraints>
    1. 必须完全基于输入文本提取，不要虚构事实。
    2. 如果某个信息（如工作年限）在文本中未提及，请将其值设为 null，不要瞎猜。
    3. 你必须【严格】返回一个 JSON 对象，不能包含任何 Markdown 格式包裹（如 ```json ... ```），也不要包含任何解释性文字。
    </Constraints>

    <Output_Format>
    返回的 JSON 必须符合以下键值结构：
    {
        "name": "字符串，姓名",
        "category": "字符串，岗位分类",
        "skills": ["数组，技能列表"],
        "experience_years": 整数或 null，工作年限,
        "confidence_score": 浮点数，你对这份解析的置信度(0.0 - 1.0)
    }
    </Output_Format>
    """

    # 🌟 重点学习 2: 动态 Context 注入 (Dynamic Context Injection)
    # 根据用户传入的岗位类型，从示例库中挑选出最匹配的 One-shot/Few-shot
    messages = [{"role": "system", "content": system_prompt.strip()}]

    # 动态匹配并注入历史示例（模拟模型的少样本学习）
    if job_type in FEW_SHOT_POOL:
        example = FEW_SHOT_POOL[job_type]
        messages.append(
            {"role": "user", "content": f"请解析以下简历：{example['user_input']}"}
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(example["ideal_output"], ensure_ascii=False),
            }
        )
        print(f"[Context Engine] 成功注入 [{job_type}] 类型的 Few-shot 示例。")
    else:
        print("[Context Engine] 未找到匹配的示例，将进行零样本（Zero-shot）推理。")

    # 注入当前用户真正需要处理的输入
    messages.append({"role": "user", "content": f"请解析以下简历：{raw_resume_text}"})

    # 🌟 重点学习 3: 严格的输出控制与调用
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",  # 实际使用时请确认你使用的 DeepSeek 模型名称
            messages=messages,
            # 提示：DeepSeek 支持 response_format={"type": "json_object"} 开启强力 JSON Mode
            # 这能强制大模型在语法层面吐出合法的 JSON
            response_format={"type": "json_object"},
            temperature=0.1,  # 设为低温度，确保解析任务的稳定性和确定性
        )

        raw_content = response.choices[0].message.content
        print(f"\n[LLM Raw Response]:\n{raw_content}\n")

        # 尝试将大模型返回的字符串解析为 Python 字典
        result_json = json.loads(raw_content)
        return result_json

    except json.JSONDecodeError as je:
        print(f"❌ JSON 解析失败！大模型没有返回合法的 JSON。原始返回为：{raw_content}")
        return {"error": "JSON_DECODE_ERROR", "raw": raw_content}
    except Exception as e:
        print(f"❌ 发生其他错误: {e}")
        return {"error": str(e)}


# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    # 测试用例：一个技术类简历（故意写得有些凌乱）
    test_resume = "我叫王五。写过两年 Java 爬虫，后来转做全栈，前端精通 Vue 和 React，掌握 K8s 部署。总共工作了快 4 年吧。"
    # no work years   

    print("--- 开始调用 Agent 进行解析 ---")
    parsed_data = parse_resume_agent(test_resume, job_type="tech")

    print("--- Agent 解析后的 Python 字典对象 ---")
    print(parsed_data)

    # 验证是否可以直接通过 Key 取值
    if "name" in parsed_data:
        print(
            f"\n成功提取！候选人姓名：{parsed_data.get('name')}, 工作年限：{parsed_data.get('experience_years')}年"
        )
