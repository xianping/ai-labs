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
    【优化版】Context Engineering 简历解析 Agent
    """
    # 🌟 优化点 1：加固约束，并要求模型在 JSON 内部先进行推理(Reasoning)
    system_prompt = """
    <Role>
    你是一个极其严谨的 HR 简历数据提取专家。
    </Role>

    <Constraints>
    1. 必须完全基于 <Input_Resume> 标签内的原始文本提取，严禁脑补或编造任何未提及的事实。
    2. 关于工作年限(experience_years)：如果文本中出现多处矛盾的描述、或者没有明确提及总工作年限，必须将其设为 null。
    3. 严禁盲目复制 Few-shot 示例中的数值。
    4. 你必须【严格】返回一个纯净的 JSON 对象，不得包含任何 Markdown 格式包裹或额外解释。
    </Constraints>

    <Output_Format>
    返回的 JSON 必须【严格】符合以下键值结构：
    {
        "reasoning": "字符串，你在提取各项数据时的矛盾点分析与推导过程（在此先进行思考）",
        "name": "字符串，姓名",
        "category": "字符串，岗位分类",
        "skills": ["数组，技能列表"],
        "experience_years": 整数或 null（如果文本前后矛盾或未明确说明，必须返回 null）",
        "confidence_score": 浮点数，置信度(0.0 - 1.0)
    }
    </Output_Format>
    """

    messages = [{"role": "system", "content": system_prompt.strip()}]

    # 为了防止 Few-shot 污染，我们在示例里也加上矛盾导致 null 的样本（根据需要，此处保持结构一致）
    if job_type in FEW_SHOT_POOL:
        example = FEW_SHOT_POOL[job_type]
        # 🌟 优化点 2：示例也使用严格的标签隔离
        messages.append(
            {
                "role": "user",
                "content": f"<Input_Resume>\n{example['user_input']}\n</Input_Resume>",
            }
        )
        # 为示例补充 reasoning 字段以匹配新的格式
        ideal_out = example["ideal_output"].copy()
        ideal_out["reasoning"] = "文本明确提到了工作年限，直接提取。"
        messages.append(
            {"role": "assistant", "content": json.dumps(ideal_out, ensure_ascii=False)}
        )

    # 🌟 优化点 3：对用户的真实输入使用严格的标签进行隔离，防止模型混淆指令和内容
    messages.append(
        {
            "role": "user",
            "content": f"请解析以下简历：\n<Input_Resume>\n{raw_resume_text}\n</Input_Resume>",
        }
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",  # 确保使用的是你的 v4-flash 终结点
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,  # 🌟 优化点 4：降到 0.0，极致压榨模型的确定性，减少随机性
        )

        raw_content = response.choices[0].message.content
        print(f"\n[LLM Raw Response]:\n{raw_content}\n")

        result_json = json.loads(raw_content)
        return result_json

    except json.JSONDecodeError:
        print("❌ JSON 解析失败！")
        return {"error": "JSON_DECODE_ERROR", "raw": raw_content}
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    # 测试用例：一个技术类简历（故意写得有些凌乱）
    # no work years
    test_resume = "我叫王五。写过两年 Java 爬虫，后来转做全栈，前端精通 Vue 和 React，掌握 K8s 部署, 又做了3年。后来又做agent开发"

    print("--- 开始调用 Agent 进行解析 ---")
    parsed_data = parse_resume_agent(test_resume, job_type="tech")

    print("--- Agent 解析后的 Python 字典对象 ---")
    print(parsed_data)

    # 验证是否可以直接通过 Key 取值
    if "name" in parsed_data:
        print(
            f"\n成功提取！候选人姓名：{parsed_data.get('name')}, 工作年限：{parsed_data.get('experience_years')}年"
        )
