import os
import re
from typing import Dict, Any
# 假设使用 deepseek 官方或兼容 OpenAI 的 SDK
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(encoding='utf-8')
# 初始化 DeepSeek 客户端 (这里以官方标准接口为例)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

# ==========================================
# 1. 模拟的大数据测试床基础设施 (Harness 隔离环境)
# ==========================================
def mock_bigdata_executor(sql: str) -> Dict[str, Any]:
    """模拟大数据引擎（如 Hive/Spark）执行 SQL 的沙箱"""
    sql_clean = sql.strip().lower()
    
    # 故意刁难：如果 LLM 使用了 MySQL 的 LIMIT，大数据引擎会报语法错误
    if "limit" in sql_clean and not re.search(r"row_number\(\)|limit \d+$", sql_clean):
        return {"success": False, "error": "FAILED: SemanticException [Error 10004]: Line 1: OFFSET clause is not supported in HiveQL. Please use ROW_NUMBER()."}
    
    # 如果漏了分区字段，报错
    if "where" not in sql_clean or "dt=" not in sql_clean:
        return {"success": False, "error": "FAILED: ValidationException: Querying partitioned table 'user_behavior' without 'dt' (date) partition filter is blocked for performance safety."}
    
    return {"success": True, "data": [{"count": 1024}]}

# ==========================================
# 2. Loop Engineering 核心控制流 (原生实现)
# ==========================================
def text_to_sql_loop(user_query: str, schema_info: str, max_turns: int = 3) -> Dict[str, Any]:
    """
    Loop Engineering 核心函数：
    管理状态(State)、消息流、断路器(Max Turns)与运行时自我修正逻辑。
    """
    # 状态初始化 (Agent State)
    messages = [
        {"role": "system", "content": f"你是一个大数据专家。根据以下表结构生成 Hive SQL。必须包含 dt 分区过滤。\n表结构:\n{schema_info}"},
        {"role": "user", "content": f"需求: {user_query}"}
    ]
    
    turn = 0
    current_sql = ""
    
    # 显式循环控制 (The Runtime Loop)
    while turn < max_turns:
        turn += 1
        print(f"\n[🔄 Loop Turn {turn}] 正在请求 DeepSeek 模型生成/修正 SQL...")
        
        # 1. Action: 调用 DeepSeek 模型 (使用 deepseek-chat 或 deepseek-reasoner)
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0.1, # 降低随机性
        )
        
        response_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_text})
        
        # 正则提取 SQL 语句
        sql_match = re.search(r"```sql\n(.*?)\n```", response_text, re.DOTALL)
        current_sql = sql_match.group(1) if sql_match else response_text
        print(f"-> 提炼出的 SQL:\n{current_sql}")
        
        # 2. Perception & Execution: 在沙箱环境中实际执行 SQL
        print("-> 正在生产集群沙箱中验证该 SQL...")
        execution_result = mock_bigdata_executor(current_sql)
        
        # 3. Guard & Condition (条件路由与断路器判断)
        if execution_result["success"]:
            print("======== [✓] Loop 成功结束 ========")
            return {
                "status": "success",
                "turns_used": turn,
                "sql": current_sql,
                "data": execution_result["data"]
            }
        else:
            # 遭遇失败，更新状态，准备进行下一次循环（反思与纠错）
            error_msg = execution_result["error"]
            print(f"❌ 执行失败！数据库报错:\n{error_msg}")
            
            # 把报错作为上下文喂回给大模型，驱动下一次 Loop 的“反思”
            feedback_content = f"你生成的 SQL 执行失败，报错信息如下：\n{error_msg}\n请分析原因，并重新生成修正后的完整 SQL 语句，包裹在 ```sql ... ``` 中。"
            messages.append({"role": "user", "content": feedback_content})
            
    # 触发断路器安全退出
    print("======== [⚠️] 触发断路器：达到最大尝试次数，无法自我修正 ========")
    return {
        "status": "failed",
        "turns_used": turn,
        "last_sql": current_sql,
        "error": "Max retries reached without successful execution."
    }

# ==========================================
# 3. 运行演示
# ==========================================
if __name__ == "__main__":
    table_schema = """
    CREATE TABLE user_behavior (
        user_id STRING,
        item_id STRING,
        behavior_type STRING, -- 'click', 'buy'
        dt STRING             -- 分区字段, 格式 'YYYY-MM-DD'
    );
    """
    user_request = "帮我查询昨天购买量最高的前 10 个商品"
    
    # 启动 Loop Agent
    final_output = text_to_sql_loop(user_query=user_request, schema_info=table_schema)
    print("\n[🏁 最终输出报告]:", final_output)