import os
import time
from dotenv import load_dotenv
# from langfuse import observe, langfuse_context
import openai  # 或者是你之前配置好的兼容 DeepSeek 协议的客户端
import sys, langfuse
load_dotenv()
import langfuse

# 看看里面到底有哪些导出的名字
print("===== 🔍 Langfuse 内部导出属性定义 =====")
print(dir(langfuse))
print("========================================")
# 💡 强行打印当前运行时真正加载的路径和版本
print("===== 🐍 Python 运行时环境审计 =====")
print(f"当前使用的 Python 解释器: {sys.executable}")
print(f"Langfuse 实际加载路径: {langfuse.__file__}")
try:
    print(f"Langfuse 实际探测版本: {langfuse.__version__}")
except AttributeError:
    print("Langfuse 实际探测版本: 极老版本（连 __version__ 属性都没有）")
print("==================================\n")
# 初始化 DeepSeek 客户端（复用你之前的配置）
client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL") # 或者是官方/网关地址
)

@observe(as_type="span")
def mock_chromadb_retrieve(query: str):
    """模拟你之前的 ChromaDB 长期记忆检索节点"""
    langfuse_context.update_current_span(name="ChromaDB_Memory_Retrieval", input=query)
    time.sleep(0.4) # 模拟硬件或网络IO延迟
    context = "根据公司内部规定，下周二起各部门可申请远程居家办公。"
    langfuse_context.update_current_span(output=context)
    return context

@observe(as_type="generation")
def call_llm_generation(prompt: str, context: str):
    """模拟大模型生成节点（自动捕获 Token 账单）"""
    # 显式更新 Generation 的元数据，方便 Langfuse 精准统计
    langfuse_context.update_current_generation(
        name="DeepSeek_Pro_Response",
        model="deepseek-v4-flash",
        input=f"Context: {context}\nUser Query: {prompt}"
    )
    
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": f"你是一个企业助理，请基于背景知识回答：{context}"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    output_text = response.choices[0].message.content
    
    # 喂给 Langfuse 原始的 Token 消耗数据
    langfuse_context.update_current_generation(
        output=output_text,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
    )
    return output_text

@observe() # 最外层默认作为整个 Trace 根节点
def main_rag_pipeline(user_query: str):
    # 为当前 Trace 设置一个独特的会话或标签，方便 UI 检索
    langfuse_context.update_current_trace(
        name="Enterprise_RAG_Workflow",
        session_id="session_user_9527",
        tags=["Production_Test"]
    )
    
    print(" [1/2] 正在检索长期记忆数据库...")
    context = mock_chromadb_retrieve(user_query)
    
    print(" [2/2] 正在调度大模型合成最终解答...")
    answer = call_llm_generation(user_query, context)
    
    return answer

if __name__ == "__main__":
    query = "下周二我能居家办公吗？"
    print(f"用户提问: {query}\n")
    final_res = main_rag_pipeline(query)
    print(f"\n系统最终答复: {final_res}")
    
    # 异步上报需要一点时间，脚本结束前强制 flush
    langfuse_context.flush()
    print("\n 可观测性数据已同步上报至本地 Langfuse 面板。")