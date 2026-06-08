import os
import math
from dotenv import load_dotenv
import openai

"""
failed due to Deepseek has no embeding model provided.
"""
# 1. 严格加载环境变量（确保 Windows 环境下编码安全）
load_dotenv(encoding="utf-8")

# 规范化 Base URL 的处理
base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
# 如果用户写的是官方旧版或带错后缀的地址，进行微调
if "api.deepseek.com" in base_url and not base_url.endswith("/v1"):
    base_url = base_url.rstrip("/") + "/v1"

# 2. 初始化 OpenAI/DeepSeek 客户端
client = openai.OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=base_url
)

# 确保配置存在
if not os.environ.get("DEEPSEEK_API_KEY"):
    raise ValueError("错误：未在 .env 文件中检测到 DEEPSEEK_API_KEY，请检查路径或文件内容！")

# ---------------------------------------------------------------------------
# 🛠 核心数学与向量引擎模块（纯 Python 手写，0 外部库依赖）
# ---------------------------------------------------------------------------

def get_embedding(text: str, model: str = "text-embedding-3") -> list:
    """
    调用兼容 API 获取文本的向量表示 (Embedding)
    💡 注意：DeepSeek 官方或主流转发标准向量模型一般为 text-embedding-3 
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    except openai.NotFoundError as e:
        print(f"\n❌ 404 错误定位：当前 API 节点找不到模型或路由 [{model}]。")
        print(f"👉 解决方案提示：如果是本地 Ollama，请确保已执行 `ollama pull nomic-embed-text` 并修改此处模型名。")
        print(f"👉 当前请求的 Base URL 为: {base_url}")
        return []
    except Exception as e:
        print(f"❌ 获取 Embedding 失败: {e}")
        return []

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """纯 Python 手写：计算两个高维向量的余弦相似度"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    
    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b
        
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

# ---------------------------------------------------------------------------
# 🗄 纯手写内存级向量数据库 (Vanilla Vector DB)
# ---------------------------------------------------------------------------

class SimpleVectorDB:
    def __init__(self):
        self.documents = []

    def add_document(self, text: str):
        """将文档向量化并存入内存库"""
        vector = get_embedding(text)
        if vector:
            print(f"✅ 成功向量化并导入文档: 【{text[:20]}...】")
            self.documents.append({
                "text": text,
                "vector": vector
            })
        else:
            print(f"⚠️ 无法导入文档：【{text[:20]}...】，因为 Embedding 获取为空。")

    def query(self, query_text: str, top_k: int = 2) -> list:
        """查询与用户问题最相关的 Top-K 个文档"""
        query_vector = get_embedding(query_text)
        if not query_vector:
            return []
            
        scored_docs = []
        for doc in self.documents:
            score = cosine_similarity(query_vector, doc["vector"])
            scored_docs.append((score, doc["text"]))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return scored_docs[:top_k]

# ---------------------------------------------------------------------------
# 🤖 RAG 长期记忆 Agent 实现
# ---------------------------------------------------------------------------

class RAGMemoryAgent:
    def __init__(self, vector_db: SimpleVectorDB):
        self.db = vector_db
        self.model_name = "deepseek-chat"

    def chat(self, user_question: str) -> str:
        print(f"\n🔍 [RAG Engine] 收到用户提问: '{user_question}'")
        
        relevant_mems = self.db.query(user_question, top_k=2)
        
        context_str = ""
        if relevant_mems:
            print("💡 [RAG Engine] 成功检索到相关知识碎片：")
            for i, (score, text) in enumerate(relevant_mems):
                print(f"   -> [Top {i+1}] (相似度: {score:.4f}): {text[:40]}...")
                context_str += f"- 知识源{i+1}（置信度: {score:.2f}）：{text}\n"
        else:
            print("⚠️ [RAG Engine] 未能匹配到任何有效的长期记忆。")
            context_str = "未找到相关的长期记忆和业务知识。"

        system_prompt = f"""你是一个拥有长期记忆与企业私有知识库的 AI 高级助理。
请根据 <long_term_memory> 标签中为你检索出来的业务知识，严谨、专业地回答用户的问题。
如果知识库中没有相关信息，请委婉说明，切勿编造虚假事实。

<long_term_memory>
{context_str}
</long_term_memory>
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 运行报错: {e}"

# ---------------------------------------------------------------------------
# 🚀 模拟运行与测试流程
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 🛠 步骤 1：初始化纯手写向量数据库 ===")
    vector_db = SimpleVectorDB()
    
    # 尝试导入数据
    vector_db.add_document("项目CodeNexus的负责人是张三，当前项目进度为80%，预计下个月底上线。")
    vector_db.add_document("公司关于远程办公的规定：员工每周一和周五必须在办公室，周二至周四可申请居家。")
    
    # 如果数据库为空，说明 Embedding 服务仍然调不通，拦截报错提示
    if not vector_db.documents:
        print("\n❌ 严重错误中断：无法获取 Embedding 向量，请检查：")
        print("1. 你是否使用的是本地大模型（如 Ollama）？如果是，请确保把代码第 22 行的 model 改为本地已拉取的模型名（如 'nomic-embed-text' 或 'bge-m3'）。")
        print("2. 检查 .env 文件中的 DEEPSEEK_BASE_URL 是否正确。")
    else:
        print("\n=== 🛠 步骤 2：启动 RAG 记忆智能体 ===")
        agent = RAGMemoryAgent(vector_db)
        
        q1 = "请问公司现在允许天天居家办公吗？有什么具体的硬性规定？"
        reply1 = agent.chat(q1)
        print(f"\n🤖 Agent 回答:\n{reply1}")