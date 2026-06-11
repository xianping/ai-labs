import os
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 自定义 Sentence Transformer Embedding 类
# ==========================================
class CustomHuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化本地 SentenceTransformer 模型
        如果本地没有该模型，程序会自动从 HuggingFace Hub 下载并缓存到本地
        """
        print(f"[系统提示] 正在加载本地库模型: {model_name} ...")
        # 默认会下载到 ~/.cache/huggingface/hub/，你也可以传入本地绝对路径
        self.model = SentenceTransformer(model_name)
        print("[系统提示] 模型加载完毕！")

    def __call__(self, input: Documents) -> Embeddings:
        """
        ChromaDB 规定必须实现 __call__ 方法
        系统会自动传入一个文本列表 (Documents)，要求返回一个向量列表 (Embeddings)
        """
        # self.model.encode 直接将文本转换为 list of float
        embeddings = self.model.encode(input, convert_to_numpy=False)
        # 确保转换为标准的 Python float 列表格式
        return [list(map(float, e)) for e in embeddings]

    def name(self) -> str:
        """
        强效避坑：新版 ChromaDB 会强制校验 EmbeddingFunction 的 name() 方法
        显式写出此方法，能够 100% 免疫 'AttributeError: ... object has no attribute name' 报错！
        """
        return "CustomHuggingFaceEmbeddingFunction"


def main():
    print("=== 2. 初始化自定义本地 Embedding 引擎 ===")
    # 这里我们选用对中文极为友好的 BGE 小型模型，推理速度极快
    hf_ef = CustomHuggingFaceEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")

    print("\n=== 3. 绑定本地引擎创建持久化集合 ===")
    db_path = os.path.join(os.getcwd(), "chroma_hf_db")
    client = chromadb.PersistentClient(path=db_path)
    
    # 将我们手写的 hf_ef 引擎绑定到集合中
    collection = client.get_or_create_collection(
        name="local_knowledge_memory",
        embedding_function=hf_ef
    )

    print("\n=== 4. 录入数据 ===")
    documents = [
        "知识库：LangGraph 是 LangChain 团队推出的多智能体（Multi-Agent）图拓扑状态机框架。",
        "知识库：ChromaDB 是一个专门为 AI 智能体打造的轻量级向量数据库。",
        "知识库：Sentence Transformers 可以高效地将任意句子转化为固定维度的密集向量。"
    ]
    ids = ["doc_1", "doc_2", "doc_3"]
    
    # 写入数据（ChromaDB 内部会自动调用 hf_ef(documents)）
    collection.add(documents=documents, ids=ids)
    print("数据本地向量化并存盘成功！")

    print("\n=== 5. 语义检索测试 ===")
    query = "多智能体框架和图状态机推荐用什么开发？"
    print(f"用户查询: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    print("\n[检索结果]")
    print(f" 匹配到的文档: {results['documents'][0][0]}")
    print(f" 语义距离(越小越相似): {results['distances'][0][0]:.4f}")


if __name__ == "__main__":
    main()