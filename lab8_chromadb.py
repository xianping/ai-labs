import os
import chromadb

def main():
    print("=== 1. 初始化 ChromaDB 客户端 ===")
    # 方式 A（内存模式）：数据只在内存中，程序结束即销毁，适合测试
    # client = chromadb.EphemeralClient()
    
    # 方式 B（持久化模式）：数据会存盘到当前目录下的 ./chroma_db 文件夹，推荐！
    db_path = os.path.join(os.getcwd(), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    print(f"ChromaDB 已连接，持久化路径: {db_path}")

    print("\n=== 2. 创建或获取集合 (Collection) ===")
    # Collection 类似于传统数据库里的“表 (Table)”
    # get_or_create_collection 可以防止因为重复创建而报错
    collection = client.get_or_create_collection(name="agent_memory")
    print(f"成功获取集合: {collection.name}")

    print("\n=== 3. 向集合中添加数据 (Add) ===")
    # 模拟 Agent 收集到的关于用户的记忆片段
    memories = [
        "用户喜欢在晚上编写 Python 代码，习惯用 VS Code 编译器。",
        "用户目前正在深入学习 AI Agent 开发，重点关注记忆管理机制。",
        "用户下周二有居家办公的计划，需要提醒他整理公司资产。",
        "今天天气真好，适合出去散步。"
    ]
    # 每一条记忆必须有唯一的 ID
    ids = ["mem_001", "mem_002", "mem_003", "mem_004"]
    # 可以附带元数据 (Metadata)，用于后续的条件过滤
    metadatas = [
        {"category": "habit"},
        {"category": "study"},
        {"category": "work"},
        {"category": "casual"}
    ]

    # 注意：因为我们没有传入 embeddings 参数，ChromaDB 会自动调用默认模型把 documents 变成向量
    collection.add(
        documents=memories,
        ids=ids,
        metadatas=metadatas
    )
    print(f"成功存入 {len(memories)} 条记忆片段！")

    print("\n=== 4. 向量相似度检索 (Query) ===")
    # 模拟 Agent 接收到用户的提问
    user_query = "我想知道关于我学习和工作方面的计划"
    print(f"用户问题: '{user_query}'")
    
    # n_results=2 表示返回最相似的前 2 条数据
    results = collection.query(
        query_texts=[user_query],
        n_results=2
    )

    # 打印检索结果
    print("\n检索出的最相关的长期记忆：")
    for doc, idx, meta, dist in zip(results['documents'][0], results['ids'][0], results['metadatas'][0], results['distances'][0]):
        # distance 是距离（余弦距离或L2距离），数值越小说明越相似
        print(f"  - [{idx}] 记忆内容: {doc} | 标签: {meta['category']} | 距离(越小越近): {dist:.4f}")

    print("\n=== 5. 条件过滤查询 (Where Filter) ===")
    # 有时候我们不仅要算向量相似度，还要通过标签精确过滤
    print("目标：只在 category 为 'habit' 的记忆里寻找与 '编程' 相关的")
    filtered_results = collection.query(
        query_texts=["编程"],
        n_results=1,
        where={"category": "habit"} # 元数据精确匹配
    )
    if filtered_results['documents'][0]:
        print(f"  - 过滤结果: {filtered_results['documents'][0][0]}")

    print("\n=== 6. 数据更新与删除 (Update & Delete) ===")
    # 更新一条记忆
    collection.update(
        ids=["mem_001"],
        documents=["用户习惯在晚上用 VS Code 编写 Python 和 Go 代码。"]
    )
    print("已成功更新 mem_001 的记忆内容。")

    # 删除无用记忆（比如闲聊的那条）
    collection.delete(ids=["mem_004"])
    print("已成功删除 mem_004 (闲聊) 记忆。")
    print(f"当前集合内总条数: {collection.count()}")

if __name__ == "__main__":
    main()