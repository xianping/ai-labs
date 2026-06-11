## 基础概念学习
lab1: learn langchain LCEL chain expression

lab2: 结构化输出，用PydanticOutputParser 强制model output 符合object 定义格式。

lab3 tools：langchain community 三方的DuckDuckGoSearchRun tool 测试工具调用需求。最终结果model 和 model binding tools 需要分开，不然tool结果污染model 最后返回总结内容。Search

lab3 ducktool：构建并发异步。

lab4： 工业级code，使用tool，限制max loop，防止无限tool calling。分离router 
model for tool calling and base model for summary

lab5： react mode

lab6：short memory implementation。同时测试deepseek context caching

lab7：一个完整agent with tool and short memory

lab7 v2: 尝试实现一个行业best 方案，但是去重检查直接匹配有问题

lab7 v3：用更高级的方法实现了多次query去重。

## 高级
1. context engineering
学习如何做context eng

2. memory
学习如何设计model的记忆机制

3. rag
学习利用RAG 存储文档和长期记忆

## 综合项目学习
projects下放置综合概念项目
