## vanilla_rag.py 
手写RAG，embedding 用model API （Deepseek没有embeding model，等后续尝试其他model API）


## langgraph_hf_rage.py
use hugging face local model as embedding.
pip install langchain-huggingface sentence-transformers
- 用local model 做embedding
- langGraph做了retrive doc和generate doc 两个node。
- 用指定retrive的docs 进行总结回答。

## langgraph_router_rag.py
带病版本。
企业版本Agent，用指定的docs in VecoterDB回答企业问题。如果非企业问题，用普通的chat。（其实也可以拒绝回答）
增加了一个Router Node，来判断企业问题还是general 问题。

## langgraph_router_rag2.py
fix json issue by using pydantic output parser. PydanticOutputParser
- Router Node 判断如果企业问题走Retrive Node，获取企业知识库。
- RouterNOde 可以防止结论污染。全部走retrive 一定会获取docs。