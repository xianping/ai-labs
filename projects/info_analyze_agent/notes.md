1. ChromaHFEmbeddingFunction 这里面name 必须是函数，但是deepseek chat 给我代码建议是property
修改过成name（）后，几轮讨论，他仍然给property。
说明：deepseek的记忆“漂移”有问题。

2. 方便debug，用chatdeepseek取代openai client

3. DeepSeek thinking模式与结构化输出冲突、openai 支持thinking 参数，DS 不支持
4. DuckDuckGo包名变更ddgs
5. ChromaHFEmbeddingFunction 的override 函数embed_query 可能传入list
6. 事实证明，critic模型经常让action模型一直做，达不到要求，需要强制收敛。这个质量堪忧。

## critic_research_agent.py
使用强制max loop收敛机制，单一search tool。比较粗暴

## critic_research_agent2.py
让critic给予一些澄清prompt，有利于researcher进一步精准
在轮次增大之后，逐渐给收敛压迫的promot，渐进式，让critic 放松要求。

## critic_research_agent2.py
放弃之前的“领导指挥，员工干活”模式，改成多轮对话，在debate room中，researcher与critic
进行讨论，为啥critic认为结论不行？researcher需要调整数据源，还是深入某个维度方向？

## critic_research_agent3.py
Add one **debate_room** node, to let researcher and cirtic to debate if the collections
from researcher can match Critic requires.
The drawback is there is only one round of debating.
This leads us to think of how to design the REAL multi-agent, muti-round debating.

