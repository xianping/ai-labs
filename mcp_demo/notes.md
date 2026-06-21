# server.py
启动后streamble http listening： localhost：8000/mcp

# 启动inspector UI
npx @modelcontextprotocol/inspector，配置连接 http://localhost：8000/mcp

# start Agent.py
Agent list all tools, 然后给model，model判断是否调用tool。
tool 调用还是解释model response，用fastmcp tool_call 调用。
结果返回给model 做回答语言组织。

<!-- 启动mcp inspector后，windows下有个问题 **.\mcp_server_basic.py** y要改成/ 不然变成.mcp_server_basic.py
run --with mcp mcp run ./mcp_server_basic.py -->