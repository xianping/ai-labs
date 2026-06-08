from openai import OpenAI
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
import os

load_dotenv(encoding='utf-8')
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com")
print(client.models.list())
"""
output:
SyncPage[Model](data=[Model(id='deepseek-v4-flash', created=None, object='model', owned_by='deepseek'), Model(id='deepseek-v4-pro', created=None, object='model', owned_by='deepseek')], object='list')

"""