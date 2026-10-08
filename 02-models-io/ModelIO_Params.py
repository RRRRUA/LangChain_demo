"""
【案例】模型标准化参数：temperature、max_tokens 等

对应教程章节：第 11 章 - Model I/O 与模型接入 → 2.5 常用模型参数、2.6 Token、max_tokens 与计费的关系、2.9 调用后的返回信息

知识点速览：
- 这是一个“观察型案例”，重点不是业务功能，而是帮助你理解模型参数与返回对象结构。
- 演示 `temperature` 如何影响输出随机性，以及 `max_tokens` 与回复长度 / 成本控制的关系。
- 也适合配合第 11 章里关于 `AIMessage`、`response.content`、`response_metadata`、`usage_metadata` 的讲解一起看。
- 依赖 `langchain`、`langchain-openai`，运行前在 `.env` 中配置 `deepseek-api`。
"""

# ========== 1. 导入与环境 ==========
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

model = init_chat_model(
    model = os.getenv("GLM_MODEL"),
    model_provider = "openai",
    api_key = os.getenv("GLM_API_KEY"),
    base_url = os.getenv("GLM_BASE_URL"),
    temperature = 0.7,  # 控制「随机程度」：0 更确定、重复性高；1 更随机、更有创意。一般 0.5～0.8 即可。
)

# 直接打印完整 response，便于观察 AIMessage 结构：
# - content：正文
# - response_metadata：厂商原始元数据
# - usage_metadata：统一整理后的 token 用量
print(model.invoke("写一句关于春天的词，14 字以内"))
# <class 'langchain_openai.chat_models.base.ChatOpenAI'>
print(type(model))
# <class 'str'>
print(type(model.invoke("写一句关于春天的词，14 字以内").content))
# <class 'langchain_core.messages.ai.AIMessage'>
print(type(model.invoke("写一句关于春天的词，14 字以内")))