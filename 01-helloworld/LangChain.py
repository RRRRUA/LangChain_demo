"""
【案例】LangChain 1.0 写法：init_chat_model 统一入口调用大模型

对应教程章节：第 10 章 - LangChain 快速上手与 HelloWorld → 4、实战：基于阿里百炼的 HelloWorld

知识点速览：
- 1.0 推荐用 init_chat_model 作为统一入口，通过 model_provider（如 "openai"）指定厂商，同一套写法可切换模型。
- 接国内平台（阿里百炼、通义等）时需显式写 model_provider="openai"，否则会报错无法推断 provider。
- 调用三件套：API Key、模型名、Base URL；invoke(问题) 返回消息对象，.content 取正文。
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

_ = load_dotenv()


model03 = ChatOpenAI(
    model = os.getenv("MIMO_MODEL"),
    api_key = os.getenv("MIMO_API_KEY"),
    base_url = os.getenv("MIMO_BASE_URL"),
)

model1 = init_chat_model(
    model = os.getenv("MIMO_MODEL"),
    model_provider = "openai",
    api_key = os.getenv("MIMO_API_KEY"),
    base_url = os.getenv("MIMO_BASE_URL"),
)

modelPixel = init_chat_model(
    model = os.getenv("MODEL"),
    model_provider = "openai",
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL"),
)
# print(model1.invoke("你好，MIMO！").content)
# print(model03.invoke("你好，MIMO！").content)
try:
    response = modelPixel.invoke("你好，你是谁")
    print(response.content)
except Exception as e:
    print("原始错误内容:", e.__cause__)  # 或查看完整堆栈