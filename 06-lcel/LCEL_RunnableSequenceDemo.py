"""
【案例】顺序链：Prompt → Model → Parser 一条线执行

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.1 RunnableSequence（顺序链）

知识点速览：
- 这是最经典的 LCEL 入门案例：`prompt | model | parser`。
- 这里要区分两个概念：LCEL 是“把多个 Runnable 连起来的写法”，而真正得到的可执行对象是 Chain；这个 Chain 的具体类型通常就是 `RunnableSequence`。
- prompt、model、parser 都实现了 Runnable 接口，所以既可以分步 `invoke()`，也可以先用 `|` 组合后再整体 `invoke()`。
"""

import os

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，回答要简短。"),
    ("human", "please answer the following question: {question}"),
])

prompt = chat_prompt.invoke(
    {
        "role": "专业的数学助手",
        "question": "请用一句话说明什么是素数",
    }
)
logger.info(prompt)

model = init_chat_model(
    model=os.getenv("GLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_BASE_URL"),
)

result = model.invoke(prompt)

logger.info(f"模型输出：{result.content}")

parser = StrOutputParser()
parsed_result = parser.invoke(result)

logger.info(f"解析结果：{parsed_result}")

print("*" * 50)

chain = chat_prompt | model | parser
chain_result = chain.invoke(
    {
        "role": "专业的数学助手",
        "question": "请用一句话说明什么是素数",
    }
)
logger.info(f"链式调用结果：{chain_result}")

print(type(chain))