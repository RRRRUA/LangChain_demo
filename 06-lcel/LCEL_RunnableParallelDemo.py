"""
【案例】并行链：同时运行多条子链，汇总结果

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.4 RunnableParallel（并行链）

知识点速览：
- `RunnableParallel` 解决的是“同一输入，要同时跑多条子链”的问题。
- 结果会以 `dict` 形式汇总返回，键名对应并行结构里的键，值对应每条子链的输出。
- 除了显式写 `RunnableParallel({...})`，LCEL 里也常直接用字典表达并行结构；并行完成后，还可以继续把这个字典交给后续链做总结或比较。
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from loguru import logger

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

model = init_chat_model(
    model=os.getenv("GLM_MODEL"),
    model_provider="openai",
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_BASE_URL"),
)

# 子链 1：中文简短介绍
prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
        ("human", "请简短介绍什么是{topic}"),
    ]
)
parser1 = StrOutputParser()
chain1 = prompt1 | model | parser1

# 子链 2：英文简短介绍（与 chain1 同结构，仅提示词语言不同）
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个知识渊博的计算机专家，请用英文简短回答"),
        ("human", "请简短介绍什么是{topic}"),
    ]
)
parser2 = StrOutputParser()
chain2 = prompt2 | model | parser2

parallel_chain = RunnableParallel(
    {
        "中文介绍": chain1,
        "英文介绍": chain2,
    }
)
result = parallel_chain.invoke({"topic": "LangChain"})
logger.info(f"并行链结果：{result}")

parallel_chain.get_graph().print_ascii()