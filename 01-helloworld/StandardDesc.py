"""
【案例】标准/工程化写法：用 LangChain 调用大模型（invoke + stream）

对应教程章节：第 10 章 - LangChain 快速上手与 HelloWorld → 6、实战：企业级封装与流式输出

本案例演示从零到一的完整工程化写法：
- 用通义/阿里云兼容接口通过 LangChain 发问，掌握 invoke（一次性返回）与 stream（流式返回）两种调用方式。
- 将「初始化模型」封装成函数便于复用；用 .env 存密钥、logging 打日志、try/except 区分错误，符合正式项目习惯。
- 运行前在项目根目录配置 .env 中的 MIMO_API_KEY，执行：python 案例与源码-2-LangChain框架/01-helloworld/StandardDesc.py

补充说明：
- 为了让工程化示例更直观，这里继续使用很多同学在旧资料里更常见的 `ChatOpenAI` 写法；若你想看 1.x 统一入口，请对照同目录下的 `LangChainV1.0.py`。
- 当前脚本使用的是“阿里百炼兼容端点 + DeepSeek 模型”这组组合，重点仍然是学习工程化写法，而不是限定某一个具体模型。
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# LangChainException：LangChain 在调用模型失败时会抛出的异常类型。
# 在 main() 里用 except LangChainException 单独接住这类错误，就能打出「模型调用失败」的日志，和配置错误、其他未知错误区分开。
from langchain_core.exceptions import LangChainException

import os
import logging

load_dotenv()

# ----- 日志配置 -----
# logging 是 Python 自带的日志库，不用 pip 安装。
# 用 logger.info() / logger.error() 代替 print，方便区分「普通信息」和「错误」，且可统一格式、写文件等。
# 通过环境变量 LOG_LEVEL 控制输出多少：开发时用 INFO（看得到调试信息），生产时在 .env 里设 LOG_LEVEL=WARNING，就只打警告和错误，减少刷屏。
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 当前模块的 logger，后面用 logger.info(...) 即可
logger = logging.getLogger(__name__)

def init_llm_client(api_key,model_name,base_url) -> ChatOpenAI:
    """
    初始化 LLM 客户端（封装成函数，提高复用性）。

    Returns:
        ChatOpenAI: 初始化好的「对话客户端」，可以对其调用 .invoke(问题) 或 .stream(问题)。
    """
    if not api_key:
        raise ValueError("环境变量未配置，请检查 .env 文件")

    # 创建客户端：指定用哪个模型、密钥、接口地址，以及「回复风格」相关参数。
    llm = ChatOpenAI(
        model=model_name,  # 模型名称（这里演示的是“DeepSeek 模型 + 阿里百炼兼容接口”）
        api_key=api_key,
        base_url=base_url,  # 阿里云提供的兼容 OpenAI 的地址
        temperature=0.7,  # 控制「随机程度」：0 更确定、重复性高；1 更随机、更有创意。一般 0.5～0.8 即可。
        max_tokens=2048,  # 单次回复最多生成多少个 token（约等于字数），防止回复过长或超限。
    )
    return llm


def main():
    api_key = os.getenv("GLM_API_KEY")
    model_name = os.getenv("GLM_MODEL", "glm-5.2")
    base_url = os.getenv("GLM_BASE_URL", "https://tokenhub.tencentmaas.com/v1")

    mimp_api_key = os.getenv("MIMO_API_KEY")
    mimp_model_name = os.getenv("MIMO_MODEL", "mimo-v2.5")
    mimp_base_url = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

    """主函数：封装核心逻辑，符合 Python 工程化规范。"""
    try:
        # 先拿到「可对话的客户端」
        llm = init_llm_client(api_key, model_name, base_url)
        llm2 = init_llm_client(mimp_api_key, mimp_model_name, mimp_base_url)
        logger.info("LLM客户端初始化成功")

        # ----- 方式一：invoke（一次性拿完整回复） -----
        # 发一个问题，程序会等模型全部答完，再一次性把 response 给你。适合短问答。
        question = "你是谁"
        response = llm.invoke(question)
        logger.info(f"问题：{question}")
        logger.info(f"回答：{response.content}")  # .content 里是模型的纯文字回复

        # ----- 方式二：stream（流式，边生成边输出） -----
        # 模型边想边返回，每次返回一小段（chunk），用 for 循环一段段打印，就像打字机效果。适合长文或需要「实时看到输出」的场景。
        print("==================== 以下是流式输出（另一种调用方式）")
        print("*" * 50)
        response_stream = llm.stream("介绍下 langchain，300字以内")
        for chunk in response_stream:
            print(chunk.content, end="")  # end="" 表示不换行，紧挨着打
        print()  # 流式结束后补一个换行，避免和后续输出粘在一起

    # ----- 异常处理：根据错误类型打不同日志，方便排查 -----
    # try 里面的代码一旦报错，会跳到下面某个 except；若都不匹配，再往上抛。
    except ValueError as e:
        # 例如：.env 里没配 QWEN_API_KEY，init_llm_client 里会 raise ValueError
        logger.error(f"配置错误：{str(e)}")
    except LangChainException as e:
        # 例如：网络失败、API 限流、模型返回异常等，LangChain 会抛出 LangChainException
        logger.error(f"模型调用失败：{str(e)}")
    except Exception as e:
        # 其他没预料到的错误都归到这里，避免程序静默崩溃
        logger.error(f"未知错误：{str(e)}")


# ========== 脚本入口 ==========
# __name__ 是 Python 给每个模块自动设置的内置变量，表示「当前模块的名字」：
#   - 直接运行本文件时（如 python StandardDesc.py），Python 会把 __name__ 设为字符串 "__main__"（前后各两个下划线），
#     于是下面的条件为真，会执行 main()；
#   - 被别的文件 import 时，__name__ 是模块名（如 "01_helloworld.StandardDesc"），不等于 "__main__"，不会执行 main()，
#     避免一导入就自动跑一遍问问题。
#
# 注意：必须写 "__main__" 不能写成 "main"。Python 规定「主程序」的 __name__ 就是 "__main__"，
# 若写成 if __name__ == "main": 则条件永远为假（因为 __name__ 实际是 "__main__"），直接运行脚本时 main() 也不会被调用。
#
# 这里直接写 main() 即可，因为本文件的 main 是普通函数（def main），调用即执行。
# 若 main 是异步函数（async def main），则必须写 asyncio.run(main())，否则协程不会真正运行。
if __name__ == "__main__":
    main()