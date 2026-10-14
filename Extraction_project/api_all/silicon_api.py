# 文件路径: Extraction_project/api_all/silicon_api.py
from openai import AsyncOpenAI
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List

# 导入服务配置
try:
    from Extraction_project.config import API_CONFIG
except ImportError:
    try:
        from Extraction_project.config import API_CONFIG
    except ImportError:
        # print("注意: 无法导入 API_CONFIG，请确保 model_config 参数传入完整配置")
        API_CONFIG = {}

DEFAULT_TIMEOUT = 150
_async_clients_cache: Dict[str, AsyncOpenAI] = {}

def get_silicon_client(config: Dict[str, Any]) -> AsyncOpenAI:
    """获取或创建 SiliconFlow 客户端 (带缓存)"""
    api_key = config.get("api_key")
    base_url = config.get("base_url")
    if not api_key or not base_url:
        raise ValueError("Set PV_API_KEY and PV_API_BASE_URL before making model calls")
    cache_key = f"{api_key}@{base_url}"

    if cache_key not in _async_clients_cache:
        try:
            new_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            _async_clients_cache[cache_key] = new_client
        except Exception as e:
            raise e
    return _async_clients_cache[cache_key]

async def call_model(
        prompt: str = None,
        messages: List[Dict[str, str]] = None, # 支持 messages
        stream: bool = False,
        timeout: Optional[int] = DEFAULT_TIMEOUT,
        model_config: Dict[str, Any] = {}
) -> str | AsyncGenerator[str, None]:
    """
    通用调用入口：支持 prompt 字符串 或 messages 列表
    """
    if not model_config:
        raise ValueError("model_config 不能为空")

    # 1. 获取客户端
    async_client = get_silicon_client(model_config)
    model_name = model_config.get("model")
    if not model_name:
        raise ValueError(f"配置缺少 'model' 字段: {model_config}")

    # 2. 构造消息 (优先使用 messages，其次用 prompt)
    final_messages = []
    if messages:
        final_messages = messages
    elif prompt:
        # 单独传入 prompt 时作为 user 消息发送，保留上层提供的 system 消息。
        final_messages = [
            {"role": "user", "content": prompt}
        ]
    else:
        raise ValueError("call_model 必须提供 prompt 或 messages")

    # 3. 发起请求
    try:
        response = await asyncio.wait_for(
            async_client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                stream=stream,
                temperature=0.1,
                max_tokens=16000
            ),
            timeout=timeout
        )

        if not stream:
            content = response.choices[0].message.content
            return content.strip() if content else ""

        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return stream_generator()

    except asyncio.TimeoutError:
        raise ValueError(f"调用超时 ({timeout}s)")
    except Exception as e:
        raise ValueError(f"API 调用失败: {str(e)}")
