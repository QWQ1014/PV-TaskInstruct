import os
from typing import Optional, AsyncGenerator, List, Dict, Any
import logging

# 导入服务接口
try:
    from Extraction_project.api_all.silicon_api import call_model as silicon_call_model
except ImportError:
    silicon_call_model = None
try:
    from Extraction_project.api_all.baidu_api import call_model as baidu_call_model
except ImportError:
    baidu_call_model = None
try:
    from Extraction_project.api_all.zhipu_api import call_model as zhipu_call_model
except ImportError:
    zhipu_call_model = None
try:
    from Extraction_project.api_all.spark_api import call_model as spark_call_model
except ImportError:
    spark_call_model = None

# 导入统一模型接口
from Extraction_project.llm_api.model_api_interface import BaseModelAPI

# 导入服务配置
try:
    from Extraction_project.config import API_CONFIG
except ImportError:
    # 尝试从上级导入或留空
    try:
        from Extraction_project.config import API_CONFIG
    except ImportError:
        print("错误: 无法导入 Extraction_project/config.py。")
        API_CONFIG = {}

# 可选依赖
try:
    import openai
except ImportError:
    openai = None
try:
    import anthropic
except ImportError:
    anthropic = None


# --- 1. SiliconModelAPI（支持 messages）---
class SiliconModelAPI(BaseModelAPI):
    def __init__(self, model: Optional[str] = None):
        self.api_name = model
        self.model_config = API_CONFIG.get(self.api_name)
        if not self.model_config:
            raise ValueError(f"配置 '{self.api_name}' 未在 Extraction_project/config.py 中找到")

        super().__init__(model=self.model_config.get("model"))

    def validate_dependencies(self):
        if silicon_call_model is None:
            raise ImportError("Silicon API (Extraction_project/api_all/silicon_api) 未能导入。")

    # 同时支持 prompt 和 messages
    async def call(self, prompt: str = None, messages: List[Dict[str, str]] = None, stream: bool = False,
                   timeout: Optional[int] = None, **kwargs) -> str | AsyncGenerator[str, None]:
        effective_timeout = timeout
        if effective_timeout is None:
            effective_timeout = int(self.model_config.get("timeout", 300))
        # 兼容性处理：构建有效参数
        # 如果上层没传 prompt 但传了 messages，silicon_call_model 会优先处理 messages

        if prompt is None and messages is None:
            raise ValueError("SiliconModelAPI.call 必须提供 prompt 或 messages")

        return await silicon_call_model(
            prompt=prompt,
            messages=messages,
            stream=stream,
            timeout=effective_timeout,
            model_config=self.model_config
        )

# --- 2. BaiduQianfanAPI ---
class BaiduQianfanAPI(BaseModelAPI):
    def __init__(self, model: Optional[str] = None):
        self.api_name = model
        self.model_config = API_CONFIG.get(self.api_name)
        if not self.model_config:
            raise ValueError(f"配置 '{self.api_name}' 未在 Extraction_project/config.py 中找到")
        super().__init__(model=self.model_config.get("model"))

    def validate_dependencies(self):
        if baidu_call_model is None:
            raise ImportError("Baidu API (Extraction_project/api_all/baidu_api) 未能成功导入。")

    async def call(self, prompt: str = None, messages: List[Dict[str, str]] = None, stream: bool = False,
                   timeout: Optional[int] = 60, **kwargs) -> str | AsyncGenerator[str, None]:

        # 该适配器将消息列表转换为单个 prompt。

        effective_prompt = prompt
        if effective_prompt is None and messages is not None:
            # 简单回退策略：把 messages 拼成字符串 (如果底层不支持 messages)
            effective_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        return await baidu_call_model(
            prompt=effective_prompt,
            stream=stream,
            timeout=timeout,
            model_config=self.model_config
        )


# --- 3. ZhipuModelAPI ---
class ZhipuModelAPI(BaseModelAPI):
    def __init__(self, model: Optional[str] = None):
        self.api_name = model
        self.model_config = API_CONFIG.get(self.api_name)
        if not self.model_config:
            raise ValueError(f"配置 '{self.api_name}' 未在 Extraction_project/config.py 中找到")
        super().__init__(model=self.model_config.get("model"))

    def validate_dependencies(self):
        if zhipu_call_model is None:
            raise ImportError("Zhipu API (Extraction_project/api_all/zhipu_api) 未能成功导入。")

    async def call(self, prompt: str = None, messages: List[Dict[str, str]] = None, stream: bool = False,
                   timeout: Optional[int] = 60, **kwargs) -> str | AsyncGenerator[str, None]:
        effective_prompt = prompt
        if effective_prompt is None and messages is not None:
            effective_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        return await zhipu_call_model(
            prompt=effective_prompt,
            stream=stream,
            timeout=timeout,
            model_config=self.model_config
        )


# --- 4. SparkModelAPI ---
class SparkModelAPI(BaseModelAPI):
    def __init__(self, model: Optional[str] = None):
        self.api_name = model
        self.model_config = API_CONFIG.get(self.api_name)
        if not self.model_config:
            raise ValueError(f"配置 '{self.api_name}' 未在 Extraction_project/config.py 中找到")
        super().__init__(model=self.model_config.get("model"))

    def validate_dependencies(self):
        if spark_call_model is None:
            raise ImportError("Spark API (Extraction_project/api_all/spark_api) 未能成功导入。")

    async def call(self, prompt: str = None, messages: List[Dict[str, str]] = None, stream: bool = False,
                   timeout: Optional[int] = 60, **kwargs) -> str | AsyncGenerator[str, None]:
        effective_prompt = prompt
        if effective_prompt is None and messages is not None:
            effective_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        return await spark_call_model(
            prompt=effective_prompt,
            stream=stream,
            timeout=timeout,
            model_config=self.model_config
        )


# --- 5. OpenAIModelAPI（支持 messages）---
class OpenAIModelAPI(BaseModelAPI):
    def __init__(self, model: Optional[str] = None):
        self.api_name = model
        self.model_config = API_CONFIG.get(self.api_name)
        if not self.model_config:
            raise ValueError(f"配置 '{self.api_name}' 未在 Extraction_project/config.py 中找到")

        specific_model = self.model_config.get("model")
        super().__init__(model=specific_model)
        self.client = None

    def validate_dependencies(self):
        if openai is None: raise ImportError("OpenAI 库未安装。")
        api_key = self.model_config.get("api_key")
        base_url = self.model_config.get("base_url")
        if not api_key:
            raise ValueError(f"OpenAI 'api_key' 未在 Extraction_project/config.py 的 '{self.api_name}' 条目中配置")

        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )

    async def call(self, prompt: str = None, messages: List[Dict[str, str]] = None, stream: bool = False,
                   timeout: Optional[int] = 60, **kwargs) -> str | AsyncGenerator[str, None]:
        if self.client is None: raise ValueError("OpenAI 客户端未初始化。")
        if stream: raise NotImplementedError("OpenAI 流式调用暂未实现")

        # 构造消息列表
        request_messages = []
        if messages:
            request_messages = messages
        elif prompt:
            request_messages = [
                {"role": "system", "content": "你是一个辅助助手，严格按照要求输出JSON。"},
                {"role": "user", "content": prompt}
            ]
        else:
            raise ValueError("OpenAIModelAPI.call 需要 prompt 或 messages")

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,  # 使用请求消息
                response_format={"type": "json_object"},
                timeout=timeout
            )
            response_content = completion.choices[0].message.content
            return response_content or ""
        except Exception as e:
            logging.error(f"OpenAI API ({self.api_name}) 调用失败: {e}")
            raise


# --- 6. AnthropicModelAPI ---
class AnthropicModelAPI(BaseModelAPI):
    def __init__(self, model: Optional[str] = None):
        self.api_name = model
        self.model_config = API_CONFIG.get(self.api_name)
        if not self.model_config:
            raise ValueError(f"配置 '{self.api_name}' 未在 Extraction_project/config.py 中找到")

        specific_model = self.model_config.get("model")
        super().__init__(model=specific_model)
        self.client = None

    def validate_dependencies(self):
        if anthropic is None: raise ImportError("Anthropic 库未安装。")
        api_key = self.model_config.get("api_key")
        if not api_key:
            raise ValueError(f"Anthropic 'api_key' 未在 Extraction_project/config.py 的 '{self.api_name}' 条目中配置")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def call(self, prompt: str = None, messages: List[Dict[str, str]] = None, stream: bool = False,
                   timeout: Optional[int] = 60, **kwargs) -> str | AsyncGenerator[str, None]:
        if self.client is None: raise ValueError("Anthropic 客户端未初始化。")
        if stream: raise NotImplementedError("Anthropic 流式调用暂未实现")

        system_prompt = "你是一个辅助助手，严格按照要求输出JSON..."

        # Anthropic 的 messages 格式略有不同 (System 必须是单独参数)
        request_messages = []
        if messages:
            # 过滤掉 role='system' 的消息，提取出来
            for m in messages:
                if m['role'] == 'system':
                    system_prompt = m['content']
                else:
                    request_messages.append(m)
        elif prompt:
            request_messages = [{"role": "user", "content": prompt}]

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=request_messages,
                timeout=timeout
            )
            return message.content[0].text or ""
        except Exception as e:
            logging.error(f"Anthropic API ({self.api_name}) 调用失败: {e}")
            raise
