import logging
from typing import Optional, Dict, Type

# 导入模型服务适配器
from Extraction_project.llm_api.model_api_interface import BaseModelAPI
from Extraction_project.llm_api.model_api_providers import (
    SiliconModelAPI,
    OpenAIModelAPI,
    AnthropicModelAPI,
    BaiduQianfanAPI,
    ZhipuModelAPI,
    SparkModelAPI
)

# 服务名称需与 config.py 中的配置键一致
API_PROVIDERS_REGISTRY: Dict[str, Type[BaseModelAPI]] = {
    # 硅基 (SiliconFlow)
    "silicon": SiliconModelAPI,
    "silicon_deepseek-v3.2": SiliconModelAPI,
    "silicon_deepseek_r1_0528": SiliconModelAPI,
    "silicon_01": SiliconModelAPI,
    "silicon_glm_z1_01": SiliconModelAPI,
    "silicon_deepseek_r1_0528_01": SiliconModelAPI,
    "silicon_glm_4": SiliconModelAPI,

    # 百度 (Baidu)
    "baidu": BaiduQianfanAPI,

    # 智谱 (Zhipu)
    "zhipu": ZhipuModelAPI,

    # 讯飞 (Spark)
    "spark": SparkModelAPI,

    # OpenAI
    "openai": OpenAIModelAPI,

    # Anthropic
    "anthropic": AnthropicModelAPI
}


def get_model_api(
        api_provider: str,
        model_name: Optional[str] = None  # 保留参数以兼容旧调用
) -> BaseModelAPI:
    """
    API工厂函数。
    """
    provider_name = api_provider.lower()

    provider_class = API_PROVIDERS_REGISTRY.get(provider_name)

    if provider_class is None:
        raise ValueError(
            f"未知的 API provider: '{api_provider}'. "
            f"支持的 providers: {list(API_PROVIDERS_REGISTRY.keys())}"
        )

    try:
        # 将服务配置键传入适配器以读取对应配置
        return provider_class(model=api_provider)

    except Exception as e:
        logging.error(f"初始化 API provider '{api_provider}' 失败: {e}")
        raise