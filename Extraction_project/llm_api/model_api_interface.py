from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator

class BaseModelAPI(ABC):
    """
    模型API的抽象基类 (接口)。
    """
    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.validate_dependencies()

    @abstractmethod
    # 可选超时时间
    async def call(
        self,
        prompt: str,
        stream: bool = False,
        timeout: Optional[int] = 60
    ) -> str | AsyncGenerator[str, None]:
        """
        异步调用模型API。
        """
        pass

    def validate_dependencies(self):
        """
        (可选) 检查所需的环境变量或库是否已安装。
        """
        pass