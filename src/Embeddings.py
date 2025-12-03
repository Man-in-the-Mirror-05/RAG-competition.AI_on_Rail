import os
import time
from typing import Dict, List, Optional
import numpy as np
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        """
        初始化嵌入基类
        Args:
            path (str): 模型或数据的路径
            is_api (bool): 是否使用API方式。True表示使用在线API服务，False表示使用本地模型
        """
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        """
        获取文本的嵌入向量表示
        Args:
            text (str): 输入文本
            model (str): 使用的模型名称
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 该方法需要在子类中实现
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度
        Args:
            vector1 (List[float]): 第一个向量
            vector2 (List[float]): 第二个向量
        Returns:
            float: 两个向量的余弦相似度，范围在[-1,1]之间
        """
        # 将输入列表转换为numpy数组，并指定数据类型为float32
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # 检查向量中是否包含无穷大或NaN值
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0

        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        # 计算向量的范数（长度）
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算分母（两个向量范数的乘积）
        magnitude = norm_v1 * norm_v2
        # 处理分母为0的特殊情况
        if magnitude == 0:
            return 0.0
            
        # 返回余弦相似度
        return dot_product / magnitude
    
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        self.max_retries = max(1, int(os.getenv("OPENAI_EMBEDDING_MAX_RETRIES", "8")))
        self._base_retry_delay = max(1.0, float(os.getenv("OPENAI_EMBEDDING_RETRY_DELAY", "5")))
        self._max_retry_delay = max(
            self._base_retry_delay,
            float(os.getenv("OPENAI_EMBEDDING_RETRY_MAX_DELAY", "60")),
        )
        rpm_limit = float(os.getenv("OPENAI_EMBEDDING_MAX_RPM", "0"))
        min_interval = float(os.getenv("OPENAI_EMBEDDING_MIN_INTERVAL", "0"))
        self._min_interval = max(min_interval, 60.0 / rpm_limit if rpm_limit > 0 else 0.0)
        self._last_request_ts = 0.0
        if self.is_api:
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def _sleep_for_rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.time()
        delta = now - self._last_request_ts
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_request_ts = time.time()

    def _should_retry(self, error: Exception) -> bool:
        if isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError, APIError)):
            return True
        if isinstance(error, PermissionDeniedError):
            message = str(error).lower()
            return "rpm limit" in message or "rate limit" in message
        return False

    def get_embedding(self, text: str, model: str = os.getenv("OPENAI_API_MODEL")) -> List[float]:
        if not self.is_api:
            raise NotImplementedError
        clean_text = text.replace("\n", " ")
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._sleep_for_rate_limit()
            try:
                response = self.client.embeddings.create(input=[clean_text], model=model)
                return response.data[0].embedding
            except Exception as exc:  # pragma: no cover - 仅运行时触发
                last_error = exc
                if not self._should_retry(exc) or attempt == self.max_retries:
                    break
                delay = min(self._base_retry_delay * (2 ** (attempt - 1)), self._max_retry_delay)
                print(
                    f"OpenAI embeddings request throttled ({attempt}/{self.max_retries}): {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Failed to generate embedding after {self.max_retries} attempts: {last_error}"
        ) from last_error
        
if __name__ == "__main__":
    embedding_model = OpenAIEmbedding()
    vector1 = embedding_model.get_embedding("你好，世界！")
    vector2 = embedding_model.get_embedding("Hello, world!")
    similarity = OpenAIEmbedding.cosine_similarity(vector1, vector2)
    print(f"Cosine Similarity: {similarity}")

