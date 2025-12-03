import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from src.Embeddings import BaseEmbeddings, OpenAIEmbedding
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: Optional[List[str]] = None) -> None:
        self.document = document or []
        self.vectors: List[List[float]] = []

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        document_path = path_obj / "document.json"
        with open(document_path, 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        legacy_path = path_obj / "doecment.json"
        if legacy_path.exists():
            legacy_path.unlink()
        if self.vectors:
            with open(path_obj / "vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        path_obj = Path(path)
        vectors_path = path_obj / "vectors.json"
        if not vectors_path.exists():
            raise FileNotFoundError(f"未找到向量文件：{vectors_path}")
        with open(vectors_path, 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        document_path = path_obj / "document.json"
        legacy_path = path_obj / "doecment.json"
        if document_path.exists():
            with open(document_path, 'r', encoding='utf-8') as f:
                self.document = json.load(f)
        elif legacy_path.exists():
            with open(legacy_path, 'r', encoding='utf-8') as f:
                self.document = json.load(f)
            # upgrade legacy file to new name
            with open(document_path, 'w', encoding='utf-8') as f:
                json.dump(self.document, f, ensure_ascii=False)
            legacy_path.unlink()
        else:
            raise FileNotFoundError(f"未找到文档文件：{document_path}")

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        if not getattr(self, "vectors", None):
            raise ValueError("向量库为空，请先调用 get_vector 或 load_vector 加载向量")
        if not self.document:
            return []
        k = max(1, min(k, len(self.document)))
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        if result.size == 0:
            return []
        indices = result.argsort()[-k:][::-1]
        return np.array(self.document, dtype=object)[indices].tolist()
