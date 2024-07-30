from typing import Any, Dict
from transformers import PretrainedConfig


class FastTextConfig(PretrainedConfig):
    model_type = "fasttext_wrapper"

    def __init__(
        self,
        dim: int = 100,
        ws: int = 5,
        min_count: int = 5,
        min_n: int = 1,
        max_n: int = 1,
        word_ngrams: int = 1,
        bucket: int = 2000000,
        pretrained_vectors: str = "",
    ):
        self.dim = dim
        self.ws = ws
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.word_ngrams = word_ngrams
        self.bucket = bucket
        self.pretrained_vectors = pretrained_vectors
        super().__init__()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "ws": self.ws,
            "minCount": self.min_count,
            "minn": self.min_n,
            "maxn": self.max_n,
            "wordNgrams": self.word_ngrams,
            "bucket": self.bucket,
            "pretrainedVectors": self.pretrained_vectors
        }
