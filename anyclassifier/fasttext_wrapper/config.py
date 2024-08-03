from typing import Any, Dict, Set, Optional, Union
from transformers import PretrainedConfig


class FastTextConfig(PretrainedConfig):
    model_type = "fasttext"

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
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        self.dim = dim
        self.ws = ws
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.word_ngrams = word_ngrams
        self.bucket = bucket
        self.pretrained_vectors = pretrained_vectors
        self.label2id = label2id
        self.id2label = id2label
        if self.id2label is not None:
            # json key is always str but id2label key is always int. Conversion is necessary.
            self.id2label = {int(k): v for k, v in self.id2label.items()}

    def to_fasttext_args(self) -> Dict[str, Any]:
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

    def to_dict(self, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
        output = {
            "dim": self.dim,
            "ws": self.ws,
            "min_count": self.min_count,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "word_ngrams": self.word_ngrams,
            "bucket": self.bucket,
            "pretrained_vectors": self.pretrained_vectors,
            "label2id": self.label2id,
            "id2label": self.id2label
        }
        if exclude is not None:
            return {k: v for k, v in output.items() if k not in exclude}
        return output
