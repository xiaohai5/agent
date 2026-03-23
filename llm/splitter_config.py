from dataclasses import dataclass
from typing import Any


@dataclass
class SplitterConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    max_splite_num: int = 100

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SplitterConfig":
        if not data:
            return cls()
        return cls(
            chunk_size=int(data.get("chunk_size", cls.chunk_size)),
            chunk_overlap=int(data.get("chunk_overlap", cls.chunk_overlap)),
            top_k=int(data.get("top_k", cls.top_k)),
            max_splite_num=int(data.get("max_splite_num", cls.max_splite_num)),
        )

    def get_config(self) -> tuple[int, int, int, int]:
        return self.chunk_size, self.chunk_overlap, self.top_k, self.max_splite_num
