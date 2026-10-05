from dataclasses import dataclass
from typing import Mapping, Any

@dataclass(frozen=True)
class KernelConfig:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 32
    stages: int = 2
    warps: int = 4
    producer_warps: int = 1
    consumer_warpgroups: int = 1
    chunk_size: int = 256

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.stages > 8:
            raise ValueError("The public lab limits stages to 1..8")

    @classmethod
    def parse(cls, config: Mapping[str, Any] | None = None):
        return cls(**dict(config or {}))
