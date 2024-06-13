from typing import Iterable, Self
import torch


class Source:
    position: torch.Tensor = None
    intensity: float = None

    def __init__(self, position: torch.Tensor, intensity: torch.Tensor = None) -> None:
        if position.ndim == 1:
            position = position.unsqueeze(0)
        self.position = position
        self.intensity = (
            intensity if intensity is not None else torch.ones_like(position[..., 0])
        )

    @property
    def p(self) -> torch.Tensor:
        return self.position

    def delay(self, p: torch.Tensor, speed_of_sound: float = 343.0) -> torch.Tensor:
        return self.distance_to(p) / speed_of_sound

    def distance_to(self, p: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.position - p, dim=-1)

    @classmethod
    def merge(cls, s_list: Iterable[Self]) -> Self:
        return cls(
            position=torch.cat(tuple(s.position for s in s_list), dim=0),
            intensity=torch.cat(tuple(s.intensity for s in s_list), dim=0),
        )

    def chunk(self, n_chunks: int) -> Iterable[Self]:
        for p, i in zip(self.position.chunk(n_chunks), self.intensity.chunk(n_chunks)):
            yield Source(p, i)
