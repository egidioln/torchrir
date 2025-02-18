"""Source Module"""

from typing import Iterable, Self
import torch


class Source:
    r"""A point source $S=(p, I)$ is defined by its position $p\in\mathbb{R}^3$ and it's intensity $I$"""

    position: torch.Tensor
    """A 3D vector $p$ defining position of the point source"""
    intensity: float = None
    """An real number defining its intensity (in Watts)"""

    def __init__(self, position: torch.Tensor, intensity: torch.Tensor = None) -> None:
        if position.ndim == 1:
            position = position.unsqueeze(0)
        self.position = position
        self.intensity = (
            intensity if intensity is not None else torch.ones_like(position[..., 0])
        )

    @property
    def p(self) -> torch.Tensor:
        """An alias for [`position`](torchrir.source.Source.position) $p$"""
        return self.position

    def delay(self, p0: torch.Tensor, speed_of_sound: float = 343.0) -> torch.Tensor:
        """Computes time (in s) that sounds take to travel from source until a given point $p_0\in\mathbb{R}^3.$

        Args:
            p0: point $p_0\in\mathbb{R}^3$
            speed_of_sound: speed of sound $c$, by default $343.0~\mathrm{m}/\mathrm{s}.$

        Returns:
            Time delay that takes to sound from source to reach $p_0$
        """
        return self.distance_to(p0) / speed_of_sound

    def distance_to(self, p0: torch.Tensor) -> torch.Tensor:
        """Computes distance to a given point $p_0\in\mathbb{R}^3$

        Args:
            p0: point $p_0$

        Returns:
            Distance between source and $p_0$
        """
        return torch.norm(self.position - p0, dim=-1)

    @classmethod
    def merge(cls, s_list: Iterable[Self]) -> Self:
        return cls(
            position=torch.cat(tuple(s.position for s in s_list), dim=0),
            intensity=torch.cat(tuple(s.intensity for s in s_list), dim=0),
        )

    def chunk(self, n_chunks: int) -> Iterable[Self]:
        for p, i in zip(self.position.chunk(n_chunks), self.intensity.chunk(n_chunks)):
            yield Source(p, i)
