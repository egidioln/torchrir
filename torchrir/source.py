"""Source Module"""

from typing import Iterable, Optional, Self, TYPE_CHECKING
import torch


if TYPE_CHECKING:
    from torchrir.geometry import Patch, Ray


class Source:
    r"""A point source $S=(p, I)$ is defined by its position $p\in\mathbb{R}^3$ and it's intensity $I$"""

    position: torch.Tensor
    """A 3D vector $p$ defining position of the point source"""
    intensity: torch.Tensor
    """A real number defining its intensity (in Watts)"""
    root_patch_indices: Optional[torch.Tensor]
    """For each virtual source, the index of the patch that was used to create its oldest virtual ancestor."""
    root_patch: Optional["Patch"]
    """The patch that was used to create the oldest virtual ancestor of this source."""

    def __init__(
        self,
        position: torch.Tensor,
        intensity: Optional[torch.Tensor] = None,
        root_patch_indices: Optional[torch.Tensor] = None,
        root_patch: Optional["Patch"] = None,
    ) -> None:
        """Initializes a point source.

        Args:
            position: tensor of shape (..., 3, 1) position of the point source.
            intensity: tensor of shape (...) defining relative intensity of the sources. Defaults to ones.
        """
        if position.ndim == 1:
            position = position.unsqueeze(1)
        self.position = position

        if intensity is None:
            intensity = torch.ones_like(position[..., 0, 0])
        if not isinstance(intensity, torch.Tensor):
            intensity = torch.tensor(intensity, dtype=position.dtype)
        self.intensity = intensity

        self.root_patch_indices = root_patch_indices
        self.root_patch = root_patch

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
        return torch.norm(self.position - p0, dim=-2)[..., 0]

    @classmethod
    def merge(cls, s_list: Iterable[Self]) -> Self:
        return cls(
            position=torch.cat(tuple(s.position for s in s_list), dim=0),
            intensity=torch.cat(tuple(s.intensity for s in s_list), dim=0),
        )

    def chunk(self, n_chunks: int) -> Iterable["Source"]:
        for p, i in zip(self.position.chunk(n_chunks), self.intensity.chunk(n_chunks)):
            yield Source(p, i)

    def can_see(self, p: torch.Tensor) -> "Source":
        """Checks if source can see a point $p\in\mathbb{R}^3 through it's parent patch$.

        Args:
            p: point $p\in\mathbb{R}^3$

        Returns:
            A boolean tensor indicating whether source can see the point $p$.
        """
        from torchrir.geometry import Ray

        if p.ndim == 1:
            p = p.unsqueeze(1)

        if self.root_patch_indices is None:
            return self
        direction = (_ := p - self.position) / torch.norm(_, dim=-2, keepdim=True)
        ray = Ray(direction, self.position)
        idx, _ = ray.intersects(self.root_patch[self.root_patch_indices])
        return Source(
            position=self.position[idx],
            intensity=self.intensity[idx],
            root_patch_indices=self.root_patch_indices[idx],
            root_patch=self.root_patch,
        )

    def sample_rays(self) -> "Ray":
        """Samples rays from the source to its parent patch.

        Returns:
            An iterable of rays originating from the source position towards the parent patch.
        """
        from torchrir.geometry import Ray

        direction = (_r := torch.randn_like(self.position)) / _r.norm(
            dim=-2, keepdim=True
        )
        r = Ray(
            direction=direction,
            origin=self.position,
        )
        return r
