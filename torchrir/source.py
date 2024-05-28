
import torch


class Source:
    position: torch.Tensor = None
    intensity: float = None
    reflection: torch.Tensor = None
    def __init__(self, position: torch.Tensor, intensity: torch.Tensor = None, reflection: torch.Tensor=None) -> None:
        if position.ndim == 1:
            position = position.unsqueeze(0)
        self.position = position
        self.intensity = intensity if intensity is not None else torch.ones_like(position[..., 0])
        self.reflection = reflection

    @property
    def p(self) -> torch.Tensor:
        return self.position


    def distance_to(self, p) -> torch.Tensor:
        return torch.norm(self.position - p, dim=-1)
    