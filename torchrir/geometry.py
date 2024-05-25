from typing import Any, Iterable, Tuple

import torch
from torch import BoolTensor, Tensor
from torch.linalg import matrix_rank


class Patch:
    _tensor: Tensor = None
    _origin: Tensor = None
    _rel_vertices: Tensor = None
    _is_planar: bool = None
    _is_convex: bool = None
    _matrix_plane: Tensor = None
    _normal_plane: Tensor = None

    def __init__(self, vertices: Tensor):
        for obj in vertices:
            if obj.shape[-1] != 3:
                raise ValueError(
                    f"Expected tensors of shape (..., 3,), got {obj.shape}"
                )
        self._t = vertices.unsqueeze(0) if vertices.ndim == 2 else vertices

        # Set inner properties
        self._origin = torch.tensor(self._t[:, :1])
        self._rel_vertices = torch.tensor(self._t) - self._origin

        self._is_planar = matrix_rank(self._rel_vertices) == 2

        def _if_planar(x: Tensor, other=torch.nan) -> Tensor:
            return torch.where(
                self.is_planar.view(-1, *((1,) * (x.ndim - 1))), x, other
            )

        self._matrix_plane = _if_planar((torch.tensor(self._t[:, 1:3]) - self._origin))
        self._normal_plane = _if_planar(torch.cross(*self._matrix_plane.moveaxis(1, 0))).unsqueeze(1)
        self._normal_plane /= self._normal_plane.norm(dim=-1).unsqueeze(-1)

        # Try to define if is convex
        self._is_convex = _if_planar(torch.tensor(torch.nan), False)
        if self._t.shape[1] <= 3:
            self._is_convex[:] = True

    @property
    def is_planar(self) -> BoolTensor:
        return self._is_planar

    @property
    def is_convex(self):
        if self._is_convex is not None:
            return self._is_convex
        raise NotImplementedError(
            "Unable to determine if poly is convex for more than 4 vertices"
        )

    @property
    def origin(self):
        return self._origin

    @property
    def normal_plane(self):
        if self.is_planar.all():
            return self._normal_plane
        raise ValueError("Non-planar patch has no normal")

    def contains(self, arg: Any) -> Tuple[BoolTensor, Tensor]:
        """Check if a point is contained in the patch."""
        if self.is_convex.all():
            return self._convex_contains(*arg)
        raise NotImplementedError()

    def _convex_contains(
        self, p: Tensor, mask: BoolTensor = None, atol: float = 1e-4
    ) -> BoolTensor:
        """Tests if a point p is inside a convex 3D polygon (self) by computing the winding number.

        See https://en.wikipedia.org/wiki/Point_in_polygon

        Args:
            p: point p to be tested
            mask: boolean mask flagging batch elements to be tested
            atol: see torch.isclose. Defaults to 1e-4.

        Returns:
            bool: true iff p in self
        """
        rel_vertices = self._rel_vertices
        origin = self._origin
        if mask is not None:
            if not mask.any():
                return False
            p = p[mask]
            rel_vertices = rel_vertices[... if rel_vertices.shape[0] == 1 else mask]
            origin = origin[... if rel_vertices.shape[0] == 1 else mask]

        def _circle_pairwise(x: Tensor):
            x = x.moveaxis(1, 0)
            return zip(x, (*x[1:], x[0]))

        # Inside outside problem, solution 4: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
        angle = torch.zeros(1, device=p.device)

        for v0, v1 in _circle_pairwise(rel_vertices - p + origin):
            m0, m1 = v0.norm(dim=1), v1.norm(dim=1)  # p is numerically at a vertex
            at_vertex = torch.isclose(
                den := m0 * m1, torch.zeros(1, device=den.device), atol=atol
            )
            angle = angle + torch.acos((v1 * v0).sum(dim=1) / den)
            angle[at_vertex] = 2 * torch.pi
        return angle >= (2 * torch.pi - atol)

    def mirror(self, p: Tensor):
        """Mirror a point p through the plane of self"""
        if p.ndim == 1:
            p = p.unsqueeze(0)
        if p.ndim == 2:
            p = p.unsqueeze(1)
        p = p - self._origin
        return self._origin + p - 2 * _broadcast_dot(p, self.normal_plane).unsqueeze(-1) * self.normal_plane

class Ray:
    def __init__(self, direction: Tensor, origin: Tensor = None) -> None:
        if direction.ndim == 2:
            direction = direction.unsqueeze(1)
        if origin is None:
            origin = torch.zeros_like(direction)
        elif origin.ndim == 2:
            origin = origin.unsqueeze(1)
        if (norm_direction := direction.norm()) == 0:
            raise ValueError("Null direction: Degenerated ray instantiated")
        self.origin = origin
        self.direction = direction / norm_direction

    @torch.compile
    def intersects(self, patch: Patch) -> Tuple[BoolTensor, Tensor]:
        if torch.all(patch.is_planar):
            return self._intersects_planar_patch(patch)
        raise NotImplementedError("Ray.intersects only suports planar patches")

    def _intersects_planar_patch(
        self, patch: Patch, two_sided_ray: bool = False
    ) -> Tuple[BoolTensor, Tensor]:
        # Ray plane intersection: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html
        # Computes t such that the ray intersects the plane generated by the patch at p = origin + t * direction
        t = -_broadcast_dot(
            self.origin - patch.origin, patch.normal_plane
        ) / _broadcast_dot(self.direction, patch.normal_plane)

        if not two_sided_ray:  # t < 0  and one sided ray => missed it
            sure_missed = (t < 0)[..., 0]
        else:
            sure_missed = torch.full(t.shape[:1], False, dtype=torch.bool)

        tbd = ~sure_missed
        p = torch.zeros(t.shape + (3,), device=t.device)

        p[tbd, :] = (self.origin + t.unsqueeze(-1) * self.direction)[
            tbd
        ]  # intersection points p
        #   Check if point inside poly patch
        tbd[tbd.clone()] *= patch.contains((p, tbd))
        return tbd, p

def _broadcast_dot(x: Tensor, y: Tensor) -> Tensor:
            return (x * y).sum(dim=-1)

