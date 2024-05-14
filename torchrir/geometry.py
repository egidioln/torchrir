import math
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.linalg import matrix_rank


class Patch(tuple):
    _origin: Tensor = None
    _rel_vertices: Tensor = None
    _is_planar: bool = None
    _is_convex: bool = None
    _matrix_plane: Tensor = None
    _normal_plane: Tensor = None

    def __new__(cls, vertices: Iterable[Tensor]) -> "Self":
        return super(Patch, cls).__new__(
            cls, tuple(obj.detach().squeeze() for obj in vertices)
        )

    def __init__(self, vertices: Iterable[Tensor]):

        for obj in self:
            if obj.shape != (3,):
                raise ValueError(f"Expected tensor of shape (3,), got {obj.shape}")

        # Set inner properties
        cat_vertices = torch.cat(self).view(3, -1)
        self._origin = cat_vertices[0]
        self._rel_vertices = cat_vertices - self._origin
        self._is_planar = matrix_rank(self._rel_vertices).item() == 2

        if self.is_planar:
            self._matrix_plane = cat_vertices[1:3] - self._origin
            self._normal_plane = torch.cross(*self._matrix_plane)
            self._normal_plane /= self._normal_plane.norm()

        # Try to define if is convex
        if not self.is_planar:
            self._is_convex = False
        if self.is_planar and len(self) <= 4:
            self._is_convex = True

    @property
    def is_planar(self):
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
        if self.is_planar:
            return self._normal_plane
        raise ValueError("Non-planar patch has no normal")

    def __contains__(self, p: Tensor) -> bool:
        if self.is_convex:
            return self._convex_contains(p)
        raise NotImplementedError()

    def _convex_contains(self, p: Tensor, epsilon: float = 1e-4) -> bool:
        def _circle_pairwise(x: Tensor):
            return zip(x, [*x[1:], x[0]])

        # Inside outside problem, solution 4: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
        angle = 0

        for v0, v1 in _circle_pairwise(self._rel_vertices - p + self._origin):
            m0, m1 = v0.norm(), v1.norm()
            if (den := m0 * m1) <= epsilon:  # numerically at a vertex
                return True
            angle += torch.acos(torch.dot(v0, v1) / den)
        return abs(angle - 2 * math.pi) <= epsilon


class Ray:
    origin: Tensor
    direction: Tensor

    def __init__(self, direction: Tensor, origin: Tensor = None) -> None:
        if origin is None:
            origin = torch.zeros(3)
        if norm_direction := direction.norm() == 0:
            raise ValueError("Null direction: Degenerated ray instantiated")
        self.origin = origin
        self.direction = direction / norm_direction

    def intersects(self, patch: Patch) -> bool:
        if patch.is_planar:
            return self._intersects_planar_patch(patch)
        raise NotImplementedError("Ray.intersects only suports planar patches")

    def _intersects_planar_patch(
        self, patch: Patch, two_sided_ray: bool = False
    ) -> bool:
        # Ray plane intersection: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html
        # Computes t such that the ray intersects the plane generated by the patch at p = origin + t * direction
        t = -torch.dot(self.origin - patch.origin, patch.normal_plane) / torch.dot(
            self.direction, patch.normal_plane
        )
        if not two_sided_ray and t < 0:
            return False

        p = self.origin + t * self.direction

        # Check if point inside poly
        return p in patch
