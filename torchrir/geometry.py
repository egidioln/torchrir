from collections import deque
from typing import Any, Callable, Iterable, List, Tuple
from warnings import warn

import scipy
import scipy.ndimage
import torch
from torch import BoolTensor, Tensor
from torch.linalg import matrix_rank

import torchist


from torchrir.source import Source

NDArray = type("NDArray")
ImpulseResponseMethod = Callable[[Tensor, Source, int, float, float], Tensor]


class Patch:
    r"""A patch is defined as a set of $n$ co-planar points $P=\{p_0, ... p_{n-1}\} \in 2^{\mathrm{R}^3}$ and an interpolating function $f(x)$ defining a surface at the locus $S_P = \{x\in\mathrm{R}^3~:~f(x)=0\}$ among them that"""

    _vertices: Tensor = None
    _reflection_coeff: Tensor = None
    _origin: Tensor = None
    _rel_vertices: Tensor = None
    _is_planar: bool = None
    _is_convex: bool = None
    _matrix_plane: Tensor = None
    _normal_plane: Tensor = None

    __oom_retry_count: int = 0

    def __init__(self, vertices: Tensor, reflection_coeff: Tensor | float = None):
        for obj in vertices:
            if obj.shape[-1] != 3:
                raise ValueError(
                    f"Expected tensors of shape (..., 3,), got {obj.shape}"
                )
        self._vertices = (
            torch.tensor(vertices).unsqueeze(0) if vertices.ndim == 2 else vertices
        )

        self._reflection_coeff = reflection_coeff
        if self._reflection_coeff is None:
            self._reflection_coeff = torch.full_like(vertices[..., 0, 0], 0.5)
        if isinstance(self._reflection_coeff, float):
            self._reflection_coeff = torch.full_like(
                vertices[..., 0, 0], self._reflection_coeff
            )

        # Set inner propertiesPatch
        self._origin = torch.tensor(self._vertices[:, :1])
        self._rel_vertices = torch.tensor(self._vertices) - self._origin

        self._is_planar = matrix_rank(self._rel_vertices) == 2

        def _if_planar(x: Tensor, other=torch.nan) -> Tensor:
            return torch.where(
                self.is_planar.view(-1, *((1,) * (x.ndim - 1))), x, other
            )

        self._matrix_plane = _if_planar(
            (self._vertices[:, 1:3].clone().detach() - self._origin)
        )
        self._normal_plane = _if_planar(
            torch.linalg.cross(*self._matrix_plane.moveaxis(1, 0))
        ).unsqueeze(1)
        self._normal_plane /= self._normal_plane.norm(dim=-1, keepdim=True)

        # Try to define if is convex
        self._is_convex = _if_planar(torch.tensor(torch.nan), False)
        if self._vertices.shape[1] <= 3:
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

    def mirror(
        self, s: Source, force_product: bool = False, if_inside: bool = False
    ) -> Source:
        """Mirror a point p through the plane of self

        Args:
            s (Source): Source to mirror
            force_product (bool, optional): Forces Cartesian Products between the source and the
                                            mirror batches. Defaults to False.
            if_inside (bool, optional): Mirror only if inner product between source and plane
                                        normal is positive. Defaults to False.

        Returns:
            Source: Mirrors of source
        """
        failed = False
        try:
            yield self._mirror(s, force_product=force_product, if_inside=if_inside)
        except torch.cuda.OutOfMemoryError:
            failed = True
        # OutOfMemory Fallback, try splitting the tensor for a couple of times
        if failed:
            if self.__oom_retry_count > 10:
                raise torch.cuda.OutOfMemoryError()
            self.__oom_retry_count += 1
            # print(torch.cuda.memory_summary())
            for _ in s.chunk(2):
                yield from self.mirror(
                    _, force_product=force_product, if_inside=if_inside
                )
            self.__oom_retry_count -= 1

    def _mirror(
        self, s: Source, force_product: bool = False, if_inside: bool = False
    ) -> Source:
        p = s.p
        intensity = s.intensity
        if p.ndim == 1:
            p = p.unsqueeze(0)
        if p.ndim == 2:
            p = p.unsqueeze(1)
        if (
            p.shape[-3] != 1
            and (p.shape[-3] != self._origin.shape[-3] or force_product)
        ):  # forces cartesian product of sources and patches by expanding a dimension, if needed
            p = p.unsqueeze(-3)
            intensity = intensity.unsqueeze(-1)
        p = p - self._origin
        inner_product_p_normal = _dot(p, self.normal_plane, keepdim=True)

        valid = (inner_product_p_normal >= 0)[..., 0] if if_inside else ...
        # I tried using masked tensors here but it didn't help
        # p = torch.masked.masked_tensor(p, valid.expand_as(p))
        # inner_product_p_normal = torch.masked.masked_tensor(inner_product_p_normal, valid.expand_as(inner_product_p_normal))

        new_positions = (
            self._origin + p - 2 * inner_product_p_normal * self.normal_plane
        )[valid]
        new_intensities = (intensity * self._reflection_coeff).unsqueeze(-1)[valid]
        # if torch.masked.is_masked_tensor(new_positions):
        #     new_positions = new_positions.get_data()[new_positions.get_mask()[..., 0]]
        return Source(new_positions, new_intensities)

    def turn_normal_towards(self, point: Tensor):
        """If necessary, flips the normal of the plane towards a point."""
        self._normal_plane *= _dot(
            self._normal_plane, point - self._origin, keepdim=True
        ).sign()

    def can_see(self, other: "Patch") -> bool:
        """Checks if this patch can see the other one."""
        if self.is_convex and other.is_convex:
            return self._convex_can_see(other)
        return NotImplementedError("Method not implemented for non-convex patches.")

    def _convex_can_see(self, other: "Patch"):
        """Checks if this convex patch can see the other one.

        Convex Patch A can see the convex patch B if and only if

        """
        pass


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

    # @torch.compile
    def intersects(self, patch: Patch) -> Tuple[BoolTensor, Tensor]:
        if torch.all(patch.is_planar):
            return self._intersects_planar_patch(patch)
        raise NotImplementedError("Ray.intersects() only supports planar patches")

    def _intersects_planar_patch(
        self, patch: Patch, two_sided_ray: bool = False
    ) -> Tuple[BoolTensor, Tensor]:
        # Ray plane intersection: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html  # noqa: E501
        # Computes t such that the ray intersects the plane generated by the patch at p = origin + t * direction
        t = -_dot(self.origin - patch.origin, patch.normal_plane, keepdim=True) / _dot(
            self.direction, patch.normal_plane, keepdim=True
        )

        if not two_sided_ray:  # t < 0  and one sided ray => missed it
            sure_missed = (t < 0)[..., 0, 0]
        else:
            sure_missed = torch.full(t.shape[:1], False, dtype=torch.bool)

        tbd = ~sure_missed
        p = torch.zeros(t.shape[:-1] + (3,), device=t.device)

        p[tbd, :] = (self.origin + t * self.direction)[tbd]  # intersection points p
        #  Check if point inside poly patch
        tbd[tbd.clone()] *= patch.contains((p, tbd))
        return tbd, p


def _dot(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    """Broadcastable version of torch.dot


    Args:
        x: tensor $x$
        y: tensor $y$
        keepdim : whether to keep the last dimension. Default, False

    Returns:
        dot product <x, y>
    """
    # return (x * y).sum(dim=-1, keepdim=keepdim)
    # res = torch.linalg.vecdot(x, y, dim=-1)
    res = torch.einsum("...i,...i", x, y)
    if keepdim:
        res.unsqueeze_(-1)
    return res


class Room:
    _walls: Patch | List[Patch]

    @property
    def walls(self) -> Patch | List[Patch]:
        return self._walls

    try:
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        def plot(self, *args, **kwargs) -> Tuple[Figure, Axes3D]:
            import matplotlib.pyplot as plt

            fig = plt.figure("room")
            # 3d plot of all patches
            ax = fig.add_subplot(111, projection="3d")
            for facet in self.facets:
                x, y, z = facet.T.tolist()
                ax.plot_trisurf(x, y, z, triangles=[[0, 1, 2], [1, 2, 0]], **kwargs)
            return fig, ax

    except ImportError:
        warn("No plotting support")


class ConvexRoom(Room):
    _points: Iterable[Tensor] = None
    _convex_hull: scipy.spatial.ConvexHull = None

    def __init__(self, points: Iterable[Tensor | NDArray], reflection_coeff: Tensor):
        """Constructor of the class `ConvexRoom`

        Args:
            points (Iterable[Tensor  |  NDArray]): the vertices of the room
        """
        device = None
        if isinstance(next(iter(points)), Tensor):
            points = points.detach()
            device = points.device
        self._convex_hull = scipy.spatial.ConvexHull(points.detach().cpu())
        self._points = torch.tensor(
            self._convex_hull.points, device=device, dtype=points.dtype
        )
        self._facets = self._points[self._convex_hull.simplices]
        self._walls = Patch(self._facets, reflection_coeff=reflection_coeff)
        self._force_walls_normal_to_point_inwards()

    @property
    def points(self) -> Tensor:
        """Returns the vertices of the room"""
        return self._points

    @property
    def facets(self) -> Tensor:
        """Returns the facets of the room"""
        return self._facets

    def _force_walls_normal_to_point_inwards(self) -> None:
        # centroid of points is a inner_point
        inner_point = self.points.sum(dim=-2) / self.points.shape[-2]
        if isinstance(self.walls, Patch):
            self._walls.turn_normal_towards(inner_point)
            return
        if isinstance(self.walls, Iterable):
            for w in self._walls:
                w.make_normal_to(inner_point)
            return
        raise RuntimeError("Unknown type of walls in room")

    def compute_k_reflected_sources(
        self,
        sources: Iterable[Source],
        k: int,
        force_batch_product: bool = False,
    ) -> List[Iterable[Source]]:
        if not isinstance(sources, Source):
            return [
                self.compute_k_reflected_sources(
                    s, k, force_batch_product=force_batch_product
                )
                for s in sources
            ]
        sources = [sources]
        tuple(
            sources.append(
                iter(
                    self.compute_reflected_sources(
                        sources[-1], force_batch_product=force_batch_product
                    )
                )
            )
            for _ in range(k)
        )
        return sources

    def compute_reflected_sources(
        self,
        s_list: Source | Iterable[Source],
        force_batch_product: bool = False,
    ) -> Iterable[Source]:
        if not isinstance(s_list, Source):
            return [
                self.walls.mirror(s, force_product=force_batch_product, if_inside=True)
                for s in s_list
            ]
        for _ in self.walls.mirror(
            s_list, force_product=force_batch_product, if_inside=True
        ):
            yield _

    # @torch.compile
    def compute_rir(
        self,
        p: Tensor,
        s: Source,
        k: int,
        t_final: float = 2.0,
        fs: float = 10000.0,
        impulse_response: ImpulseResponseMethod = None,
    ) -> Tensor:
        impulse_response = (
            impulse_response
            if impulse_response is not None
            else ImpulseResponseStrategies.sinc
        )
        n_samples = int(t_final * fs)
        dt = 1 / fs

        h = impulse_response(p, s, dt, n_samples)

        # For efficiency, compute_reflected_sources is a generator
        s_list = deque([s])
        for _ in range(k):
            print(f"Reflection {_}, reflecting...")
            for __ in range(len(s_list)):
                for s in self.compute_reflected_sources(
                    s_list.pop(), force_batch_product=True
                ):
                    print(
                        f"Reflection {_}, adding {s.intensity.nelement()} impulse responses..."
                    )
                    h += impulse_response(p, s, dt, n_samples)
                    if _ <= k - 1:
                        s_list.appendleft(s)

        return h.cpu(), torch.arange(len(h), device="cpu") * dt


class ImpulseResponseStrategies:
    @staticmethod
    def histogram(
        p: Tensor, s: Source, dt: float, n_samples: int, speed_of_sound: float = 343.0
    ) -> Tensor:
        d = s.distance_to(p)
        return torchist.histogram(
            d / speed_of_sound / dt,
            bins=n_samples,
            low=0,
            upp=n_samples,
            weights=s.intensity / (4 * torch.pi * d),
        )

    @staticmethod
    def sinc(
        p: Tensor,
        s: Source,
        dt: float,
        n_samples: int,
        speed_of_sound: float = 343.0,
        tw: int = 20,
    ) -> Tensor:
        """Method described in https://arxiv.org/pdf/1710.04196

        Args:
            p (Tensor): _description_
            s (Source): _description_
            dt (float): _description_
            n_samples (int): _description_

        Returns:
            Tensor: _description_
        """

        def _delta_lp(x):
            mask = x.abs() < tw / 2
            out = torch.zeros_like(x)
            if mask.any():
                t = x[mask]
                out[mask] = 0.5 * (1 + torch.cos(2 * torch.pi / tw * t)) * torch.sinc(t)
            return out

        d = s.distance_to(p)

        def _get_rel_idx(_float_idx):
            idx = torch.round(_float_idx)
            return idx - float_idx

        window_radius = tw // 2

        float_idx = d / (speed_of_sound * dt)
        rel_idx = _get_rel_idx(float_idx)
        h = torchist.histogram(
            float_idx,
            bins=n_samples,
            low=0,
            upp=n_samples,
            weights=s.intensity / (4 * torch.pi * d) * _delta_lp(rel_idx),
        )

        for _ in range(window_radius):
            for m in (-_, _):
                h += torchist.histogram(
                    float_idx + m,
                    bins=n_samples,
                    low=0,
                    upp=n_samples,
                    weights=s.intensity / (4 * torch.pi * d) * _delta_lp(rel_idx + m),
                )

        return h
