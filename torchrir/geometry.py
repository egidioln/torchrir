"""Geometry module"""

from collections import deque
from typing import Any, Iterable, List, Optional, Protocol, Tuple
from warnings import warn

import scipy
from einops import einsum, rearrange
import torch
from torch import Tensor
from torch.linalg import matrix_rank

from numpy.typing import NDArray
import torchist
from torchaudio.functional import highpass_biquad

from torchrir.source import Source


class ImpulseResponseMethod(Protocol):
    """A callable that computes the impulse response of a source at a given point.

    Args:
        p: point at which the impulse response is computed
        s: Source
        dt: Sampling time ( 1 / sampling frequency)
        n_samples: Number of samples in the impulse response

    Returns:
        A tensor with shape (..., n_samples) containing the impulse response.
    """

    @staticmethod
    def __call__(
        p: Tensor,
        s: Source,
        dt: float,
        n_samples: int,
        speed_of_sound: float,
        tw: int,
    ) -> Tensor: ...


class ImpulseResponseStrategies:
    """A class containing strategies to build an impulse response."""

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
        tw: int = 200,
    ) -> Tensor:
        """Method described in https://arxiv.org/pdf/1710.04196

        Args:
            p (Tensor): point at which the impulse response is computed
            s (Source): Source
            dt (float): Sampling time ( 1 / sampling frequency)
            n_samples (int): Number of samples in the impulse response

        Returns:
            Tensor: a time-sampled impulse response perceived at position p from a (virtual or real) source s
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
            return idx - delay_idx_float

        window_radius = tw // 2

        delay_idx_float = d / (speed_of_sound * dt)
        delay_idx_error = _get_rel_idx(delay_idx_float)

        # This is a more effficient way to compute the impulse response of a big number of sources.
        # Computing the impulse response for each source separately is not efficient because most of the
        # samples are evaluated to 0 and it requires instantiating n_sources tensors of size n_samples.
        # Instead, we build the impulse response as a sequence of histograms, each of which contains
        # n_samples bins and the bins match the time indices of the output. The data accumulated in these
        # histograms is "pieces" of each individual impulse response, each of which calculcated for a integer
        # time delay n in [-window_radius, window_radius]. That way, we only compute n_window_samples histograms
        # instead of n_sources histograms, which normally n_window_samples << n_sources.

        h = torchist.histogram(
            delay_idx_float,
            bins=n_samples,
            low=0,
            upp=n_samples,
            weights=s.intensity / (4 * torch.pi * d) * _delta_lp(delay_idx_error),
        )

        for _ in range(window_radius):
            for n in (-_, _):
                h += torchist.histogram(
                    delay_idx_float + n,
                    bins=n_samples,
                    low=0,
                    upp=n_samples,
                    weights=s.intensity
                    / (4 * torch.pi * d)
                    * _delta_lp(delay_idx_error + n),
                )

        return h


class Patch:
    r"""A 3D patch $P$ of surface

    A patch is defined as the convex-hull of set of $n$ co-planar points $P:=\mathrm{co}\Big(\{p_0, ... p_{n-1}\}\Big)
    \in 2^{\mathbb{R}^3}$.

    Args:
        vertices: tensor with shape $(..., 3, n)$ where $n$ is the number of points defining the patch $P$. Requires $n>3$
        reflection_coeff: tensor with shape $(...)$ defining the reflectivity coefficient of the patch.
    """

    _vertices: Tensor
    _reflection_coeff: Tensor | float
    _rel_vertices: Tensor
    _is_planar: Tensor
    _is_convex: Tensor
    _normal_vector: Tensor

    __oom_retry_count: int = 0

    def __init__(
        self,
        vertices: Tensor,
        reflection_coeff: Tensor | float = 0.5,
    ):
        if vertices.shape[-2] != 3:
            raise ValueError(
                f"Expected tensors of shape (..., 3, n), got {vertices.shape}"
            )
        if vertices.shape[-1] < 3:
            raise ValueError(
                f"Expected tensors of shape (..., 3, n) with n >= 3, got {vertices.shape}"
            )

        if isinstance(reflection_coeff, float):
            reflection_coeff = torch.full_like(vertices[..., 0, 0], reflection_coeff)

        self._reflection_coeff = reflection_coeff
        self._vertices = vertices

        # Set inner propertiesPatch
        self._rel_vertices = _as_tensor(self._vertices) - self.origin

        self._is_planar = matrix_rank(self._rel_vertices) == 2

        if not torch.all(self._is_planar):
            raise NotImplementedError(
                "Non-planar patch detected, only planar patches are currently supported."
            )
        self._is_convex = self._is_planar  # all planar patches are convex

        v1, v2, *_ = torch.split(
            (self._vertices[..., :, 1:3] - self.origin),
            1,
            dim=-1,
        )
        normal_vector = _cross_movedim(v1, v2, (-1, -2))
        self._normal_vector = normal_vector / normal_vector.norm(dim=-2, keepdim=True)

    @property
    def is_planar(self) -> Tensor:
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
        return self._vertices[..., :, :1]

    @property
    def normal_vector(self):
        if self.is_planar.all():
            return self._normal_vector
        raise ValueError("Non-planar patch has no normal")

    def contains(self, arg: Any) -> Tensor:
        """Check if a point is contained in the patch."""
        if torch.all(self.is_convex):
            return self._convex_contains(*arg)
        raise NotImplementedError()

    def _convex_contains(
        self,
        p: Tensor,
        mask: Optional[Tensor] = None,
        atol: float = 1e-4,
    ) -> Tensor:
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
        origin = self.origin
        if mask is not None:
            if not mask.any():
                return torch.full_like(mask, False, dtype=torch.bool)
            p = p[mask]
            rel_vertices = rel_vertices[mask]
            origin = origin[mask]

        def _circle_pairwise(x: Tensor):
            x = rearrange(x, "... d n -> n ... d 1")
            return zip(x, (*x[1:], x[0]))

        # Inside outside problem, solution 4: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
        angle = torch.zeros(1, device=p.device)

        for v0, v1 in _circle_pairwise(rel_vertices - p + origin):
            m0, m1 = v0.norm(dim=-2), v1.norm(dim=-2)  # p is numerically at a vertex
            at_vertex = torch.isclose(
                den := m0 * m1, torch.zeros(1, device=den.device), atol=atol
            )
            angle = angle + torch.acos((v1 * v0).sum(dim=-2) / den)
            angle[at_vertex] = 2 * torch.pi
        return angle[..., 0] >= (2 * torch.pi - atol)

    def mirror(
        self, s: Source, force_product: bool = False, if_valid: bool = False
    ) -> Source | Iterable[Source]:
        """Mirror a point p through the plane of self

        Args:
            s (Source): Source to mirror
            force_product (bool, optional): Forces Cartesian Products between the source and the
                                            mirror batches. Defaults to False.
            if_valid (bool, optional): Mirror only if inner product between source and plane
                                        normal is positive. Defaults to False.

        Returns:
            Source: Mirrors of source
        """
        failed = False
        try:
            yield self._mirror(s, force_product=force_product, if_valid=if_valid)
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
                    _, force_product=force_product, if_valid=if_valid
                )
            self.__oom_retry_count -= 1

    def _mirror(
        self, s: Source, force_product: bool = False, if_valid: bool = False
    ) -> Source:
        p = s.p
        intensity = -s.intensity
        reflection_coeff = self._reflection_coeff
        if force_product:  # forces cartesian product of sources and patches by expanding a dimension, if needed
            for _ in range(self.origin.ndim - 2):
                p = p.unsqueeze(-3)
                intensity = intensity.unsqueeze(-1)
        p = p - self.origin
        inner_product_p_normal = dot(p, self.normal_vector, keepdim=True)

        valid = (inner_product_p_normal >= 0)[..., 0, 0] if if_valid else ...
        # I tried using masked tensors here but it didn't help
        # p = torch.masked.masked_tensor(p, valid.expand_as(p))
        # inner_product_p_normal = torch.masked.masked_tensor(inner_product_p_normal, valid.expand_as(inner_product_p_normal))

        new_positions = (
            self.origin + p - 2 * inner_product_p_normal * self.normal_vector
        )[valid]
        new_intensities = (intensity * reflection_coeff)[..., valid]
        # if torch.masked.is_masked_tensor(new_positions):
        #     new_positions = new_positions.get_data()[new_positions.get_mask()[..., 0]]
        return Source(new_positions, new_intensities)

    def turn_normal_towards(self, point: Tensor):
        """If necessary, flips the normal of the plane towards a point."""
        self._normal_vector *= dot(
            self._normal_vector, point - self.origin, keepdim=True
        ).sign()

    def can_see(self, other: "Patch") -> bool:
        """Checks if this patch can see the other one."""
        if self.is_convex and other.is_convex:
            return self._convex_can_see(other)
        return NotImplementedError("Method not implemented for non-convex patches.")

    def _convex_can_see(self, other: "Patch") -> bool:
        """Checks if this convex patch can see the other one.

        Convex Patch A can see the convex patch B if and only if

        """
        return NotImplementedError("Method not implemented patches.")

    def plot(
        self,
        *args,
        fig: Optional["matplotlib.pyplot.Figure"] = None,  # noqa: F821
        ax: Optional["matplotlib.pyplot.Axes"] = None,  # noqa: F821
        **kwargs,
    ):
        """Plots the patch as a 3D polygon."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from mpl_toolkits.mplot3d import Axes3D

        fig = fig or plt.gcf()

        ax = ax or plt.gca()
        if not isinstance(ax, Axes3D):
            ax = fig.add_subplot(111, projection="3d")
        poly = Poly3DCollection([self._vertices.T.cpu().numpy()], **kwargs)
        ax.add_collection3d(poly)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig, ax


class Room:
    """A base class for the room object, made out of a set of walls, each of which is a [`Patch`](torchrir.geometry.Patch)

    Args:
        walls: a list of patches defining the walls
    """

    _walls: Patch | Iterable[Patch]

    def __init__(self, walls: Patch | Iterable[Patch]):
        self._walls = walls

    @property
    def walls(self) -> Patch | Iterable[Patch]:
        return self._walls

    try:
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        def plot(self, *args, **kwargs) -> Tuple[Figure, Axes3D]:
            """Plots a the room by drawing each wall as a 3d patch.

            Returns:
                Matplotlib figure handler
                Matplotlib axes handler
            """
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
    """A room defined by the convex hull of a set of points

    Args:
        points: the vertices of the room
        reflection_coeff: the reflections coefficient of every wall
    """

    _points: Iterable[Tensor] = None
    _convex_hull: scipy.spatial.ConvexHull = None

    def __init__(
        self,
        points: Iterable[Tensor | NDArray],
        reflection_coeff: Tensor | float,
    ):
        device = None
        if isinstance(next(iter(points)), Tensor):
            points = points.detach()
            device = points.device
        self._convex_hull = scipy.spatial.ConvexHull(points.detach().cpu().T)
        self._points = torch.tensor(
            self._convex_hull.points.T, device=device, dtype=points.dtype
        )
        self._facets = rearrange(
            self._points[:, self._convex_hull.simplices.T], "... v t -> t v ..."
        )
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
        inner_point = self.points.sum(dim=-1, keepdim=True) / self.points.shape[-1]
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
                self.walls.mirror(s, force_product=force_batch_product, if_valid=True)
                for s in s_list
            ]
        for _ in self.walls.mirror(
            s_list, force_product=force_batch_product, if_valid=True
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
        impulse_response_fn: ImpulseResponseMethod = ImpulseResponseStrategies.sinc,
    ) -> tuple[Tensor, Tensor]:
        if p.shape[-1] == 3:
            p.unsqueeze(-1)

        n_samples = int(t_final * fs)
        dt = 1 / fs

        impulse_response = impulse_response_fn(p, s, dt, n_samples)

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
                    impulse_response += impulse_response_fn(p, s, dt, n_samples)
                    if _ <= k - 1:
                        s_list.appendleft(s)

        impulse_response = highpass_filtering(
            impulse_response,
            cutoff_hz=20.0,
            fs=fs,
            ir_len=len(impulse_response) // 10,
            order=4,
        )
        return impulse_response.cpu(), torch.arange(
            len(impulse_response), device="cpu"
        ) * dt


class Ray:
    """A ray $R(p_o, d)= \{p_o+dt~:~t\geq0\}$ defined by an origin $p_o\in\mathrm{R}^3$ and a direction $d\in\mathrm{R}^3$.


    Args:
        direction: tensor of shape $(..., 3, 1)$ defining directions towards which the rays are shot.
        origin: optional tensor of shape $(..., 3, 1)$ defining the origin of the rays. Defaults to $0$.

    """

    def __init__(self, direction: Tensor, origin: Tensor = None) -> None:
        if direction.ndim == 1:
            direction = direction.unsqueeze(-1)
        if origin is None:
            origin = torch.zeros_like(direction)
        if direction.shape[-2:] != (3, 1):
            raise ValueError(
                f"Expected shape (..., 3, 1) for direction, got {direction.shape}"
            )
        if origin.shape != direction.shape:
            raise ValueError(
                f"direction and origin must have the same shape, got {direction.shape} and {origin.shape}"
            )
        if (norm_direction := direction.norm()) == 0:
            raise ValueError("Null direction: Degenerated ray instantiated")
        self.origin = origin
        self.direction = direction / norm_direction

    # @torch.compile
    def intersects(self, patch: Patch) -> Tuple[Tensor, Tensor]:
        if torch.all(patch.is_planar):
            return self._intersects_planar_patch(patch)
        raise NotImplementedError("Ray.intersects() only supports planar patches")

    def _intersects_planar_patch(
        self, patch: Patch, two_sided_ray: bool = False
    ) -> Tuple[Tensor, Tensor]:
        # Ray plane intersection: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html  # noqa: E501
        # Computes t such that the ray intersects the plane generated by the patch at p = origin + t * direction
        t = -dot(
            self.origin - patch.origin,
            patch.normal_vector,
            keepdim=True,
        ) / dot(
            self.direction,
            patch.normal_vector,
            keepdim=True,
        )

        if not two_sided_ray:  # t < 0  and one sided ray => missed it
            sure_missed = (t < 0)[..., 0, 0]
        else:
            sure_missed = torch.full(t.shape[..., 0, 0], False, dtype=torch.bool)

        undecided = ~sure_missed
        p = torch.zeros(t.shape[:-2] + (3, 1), device=t.device)

        p[undecided, ...] = (self.origin + t * self.direction)[
            undecided, ...
        ]  # intersection points p
        #  Check if point inside poly patch
        undecided[undecided.clone()] *= patch.contains((p, undecided))
        return undecided, p


def dot(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    r"""Broadcastable version of [`torch.dot`](torch.dot)


    Args:
        x: tensor $x$
        y: tensor $y$
        keepdim : whether to keep the last dimension. Default, False

    Returns:
        dot product $\langle x, y\rangle = {x^\top y}$
    """

    trailing = "t" if x.shape[-1] == 1 else ""  # is a column vector
    input_shape = f"... i {trailing}"
    output_shape = f"... {trailing}"
    res = einsum(x, y, f"{input_shape}, {input_shape} -> {output_shape}")
    if keepdim:
        res = rearrange(res, f"... {trailing} -> ... 1 {trailing}")
    return res


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.tensor(x)


def _cross_movedim(v1: Tensor, v2: Tensor, moveaxis_arg: Tuple[int, int]):
    return torch.linalg.cross(
        v1.moveaxis(*moveaxis_arg),
        v2.moveaxis(*moveaxis_arg),
    ).moveaxis(*reversed(moveaxis_arg))


def highpass_filtering(
    signal: torch.Tensor, cutoff_hz: float, fs: float, order: int = 4, ir_len: int = 512
) -> torch.Tensor:
    """
    Filters a 1D signal using a Butterworth low-pass filter via convolution with the impulse response.

    Args:
        signal (torch.Tensor): Input 1D signal.
        cutoff_hz (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order.
        ir_len (int): Length of the impulse response used for convolution.

    Returns:
        torch.Tensor: Filtered signal.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be 1D")

    filtered = highpass_biquad(
        signal,
        sample_rate=int(fs),
        cutoff_freq=cutoff_hz,
        Q=0.99,
    )
    return filtered
