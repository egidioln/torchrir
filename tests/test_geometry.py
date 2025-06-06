from itertools import product
import time
from typing import Iterable
from torchrir.geometry import ConvexRoom, ImpulseResponseStrategies, Patch, Ray
import torch
import pytest

from torchrir.source import Source


def test_create_patch():
    vertices = torch.randn(3, 3)
    _ = Patch(vertices)

    with pytest.raises(ValueError):
        vertices = torch.randn(4, 3)
        _ = Patch(vertices)

    with pytest.raises(ValueError):
        vertices = torch.randn(3, 2)
        _ = Patch(vertices)


def test_patch_is_planar():
    vertices = torch.eye(3)
    assert Patch(vertices).is_planar

    vertices[2, 2] = 0
    assert Patch(vertices).is_planar

    vertices[1, 1] = 0
    with pytest.raises(NotImplementedError):
        Patch(vertices).is_planar


def test_ray_intersects_convex_patch():
    patch = Patch(vertices=torch.eye(3)[None, ...])

    intersects, p0 = Ray(torch.ones(3, 1)).intersects(patch)
    assert intersects
    assert torch.allclose(p0, torch.ones(3, 1) / 3)
    intersects, p0 = Ray(torch.tensor([[1.0], [0.0], [0.0]])).intersects(patch)
    assert torch.allclose(p0, torch.tensor([[1.0], [0.0], [0.0]]))
    assert intersects

    intersects, p0 = Ray(torch.tensor([[-1.0], [1.0], [1.0]])).intersects(patch)
    assert not intersects

    intersects, p0 = Ray(-torch.ones(3)).intersects(patch)
    assert not intersects


# mark as flaky
@pytest.mark.flaky(max_runs=5)
def test_ray_intersects_convex_patch_broadcasting():
    n_batch = 1000
    n_dim_patches = 3  # all simplices
    n_reps = 10
    hits = 0

    for _ in range(n_reps):
        patches = Patch(vertices=10 * torch.randn(n_batch, n_dim_patches, 3))

        rays = Ray(*torch.randn(2, n_batch, 3, 1))

        hits += rays.intersects(patches)[0].sum() / n_batch / n_reps

    assert 0.120 < hits < 0.13


def test_mirror_point_on_patch():
    patch = Patch(torch.eye(3))

    s = Source(torch.zeros(3))
    assert torch.allclose(patch._mirror(s).p, 2 * torch.ones(3) / 3)

    s = Source(2 * torch.ones(3) / 3)
    assert torch.allclose(patch._mirror(s).p, torch.zeros(3), atol=1e-7)

    patch = Patch(
        torch.tensor(
            [
                [1.0, 4.0, 0.0],
                [1.0, 4.0, 1.0],
                [0.0, 4.0, 1.0],
            ]
        ).T
    )

    s = Source(2 * torch.ones(3) / 3)
    assert torch.allclose(
        patch._mirror(s).p,
        torch.tensor([[s.p[..., 0, :]], [8 - s.p[..., 1, :]], [s.p[..., 2, :]]]),
        atol=1e-7,
    )


def test_mirror_point_on_patch_broadcasting():
    n_batches = 3
    s = Source(torch.randn(n_batches, n_batches, 3, 1))
    patch = Patch(torch.randn(3, 3))
    p_r = patch._mirror(s, force_product=True)
    assert p_r.p.shape == (n_batches, n_batches, 3, 1)
    assert p_r.intensity.shape == (n_batches, n_batches)

    n_batches = 100
    s = Source(torch.randn(n_batches, 3, 1))
    patch = Patch(torch.randn(n_batches, 3, 3))
    p_rr = patch._mirror(patch._mirror(s))
    assert torch.allclose(s.p, p_rr.p, atol=1e-3)

    n_batches = 100
    s = Source(torch.randn(3, 1))
    patch = Patch(torch.randn(n_batches, 3, 3))
    p_rr = patch._mirror(patch._mirror(s))
    assert torch.allclose(s.p, p_rr.p, atol=1e-5)

    n_batches = 100
    s = Source(torch.randn(n_batches, 3, 1))
    patch = Patch(torch.randn(3, 3))
    p_rr = patch._mirror(patch._mirror(s))
    assert torch.allclose(s.p, p_rr.p, atol=1e-6)


def test_room_mirrors():
    room_geometry = torch.tensor([5, 4.3, 2.4])
    points: Iterable[torch.Tensor] = (
        room_geometry * torch.tensor(list(product((-1, 1), repeat=3))) / 2
    ).T
    room: ConvexRoom = ConvexRoom(points, reflection_coeff=0.3)
    n_vertices = len(room.points)
    _ = torch.rand(n_vertices)
    sources = Source(torch.zeros(3), intensity=torch.ones(1) * 5)

    reflected_sources = next(
        room.compute_reflected_sources(sources, force_batch_product=True)
    )
    dtype = reflected_sources.intensity.dtype
    assert torch.allclose(
        reflected_sources.intensity, -torch.ones(12, dtype=dtype) * 0.3 * 5
    )

    reflected_sources = next(
        room.compute_reflected_sources(reflected_sources, force_batch_product=True)
    )
    assert torch.allclose(
        reflected_sources.intensity, torch.ones(77, dtype=dtype) * 0.3 * 0.3 * 5
    )

    reflected_sources = next(
        room.compute_reflected_sources(reflected_sources, force_batch_product=True)
    )
    assert torch.allclose(
        reflected_sources.intensity,
        -torch.ones(447, dtype=dtype) * 0.3 * 0.3 * 0.3 * 5,
    )


def test_convexroom_rir_hist():
    room_geometry = torch.tensor([5, 4.3, 2.4], dtype=torch.float32)
    points: Iterable[torch.Tensor] = (
        room_geometry * torch.tensor(list(product((-1, 1), repeat=3))) / 2
    ).T
    room: ConvexRoom = ConvexRoom(points, reflection_coeff=0.1)
    source = Source(
        torch.zeros(3), intensity=torch.tensor(5, dtype=room_geometry.dtype)
    )

    p = torch.tensor([[2], [2], [0.8]], dtype=room_geometry.dtype)

    t0 = time.perf_counter_ns()
    rir, t = room.compute_rir(
        p, source, k=7, impulse_response_fn=ImpulseResponseStrategies.histogram
    )
    _ = time.perf_counter_ns() - t0  # dt
    # warn(_ / 1e9)
    assert torch.isclose(rir.mean(), torch.tensor(0.0), atol=1e-5), (
        "RIR mean should be zero "
    )
    assert torch.isclose(rir.norm(), torch.tensor(0.1353), atol=1e-3), (
        "RIR norm shouldd doesn't match expected value"
    )


def test_convexroom_rir_sinc():
    room_geometry = torch.tensor([5, 4.3, 2.4], dtype=torch.float32)
    points: Iterable[torch.Tensor] = (
        room_geometry * torch.tensor(list(product((-1, 1), repeat=3))) / 2
    ).T
    room: ConvexRoom = ConvexRoom(points, reflection_coeff=0.1)
    source = Source(
        torch.zeros(3), intensity=torch.tensor(5, dtype=room_geometry.dtype)
    )

    p = torch.tensor([2, 2, 0.8], dtype=room_geometry.dtype)

    t0 = time.perf_counter_ns()
    rir, t = room.compute_rir(
        p, source, k=7, impulse_response_fn=ImpulseResponseStrategies.sinc
    )
    _ = time.perf_counter_ns() - t0
    # warn(dt / 1e9)
    assert torch.isclose(rir.mean(), torch.tensor(0.0), atol=1e-5), (
        "RIR mean should be zero"
    )
    assert torch.isclose(rir.norm(), torch.tensor(0.3439), atol=1e-3), (
        "RIR norm shouldd doesn't match expected value"
    )

    # import matplotlib.pyplot as plt
    # plt.plot(t, rir )


# def test_patch_can_see():
#     patch_a = Patch()
#     patch_b = Patch()
#     patch_o = Patch()
