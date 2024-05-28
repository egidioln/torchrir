from torchrir.geometry import Patch, Ray
import torch
import pytest

from torchrir.source import Source


def test_create_patch():
    vertices = torch.randn(3, 3)
    p = Patch(vertices)

    with pytest.raises(ValueError):
        vertices = torch.randn(3, 4)
        p = Patch(vertices)

    with pytest.raises(ValueError):
        vertices = torch.randn(3, 2)
        p = Patch(vertices)


def test_patch_is_planar():
    vertices = torch.eye(3)
    assert Patch(vertices).is_planar

    vertices[2, 2] = 0
    assert Patch(vertices).is_planar

    vertices[2, 0:2] = 0.5
    assert not Patch(vertices).is_planar


def test_ray_intersects_convex_patch():
    patch = Patch(vertices=torch.eye(3)[None, ...])

    intersects, p0 = Ray(torch.ones(1, 3)).intersects(patch)
    assert intersects
    assert torch.allclose(p0, torch.ones(1, 3) / 3)
    intersects, p0 = Ray(torch.tensor([1.0, 0.0, 0.0])).intersects(patch)
    assert torch.allclose(p0, torch.tensor([1.0, 0.0, 0.0]))
    assert intersects

    intersects, p0 = Ray(torch.tensor([-1.0, 1.0, 1.0])).intersects(patch)
    assert not intersects

    intersects, p0 = Ray(-torch.ones(3)).intersects(patch)
    assert not intersects


def test_ray_intersects_convex_patch_broadcasting():
    n_batch = 100000
    n_dim_patches = 3  # all simplices
    n_reps = 10
    hits = 0

    for _ in range(n_reps):
        patches = Patch(vertices=10 * torch.randn(n_batch, n_dim_patches, 3))

        rays = Ray(*torch.randn(2, n_batch, 3))

        hits += rays.intersects(patches)[0].sum() / n_batch / n_reps

    assert 0.121 < hits < 0.124


def test_mirror_point_on_patch():
    patch = Patch(torch.eye(3))

    s = Source(torch.zeros(3))
    assert torch.allclose(patch.mirror(s).p, 2 * torch.ones(3) / 3)

    s = Source(2 * torch.ones(3) / 3)
    assert torch.allclose(patch.mirror(s).p, torch.zeros(3), atol=1e-7)

    patch = Patch(
        torch.tensor(
            [
                [1.0, 4.0, 0.0],
                [1.0, 4.0, 1.0],
                [0.0, 4.0, 1.0],
            ]
        )
    )

    s = Source(2 * torch.ones(3) / 3)
    assert torch.allclose(
        patch.mirror(s).p, torch.tensor([s.p[..., 0], 8 - s.p[...,  1], s.p[..., 2]]), atol=1e-7
    )


def test_mirror_point_on_patch_broadcasting():
    n_batches = 1000
    s =  Source(torch.randn(n_batches, 3))
    patch = Patch(torch.randn(n_batches, 3, 3))
    p_rr = patch.mirror(patch.mirror(s))
    assert torch.allclose(s.p, p_rr.p.squeeze(), atol=1e-6)

    n_batches = 1000
    s =  Source(torch.randn(3))
    patch = Patch(torch.randn(n_batches, 3, 3))
    p_rr = patch.mirror(patch.mirror(s))
    assert torch.allclose(s.p, p_rr.p.squeeze(), atol=1e-5)

    n_batches = 1000
    s =  Source(torch.randn(n_batches, 3))
    patch = Patch(torch.randn(3, 3))
    p_rr = patch.mirror(patch.mirror(s))
    assert torch.allclose(s.p, p_rr.p.squeeze(), atol=1e-6)

    n_batches_a = 100
    n_batches_b = 200
    s =  Source(torch.randn(n_batches_a, 3))
    patch = Patch(torch.randn(n_batches_b, 3, 3))
    p_r = patch.mirror(s)
    assert p_r.p.shape == (n_batches_a, n_batches_b, 1, 3)


def test_room_mirrors():
    pass
