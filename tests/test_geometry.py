from torchrir.geometry import Patch, Ray
import torch
import pytest

def test_create_patch():
    vertices = torch.randn(3,3)
    p = Patch(vertices)

    with pytest.raises(ValueError):
        vertices = torch.randn(3,4)
        p = Patch(vertices)

    with pytest.raises(ValueError):
        vertices = torch.randn(3,2)
        p = Patch(vertices)
    
def test_patch_is_planar():
    vertices = torch.eye(3)
    assert Patch(vertices).is_planar

    vertices[2,2] = 0
    assert Patch(vertices).is_planar

    vertices[2,0:2] = 0.5
    assert not Patch(vertices).is_planar


def test_ray_intersects_convex_patch():
    patch = Patch(vertices=torch.eye(3)[None, ...])
    
    intersects, p0 = Ray(torch.ones(1, 3)).intersects(patch)
    assert intersects
    assert torch.allclose(p0, torch.ones(1, 3)/3)
    intersects, p0 = Ray(torch.tensor([1., 0., 0.])).intersects(patch)
    assert torch.allclose(p0, torch.tensor([1., 0., 0.]))
    assert intersects

    intersects, p0 = Ray(torch.tensor([-1., 1., 1.])).intersects(patch)
    assert not intersects

    intersects, p0 = Ray(-torch.ones(3)).intersects(patch)
    assert not intersects



def test_ray_intersects_convex_patch_broadcasting():
    n_batch = 100000
    n_dim_patches = 3 # all simplices
    n_reps = 10
    hits = 0

    for _ in range(n_reps):
        patches = Patch(vertices=10 * torch.randn(n_batch, n_dim_patches, 3))

        rays = Ray(*torch.randn(2, n_batch, 3))

        hits += rays.intersects(patches)[0].sum() / n_batch / n_reps

    assert  0.121 < hits < 0.124

def test_mirror_point_on_patch():
    patch = Patch(torch.eye(3))

    p = torch.zeros(3)
    assert torch.allclose(patch.mirror(p), 2 * torch.ones(3)/3)

    p = 2 * torch.ones(3) / 3
    assert torch.allclose(patch.mirror(p), torch.zeros(3), atol=1e-7)

    patch = Patch(torch.tensor(
        [
            [1., 4., 0.],
            [1., 4., 1.],
            [0., 4., 1.],
        ]
    ))

    p = 2 * torch.ones(3) / 3
    assert torch.allclose(patch.mirror(p), torch.tensor([p[0], 8-p[1], p[2]]), atol=1e-7)



def test_mirror_point_on_patch_broadcasting():
    n_batches = 1000
    patch = Patch(torch.randn(n_batches, 3, 3))
    p = torch.randn(n_batches, 3)
    p_rr = patch.mirror(patch.mirror(p))
    assert torch.allclose(p, p_rr.squeeze(), atol=1e-6)

    n_batches = 1000
    patch = Patch(torch.randn(n_batches, 3, 3))
    p = torch.randn(3)
    p_rr = patch.mirror(patch.mirror(p))
    assert torch.allclose(p, p_rr.squeeze(), atol=1e-5)

    n_batches = 1000
    patch = Patch(torch.randn(3, 3))
    p = torch.randn(n_batches, 3)
    p_rr = patch.mirror(patch.mirror(p))
    assert torch.allclose(p, p_rr.squeeze(), atol=1e-6)
