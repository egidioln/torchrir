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
    p = Patch(vertices=torch.eye(3)[None, ...])
    
    assert Ray(torch.ones(1, 3)).intersects(p)
    assert Ray(torch.tensor([1., 0., 0.])).intersects(p)
    
    assert not Ray(torch.tensor([-1., 1., 1.])).intersects(p)
    assert not Ray(-torch.ones(3)).intersects(p)


def test_ray_intersects_convex_patch_broadcasting():
    n_batch = 100000
    n_dim_patches = 3 # all simplices
    n_reps = 10
    hits = 0

    for _ in range(n_reps):
        patches = Patch(vertices=10 * torch.randn(n_batch, n_dim_patches, 3))

        rays = Ray(*torch.randn(2, n_batch, 3))

        hits += rays.intersects(patches).sum() / n_batch / n_reps

    assert  0.121 < hits < 0.124