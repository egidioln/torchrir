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
    p = Patch(vertices=torch.eye(3))
    
    assert Ray(torch.ones(3)).intersects(p)
    assert Ray(torch.tensor([1., 0., 0.])).intersects(p)
    
    assert not Ray(torch.tensor([-1., 1., 1.])).intersects(p)
    assert not Ray(-torch.ones(3)).intersects(p)

