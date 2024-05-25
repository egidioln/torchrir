from typing import Iterable
import torch

from torchrir.geometry import ConvexRoom, Room
from torchrir.source import Source

torch.set_default_device("cuda")

REF_DEGREE: int = 3
ROOM_VERTEX: int = 8


def random_source_images():
    points: Iterable[torch.Tensor] = (torch.rand(ROOM_VERTEX, 3) - 0.5) * 2
    room: ConvexRoom = ConvexRoom(points)
    n_vertices = len(room.points)
    _ = torch.rand(n_vertices)
    cvx_combination = _ / sum(_)
    sources = Source(torch.sum(room.points.T * cvx_combination, dim=1))

    all_sources = room.compute_k_reflected_sources(sources, REF_DEGREE, force_product=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    room_fig = plt.figure("room")
    # 3d plot of all patches
    ax = room_fig.add_subplot(111, projection='3d')

    for facet in room.facets:
        x, y, z = facet.T.tolist()
        ax.plot_trisurf(x, y, z, color="blue", alpha=0.2, edgecolor="black")
    
    for source in all_sources:
        ax.scatter(*source.p.view(-1, 3).T.detach().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    random_source_images()
