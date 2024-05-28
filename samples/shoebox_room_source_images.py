import timeit
from typing import Iterable, Tuple
from itertools import product
import torch

from torchrir.geometry import ConvexRoom
from torchrir.source import Source

torch.set_default_device("cuda")
torch.torch.set_default_dtype(torch.float)
REF_DEGREE: int = 5
N_REP_BENCHMARK: int = 10
ROOM_GEOMETRY: Tuple[int, int, int] = (4, 3, 2)
WILL_PLOT_ROOM: bool = True

def shoebox_room_source_images():
    points: Iterable[torch.Tensor] = (
        torch.tensor(ROOM_GEOMETRY) * torch.tensor(list(product((-1, 1), repeat=3))) / 2
    )
    room: ConvexRoom = ConvexRoom(points)
    n_vertices = len(room.points)
    _ = torch.rand(n_vertices)
    cvx_combination = _ / sum(_)
    sources = Source(torch.sum(room.points.T * cvx_combination, dim=1))

    all_sources = room.compute_k_reflected_sources(
        sources, REF_DEGREE, force_batch_product=True
    )
    print(
        timeit.timeit(
            lambda: room.compute_k_reflected_sources(
                sources, REF_DEGREE, force_batch_product=True
            ),
            number=N_REP_BENCHMARK,
        )
        / N_REP_BENCHMARK
    )

    if not WILL_PLOT_ROOM:
        return
    import matplotlib.pyplot as plt
    room_fig, ax = room.plot(color="blue", alpha=0.2, edgecolor="black")

    for source in all_sources:
        ax.scatter(*source.p.view(-1, 3).T.detach().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    shoebox_room_source_images()
