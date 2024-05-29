import timeit
from typing import Iterable, Tuple
from itertools import product
import torch

from torchrir.geometry import ConvexRoom
from torchrir.source import Source

torch.set_default_device("cuda")
torch.torch.set_default_dtype(torch.float)
REF_DEGREE: int = 9
SAMPLING_FREQ: float = 22e3
T_FINAL: float = 1.0
N_REP_BENCHMARK: int = 10
ROOM_GEOMETRY: Tuple[int, int, int] = (5, 4, 2.3)
WILL_PLOT_RIR: bool = True

def shoebox_room_source_images():
    points: Iterable[torch.Tensor] = (
        torch.tensor(ROOM_GEOMETRY) * torch.tensor(list(product((-1, 1), repeat=3))) / 2
    )
    room: ConvexRoom = ConvexRoom(points, reflection_coeff=0.2)
    n_vertices = len(room.points)
    _ = torch.rand(n_vertices)
    cvx_combination = _ / sum(_)
    sources = Source(torch.sum(room.points.T * cvx_combination, dim=1))

    rir, t = room.compute_rir(torch.zeros(3), sources, k=REF_DEGREE)
    print(
        timeit.timeit(
            lambda: room.compute_rir(torch.zeros(3), sources, k=REF_DEGREE, fs=SAMPLING_FREQ, t_final=T_FINAL),
            number=N_REP_BENCHMARK,
        )
        / N_REP_BENCHMARK
    )

    if not WILL_PLOT_RIR:
        return
    import matplotlib.pyplot as plt
    
    plt.plot(t, rir)
    plt.show()
    
    from scipy.io import wavfile
    wavfile.write("rir_.wav", rate=int(SAMPLING_FREQ), data=rir.cpu().numpy())

if __name__ == "__main__":
    shoebox_room_source_images()
