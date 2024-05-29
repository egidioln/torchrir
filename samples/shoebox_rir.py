import timeit
from typing import Iterable, Tuple
from itertools import product
import torch

from torchrir.geometry import ConvexRoom
from torchrir.source import Source

torch.set_default_device("cuda")
torch.torch.set_default_dtype(torch.float)
REF_DEGREE: int = 9
SAMPLING_FREQ: float = 5e3
T_FINAL: float = 0.5
N_REP_BENCHMARK: int = 2
ROOM_GEOMETRY: Tuple[int, int, int] = (3, 3, 3)
WILL_PLOT_RIR: bool = True

def shoebox_room_source_images():
    points: Iterable[torch.Tensor] = (
        torch.tensor(ROOM_GEOMETRY) * torch.tensor(list(product((-1, 1), repeat=3))) / 2
    )
    room: ConvexRoom = ConvexRoom(points, reflection_coeff=0.3)
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
    f = torch.fft.fftshift(torch.fft.fftfreq(len(t), t[1])).cpu()

    plt.semilogx(f[f>0].cpu(), (torch.fft.fftshift(torch.fft.fft(rir-rir.mean())))[f>0].abs().cpu())
    plt.xlim(20, 1000)
    plt.show()
    
    from scipy.io import wavfile
    wavfile.write("rir_.wav", rate=int(SAMPLING_FREQ), data=rir.cpu().numpy())


if __name__ == "__main__":
    shoebox_room_source_images()
