import timeit
import torch
from torchrir.geometry import Patch, Ray

_DV = torch.device("cuda")

N_RAYS = 10_000_000


def benchmark_random_rays_on_1_simplice():
    results = dict()

    p = Patch(torch.eye(3, device=_DV))

    def benchmark_sequential():
        hit = 0.0
        for _ in range(N_RAYS):
            hit += Ray(torch.randn(3, device=_DV)).intersects(p)

        if abs(hit / N_RAYS - 0.125) > 1e-1:
            raise AssertionError(
                "Flaky assertion failed; running benchmark again may solve it, "
                "otherwise check installation integrity and that N_RAYS is 10_000_000."
            )

        print(hit / N_RAYS)


    def benchmark_broadcast():
        hit = Ray(torch.randn(N_RAYS, 3, device=_DV)).intersects(p).sum().item()

        if abs(hit / N_RAYS - 0.125) > 1e-1:
            raise AssertionError(
                "Flaky assertion failed; running benchmark again may solve it, "
                "otherwise check installation integrity and that N_RAYS is 10_000_000."
            )

        print(hit / N_RAYS)

    for benchmark in [
        benchmark_broadcast,
        # benchmark_sequential,
    ]:
        # benchmark_sequential = torch.compile(benchmark_sequential)
        benchmark()
        results[benchmark] = timeit.timeit(benchmark, number=2)
        print(benchmark.__name__, results[benchmark])


if __name__ == "__main__":
    benchmark_random_rays_on_1_simplice()