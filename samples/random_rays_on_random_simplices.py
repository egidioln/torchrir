import timeit
import torch
from torchrir.geometry import Patch, Ray

_DV = torch.device("cuda")
_N = 5_000_000
N_BENCHMARK = 100


def benchmark_random_rays_on_random_simplices():
    results = dict()


    def benchmark_sequential():
        hit = 0.0
        for _ in range(_N):
            p = Patch(torch.randn(3, 3, device=_DV))
            hit += Ray(torch.randn(3, device=_DV)).intersects(p)

        if abs(hit / _N - 0.125) > 1e-1:
            raise AssertionError(
                "Flaky assertion failed; running benchmark again may solve it, "
                "otherwise check installation integrity and that N_RAYS is 10_000_000."
            )

        print(hit / _N)


    p = Patch(torch.randn(_N, 3, 3, device=_DV))
    ray = Ray(torch.randn(_N, 3, device=_DV))
    def benchmark_broadcast():
        hit = ray.intersects(p)[0].sum().item()
        if abs(hit / _N - 0.125) > 1e-1:
            raise AssertionError(
                "Flaky assertion failed; running benchmark again may solve it, "
                "otherwise check installation integrity and that N_RAYS is 10_000_000."
            )

        print(hit / _N)

    for benchmark in [
        benchmark_broadcast,
        # torch.compile(benchmark_broadcast),
        # benchmark_sequential,
    ]:
        # benchmark_sequential = torch.compile(benchmark_sequential)
        benchmark()
        
        results[benchmark] = timeit.timeit(benchmark, number=N_BENCHMARK) / N_BENCHMARK
        print(benchmark.__name__, results[benchmark])


if __name__ == "__main__":
    benchmark_random_rays_on_random_simplices()
