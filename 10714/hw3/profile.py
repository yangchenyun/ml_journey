# %%
import sys

sys.path.append("/home/steveyang/.jupyternb/hw3/python")
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

# %%
import triton


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(5, 14)],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "cpu",
            "cuda",
            'triton',
        ],  # possible values for `line_arg``
        line_names=[
            "CPU",
            "CUDA",
            "Triton",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "--")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="setitem-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(N, provider):
    shape = (N, N)
    device = None
    if provider == "cpu":
        device = nd.cpu()
    if provider == "cuda":
        device = nd.cuda()
    if provider == "triton":
        device = nd.triton()

    _A = np.random.randint(low=0, high=10, size=shape)
    _B = np.random.randint(low=0, high=10, size=shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)

    def matmul():
        return A @ B

    def ewise_setitem():
        A[:, :] = B[:, :]

    def reduce_max():
        A.max(axis=0)

    ms, min_ms, max_ms = triton.testing.do_bench(reduce_max)

    gbps = lambda ms: _A.size**2 * _A.itemsize * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
benchmark.run(show_plots=True, print_data=True)
# %%
