
from triton_util import *
from kek import matmul
import torch

# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def naive_mm_kernel(
    a_ptr,
    b_ptr,
    res_ptr,
    block_size: tl.constexpr,
    inter_dim: tl.constexpr,
    b_n_cols: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    a_offs = (
        inter_dim * (pid_0 * block_size + tl.arange(0, block_size))[:, None]
        + tl.arange(0, inter_dim)[None, :]
    )
    a_rows = tl.load(a_ptr + a_offs)

    b_offs = (
        b_n_cols * tl.arange(0, inter_dim)[:, None] + tl.arange(0, b_n_cols)[None, :]
    )
    b = tl.load(b_ptr + b_offs)
    res = tl.dot(a_rows, b)
    res_offs = (
        b_n_cols * (pid_0 * block_size + tl.arange(0, block_size))[:, None]
        + tl.arange(0, b_n_cols)[None, :]
    )
    tl.store(res_ptr + res_offs, res)


@triton.jit
def tiling_mm_kernel(
    a_ptr,
    b_ptr,
    res_ptr,
    block_size: tl.constexpr,
    inter_dim: tl.constexpr,
    b_n_cols: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    a_offs = (
        inter_dim * (pid_0 * block_size + tl.arange(0, block_size))[:, None]
        + tl.arange(0, inter_dim)[None, :]
    )
    a_rows = tl.load(a_ptr + a_offs)

    b_offs = (pid_1 * block_size + tl.arange(0, block_size))[
        :, None
    ] + b_n_cols * tl.arange(0, inter_dim)[None, :]
    b_cols = tl.load(b_ptr + b_offs)
    b_cols = tl.trans(b_cols, 1, 0)

    res = tl.dot(a_rows, b_cols)
    res_offs = (
        b_n_cols * (pid_0 * block_size + tl.arange(0, block_size))[:, None]
        + pid_1 * block_size
        + tl.arange(0, block_size)[None, :]
    )
    tl.store(res_ptr + res_offs, res)


def mm_naive(a, b, bs=16):
    res = torch.empty((a.shape[0], b.shape[1]),  device=a.device, dtype=a.dtype)
    grid = lambda meta: (cdiv(a.shape[0], bs),)
    naive_mm_kernel[grid](a, b, res, bs, a.shape[1], b.shape[1])

    return res


def mm_tiling(a, b, bs=16):
    res = torch.empty((a.shape[0], b.shape[1]),  device=a.device, dtype=a.dtype)
    grid = lambda meta: (cdiv(a.shape[0], bs), cdiv(b.shape[1], bs))
    print('ХУЙ', grid('говно'))
    tiling_mm_kernel[grid](a, b, res, bs, a.shape[1], b.shape[1])
    return res


def torch_mm(a, b):
    return torch.matmul(a, b)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "square_matrix_size"
        ],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(4, 8, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["naive", "tiling", "hui"],  # Possible values for `line_arg`.
        line_names=["Naive", "Tiling", "hui"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-"), ("pink", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="matmul-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(square_matrix_size, provider):
    sz = square_matrix_size
    a = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
    b = torch.rand((sz, sz), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: mm_naive(a, b), quantiles=quantiles, rep=2048
        )
    if provider == "tiling":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: mm_tiling(a, b), quantiles=quantiles, rep=2048
        )
    if provider == "hui":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b,16,16,16,16), quantiles=quantiles, rep=2048
        )
    gbps = lambda ms: 12 * sz / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
