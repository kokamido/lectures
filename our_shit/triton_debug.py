import torch
import os
import triton
import triton.language as tl

from triton_util import *
from load_weights import load_weights


os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def jopa_kernel(input_ptr, bs: tl.constexpr):
    pid_0 = tl.program_id(0)

    offs = pid_0 * bs + tl.arange(0, bs)
    data = tl.load(input_ptr + offs)
    print(f"pid_0={pid_0}, data={data}, offs={offs}")
    pass


def jopa():
    a = torch.randn((8))
    print(a)
    grid = (2,)
    jopa_kernel[grid](a, 4)


jopa()
