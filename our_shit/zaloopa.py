import os
import numpy as np

#os.environ['TRITON_INTERPRET']='1'

import torch

import triton
import triton.language as tl


@triton.jit
def my_kernel_1d(debug_ptr, counter_ptr):
    pid = tl.program_id(axis=0)  

    # Atomically increment the counter to get a unique index
    idx = tl.atomic_add(counter_ptr, 1)

    # Store block and thread IDs in the debug array
    tl.store(debug_ptr + idx, pid)

@triton.jit
def my_kernel_2d(debug_ptr, counter_ptr):
    pid = tl.program_id(axis=0)  # Current block ID
    tid = tl.program_id(axis=1)  # Current thread ID within the block

    # Atomically increment the counter to get a unique index
    idx = tl.atomic_add(counter_ptr, 1)

    # Store block and thread IDs in the debug array
    tl.store(debug_ptr + idx * 2, pid)  # Store block ID
    tl.store(debug_ptr + idx * 2 + 1, tid)  # Store thread ID


    

def run_2d():
    grid = lambda META: (16,12)
    counter = torch.zeros(1, dtype=int).to('cuda')
    debug = torch.rand(16*12).to('cuda')
    my_kernel_2d[grid](debug,counter)
    resized = debug.to('cpu').reshape((-1,2))
    assert len(resized) == len(set(tuple(x) for x in resized))
    print('2d', resized)

def run_1d(хуй):
    grid = lambda META: (хуй,)
    counter = torch.zeros(1, dtype=int).to('cuda')
    debug = torch.rand(хуй).to('cuda')
    my_kernel_1d[grid](debug, counter)
    return debug

print(np.mean([run_1d(2).to('cpu').numpy() for x in range(42800)], axis=0))


