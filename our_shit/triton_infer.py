import torch
import os
import math
import time
import torch.nn.functional as F

# os.environ["TRITON_INTERPRET"] = "1"
# os.environ["TRITON_MAX_TENSOR_NUMEL"] = "2097152"
# os.environ["TRITON_KERNEL_DUMP"] = "1"
# os.environ["TRITON_ALWAYS_COMPILE"] = "1"
# os.environ["TRITON_DUMP_DIR"] = (
#     "/workspaces/obisidian/flash_attention/triton/code/jopa_dump"
# )
from triton_util import *
from load_weights import load_weights
from real_image_classifier import ImageClassifier

KERNEL_SIZE = 3


def triton_infer(model, x, block_size=64):
    filter_weights = model.conv_layers[0].weight.data
    bias_weights = model.conv_layers[0].bias.data

    res_0 = conv(
        x,
        filter_weights.reshape((32, 1, KERNEL_SIZE, KERNEL_SIZE)),
        bias_weights,
    )
    res_1 = torch.empty_like(res_0)

    grid = (cdiv(res_1.numel(), block_size),)

    relu_kernel[grid](res_0, res_0.numel(), get_jopa(res_0.numel()), res_1, block_size)

    filter_weights = model.conv_layers[2].weight.data
    bias_weights = model.conv_layers[2].bias.data

    res_2 = conv(
        res_1,
        filter_weights.reshape((64, 32, KERNEL_SIZE, KERNEL_SIZE)),
        bias_weights,
    )
    res_3 = torch.empty_like(res_2, device="cuda")

    grid = (cdiv(res_3.numel(), block_size),)
    relu_kernel[grid](res_2, res_2.numel(), get_jopa(res_2.numel()), res_3, block_size)

    filter_weights = model.conv_layers[4].weight.data
    bias_weights = model.conv_layers[4].bias.data

    res_4 = conv(
        res_3,
        filter_weights.reshape((64, 64, KERNEL_SIZE, KERNEL_SIZE)),
        bias_weights,
    )
    res_5 = torch.empty_like(res_4, device="cuda")

    grid = (cdiv(res_5.numel(), block_size),)
    relu_kernel[grid](res_4, res_4.numel(), get_jopa(res_4.numel()), res_5, block_size)

    return res_5


@triton.jit
def conv_kernel(
    x_ptr,
    x_c: tl.constexpr,
    x_h: tl.constexpr,
    x_w: tl.constexpr,
    filter_ptr,
    bias_ptr,
    out_ptr,
    kernel_size: tl.constexpr,
    alloc_kernel_size: tl.constexpr,
    bias_size: tl.constexpr,
    bias_alloc_size: tl.constexpr,
):
    # берём сразу весь x и умножаем на один фильтр + добавляем bias
    # единственный параллелизм - это по фильтрам в свёртке
    pid_0 = tl.program_id(0)  # номер нового канала (или номер кернела)
    pid_1 = tl.program_id(1)  # координата по высоте
    pid_2 = tl.program_id(2)  # координата по ширине

    res = tl.zeros((x_c,), dtype=tl.float32)

    # оффсеты и маски для x
    x_offs_h = pid_1 + tl.arange(0, alloc_kernel_size)
    x_offs_w = pid_2 + tl.arange(0, alloc_kernel_size)
    x_mask_0 = x_offs_h < pid_1 + kernel_size  # 1d vector
    x_mask_1 = x_offs_w < pid_2 + kernel_size  # 1d vector
    x_mask = x_mask_0[:, None] & x_mask_1[None, :]
    x_offs = x_w * x_offs_h[:, None] + x_offs_w[None, :]

    # оффсеты и маски для filter
    filter_offs_h = tl.arange(0, alloc_kernel_size)
    filter_offs_w = tl.arange(0, alloc_kernel_size)
    filter_mask_0 = filter_offs_h < kernel_size  # 1d vector
    filter_mask_1 = filter_offs_w < kernel_size  # 1d vector
    filter_mask = filter_mask_0[:, None] & filter_mask_1[None, :]
    filter_offs = kernel_size * filter_offs_h[:, None] + filter_offs_w[None, :]

    for channel_number in range(x_c):
        filter_weights = tl.load(
            filter_ptr
            + pid_0 * kernel_size * kernel_size * x_c
            + channel_number * kernel_size * kernel_size
            + filter_offs,
            mask=filter_mask,
        )

        x_channel = tl.load(x_ptr + channel_number * x_h * x_w + x_offs, mask=x_mask)
        res = res + tl.sum(x_channel * filter_weights)

    # потом надо здесь прибавлять bias
    out_offs = (
        pid_0 * tl.num_programs(1) * tl.num_programs(2)  # скипаем прошлые каналы
        + pid_1 * tl.num_programs(1)  # скипаем строки
        + pid_2  # скипаем элементы в текущей строке
        + tl.zeros((x_c,), dtype=tl.int32)
    )

    bias_weights = tl.load(bias_ptr + pid_0)
    res = res + bias_weights

    tl.store(out_ptr + out_offs, res)


@triton.jit
def relu_kernel(
    input_ptr,
    input_numel: tl.constexpr,
    alloc_input_numel: tl.constexpr,
    output_ptr,
    block_size: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    input_offs = tl.arange(0, block_size)
    input_data = tl.load(input_ptr + pid_0 * block_size + input_offs)
    zeros = tl.zeros_like(input_data)
    out = tl.maximum(input_data, zeros)
    tl.store(output_ptr + pid_0 * block_size + tl.arange(0, block_size), out)


def conv(x, filter_weights, bias_weights, kernel_size=KERNEL_SIZE, stride=1):
    x_c, x_h, x_w = x.shape
    w_out_c, w_in_c, w_h, w_w = filter_weights.shape
    # формула работает только в этом случае
    o_c, o_h, o_w = w_out_c, x_h - (kernel_size - stride), x_w - (kernel_size - stride)

    result = torch.empty((o_c, o_h, o_w))
    if os.environ.get("TRITON_INTERPRET") != "1":
        result = result.to("cuda")

    grid = lambda meta: (o_c, o_h, o_w)
    alloc_kernel_size = int(math.exp2(math.ceil(math.log2(kernel_size))))
    conv_kernel[grid](
        x,
        x_c,
        x_h,
        x_w,
        filter_weights,
        bias_weights,
        result,
        kernel_size=kernel_size,
        alloc_kernel_size=alloc_kernel_size,
        bias_size=o_c,
        bias_alloc_size=int(math.exp2(math.ceil(math.log2(o_c)))),
    )
    return result


def get_jopa(number):
    return int(math.exp2(math.ceil(math.log2(number))))


torch.manual_seed(123)
model = load_weights().to("cuda")
# model = ImageClassifier().to("cuda")
model.eval()


x = torch.rand((1, 28, 28)).to("cuda")
with torch.inference_mode():
    start = time.time()
    res_torch = model.conv_layers(x)
    print("naive torch: ", time.time() - start)

res_our = triton_infer(model, x)

start = time.time()
res_our = triton_infer(model, x)
print(time.time() - start)

model = torch.compile(model)
with torch.inference_mode():
    start = time.time()
    res_torch_speed = model.conv_layers(x)
    print("compile torch: ", time.time() - start)

print((res_our - res_torch).cpu().abs().max())
