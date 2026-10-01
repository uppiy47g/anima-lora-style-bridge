"""Optional Triton kernel; imported only after dependency probing."""

import triton
import triton.language as tl


@triton.jit
def bf16_linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    input_block = tl.load(
        input_ptr + offsets_m[:, None] * stride_im + offsets_k[None, :] * stride_ik
    )
    weight_block = tl.load(
        weight_ptr + offsets_k[:, None] * stride_wk + offsets_n[None, :] * stride_wn
    )
    output_block = tl.dot(input_block, weight_block)
    tl.store(
        output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
        output_block,
    )


@triton.jit
def bf16_head_matmul_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ih: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    input_block = tl.load(
        input_ptr
        + HEAD * stride_ih
        + offsets_m[:, None] * stride_im
        + offsets_k[None, :] * stride_ik
    )
    weight_block = tl.load(
        weight_ptr
        + HEAD * stride_wh
        + offsets_k[:, None] * stride_wk
        + offsets_n[None, :] * stride_wn
    )
    output_block = tl.dot(input_block, weight_block)
    tl.store(
        output_ptr
        + HEAD * stride_oh
        + offsets_m[:, None] * stride_om
        + offsets_n[None, :] * stride_on,
        output_block,
    )
