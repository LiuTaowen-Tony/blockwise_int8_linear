import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

# This is a matmul kernel based on triton.ops.matmul
# It is modified to support rowwise quantized input and columnwise quantized weight
# It's purpose is fused matmul then dequantize
# It does support bias.
types = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.int32: tl.int32,
    torch.bfloat16: tl.bfloat16,
}
GROUP_SIZE = 128
SHARED_EXP_TORCH_TYPE = torch.float32
SHARED_EXP_TRITON_TYPE = types[SHARED_EXP_TORCH_TYPE]
ACCUMULATOR_TYPE = tl.float32
INPUT_OUTPUT_TORCH_TYPE = torch.float32

def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [1, 2, 3, 4, 5, 6]:
        for block_m in [16, 32, 64]:
            for block_k in [GROUP_SIZE]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [2, 4, 8]:
                        configs.append(
                            triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                            num_stages=num_stages, num_warps=num_warps))
                    for split_k in [2, 4, 8, 16]:
                        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                                        num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': GROUP_SIZE, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        *get_configs_io_bound(),
    ],
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _int8_matmul_block64_rowwise_dequantize(A, B, C, bias, state_x_ptr, state_w_ptr, M, N, K, divfactor, has_bias : tl.constexpr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
            ACC_TYPE: tl.constexpr
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    state_x_ptr = state_x_ptr + ram
    state_w_ptr = state_w_ptr + rbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for i in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - i * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        result = tl.dot(a, b).to(ACC_TYPE)
        
        # acc += result
        x_factor = tl.load(state_x_ptr)
        w_factor = tl.load(state_w_ptr)
        acc += (w_factor[None, :] * (x_factor[:, None] * (result * divfactor)))
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
        state_x_ptr += M
        state_w_ptr += N

    # acc = (acc * divfactor)
    acc = acc.to(C.dtype.element_ty)

    if has_bias:
        bias = tl.load(bias + rn, mask=rn < N, other=0).to(C.dtype.element_ty)
        acc = acc + bias[None, :]

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def int8_matmul_block64_rowwise_dequantize(a, b, state_x, state_w, bias=None):
    divfactor = 1. / (127. * 127.)

    has_bias = 0 if bias is None else 1

    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    print(a.shape, b.shape)
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=device, dtype=INPUT_OUTPUT_TORCH_TYPE)
    # accumulator types
    # launch int8_matmul_rowwise_dequantize kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    compiled = _int8_matmul_block64_rowwise_dequantize[grid](a, b, c, bias, state_x, state_w, M, N, K, divfactor, has_bias,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    GROUP_M=8, ACC_TYPE=ACCUMULATOR_TYPE)
    return c


@triton.autotune(
        configs=[
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
            triton.Config({}, num_stages=1, num_warps=2),
            triton.Config({}, num_stages=2, num_warps=2),
            triton.Config({}, num_stages=4, num_warps=2),
            triton.Config({}, num_stages=8, num_warps=2),
            triton.Config({}, num_stages=1, num_warps=4),
            triton.Config({}, num_stages=2, num_warps=4),
            triton.Config({}, num_stages=4, num_warps=4),
            triton.Config({}, num_stages=8, num_warps=4),
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _quantize_block_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    M: tl.constexpr,
    K: tl.constexpr,
    FBLOCK_SIZE: tl.constexpr,
    n_elements
):
    pid = tl.program_id(axis=0)
    block_start = pid * K
    offsets = block_start + tl.arange(0, FBLOCK_SIZE)
    output_maxs_offset = pid 
    
    for _ in range(0, tl.cdiv(K, FBLOCK_SIZE)):
        row_mask = offsets < block_start + K
        x = tl.load(x_ptr + offsets, mask=row_mask)
        abs_x = tl.abs(x)
        max_val = tl.max(tl.where(row_mask, abs_x, 0.), axis=0)
        output = tl.math.llrint(127. * (x / max_val))
        tl.store(output_ptr + offsets, output, mask=row_mask)
        tl.store(output_maxs + output_maxs_offset, max_val)
        offsets += FBLOCK_SIZE
        output_maxs_offset += M

def ceil_div(n, d):
    return -(n // -d)

def quantize_block_rowwise(x: torch.Tensor, fblock_size=GROUP_SIZE):
    m, k = x.shape

    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    # output_maxs is transposed
    output_maxs = torch.empty((ceil_div(k, fblock_size), m), device=x.device, dtype=SHARED_EXP_TORCH_TYPE)

    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (x.shape[0],)
    n_elements = output.numel()
    _quantize_block_rowwise[grid](x, output, output_maxs, M=m, K=k, FBLOCK_SIZE=fblock_size, n_elements=n_elements)
    return output, output_maxs


def groupwise_quantize(x, K):
    # Reshape x into (-1, K), assuming x.size(0) is divisible by K
    # Use view or reshape to adjust x to the shape (N/K, K)
    if x.numel() % K != 0:
        raise ValueError("The total number of elements in x must be divisible by K.")
    
    x_reshaped = x.view(-1, K)

    # Compute max values along the last dimension (K dimension)
    max_vals = x_reshaped.max(dim=1, keepdim=True)[0]

    # Normalize
    normalized_x = x_reshaped / max_vals

    # Quantize: scale normalized values to [0, 255] and convert to int8
    quantized_x = (normalized_x * 127).to(torch.int8)

    return quantized_x, max_vals



# TODO: autotune this better.
@triton.autotune(
        configs=[
            triton.Config({}, num_stages=1, num_warps=8),
            triton.Config({}, num_stages=2, num_warps=8),
            triton.Config({}, num_stages=4, num_warps=8),
            triton.Config({}, num_stages=8, num_warps=8),
            triton.Config({}, num_stages=1),
            triton.Config({}, num_stages=2),
            triton.Config({}, num_stages=4),
            triton.Config({}, num_stages=8),
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _dequantize_block_rowwise(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    K: tl.constexpr,
    M: tl.constexpr,
    FBLOCK_SIZE: tl.constexpr,
    n_elements,
):
    pid = tl.program_id(axis=0)
    block_start = pid * K
    offsets = block_start + tl.arange(0, FBLOCK_SIZE)
    output_maxs_offset = pid 
    
    for i in range(0, tl.cdiv(K, FBLOCK_SIZE)):
        row_mask = offsets < block_start + K
        x = tl.load(x_ptr + offsets, mask=row_mask)
        max_val = tl.load(state_x + output_maxs_offset)
        output = max_val * x * inv_127
        tl.store(output_ptr + offsets, output, mask=row_mask)
        offsets += FBLOCK_SIZE
        output_maxs_offset += M

def dequantize_block_rowwise(x: torch.Tensor, state_x: torch.Tensor, fblock_size=GROUP_SIZE):
    output = torch.empty(*x.shape, device=x.device, dtype=INPUT_OUTPUT_TORCH_TYPE)
    M, K = x.shape
    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (x.shape[0],)
    n_elements = output.numel()
    _dequantize_block_rowwise[grid](x, state_x, output, 1./127, K=K, M=M, FBLOCK_SIZE=fblock_size, n_elements=n_elements)
    return output


def copy(a):
    b = torch.clone(a)

# def quantise(a):
#     y, state_x = quantize_block_rowwise(a)



def fast_matmulT(a, b):
    a_int8, a_state = quantize_block_rowwise(a)
    b_int8, b_state = quantize_block_rowwise(b)
    return int8_matmul_block64_rowwise_dequantize(a_int8, b_int8.t(), a_state, b_state)

def copy_then_matmul(a, b):
    a = torch.clone(a)
    b = torch.clone(b)
    return torch.matmul(a, b)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[256 * i for i in range(2, 16, 2)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'blockwise_int8', 'blockwise_int8_quantise', 'cublas+copy'],
        # Label name for the lines
        line_names=["cuBLAS", "blockwise_int8", 'blockwise_int8_quantise', 'cuBLAS+copy'],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('black', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    print("here")
    a = torch.randn((M, K), device='cuda', dtype=INPUT_OUTPUT_TORCH_TYPE)
    b = torch.randn((K, N), device='cuda', dtype=INPUT_OUTPUT_TORCH_TYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas+copy':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: copy_then_matmul(a, b), quantiles=quantiles)
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'blockwise_int8':
        b = b.t().contiguous()
        a, state_x = quantize_block_rowwise(a)
        b, state_w = quantize_block_rowwise(b)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: int8_matmul_block64_rowwise_dequantize(a, b.t(), state_x, state_w), quantiles=quantiles)
    if provider == 'blockwise_int8_quantise':
        def bench_target(a, b):
                a_int8, a_state = quantize_block_rowwise(a)
                b_int8, b_state = quantize_block_rowwise(b)
                int8_matmul_block64_rowwise_dequantize(a_int8, b_int8.t(), a_state, b_state)
        b = b.t().contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: bench_target(a, b), quantiles=quantiles) 
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)
    return ms, max_ms, min_ms


if __name__ == "__main__":
    def block_matmul_is_close(w = 253, h = 257, k = 234, factor=1):
        a = torch.randn((w, k), device="cuda", dtype=INPUT_OUTPUT_TORCH_TYPE) * factor
        b = torch.randn((h, k), device="cuda", dtype=INPUT_OUTPUT_TORCH_TYPE) / factor
        a_int8, a_state = quantize_block_rowwise(a)
        b_int8, b_state = quantize_block_rowwise(b)
        c_hat = int8_matmul_block64_rowwise_dequantize(a_int8, b_int8.t(), a_state, b_state)
        c = torch.matmul(a, b.t())
        print(c)
        print(c_hat)
        print(torch.max(c - c_hat))

    # block_matmul_is_close()
    # block_matmul_is_close(129, 266, 75)
    # block_matmul_is_close(389, 138, 45)
    # block_matmul_is_close(389, 138, 45, 5)
    # block_matmul_is_close(389, 138, 45, 10)
    block_matmul_is_close(256, 512,512)
    block_matmul_is_close(1024, 512,512)

    # benchmark.run(show_plots=True, print_data=True)
    # quantise_benchmark.run(show_plots=True, print_data=True)
    # benchmark.run(show_plots=True, print_data=True)
    # benchmark.run(show_plots=True, print_data=True)
    #
# benchmark.run(show_plots=True, print_data=True)
# # benchmark.run(show_plots=True, print_data=True)






# @triton.autotune(
#         configs=[
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=1, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=2, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=4, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=8, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=1, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=2, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=4, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=8, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=1, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=2, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=4, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=8, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=1),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':16}, num_stages=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=1, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=2, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=4, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=8, num_warps=4),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=1, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=2, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=4, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=8, num_warps=2),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=1, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=2, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=4, num_warps=8),
#             # triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=8, num_warps=8),


#             triton.Config({'BLOCK_K':256, 'BLOCK_M':2}, num_stages=1),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':2}, num_stages=2),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':2}, num_stages=4),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':2}, num_stages=8),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=1),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=2),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=4),
#             triton.Config({'BLOCK_K':256, 'BLOCK_M':4}, num_stages=8),


#             triton.Config({'BLOCK_K':128, 'BLOCK_M':2}, num_stages=1),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':2}, num_stages=2),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':2}, num_stages=4),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':2}, num_stages=8),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=1),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=2),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=4),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=8),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=1),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=2),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=4),
#             triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=8),

#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=1, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=2, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=4, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=8, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=1, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=2, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=4, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=8, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=1, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=2, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=4, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':4}, num_stages=8, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=1, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=2, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=4, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=8, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=1, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=2, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=4, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=8, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=1, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=2, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=4, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':8}, num_stages=8, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=1, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=2, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=4, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=8, num_warps=4),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=1, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=2, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=4, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=8, num_warps=2),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=1, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=2, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=4, num_warps=8),
#             # triton.Config({'BLOCK_K':128, 'BLOCK_M':16}, num_stages=8, num_warps=8),
#         ],
#         key=['M', 'K']
# )
# @triton.jit
# def _quantize_blockwise(
#     x_ptr,
#     output_ptr,
#     output_maxs,
#     M: tl.constexpr,
#     K: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     FBLOCK_SIZE: tl.constexpr,
# ):
#     pid_m = tl.program_id(axis=0)
#     pid_k = tl.program_id(axis=1)
#     row_start = pid_m * K * BLOCK_M
#     ptr_start = row_start + pid_k * BLOCK_K
#     offsets = row_start + tl.arange(0, BLOCK_M)[None, :] * K + tl.arange(0, FBLOCK_SIZE)[:, None]
#     output_maxs_offset = pid_m + tl.arange(0, BLOCK_M)
    
#     for _ in range(0, tl.cdiv(BLOCK_K, FBLOCK_SIZE)):
#         x = tl.load(x_ptr + offsets)
#         abs_x = tl.abs(x)
#         # max_val = tl.max(tl.where(mask, abs_x, 0), axis=0)
#         max_val = tl.max(abs_x, axis=0).to(SHARED_EXP_TRITON_TYPE)
#         tl.static_assert(max_val.shape[0] == BLOCK_M)
#         output = tl.math.llrint(127. * (x / max_val))
#         tl.store(output_ptr + offsets, output)
#         tl.store(output_maxs + output_maxs_offset, max_val,)
#         offsets += FBLOCK_SIZE
#         output_maxs_offset += M


# def quantize_block(x: torch.Tensor, fblock_size=GROUP_SIZE):
#     m, k = x.shape

#     output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
#     # output_maxs is transposed
#     output_maxs = torch.empty(ceil_div(k, fblock_size), m, device=x.device, dtype=SHARED_EXP_TORCH_TYPE)

#     assert x.is_cuda and output.is_cuda
#     grid = lambda META: (triton.cdiv(m, META['BLOCK_M']), triton.cdiv(k, META['BLOCK_K']))
#     code = _quantize_blockwise[grid](x, output, output_maxs, M=m, K=k, FBLOCK_SIZE=fblock_size)
#     return output, output_maxs



# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
#         x_vals=[256 * i for i in range(2, 5)],  # Different possible values for `x_name`
#         line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
#         # Possible values for `line_arg`
#         line_vals=['copy', 'quantise', 'quantise_block'],
#         # Label name for the lines
#         line_names=["copy", "quantise", 'quantise_block'],
#         # Line styles
#         styles=[('green', '-'), ('blue', '-'), ('red', '-')],
#         ylabel="TFLOPS",  # Label name for the y-axis
#         plot_name="quantise-performance",  # Name for the plot, used also as a file name for saving the plot.
#         args={},
#     ))
# def quantise_benchmark(M, N, K, provider):
#     print("here")
#     print(M)
#     a = torch.randn((M, K), device='cuda', dtype=torch.float16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == 'copy':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: copy(a), quantiles=quantiles)
#     if provider == 'quantise':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantize_block_rowwise(a), quantiles=quantiles) 
#     if provider == 'quantise_block':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantize_block(a), quantiles=quantiles)
#     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#     # return perf(ms), perf(max_ms), perf(min_ms)
#     return ms, max_ms, min_ms

# # quantise_benchmark.run(show_plots=True, print_data=True)
