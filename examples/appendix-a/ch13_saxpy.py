#!/usr/bin/env python3
"""第 13 章作业 1：SAXPY y = a * x + y（需 GPU + torch + triton）"""
import sys

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    print("SKIP: triton/torch not installed")
    sys.exit(0)


@triton.jit
def saxpy_kernel(y_ptr, a, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, a * x + y, mask=mask)


def saxpy(a, x, y):
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    saxpy_kernel[grid](y, a, x, n, BLOCK_SIZE=1024)
    return y


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    a = 2.5
    n = 2048
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    y_ref = a * x + y.clone()
    saxpy(a, x, y)
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
    print("OK: saxpy")


if __name__ == "__main__":
    main()
