"""第 14 章：最小内核 + IR dump（需 GPU）"""
import os
import sys

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    print("SKIP: triton/torch not installed")
    sys.exit(0)


@triton.jit
def simple_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x * 2, mask=mask)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
    n = 256
    x = torch.randn(n, device="cuda")
    simple_kernel[(1,)](x, n, BLOCK=256)
    print("OK: simple_kernel compiled")


if __name__ == "__main__":
    main()
