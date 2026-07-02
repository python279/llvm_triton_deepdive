"""第 13 章 vector add 示例（需 GPU + triton + torch）"""
import sys

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    print("SKIP: triton/torch not installed")
    sys.exit(0)


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    vector_add_kernel[grid](x, y, out, n, BLOCK=256)
    assert torch.allclose(out, x + y)
    print("OK: vector_add_kernel")


if __name__ == "__main__":
    main()
