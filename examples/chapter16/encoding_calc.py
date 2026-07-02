#!/usr/bin/env python3
"""第 16 章作业：Blocked encoding 覆盖范围计算"""
# sizePerThread=[4], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]
size_per_thread = 4
threads_per_warp = 32
warps_per_cta = 4
cta_elements = size_per_thread * threads_per_warp * warps_per_cta
assert cta_elements == 512, cta_elements
print(f"OK: CTA covers {cta_elements} elements")
