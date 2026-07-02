#!/usr/bin/env python3
"""检查 NVIDIA 后端关键属性（第 19 章）"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
compiler = ROOT / "third_party/nvidia/backend/compiler.py"
assert compiler.exists(), compiler
text = compiler.read_text()
assert "cubin" in text
assert "add_stages" in text
print("OK: CUDABackend compiler.py contains cubin and add_stages")
