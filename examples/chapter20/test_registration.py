#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_backend import backends

b = backends["mygpu"]
assert b.supports_target(type("T", (), {"backend": "mygpu"})())
assert b.binary_ext == "mybin"
print("OK: minimal backend registered")
