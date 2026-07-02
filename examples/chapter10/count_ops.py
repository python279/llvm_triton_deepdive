#!/usr/bin/env python3
"""统计 MLIR 文件中 Operation 数量（第 10 章作业）"""
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
# 粗略统计：以 ` = dialect.` 或 `dialect.` 开头的 op 行
ops = re.findall(r"\b(\w+\.\w+)\b", text)
ops = [o for o in ops if "." in o and not o.startswith("func.")]
print(f"Total operations (approx): {len(set(ops))}")
