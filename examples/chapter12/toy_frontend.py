"""Toy 前端原型：生成等价 MLIR 文本（第 12 章）"""

def compile_toy(source: str) -> str:
    # 极简：def add(a,b): return a+b
    if "add" in source:
        return """func.func @add(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  func.return %0 : f32
}
"""
    raise ValueError("unsupported")


if __name__ == "__main__":
    mlir = compile_toy("def add(a,b): return a+b")
    print(mlir.strip())
    assert "arith.addf" in mlir
    print("OK: toy_frontend")
