// 第 8 章作业 1：分析用 MLIR（已验证：mlir-opt --canonicalize 通过）
func.func @test(%a: i32, %b: i32) -> i32 {
    %cond = arith.cmpi sgt, %a, %b : i32
    %max = scf.if %cond -> (i32) {
        scf.yield %a : i32
    } else {
        scf.yield %b : i32
    }
    func.return %max : i32
}
