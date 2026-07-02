func.func @test(%a: i32, %b: i32) -> i32 {
    %cond = arith.cmpi sgt, %a, %b : i32
    %max = scf.if %cond -> (i32) {
        scf.yield %a : i32
    } else {
        scf.yield %b : i32
    }
    func.return %max : i32
}
