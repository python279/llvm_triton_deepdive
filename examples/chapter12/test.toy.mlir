func.func @add(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    func.return %0 : f32
}
