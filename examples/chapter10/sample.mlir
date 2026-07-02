func.func @foo(%a: f32, %b: f32) -> f32 {
    %zero = arith.constant 0.0 : f32
    %sum = arith.addf %a, %zero : f32
    %out = arith.addf %sum, %b : f32
    func.return %out : f32
}
