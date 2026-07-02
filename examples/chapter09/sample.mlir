module {
  mini.func @main(%a: f32, %b: f32) -> f32 {
    %0 = mini.add %a, %b : f32
    mini.return %0 : f32
  }
}
