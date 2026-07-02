#mygpu_blocked = #mygpu.blocked<{vec = 4}>

module {
  tt.func @dot_kernel() {
    %a = tt.load %ptr : tensor<16x16xf16, #mygpu_blocked>
    mygpu.matmul %a, %b : tensor<16x16xf16, #mygpu_blocked>
    tt.return
  }
}
