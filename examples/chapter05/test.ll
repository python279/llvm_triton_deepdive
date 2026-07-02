define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}

define internal i32 @dead() {
  ret i32 0
}
