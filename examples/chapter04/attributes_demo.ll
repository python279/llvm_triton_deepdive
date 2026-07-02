; attributes_demo.ll — 函数属性示例

define i32 @add(i32 %a, i32 %b) nounwind readonly {
    %r = add i32 %a, %b
    ret i32 %r
}

declare i32 @printf(i8* nocapture, ...) nounwind

define i32 @main() {
    %r = call i32 @add(i32 1, i32 2)
    ret i32 %r
}
